import datetime
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.load_data import load_data
from data.loaders import (
    FC_SCGraphDataset,
    FC_SCVectorDataset,
    FC_SC_vec_Dataset,
    custom_collate_fn,
)
from diffusion.ddpm import ddpm
from diffusion.ddpm_graph import ddpm_graph
from diffusion.dit_FiLM import dit_film
from diffusion.dit_cat import dit_cat
from diffusion.graph_encoder import SCGraphModel1D
from fine_tuning.LoRA import apply_lora_ditwcat
from guiding_model.predictor import LinearRegression, RidgeRegression
from utils.preprocessing.transformations import (
    gaussian_resample,
    upper_elements_to_symmetric_matrix,
    get_upper_diagonal_elements,
    upper_elements_to_symmetric_matrix_no_chan,
)
from vae.unet_vae import vae_unet

# ---------------------------------------------------------------------
# Central configuration (edit in-place; no CLI needed)
# ---------------------------------------------------------------------
CONFIG = {
    "experiment_id": "4min_full_data",
    "metadata_path": "../config/metadata_full.yaml",
    "predictors": "bc",          # compact selector: b=base, v=vae, c=corr
    "time_short": 4,              # FC_t window in minutes (1-7)
    "model_type": "fm",        # "graph" or "fm"
    "standardize": False,          # subtract mean / divide std of FC20
    "use_sc": True,
    "use_fct": True,
    "use_cov": True,
    "use_resample": False,        # gaussian resample SC
    "device": "cuda:2",
    "seed": 1,
    "num_workers": 0,
    "output_root": "/data/benjamin_project/diffusion_models/experiments/times",

    # VAE
    "vae": {
        "epochs": 200,
        "batch_size": 32,
        "lr": 3e-4,
        "patience": 15,
        "accumulation_steps": 2,
    },

    # Diffusion
    "diffusion": {
        "epochs": 200,
        "lr": 8e-5,
        "patience": 15,
        "accumulation_steps": 1,
        "use_scheduler": False,
        "batch_size": 64,
    },

    # Sampling
    "sampling": {
        "denoising_steps": 50,
        "eta": 0.0,
        "precision": "bf16",
        "slice_index": 0,  # set to None to sample all slices
        "n_samples_per_subject": 10,
        "chunk_size": 512,
        "decode_batch_size": 50,
        "batch_size": 512,
        "split": "test",  # train/val/test
    },

    # Ridge predictor + LoRA fine-tuning
    "ridge": {
        "ridge_grid": [float(x) for x in torch.logspace(-2, 5, steps=200)],
        "n_latent_samples": 10,
        "plot": True,
        "batch_size": 256,
    },
    "finetune": {
        "run_name": "config_19",   # key in config/lora_config.yaml
        "epochs": 4,
        "patience": 10,
        "use_scheduler": False,
        "evaluate_baseline": True,
        "debug_lora_grads": False,   # print LoRA grad norms each optimizer step
    },

    # Flags
    "skip_sampling": False,
    "skip_finetune": False,
}


def to_ns(obj):
    """Recursively convert dicts to SimpleNamespace for dot access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
    return obj


def ns_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, dict):
        return {k: ns_to_dict(v) for k, v in ns.items()}
    return ns


DEFAULT_OUTPUT_ROOT = Path("/data/benjamin_project/diffusion_models/experiments/times")
EPS = 1e-8


def metadata_suffix(metadata_path: str) -> str:
    stem = Path(metadata_path).stem.lower()
    return "_full" if stem.endswith("full") else ""


def compact_experiment_tag(experiment_id: str, max_len: int = 24) -> str:
    """Create a short filesystem-safe experiment tag for run folder names."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(experiment_id).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = "run"
    return cleaned[:max_len]


PREDICTOR_CODE_MAP = {"b": "base", "v": "vae", "c": "corr"}


def parse_predictor_spec(spec: str | None):
    token = (spec or "bvc").strip().lower()
    selected = []
    for ch in token:
        if ch.isspace() or ch == ",":
            continue
        if ch not in PREDICTOR_CODE_MAP:
            raise ValueError(
                f"Invalid predictor selector '{ch}' in '{spec}'. Use only b, v, c (e.g. 'bvc', 'b', 'c')."
            )
        name = PREDICTOR_CODE_MAP[ch]
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("Predictor selector is empty. Use at least one of b, v, c.")
    return selected


def _extract_lora_dims(lora_cfg: Dict) -> Dict[str, Dict[str, int | float]]:
    """Return per-block LoRA dims, supporting both new and deprecated configs.

    New configs provide a ``lora_dims`` section with separate ``r``/``alpha``
    values for each targeted block. Older configs only have ``r``/``alpha`` at
    the top level; in that case we reuse them for all four block types so the
    updated apply_lora_ditwcat API still works.
    """

    if "lora_dims" in lora_cfg:
        dims = lora_cfg["lora_dims"]
        if "proj_out" not in dims and "output_proj" in dims:
            dims = {**dims, "proj_out": dims["output_proj"]}
        return dims

    if "r" in lora_cfg and "alpha" in lora_cfg:
        shared = {"r": lora_cfg["r"], "alpha": lora_cfg["alpha"]}
        return {
            "attn_proj": shared,
            "output_proj": shared,
            "proj_out": shared,
            "mlp_block": shared,
            "adaln": shared,
        }

    raise KeyError("LoRA config must include 'lora_dims' or top-level 'r' and 'alpha'.")


def ensure_target_normalization(data: Dict) -> Dict[str, torch.Tensor]:
    """Compute and cache global target mean/std on the train split.

    The statistics are computed once (non-batch) using the available training
    targets, excluding NaNs. They are stored on the original ``data`` dict
    under ``"_target_norm"`` so all loaders share the same normalization.
    """

    if data.get("_target_norm"):
        return data["_target_norm"]

    y_train = data["target"]["train"]
    mask = ~torch.isnan(y_train)
    mean = y_train[mask].mean()
    std = y_train[mask].std().clamp_min(1e-6)

    data["_target_norm"] = {"mean": mean, "std": std}
    return data["_target_norm"]


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_key(time_short: int) -> str:
    mapping = {
        1: "FC_1",
        2: "FC_2",
        3: "FC3",
        4: "FC_4",
        5: "FC_5",
        6: "FC_6",
        7: "FC_7",
        13: "FC_13",
    }
    if time_short not in mapping:
        raise ValueError(f"Unsupported time_short={time_short}. Expected one of {list(mapping)}.")
    return mapping[time_short]


@dataclass
class ExperimentPaths:
    root: Path
    time_tag: str
    model_type: str
    vae: Path = field(init=False)
    latents: Path = field(init=False)
    diffusion: Path = field(init=False)
    generated: Path = field(init=False)
    finetuned_models: Path = field(init=False)
    finetuned_data: Path = field(init=False)
    prediction: Path = field(init=False)
    logs: Path = field(init=False)

    def __post_init__(self):
        self.vae = self.root / "vae"
        self.latents = self.root / "latents" / self.time_tag
        self.diffusion = self.root / "diffusion" / self.time_tag / self.model_type
        self.generated = self.root / "generated" / self.time_tag / self.model_type
        self.finetuned_models = self.root / "finetuned_models" / self.time_tag / self.model_type
        self.finetuned_data = self.root / "finetuned_data" / self.time_tag / self.model_type
        self.prediction = self.root / "prediction_models" / self.time_tag
        self.logs = self.root / "logs"

    def make_dirs(self):
        for p in [
            self.root,
            self.vae,
            self.latents,
            self.diffusion,
            self.generated,
            self.finetuned_models,
            self.finetuned_data,
            self.prediction,
            self.logs,
        ]:
            p.mkdir(parents=True, exist_ok=True)


def select_fc_t(data: Dict, time_short: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = time_key(time_short)
    return data[key]["train"], data[key]["val"], data[key]["test"]


def compute_scaling(fc20_train: torch.Tensor, standardize: bool):
    mean_fc = fc20_train.mean(0, keepdim=True)
    std_fc = fc20_train.std(0, keepdim=True) + EPS
    if standardize:
        return mean_fc, std_fc
    return torch.zeros_like(mean_fc), torch.ones_like(std_fc)


def prepare_fc_sets(data: Dict, time_short: int, standardize: bool):
    fc20_train, fc20_val, fc20_test = data["FC"]["train"], data["FC"]["val"], data["FC"]["test"]
    fct_train, fct_val, fct_test = select_fc_t(data, time_short)
    mean_fc, std_fc = compute_scaling(fc20_train, standardize)

    fc20_train_p = (fc20_train - mean_fc) / std_fc
    fc20_val_p = (fc20_val - mean_fc) / std_fc
    fc20_test_p = (fc20_test - mean_fc) / std_fc

    fct_train_p = (fct_train - mean_fc.unsqueeze(1)) / std_fc.unsqueeze(1)
    fct_val_p = (fct_val - mean_fc.unsqueeze(1)) / std_fc.unsqueeze(1)
    fct_test_p = (fct_test - mean_fc.unsqueeze(1)) / std_fc.unsqueeze(1)

    return (
        (fc20_train_p, fc20_val_p, fc20_test_p),
        (fct_train_p, fct_val_p, fct_test_p),
        mean_fc,
        std_fc,
    )


def build_vae_loaders(
    fc20_sets,
    fct_sets,
    sc_sets,
    cov_sets,
    device,
    batch_size: int,
    sc_size: int,
):
    (fc20_train, fc20_val, fc20_test) = fc20_sets
    (fct_train, fct_val, fct_test) = fct_sets
    sc_train, sc_val, sc_test = sc_sets
    cov_train, cov_val, cov_test = cov_sets

    shape = (-1, 1, sc_size, sc_size)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        FC_SCVectorDataset(
            fc20_train,
            sc_train,
            fct_train,
            cov_train,
            age_dim=126,
            log_transform=False,
            shape=shape,
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        FC_SCVectorDataset(
            fc20_val,
            sc_val,
            fct_val,
            cov_val,
            age_dim=126,
            log_transform=False,
            shape=shape,
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        FC_SCVectorDataset(
            fc20_test,
            sc_test,
            fct_test,
            cov_test,
            age_dim=126,
            log_transform=False,
            shape=shape,
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

def build_vae_loaders_export(
    fc20_sets,
    fct_sets,
    sc_sets,
    cov_sets,
    device,
    batch_size: int,
    sc_size: int,
):
    (fc20_train, fc20_val, fc20_test) = fc20_sets
    (fct_train, fct_val, fct_test) = fct_sets
    sc_train, sc_val, sc_test = sc_sets
    cov_train, cov_val, cov_test = cov_sets

    shape = (-1, 1, sc_size, sc_size)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        FC_SCVectorDataset(
            fc20_train,
            sc_train,
            fct_train,
            cov_train,
            age_dim=126,
            log_transform=False,
            shape=shape,
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        FC_SCVectorDataset(
            fc20_val,
            sc_val,
            fct_val,
            cov_val,
            age_dim=126,
            log_transform=False,
            shape=shape,
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        FC_SCVectorDataset(
            fc20_test,
            sc_test,
            fct_test,
            cov_test,
            age_dim=126,
            log_transform=False,
            shape=shape,
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def vector_to_matrix(vec, mean_mat, std_mat):
    mat = upper_elements_to_symmetric_matrix_no_chan(vec)
    return mat * std_mat + mean_mat


def export_latents(split_name, loader, vae, device, mean_mat, std_mat, save_dir: Path, tag: str):
    frob_records = []
    xt_truth = []
    for batch in loader:
        _, _, xt, _ = batch
        xt_truth.append(xt.cpu())
    xt_truth = torch.cat(xt_truth, dim=0)

    out = vae.get_embeddings_and_reconstructions(loader, device)

    torch.save(out["x0_embeddings"], save_dir / f"{split_name}_x0_embeddings_{tag}.pt")
    torch.save(out["x0_recons"], save_dir / f"{split_name}_x0_recons_{tag}.pt")
    torch.save(out["xt_embeddings"], save_dir / f"{split_name}_xt_embeddings_{tag}.pt")
    torch.save(out["xt_recons"], save_dir / f"{split_name}_xt_recons_{tag}.pt")

    if out["xt_recons"] is not None:
        xt_recon = out["xt_recons"]
        B_total, S, _, D = xt_recon.shape
        xt_recon_vec = xt_recon.reshape(B_total * S, D)
        xt_vec = xt_truth.reshape(B_total * S, D)

        real_mat = vector_to_matrix(xt_vec, mean_mat, std_mat)
        recon_mat = vector_to_matrix(xt_recon_vec, mean_mat, std_mat)

        frob = torch.norm(recon_mat - real_mat, dim=(1, 2))
        s_indices = torch.arange(S).repeat(B_total)
        frob_records.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "s_idx": s_indices.numpy(),
                    "frob_norm": frob.detach().cpu().numpy(),
                }
            )
        )

    if frob_records:
        frob_df = pd.concat(frob_records, ignore_index=True)
        frob_df.to_csv(save_dir / f"{split_name}_frob2_{tag}.csv", index=False)
        fig = px.box(
            frob_df, x="s_idx", y="frob_norm", color="s_idx", points="all", title=f"Frobenius norm: {split_name}"
        )
        fig.write_html(save_dir / f"{split_name}_frob2_{tag}.html")


def apply_conditioning_filters(latents, data, use_sc=True, use_fct=True, use_cov=True, cfg=None):

    def _zero_like(arr):
        return torch.zeros_like(arr) if torch.is_tensor(arr) else np.zeros_like(arr)

    key = time_key(cfg.time_short)


    filt_latents = {}
    for k, v in latents.items():
        if (not use_fct) and k.endswith("_xt"):
            filt_latents[k] = torch.zeros_like(v)
        else:
            filt_latents[k] = v

    filt_data = {"SC": {}, "Cov": {}, "target": data["target"], 'FC': data["FC"], key: data[key]}
    filt_data["target"] = {sp: data["target"][sp].clone() for sp in ["train","val","test"]}
    for split in ["train", "val", "test"]:
        sc = data["SC"][split]
        cov = data["Cov"][split]
        filt_data["SC"][split] = sc if use_sc else _zero_like(sc)
        filt_data["Cov"][split] = cov if use_cov else _zero_like(cov)
    return filt_latents, filt_data


def build_diffusion_loaders(
    data,
    latents,
    device,
    batch_size,
    num_workers,
    sc_shape,
    use_resample: bool,
    seed: int,
    paths: ExperimentPaths | None = None,
):
    loaders = {}
    pin_memory = device.type == "cuda"
    sc_train, sc_val, sc_test = data["SC"]["train"], data["SC"]["val"], data["SC"]["test"]
    # Resample SC at most once and reuse across all loaders. If paths provided,
    # persist resampled SC to disk so future reruns can reuse the exact draw.
    if use_resample:
        if not data.get("_sc_resampled", False):
            sc_train = gaussian_resample(sc_train, seed=seed)
            sc_val = gaussian_resample(sc_val, seed=seed)
            sc_test = gaussian_resample(sc_test, seed=seed)
            data["SC"]["train"], data["SC"]["val"], data["SC"]["test"] = sc_train, sc_val, sc_test
            data["_sc_resampled"] = True
            if paths is not None:
                out_dir = paths.root / "resampled_sc"
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(sc_train, out_dir / "train.pt")
                torch.save(sc_val, out_dir / "val.pt")
                torch.save(sc_test, out_dir / "test.pt")
        else:
            sc_train, sc_val, sc_test = data["SC"]["train"], data["SC"]["val"], data["SC"]["test"]

    loaders["graph_train"] = DataLoader(
        FC_SCGraphDataset(
            latents["train_x0"],
            sc_train,
            latents["train_xt"],
            data["Cov"]["train"],
            data["target"]["train"],
            age_dim=126,
            transform_sc=not use_resample,
            shape=sc_shape,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
    )
    loaders["graph_val"] = DataLoader(
        FC_SCGraphDataset(
            latents["val_x0"],
            sc_val,
            latents["val_xt"],
            data["Cov"]["val"],
            data["target"]["val"],
            age_dim=126,
            transform_sc=not use_resample,
            shape=sc_shape,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
    )

    loaders["fm_train"] = DataLoader(
        FC_SC_vec_Dataset(
            latents["train_x0"],
            sc_train,
            latents["train_xt"],
            data["Cov"]["train"],
            data["target"]["train"],
            age_dim=126,
            transform_sc=not use_resample,
            shape=sc_shape,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loaders["fm_val"] = DataLoader(
        FC_SC_vec_Dataset(
            latents["val_x0"],
            sc_val,
            latents["val_xt"],
            data["Cov"]["val"],
            data["target"]["val"],
            age_dim=126,
            transform_sc=not use_resample,
            shape=sc_shape,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loaders


def build_diffusion_model(model_type, diffusion_config, device):
    ddpm_cfg = diffusion_config["DDPM_config"]
    schedule = ddpm_cfg.get("schedule", "linear")
    vector_cl = (ddpm_cfg["vector_c"], ddpm_cfg["vector_l"])

    if model_type == "fm":
        network = dit_film(
            seq_len=diffusion_config["DIT_config_cat"]["seq_len"],
            seq_channels=diffusion_config["DIT_config_cat"]["seq_channels"],
            config=diffusion_config["DIT_config_film"],
        ).to(device)
        model = ddpm(
            network=network,
            n_steps=ddpm_cfg["n_steps"],
            min_beta=ddpm_cfg["min_beta"],
            max_beta=ddpm_cfg["max_beta"],
            schedule=schedule,
            device=device,
            vector_cl=vector_cl,
        ).to(device)
        ckpt_name = "ddpm_fm.pt"
    else:
        network = dit_cat(
            seq_len=diffusion_config["DIT_config_cat"]["seq_len"],
            seq_channels=diffusion_config["DIT_config_cat"]["seq_channels"],
            config=diffusion_config["DIT_config_cat"],
        ).to(device)
        graph_enc = SCGraphModel1D(args=diffusion_config["Graph_encoder_config"]).to(device)
        model = ddpm_graph(
            network=network,
            GraphEncoder=graph_enc,
            n_steps=ddpm_cfg["n_steps"],
            min_beta=ddpm_cfg["min_beta"],
            max_beta=ddpm_cfg["max_beta"],
            schedule=schedule,
            device=device,
            vector_cl=vector_cl,
        ).to(device)
        ckpt_name = "ddpm_graph.pt"
    return model, ckpt_name


def load_latent_tensors(latent_dir: Path, tag: str):
    latents = {}
    for split in ["train", "val", "test"]:
        latents[f"{split}_x0"] = torch.load(latent_dir / f"{split}_x0_embeddings_{tag}.pt", map_location="cpu")
        latents[f"{split}_xt"] = torch.load(latent_dir / f"{split}_xt_embeddings_{tag}.pt", map_location="cpu")
    return latents


def build_prediction_model(path: Path, device, corr_ker: bool = False):
    model = LinearRegression(in_features=4950, freeze=True, corr_ker=corr_ker)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def _prediction_mask(y: torch.Tensor, split: str) -> torch.Tensor:
    mask = ~torch.isnan(y)
    valid = int(mask.sum().item())
    if valid == 0:
        raise ValueError(f"No valid prediction targets available for split='{split}' after masking.")
    return mask


def build_finetune_loader(cfg, data, latents, device, sc_shape, split="train", batch_size=4, paths: ExperimentPaths | None = None):
    # Resample SC only once for all splits; reuse thereafter. If previously
    # saved resamples exist on disk, load and reuse them to match prior runs.
    if cfg.use_resample and not data.get("_sc_resampled", False):
        if paths is not None:
            resampled_dir = paths.root / "resampled_sc"
            train_p, val_p, test_p = resampled_dir / "train.pt", resampled_dir / "val.pt", resampled_dir / "test.pt"
            if train_p.exists() and val_p.exists() and test_p.exists():
                data["SC"]["train"] = torch.load(train_p, map_location="cpu")
                data["SC"]["val"] = torch.load(val_p, map_location="cpu")
                data["SC"]["test"] = torch.load(test_p, map_location="cpu")
                data["_sc_resampled"] = True
        if not data.get("_sc_resampled", False):
            for sp in ["train", "val", "test"]:
                data["SC"][sp] = gaussian_resample(data["SC"][sp], seed=cfg.seed)
            data["_sc_resampled"] = True
    print(data.keys())
    fc20_sets, fct_sets, mean_fc, std_fc = prepare_fc_sets(data, cfg.time_short, cfg.standardize)
    if split == "train":
        fc20, _ = get_upper_diagonal_elements(fc20_sets[0].unsqueeze(1)), fct_sets[0]
    elif split == "val":
        fc20, _ = get_upper_diagonal_elements(fc20_sets[1].unsqueeze(1)), fct_sets[1]
    else:        
        fc20, _ = get_upper_diagonal_elements(fc20_sets[2].unsqueeze(1)), fct_sets[2]

    x0 = latents[f"{split}_x0"]
    xt = latents[f"{split}_xt"]
    sc = data["SC"][split]
    cov = data["Cov"][split]
    y = data["target"][split]
    real_y = data.get("target_real", {}).get(split, y)
    real_fc20 = data.get("FC", {}).get(split, None)
    mask = _prediction_mask(y, split=split)
    pin_memory = device.type == "cuda"

    if cfg.model_type == "graph":
        dataset = FC_SCGraphDataset(
            fc20[mask],
            sc[mask],
            xt[mask],
            cov[mask],
            y[mask],
            age_dim=126,
            transform_sc=(not cfg.use_resample),
            shape=sc_shape,
        )
        dataset.real_target = real_y[mask]
        dataset.real_fc20 = real_fc20[mask] if real_fc20 is not None else None
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=custom_collate_fn, pin_memory=pin_memory)

    dataset = FC_SC_vec_Dataset(
        fc20[mask],
        sc[mask],
        xt[mask],
        cov[mask],
        y[mask],
        age_dim=126,
        transform_sc=(not cfg.use_resample),
        shape=sc_shape,
    )
    dataset.real_target = real_y[mask]
    dataset.real_fc20 = real_fc20[mask] if real_fc20 is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), pin_memory=pin_memory)


def decode_samples(vae, samples, device, n_samples_per_subject: int, sc_size: int, batch_size: int):
    expanded = samples.reshape(-1, 4, 618)
    decoded = []
    with torch.no_grad():
        for i in range(0, len(expanded), batch_size):
            chunk = expanded[i : i + batch_size]
            dec = vae.decode(chunk.to(device))
            mats = upper_elements_to_symmetric_matrix(dec).reshape(-1, n_samples_per_subject, sc_size, sc_size)
            decoded.append(mats)
    return torch.cat(decoded, dim=0)


def load_trained_vae(cfg, paths: ExperimentPaths, device):
    with open("../config/vae_config.yaml", "r") as f:
        vae_config = yaml.safe_load(f)
    vae = vae_unet(im_channels=1, model_config=vae_config["VAE_params"]).to(device)
    ckpt = paths.vae / f"vae_{paths.time_tag}_{cfg.scale_tag}.pt"
    vae.load_state_dict(torch.load(ckpt, map_location=device))
    vae.eval()
    return vae


def train_ridge_models(
    cfg,
    data,
    paths: ExperimentPaths,
    device,
    sc_size: int,
    fc20_sets,
    fct_sets,
    mean_fc,
    std_fc,
    predictor_keys=None,
):
    selected_predictors = predictor_keys or ["base", "vae", "corr"]
    (fc20_train, fc20_val, _ ) = fc20_sets
    (fct_train, fct_val, _ ) = fct_sets
    sc_train, sc_val = data["SC"]["train"], data["SC"]["val"]
    cov_train, cov_val = data["Cov"]["train"], data["Cov"]["val"]
    y_train, y_val = data["target"]["train"], data["target"]["val"]

    mask_tr = ~torch.isnan(y_train)
    mask_va = ~torch.isnan(y_val)

    shape = (-1, 1, sc_size, sc_size)
    if cfg.use_resample:
        sc_train = gaussian_resample(sc_train, seed=cfg.seed)
        sc_val = gaussian_resample(sc_val, seed=cfg.seed)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        FC_SCGraphDataset(
            fc20_train[mask_tr],
            sc_train[mask_tr],
            fct_train[mask_tr],
            cov_train[mask_tr],
            y_train[mask_tr],
            age_dim=126,
            transform_sc= (not cfg.use_resample),
            shape=shape,
        ),
        batch_size=cfg.ridge.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        FC_SCGraphDataset(
            fc20_val[mask_va],
            sc_val[mask_va],
            fct_val[mask_va],
            cov_val[mask_va],
            y_val[mask_va],
            age_dim=126,
            transform_sc=(not cfg.use_resample),
            shape=shape,
        ),
        batch_size=cfg.ridge.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
    )

    vae = load_trained_vae(cfg, paths, device) if "vae" in selected_predictors else None

    def _metrics_from_preds(preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.reshape(-1).float()
        targets = targets.reshape(-1).float()
        mse = F.mse_loss(preds, targets).item()
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = float("nan") if ss_tot.item() <= 0 else (1.0 - (ss_res / ss_tot).item())
        pred_std = preds.std(unbiased=False)
        targ_std = targets.std(unbiased=False)
        corr = float("nan")
        if pred_std.item() > 0 and targ_std.item() > 0:
            corr = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
        return mse, r2, corr

    paths.prediction.mkdir(parents=True, exist_ok=True)
    ridge_ckpts = {}
    for predictor_type in selected_predictors:
        use_vae_noise_aug = predictor_type == "vae"
        use_corr_ker = predictor_type == "corr"
        ridge_plot = paths.logs / f"ridge_val_{predictor_type}_{cfg.experiment_id}_{paths.time_tag}_{cfg.scale_tag}.png"

        ridge_model = RidgeRegression(
            ridge_grid=cfg.ridge.ridge_grid,
            device=device,
            plot=cfg.ridge.plot,
            save_path=ridge_plot,
            use_upper_triangle=True,
            corr_ker=use_corr_ker,
        ).to(device)
        ridge_model.fit(
            train_loader,
            val_loader,
            vae=vae,
            use_vae_noise_aug=use_vae_noise_aug,
            n_latent_samples=cfg.ridge.n_latent_samples,
            device=device,
        )
        val_preds, val_targets = ridge_model.predict(
            val_loader,
            return_y=True,
            vae=vae if use_vae_noise_aug else None,
            use_vae_noise_aug=use_vae_noise_aug,
            n_latent_samples=cfg.ridge.n_latent_samples,
            device=device,
        )
        mse, r2, corr = _metrics_from_preds(val_preds, val_targets)
        print(f"[Ridge val metrics][{predictor_type}] mse={mse:.6f}, r2={r2:.6f}, corr={corr:.6f}")

        beta = ridge_model.W.detach().cpu().squeeze()
        intercept = ridge_model.bias.detach().cpu().squeeze()
        linear_model = LinearRegression(
            in_features=beta.numel(),
            beta_vector=beta,
            intercept=intercept,
            freeze=True,
            corr_ker=use_corr_ker,
        )
        pred_path = paths.prediction / f"ridge_linear_{predictor_type}_{paths.time_tag}_{cfg.scale_tag}.pth"
        torch.save(linear_model.state_dict(), pred_path)
        ridge_ckpts[predictor_type] = pred_path
    return ridge_ckpts


def train_and_export_vae(cfg, data, paths: ExperimentPaths, device, sc_size: int, fc20_sets, fct_sets, mean_fc, std_fc) -> Path:
    loaders = build_vae_loaders(
        fc20_sets,
        fct_sets,
        (data["SC"]["train"], data["SC"]["val"], data["SC"]["test"]),
        (data["Cov"]["train"], data["Cov"]["val"], data["Cov"]["test"]),
        device,
        cfg.vae.batch_size,
        sc_size,
    )
    train_loader, val_loader, test_loader = loaders

    with open("../config/vae_config.yaml", "r") as f:
        vae_config = yaml.safe_load(f)

    vae = vae_unet(im_channels=1, model_config=vae_config["VAE_params"]).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.vae.lr)

    ckpt_path = paths.vae / f"vae_{paths.time_tag}_{cfg.scale_tag}.pt"
    vae.train_vae_extended(
        loader=train_loader,
        loader_val=val_loader,
        n_epochs=cfg.vae.epochs,
        optim=optimizer,
        device=device,
        beta=float(vae_config["VAE_params"]["beta"]),
        patience=cfg.vae.patience,
        use_scheduler=True,
        accumulation_steps=cfg.vae.accumulation_steps,
        store_path=str(ckpt_path),
        plot_title=f"VAE train/val loss - {cfg.experiment_id} | {paths.time_tag} | {cfg.model_type} | {cfg.scale_tag}",
        plot_filename=paths.logs / f"loss_{cfg.experiment_id}_{paths.time_tag}_{cfg.model_type}_{cfg.scale_tag}.png",
    )

    # reload best checkpoint before exporting latents
    vae.load_state_dict(torch.load(ckpt_path, map_location=device))
    vae.eval()

    mean_mat = torch.as_tensor(mean_fc, dtype=torch.float32).squeeze(0)
    std_mat = torch.as_tensor(std_fc, dtype=torch.float32).squeeze(0)

    loaders_export = build_vae_loaders_export(
        fc20_sets,
        fct_sets,
        (data["SC"]["train"], data["SC"]["val"], data["SC"]["test"]),
        (data["Cov"]["train"], data["Cov"]["val"], data["Cov"]["test"]),
        device,
        cfg.vae.batch_size,
        sc_size,
    )

    for split_name, loader in zip(["train", "val", "test"], loaders_export):
        export_latents(split_name, loader, vae, device, mean_mat, std_mat, paths.latents, cfg.scale_tag)

    return ckpt_path


def train_diffusion(cfg, data, paths: ExperimentPaths, device, sc_size: int):
    with open("../config/diffusion_config.yaml", "r") as f:
        diffusion_config = yaml.safe_load(f)

    latents = load_latent_tensors(paths.latents, cfg.scale_tag)
    latents, data = apply_conditioning_filters(
        latents=latents,
        data=data,
        use_sc=cfg.use_sc,
        use_fct=cfg.use_fct,
        use_cov=cfg.use_cov,
        cfg=cfg,
    )
    target_norm = ensure_target_normalization(data)

    sc_shape = (-1, 1, sc_size, sc_size)
    loaders = build_diffusion_loaders(
        data=data,
        latents=latents,
        device=device,
        batch_size=cfg.diffusion.batch_size,
        num_workers=cfg.num_workers,
        sc_shape=sc_shape,
        use_resample=cfg.use_resample,
        seed=cfg.seed,
        paths=paths,
    )

    model, ckpt_name = build_diffusion_model(cfg.model_type, diffusion_config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.diffusion.lr)

    paths.diffusion.mkdir(parents=True, exist_ok=True)
    store_path = paths.diffusion / ckpt_name

    train_loader = loaders["graph_train"] if cfg.model_type == "graph" else loaders["fm_train"]
    val_loader = loaders["graph_val"] if cfg.model_type == "graph" else loaders["fm_val"]

    model.train_ddpm_amp(
        loader=train_loader,
        loader_val=val_loader,
        n_epochs=cfg.diffusion.epochs,
        optimizer=optimizer,
        patience=cfg.diffusion.patience,
        accumulation_steps=cfg.diffusion.accumulation_steps,
        use_scheduler=cfg.diffusion.use_scheduler,
        debug=False,
        store_path=str(store_path),
    )

    # reload best checkpoint before returning
    model.load_state_dict(torch.load(store_path, map_location=device))
    model.eval()

    return model, store_path, sc_shape


def sample_diffusion(cfg, model, data, paths: ExperimentPaths, device, sc_shape, split: str):
    latents = load_latent_tensors(paths.latents, cfg.scale_tag)
    latents, data = apply_conditioning_filters(
        latents=latents,
        data=data,
        use_sc=cfg.use_sc,
        use_fct=cfg.use_fct,
        use_cov=cfg.use_cov,
        cfg = cfg,
    )

    sc_size = sc_shape[-1]
    data_split = split
    #if cfg.use_resample:
    #    data["SC"][data_split] = gaussian_resample(data["SC"][data_split], seed=cfg.seed)
    if cfg.use_resample and not data.get("_sc_resampled", False):
        if paths is not None:
            resampled_dir = paths.root / "resampled_sc"
            train_p, val_p, test_p = resampled_dir / "train.pt", resampled_dir / "val.pt", resampled_dir / "test.pt"
            if train_p.exists() and val_p.exists() and test_p.exists():
                data["SC"]["train"] = torch.load(train_p, map_location="cpu")
                data["SC"]["val"] = torch.load(val_p, map_location="cpu")
                data["SC"]["test"] = torch.load(test_p, map_location="cpu")
                data["_sc_resampled"] = True
        if not data.get("_sc_resampled", False):
            for sp in ["train", "val", "test"]:
                data["SC"][sp] = gaussian_resample(data["SC"][sp], seed=cfg.seed)
            data["_sc_resampled"] = True

    y = data["target"][data_split]
    mask = _prediction_mask(y, split=data_split)
    x0 = latents[f"{data_split}_x0"][mask]
    xt = latents[f"{data_split}_xt"][mask]
    sc = data["SC"][data_split][mask]
    cov = data["Cov"][data_split][mask]
    y_masked = y[mask]

    if cfg.model_type == "graph":
        dataset = FC_SCGraphDataset(
            x0,
            sc,
            xt,
            cov,
            y_masked,
            age_dim=126,
            transform_sc= (not cfg.use_resample),
            shape=sc_shape,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.sampling.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=(device.type == "cuda"),
        )
    else:
        dataset = FC_SC_vec_Dataset(
            x0,
            sc,
            xt,
            cov,
            y_masked,
            age_dim=126,
            transform_sc= (not cfg.use_resample),
            shape=sc_shape,
        )
        loader = DataLoader(dataset, batch_size=cfg.sampling.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    vae = load_trained_vae(cfg, paths, device)

    out_dir = paths.generated
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if cfg.model_type == "graph":
                _, cond1, cond2, cov, _ = batch
            else:
                cond1, cond2, cov = batch[1], batch[2], batch[3]

            samples = model.sample_repeated_chunked_ddim(
                cond1_data=cond1.to(device),
                cond2=cond2.to(device),
                cov_cond=cov.to(device),
                denoising_steps=cfg.sampling.denoising_steps,
                eta=cfg.sampling.eta,
                n=cfg.sampling.n_samples_per_subject,
                chunk_size=cfg.sampling.chunk_size,
                amp=True,
                precision=getattr(cfg.sampling, "precision", "bf16"),
                grad=False,
                slice_index=getattr(cfg.sampling, "slice_index", 0),
            )
            all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)
    recon = decode_samples(
        vae,
        all_samples,
        device,
        n_samples_per_subject=cfg.sampling.n_samples_per_subject,
        sc_size=sc_size,
        batch_size=cfg.sampling.decode_batch_size,
    )

    out_path = out_dir / f"FC_gen_{cfg.model_type}_{paths.time_tag}_{cfg.scale_tag}_{cfg.sampling.split}.pt"
    out_path = out_dir / f"FC_gen_{cfg.model_type}_{paths.time_tag}_{cfg.scale_tag}_{split}.pt"
    torch.save(recon.cpu(), out_path)
    return out_path


def _predict_targets_from_fc(fc20_sets, predictor, device):
    """Use FC20 matrices to produce scalar targets with the given predictor.

    Args:
        fc20_sets: tuple(train, val, test) where each is (B, 100, 100).
        predictor: model with ``predict_from_fc_tensor``.
        device: torch device for inference.
    Returns:
        dict split -> predicted targets on CPU.
    """
    fc_map = {"train": fc20_sets[0], "val": fc20_sets[1], "test": fc20_sets[2]}
    out = {}
    for split, fc in fc_map.items():
        with torch.no_grad():
            preds = predictor.predict_from_fc_tensor(fc.to(device))
        out[split] = preds.detach().cpu()
    return out


def finetune_diffusion(
    cfg,
    data,
    paths: ExperimentPaths,
    device,
    sc_size: int,
    predictor_path: Path,
    predictor_type: str,
    fc20_sets=None,
):
    # Backfill finetune flags for snapshots/scripts that predate them
    if not hasattr(cfg.finetune, "use_scheduler"):
        cfg.finetune.use_scheduler = True
    if not hasattr(cfg.finetune, "evaluate_baseline"):
        cfg.finetune.evaluate_baseline = True
    if not hasattr(cfg.finetune, "debug_lora_grads"):
        cfg.finetune.debug_lora_grads = False
    with open("../config/diffusion_config.yaml", "r") as f:
        diffusion_config = yaml.safe_load(f)
    with open("../config/lora_config.yaml", "r") as f:
        lora_all = yaml.safe_load(f)
    if cfg.finetune.run_name not in lora_all:
        raise KeyError(f"RUN_NAME '{cfg.finetune.run_name}' not found in lora_config.yaml")
    lora_cfg = lora_all[cfg.finetune.run_name]

    print(f"Lora config: {lora_cfg}")
    print(f'Data keys before everything changing {data.keys()}')

    latents = load_latent_tensors(paths.latents, cfg.scale_tag)


    latents, data = apply_conditioning_filters(
        latents=latents,
        data=data,
        use_sc=cfg.use_sc,
        use_fct=cfg.use_fct,
        use_cov=cfg.use_cov,
        cfg = cfg
    )

    prediction_model = build_prediction_model(
        Path(predictor_path),
        device,
        corr_ker=(predictor_type == "corr"),
    )
    print(f'Data keys before changing {data.keys()}')
    data["target_real"] = {sp: data["target"][sp].clone() for sp in ["train", "val", "test"]}
    # Optionally replace targets with predictor outputs from FC20
    if fc20_sets is None:
        print("No FC20 sets provided; using original targets for finetuning.")
    if not lora_cfg.get("use_fc20_as_target", False):
        print("Using original targets for finetuning (not FC20-based predictions), as per config.")
    if fc20_sets is not None and lora_cfg.get("use_fc20_as_target", False):
        print("Using FC20-based predictions as finetune targets...")
        pred_targets = _predict_targets_from_fc(fc20_sets, prediction_model, device)
        for split, preds in pred_targets.items():
            assert preds.shape[0] == data["target"][split].shape[0]
            # keep shape consistent (B,) or (B,1) both accepted downstream
            data["target"][split] = preds

    target_norm = ensure_target_normalization(data)
    print(f'Data keys after changing {data.keys()}')
    sc_shape = (-1, 1, sc_size, sc_size)
    train_loader = build_finetune_loader(
        cfg, data, latents, device, sc_shape, split="train", batch_size=lora_cfg["batch_size"], paths=paths
    )
    val_loader = build_finetune_loader(
        cfg, data, latents, device, sc_shape, split="val", batch_size=lora_cfg["batch_size"], paths=paths
    )

    model, _ = build_diffusion_model(cfg.model_type, diffusion_config, device)
    model.load_state_dict(torch.load(paths.diffusion / ("ddpm_fm.pt" if cfg.model_type == "fm" else "ddpm_graph.pt"), map_location=device))
    lora_dims = _extract_lora_dims(lora_cfg)
    apply_lora_ditwcat(
        model,
        dims=lora_dims,
        include_mlp=lora_cfg.get("include_mlp", True),
        include_ln=lora_cfg.get("include_ln", True),
    )
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    vae = load_trained_vae(cfg, paths, device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lora_cfg["lr"],
        weight_decay=lora_cfg["weight_decay"],
    )

    steps_per_epoch = len(train_loader)
    accumulation_steps = lora_cfg["accumulation_steps"]
    updates_per_epoch = max(1, steps_per_epoch // accumulation_steps)
    effective_batch = lora_cfg["batch_size"] * accumulation_steps
    total_updates = updates_per_epoch * cfg.finetune.epochs
    warmup_steps = max(1, int(0.1 * total_updates))  # 5% of total optimizer steps
    warmup_iters = warmup_steps  # scheduler expects optimizer-step count
    steps_log = max(1, updates_per_epoch // 5)

    # --- Finetuning run sheet (shown once per predictor) ---
    print("\n=== Finetune Parameters ===")
    print(f"predictor          : {predictor_type}")
    print(f"train set size     : {len(train_loader.dataset)} examples")
    print(f"micro-batch size   : {lora_cfg['batch_size']}")
    print(f"grad accumulation  : {accumulation_steps} steps -> effective batch {effective_batch}")
    print(f"optimizer updates  : {updates_per_epoch} per epoch, {total_updates} total")
    print(f"log/val frequency  : every {steps_log} optimizer steps (~{steps_log * effective_batch} samples)")
    print(f"warmup             : {warmup_steps} optimizer steps")

    model.train()
    model.fine_tune_DRAFT_prediction(
        loader=train_loader,
        loader_val=val_loader,
        n_epochs=cfg.finetune.epochs,
        optimizer=optimizer,
        decoder=vae,
        guide_model=prediction_model,
        denoising_steps=lora_cfg["denoising_steps"],
        K=lora_cfg["K"],
        patience=cfg.finetune.patience,
        accumulation_steps=lora_cfg["accumulation_steps"],
        lambd=lora_cfg["lambd"],
        warmup_iters=warmup_iters,
        LV=lora_cfg["LV"],
        n_rep=lora_cfg["n_rep"],
        L1=lora_cfg["L1"],
        include_diff_loss=lora_cfg["include_diff_loss"],
        m=lora_cfg["m"],
        steps_log=steps_log,
        use_scheduler=cfg.finetune.use_scheduler,
        evaluate_baseline=cfg.finetune.evaluate_baseline,
        debug=cfg.finetune.debug_lora_grads,
        target_norm=target_norm,
        use_fc20_as_target=bool(lora_cfg.get("use_fc20_as_target", False)),
        store_path=str(
            paths.finetuned_models
            / f"{cfg.finetune.run_name}_finetuned_{cfg.model_type}_{predictor_type}_{paths.time_tag}.pt"
        ),
    )

    # reload best checkpoint to ensure subsequent sampling uses best epoch
    model.load_state_dict(
        torch.load(
            paths.finetuned_models / f"{cfg.finetune.run_name}_finetuned_{cfg.model_type}_{predictor_type}_{paths.time_tag}.pt",
            map_location=device,
        )
    )
    model.eval()


def sample_finetuned(cfg, data, paths: ExperimentPaths, device, sc_size: int, predictor_type: str, split: str):
    with open("../config/diffusion_config.yaml", "r") as f:
        diffusion_config = yaml.safe_load(f)
    with open("../config/lora_config.yaml", "r") as f:
        lora_all = yaml.safe_load(f)
    lora_cfg = lora_all[cfg.finetune.run_name]

    latents = load_latent_tensors(paths.latents, cfg.scale_tag)
    latents, data = apply_conditioning_filters(
        latents=latents,
        data=data,
        use_sc=cfg.use_sc,
        use_fct=cfg.use_fct,
        use_cov=cfg.use_cov,
        cfg = cfg,
    )

    sc_shape = (-1, 1, sc_size, sc_size)
    loader = build_finetune_loader(
        cfg, data, latents, device, sc_shape, split=split, batch_size=cfg.sampling.decode_batch_size
        , paths=paths
    )

    model, _ = build_diffusion_model(cfg.model_type, diffusion_config, device)
    lora_dims = _extract_lora_dims(lora_cfg)
    apply_lora_ditwcat(
        model,
        dims=lora_dims,
        include_mlp=lora_cfg.get("include_mlp", True),
        include_ln=lora_cfg.get("include_ln", True),
    )
    model.load_state_dict(
        torch.load(
            paths.finetuned_models / f"{cfg.finetune.run_name}_finetuned_{cfg.model_type}_{predictor_type}_{paths.time_tag}.pt",
            map_location=device,
        )
    )
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    vae = load_trained_vae(cfg, paths, device)

    all_samples = []
    with torch.no_grad():
        for batch in loader:
            if cfg.model_type == "graph":
                _, cond1, cond2, cov, _ = batch
            else:
                cond1, cond2, cov = batch[1], batch[2], batch[3]

            samples = model.sample_repeated_chunked_ddim(
                cond1_data=cond1.to(device),
                cond2=cond2.to(device),
                cov_cond=cov.to(device),
                denoising_steps=cfg.sampling.denoising_steps,
                eta=cfg.sampling.eta,
                n=cfg.sampling.n_samples_per_subject,
                chunk_size=cfg.sampling.chunk_size,
                amp=True,
                precision=getattr(cfg.sampling, "precision", "bf16"),
                grad=False,
                slice_index=getattr(cfg.sampling, "slice_index", 0),
            )
            all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)
    recon = decode_samples(
        vae,
        all_samples,
        device,
        n_samples_per_subject=cfg.sampling.n_samples_per_subject,
        sc_size=sc_size,
        batch_size=cfg.sampling.decode_batch_size,
    )

    out_dir = paths.finetuned_data
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / f"FC_gen_{cfg.finetune.run_name}_{predictor_type}_{cfg.model_type}_{paths.time_tag}_{split}.pt"
    )
    torch.save(recon.cpu(), out_path)
    return out_path


def snapshot_config(cfg_ns, paths: ExperimentPaths, meta: Dict = None):
    snapshot = ns_to_dict(cfg_ns)
    if meta:
        snapshot.update(meta)
    snapshot["created_at"] = datetime.datetime.now().isoformat()
    snapshot_path = paths.root / "config_snapshot.yaml"
    with open(snapshot_path, "w") as f:
        yaml.safe_dump(snapshot, f)
    return snapshot_path


def main():
    cfg = to_ns(CONFIG)
    device = torch.device(cfg.device)
    if not torch.cuda.is_available():
        print("⚠️ CUDA not detected; training may be very slow.")

    set_seed(cfg.seed)
    cfg.scale_tag = "scaled" if cfg.standardize else "raw"
    time_tag = f"{cfg.time_short}min"
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    md_suffix = metadata_suffix(cfg.metadata_path)
    run_tag = compact_experiment_tag(cfg.experiment_id)
    legacy_run_name = f"{cfg.experiment_id}_{time_tag}{md_suffix}_{stamp}"
    run_root = Path(cfg.output_root) / f"{run_tag}_{stamp}"
    paths = ExperimentPaths(root=run_root, time_tag=time_tag, model_type=cfg.model_type)
    paths.make_dirs()
    snapshot_config(
        cfg,
        paths,
        meta={
            "time_tag": time_tag,
            "run_stamp": stamp,
            "scale_tag": cfg.scale_tag,
            "legacy_run_name": legacy_run_name,
        },
    )

    print("\n=== FULL CONFIG (resolved) ===")
    print(yaml.safe_dump(ns_to_dict(cfg), sort_keys=False, default_flow_style=False))

    data = load_data(metadata_path=cfg.metadata_path)
    sc_size = data["SC"]["train"].shape[-1]
    fc20_sets, fct_sets, mean_fc, std_fc = prepare_fc_sets(data, cfg.time_short, cfg.standardize)
    predictor_keys = parse_predictor_spec(getattr(cfg, "predictors", "bvc"))

    print("\n=== RUN CONFIG ===")
    print(f"root: {paths.root}")
    print(f"metadata: {cfg.metadata_path}")
    print(f"time: {cfg.time_short} min | model: {cfg.model_type} | scale: {cfg.scale_tag} | seed: {cfg.seed}")
    print(f"use_sc={cfg.use_sc}, use_fct={cfg.use_fct}, use_cov={cfg.use_cov}, use_resample={cfg.use_resample}")

    print("\n[1/6] Training VAE + exporting latents ...")
    vae_ckpt = train_and_export_vae(cfg, data, paths, device, sc_size, fc20_sets, fct_sets, mean_fc, std_fc)
    print(f"[1/6] Done VAE. Saved: {vae_ckpt}")

    print(f"\n[2/6] Training ridge predictors ({', '.join(predictor_keys)}) ...")
    ridge_ckpts = train_ridge_models(
        cfg,
        data,
        paths,
        device,
        sc_size,
        fc20_sets,
        fct_sets,
        mean_fc,
        std_fc,
        predictor_keys=predictor_keys,
    )
    print(f"[2/6] Ridge saved -> {ridge_ckpts}")

    print("\n[3/6] Training diffusion model ...")
    model, diff_ckpt, sc_shape = train_diffusion(cfg, data, paths, device, sc_size)
    print(f"[3/6] Diffusion saved -> {diff_ckpt}")

    print(f"\n[4/6] Fine-tuning diffusion with LoRA ({', '.join(predictor_keys)} predictors) ...")
    if not cfg.skip_finetune:
        for predictor_type, pred_path in ridge_ckpts.items():
            print(f"   - Fine-tune with predictor '{predictor_type}' ...")
            finetune_diffusion(
                cfg,
                data,
                paths,
                device,
                sc_size,
                predictor_path=pred_path,
                predictor_type=predictor_type,
                fc20_sets=fc20_sets,
            )
        print("[4/6] Fine-tune completed for all predictors.")
    else:
        print("Skipping finetuning as requested.")

    print("\n[5/6] Sampling raw diffusion (val & test) ...")
    for split in ["val", "test"]:
        gen_path = sample_diffusion(cfg, model, data, paths, device, sc_shape, split=split)
        print(f"   - Raw samples [{split}] -> {gen_path}")

    if not cfg.skip_finetune:
        print(f"\n[6/6] Sampling fine-tuned models (val & test, {', '.join(predictor_keys)} predictors) ...")
        for predictor_type in ridge_ckpts.keys():
            for split in ["val", "test"]:
                ft_path = sample_finetuned(cfg, data, paths, device, sc_size, predictor_type=predictor_type, split=split)
                print(f"   - Finetuned '{predictor_type}' [{split}] -> {ft_path}")
        print("Sampling complete.")
    else:
        print("Finetune skipped; no finetuned sampling.")


if __name__ == "__main__":
    main()
