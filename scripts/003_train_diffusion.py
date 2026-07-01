import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.load_data import load_data
from data.loaders import FC_SCGraphDataset, FC_SC_vec_Dataset, custom_collate_fn
from diffusion.ddpm import ddpm
from diffusion.ddpm_graph import ddpm_graph
from diffusion.dit_FiLM import dit_film
from diffusion.dit_cat import dit_cat
from diffusion.graph_encoder import SCGraphModel1D


DEFAULT_LATENT_DIR = Path("/data/benjamin_project/diffusion_models/experiments/no_mean/latent_data")
DEFAULT_MODEL_DIR = Path("/data/benjamin_project/diffusion_models/experiments/no_mean/diffusion_models")


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_latents(latent_dir: Path):
    latents = {}
    for split in ["train", "val", "test"]:
        latents[f"{split}_x0"] = torch.load(latent_dir / f"{split}_x0_embeddings.pt", map_location="cpu")
        latents[f"{split}_xt"] = torch.load(latent_dir / f"{split}_xt_embeddings.pt", map_location="cpu")
    return latents


def _zero_like(arr):
    if torch.is_tensor(arr):
        return torch.zeros_like(arr)
    return np.zeros_like(arr)


def apply_conditioning_filters(latents, data, use_sc=True, use_fct=True, use_cov=True):
    """
    Returns copies of latents/data with non-selected conditioning sources zeroed out.
    - SC: sets SC matrices to zero so graphs/SC vectors carry no signal.
    - FCT: sets xt embeddings to zero.
    - Cov: sets covariates to zero.
    """
    latents_filt = {}
    for k, v in latents.items():
        if (not use_fct) and k.endswith("_xt"):
            latents_filt[k] = torch.zeros_like(v)
        else:
            latents_filt[k] = v

    data_filt = {"SC": {}, "Cov": {}, "target": data["target"]}
    for split in ["train", "val", "test"]:
        sc = data["SC"][split]
        cov = data["Cov"][split]
        data_filt["SC"][split] = sc if use_sc else _zero_like(sc)
        data_filt["Cov"][split] = cov if use_cov else _zero_like(cov)

    return latents_filt, data_filt


def build_loaders(data, latents, device, batch_size, num_workers, sc_shape):
    loaders = {}
    pin_memory = device.type == "cuda"

    loaders["graph_train"] = DataLoader(
        FC_SCGraphDataset(
            latents["train_x0"],
            data["SC"]["train"],
            latents["train_xt"],
            data["Cov"]["train"],
            data["target"]["train"],
            age_dim=126,
            transform_sc=True,
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
            data["SC"]["val"],
            latents["val_xt"],
            data["Cov"]["val"],
            data["target"]["val"],
            age_dim=126,
            transform_sc=True,
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
            data["SC"]["train"],
            latents["train_xt"],
            data["Cov"]["train"],
            data["target"]["train"],
            age_dim=126,
            transform_sc=True,
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
            data["SC"]["val"],
            latents["val_xt"],
            data["Cov"]["val"],
            data["target"]["val"],
            age_dim=126,
            transform_sc=True,
            shape=sc_shape,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loaders


def build_model(model_type, diffusion_config, device):
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
        default_name = "graph_fm.pt"
    elif model_type == "graph":
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
        default_name = "graph_log_transform.pt"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, default_name


def main(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Training uses CUDA autocast; please select a CUDA device.")

    with open("../config/metadata.yaml", "r") as file:
        master_config = yaml.safe_load(file)

    with open("../config/diffusion_config.yaml", "r") as file:
        diffusion_config = yaml.safe_load(file)

    set_seed(master_config["seed"])

    data = load_data(metadata_path="../config/metadata.yaml")
    sc_size = data["SC"]["train"].shape[-1]
    sc_shape = (-1, 1, sc_size, sc_size)

    latent_dir = Path(args.latent_dir)
    latents = load_latents(latent_dir)
    latents, data = apply_conditioning_filters(
        latents=latents,
        data=data,
        use_sc=not args.no_sc,
        use_fct=not args.no_fct,
        use_cov=not args.no_cov,
    )

    loaders = build_loaders(
        data=data,
        latents=latents,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sc_shape=sc_shape,
    )

    model, default_ckpt_name = build_model(args.model, diffusion_config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    store_path = Path(args.store_path) if args.store_path else model_dir / default_ckpt_name

    train_loader = loaders["graph_train"] if args.model == "graph" else loaders["fm_train"]
    val_loader = loaders["graph_val"] if args.model == "graph" else loaders["fm_val"]

    model.train_ddpm_amp(
        loader=train_loader,
        loader_val=val_loader,
        n_epochs=args.epochs,
        optimizer=optimizer,
        patience=args.patience,
        accumulation_steps=args.accumulation_steps,
        use_scheduler=not args.no_scheduler,
        debug=False,
        store_path=str(store_path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion models (fm or graph) using saved VAE latents.")
    parser.add_argument("--model", choices=["fm", "graph"], default="graph", help="Choose between fm (ddpm_fm) or graph model.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latent-dir", default=str(DEFAULT_LATENT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--store-path", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--no-scheduler", action="store_true", help="Disable ReduceLROnPlateau scheduler.")
    parser.add_argument("--no-sc", action="store_true", help="Zero out SC conditioning.")
    parser.add_argument("--no-fct", action="store_true", help="Zero out FCt embeddings conditioning.")
    parser.add_argument("--no-cov", action="store_true", help="Zero out covariate conditioning.")
    args = parser.parse_args()
    main(args)
