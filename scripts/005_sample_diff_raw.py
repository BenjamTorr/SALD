"""
SAMPLE DIFFUSION MODELS (GRAPH OR FM) USING VAE LATENTS
Everything configurable at the top of the script.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.preprocessing.transformations import gaussian_resample

# ================================================================
#                   GLOBAL CONFIG — EDIT ONLY HERE
# ================================================================

import datetime
import pprint

def print_config(config, title="CONFIGURATION"):
    """
    Pretty-print a config dictionary in a log-friendly format.
    Includes timestamp and sorts keys alphabetically.
    """
    print("\n" + "=" * 70)
    print(f"{title} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Sort keys so logs are always consistent
    for key in sorted(config.keys()):
        print(f"{key:30} : {config[key]}")

    print("=" * 70 + "\n")


CONFIG = {

    # -------------------- Model selection --------------------
    "MODEL_TYPE": "fm",            # "graph" or "fm"
    "MODEL_NAME": "fm_7min",   # your tag
    "MODEL_CKPT": "/data/benjamin_project/diffusion_models/experiments/no_mean/diffusion_models/ddpm_7min_fm_fm.pt",

    # -------------------- Directory paths --------------------
    "LATENT_DIR": "/data/benjamin_project/diffusion_models/experiments/no_mean/latent_data/7min",
    "OUTPUT_DIR": "/data/benjamin_project/diffusion_models/experiments/no_mean/generated_data/7min",
    "METADATA_CONFIG": "../config/metadata.yaml",
    "DIFFUSION_CONFIG": "../config/diffusion_config.yaml",
    "SPLIT": "test",   # which data split to sample from

    # -------------------- VAE decoder -------------------------
    "VAE_CONFIG": "../config/vae_config.yaml",
    "VAE_CKPT": "/data/benjamin_project/diffusion_models/experiments/no_mean/vae_models/VAE_7_scaled.pt",

    # -------------------- Conditioning toggles ----------------
    "USE_SC": True,    # use structural connectivity
    "USE_FCT": True,   # use xt (FCt) conditioning
    "USE_COV": True,   # use covariates (age, sex)
    "USE_RESAMPLE": True,  # whether SC is gaussian resampled

    # -------------------- Sampling parameters -----------------
    "DENOISING_STEPS": 50,
    "ETA": 0.0,
    "SLICE_INDEX": 0,  # set to None to sample all slices
    "N_SAMPLES_PER_SUBJECT": 10,
    "CHUNK_SIZE": 512,
    "DECODE_BATCH_SIZE": 50,
    "BATCH_SIZE": 512,
    "NUM_WORKERS": 0,

    # -------------------- Device ------------------------------
    "DEVICE": "cuda:2",

    # -------------------- Random seed -------------------------
    "SEED": 42,
}


print(CONFIG)

# ================================================================
#                       UTILITY FUNCTIONS
# ================================================================

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
        latents[f"{split}_x0"] = torch.load(latent_dir / f"{split}_x0_embeddings_scaled.pt", map_location="cpu")
        latents[f"{split}_xt"] = torch.load(latent_dir / f"{split}_xt_embeddings_scaled.pt", map_location="cpu")
    return latents


def _zero_like(arr):
    return torch.zeros_like(arr) if torch.is_tensor(arr) else np.zeros_like(arr)


def apply_conditioning_filters(latents, data, cfg):
    """Zero out conditioning sources depending on CFG settings."""
    filtered_latents = {}
    for k, v in latents.items():
        if not cfg["USE_FCT"] and k.endswith("_xt"):
            filtered_latents[k] = torch.zeros_like(v)
        else:
            filtered_latents[k] = v

    filtered_data = {"SC": {}, "Cov": {}, "target": data["target"]}

    for split in ["train", "val", "test"]:
        sc = data["SC"][split]
        cov = data["Cov"][split]

        filtered_data["SC"][split] = sc if cfg["USE_SC"] else _zero_like(sc)
        filtered_data["Cov"][split] = cov if cfg["USE_COV"] else _zero_like(cov)

    return filtered_latents, filtered_data


# ================================================================
#                       IMPORT REQUIRED CLASSES
# ================================================================

from data.load_data import load_data
from data.loaders import (
    FC_SCGraphDataset,
    FC_SC_vec_Dataset,
    custom_collate_fn,
)

from diffusion.ddpm import ddpm
from diffusion.ddpm_graph import ddpm_graph
from diffusion.dit_FiLM import dit_film
from diffusion.dit_cat import dit_cat
from diffusion.graph_encoder import SCGraphModel1D


# ================================================================
#                       MODEL BUILDERS
# ================================================================

def build_model(cfg, diffusion_config, device):
    ddpm_cfg = diffusion_config["DDPM_config"]
    vector_cl = (ddpm_cfg["vector_c"], ddpm_cfg["vector_l"])
    schedule = ddpm_cfg.get("schedule", "linear")

    if cfg["MODEL_TYPE"] == "fm":
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

    elif cfg["MODEL_TYPE"] == "graph":

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

    else:
        raise ValueError("MODEL_TYPE must be 'graph' or 'fm'.")

    return model


def load_vae(cfg, device):
    from vae.unet_vae import vae_unet
    with open(cfg["VAE_CONFIG"], "r") as f:
        vae_cfg = yaml.safe_load(f)

    vae = vae_unet(im_channels=1, model_config=vae_cfg["VAE_params"]).to(device)
    vae.load_state_dict(torch.load(cfg["VAE_CKPT"], map_location=device))
    vae.eval()
    return vae


def build_loader(cfg, data, latents, device, sc_shape):
    """
    Build a loader based on cfg['SPLIT'], e.g. 'train', 'val', 'test'.
    Uses MODEL_TYPE to select dataset class.
    Applies SC resampling for that split if enabled.
    """

    split = cfg["SPLIT"]   # <-- now taken from the config

    # ---- Optional SC resampling ----
    if cfg.get("USE_RESAMPLE", False):
        SEED = cfg["SEED"]
        data["SC"][split] = gaussian_resample(data["SC"][split], seed=SEED)

    # ---- Grab data for this split ----
    x0  = latents[f"{split}_x0"]
    xt  = latents[f"{split}_xt"]
    SC  = data["SC"][split]
    Cov = data["Cov"][split]
    y   = data["target"][split]
    mask = ~torch.isnan(y)
    pin_memory = device.type == "cuda"

    # ---- Build dataset depending on model type ----
    if cfg["MODEL_TYPE"] == "graph":
        dataset = FC_SCGraphDataset(
            x0[mask],
            SC[mask],
            xt[mask],
            Cov[mask],
            y[mask],
            age_dim=126,
            transform_sc=True,
            shape=sc_shape,
        )

        return DataLoader(
            dataset,
            batch_size=cfg["BATCH_SIZE"],
            shuffle=(split == "train"),      # auto shuffle only for train
            num_workers=cfg["NUM_WORKERS"],
            collate_fn=custom_collate_fn,
            pin_memory=pin_memory,
        )

    else:  # MODEL_TYPE == "fm"
        dataset = FC_SC_vec_Dataset(
            x0[mask],
            SC[mask],
            xt[mask],
            Cov[mask],
            y[mask],
            age_dim=126,
            transform_sc=True,
            shape=sc_shape,
        )

        return DataLoader(
            dataset,
            batch_size=cfg["BATCH_SIZE"],
            shuffle=(split == "train"),
            num_workers=cfg["NUM_WORKERS"],
            pin_memory=pin_memory,
        )



# ================================================================
#                               MAIN
# ================================================================

def main(cfg):
    device = torch.device(cfg["DEVICE"])
    set_seed(cfg["SEED"])

    print("\n==============================")
    print("      SAMPLING CONFIG")
    print("==============================")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("==============================\n")

    # ---- Load configs ----
    with open(cfg["METADATA_CONFIG"], "r") as f:
        master_cfg = yaml.safe_load(f)

    with open(cfg["DIFFUSION_CONFIG"], "r") as f:
        diffusion_config = yaml.safe_load(f)

    # ---- Load data ----
    data = load_data(metadata_path=cfg["METADATA_CONFIG"])
    sc_size = data["SC"]["train"].shape[-1]
    sc_shape = (-1, 1, sc_size, sc_size)

    # ---- Load latents ----
    latents = load_latents(Path(cfg["LATENT_DIR"]))

    # ---- Apply conditioning filters ----
    latents, data = apply_conditioning_filters(latents, data, cfg)

    # ---- Build test loader ----
    test_loader = build_loader(cfg, data, latents, device, sc_shape)

    # ---- Build diffusion model ----
    model = build_model(cfg, diffusion_config, device)
    model.load_state_dict(torch.load(cfg["MODEL_CKPT"], map_location=device))
    model.eval()

    # ---- Load VAE ----
    vae = load_vae(cfg, device)

    # ---- Output directory ----
    out_dir = Path(cfg["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================
    #                SAMPLING LOOP
    # ======================================================
    from utils.preprocessing.transformations import upper_elements_to_symmetric_matrix

    all_samples = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Sampling"):

            if cfg["MODEL_TYPE"] == "graph":
                _, cond1, cond2, cov, _ = batch
            else:   # fm fallback
                cond1, cond2, cov = batch[1], batch[2], batch[3]

            samples = model.sample_repeated_chunked_ddim(
                cond1_data=cond1.to(device),
                cond2=cond2.to(device),
                cov_cond=cov.to(device),
                denoising_steps=cfg["DENOISING_STEPS"],
                eta=cfg["ETA"],
                n=cfg["N_SAMPLES_PER_SUBJECT"],
                chunk_size=cfg["CHUNK_SIZE"],
                amp=True,
                grad=False,
                slice_index=cfg.get("SLICE_INDEX", 0),
            )

            all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)

    # ======================================================
    #                DECODE WITH VAE
    # ======================================================

    expanded = all_samples.reshape(-1, 4, 618)
    decoded_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(expanded), cfg["DECODE_BATCH_SIZE"]), desc="Decoding"):
            chunk = expanded[i : i + cfg["DECODE_BATCH_SIZE"]]
            dec = vae.decode(chunk.to(device))
            mats = upper_elements_to_symmetric_matrix(dec).reshape(
                -1, cfg["N_SAMPLES_PER_SUBJECT"], sc_size, sc_size
            )
            decoded_list.append(mats)

    recon = torch.cat(decoded_list, dim=0)

    # ---- Save ----
    out_path = out_dir / f"FC_gen_{cfg['MODEL_NAME']}_{cfg['SPLIT']}.pt"
    torch.save(recon.cpu(), out_path)
    print(f"\nSaved generated FC matrices to:\n{out_path}\n")


# ===============================================================
# Run
# ===============================================================

if __name__ == "__main__":
    main(CONFIG)
