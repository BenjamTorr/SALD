import argparse
from contextlib import nullcontext
import os
from pathlib import Path

import plotly.express as px
import torch
import yaml
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from data.load_data import load_data
from data.loaders import FC_SCVectorDataset
from utils.preprocessing.transformations import upper_elements_to_symmetric_matrix_no_chan
from vae.unet_vae import vae_unet


def build_loader(split_data, device, batch_size):
    FC_scaled, SC, FCt_scaled, Cov = split_data
    dataset = FC_SCVectorDataset(
        FC_scaled,
        SC,
        FCt_scaled,
        Cov,
        age_dim=126,
        log_transform=False,
        shape=(-1, 1, FC_scaled.shape[-2], FC_scaled.shape[-1]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))


def vector_to_matrix(vec, mean_mat, std_mat):
    """
    vec: (N, num_upper)
    returns: (N, L, L) with mean/std added back.
    """
    mat = upper_elements_to_symmetric_matrix_no_chan(vec)
    return mat * std_mat + mean_mat


def process_split(split_name, loader, vae, device, mean_mat, std_mat, save_dir):
    frob_records = []
    # Collect ground-truth xt for Frobenius, but use the VAE helper for embeddings/recons
    xt_truth = []
    for batch in loader:
        _, _, xt, _ = batch
        xt_truth.append(xt.cpu())
    xt_truth = torch.cat(xt_truth, dim=0)  # (B_total, S, D)

    out = vae.get_embeddings_and_reconstructions(loader, device)

    # Save tensors
    if out["x0_embeddings"] is not None:
        torch.save(out["x0_embeddings"], save_dir / f"{split_name}_x0_embeddings2.pt")
    if out["x0_recons"] is not None:
        torch.save(out["x0_recons"], save_dir / f"{split_name}_x0_recons2.pt")
    if out["xt_embeddings"] is not None:
        torch.save(out["xt_embeddings"], save_dir / f"{split_name}_xt_embeddings2.pt")
    if out["xt_recons"] is not None:
        torch.save(out["xt_recons"], save_dir / f"{split_name}_xt_recons2.pt")

    # Frobenius norms per time step using stored reconstructions
    if out["xt_recons"] is not None:
        xt_recon = out["xt_recons"]  # (B_total, S, C, D)
        B_total, S, _, D = xt_recon.shape
        xt_recon_vec = xt_recon.reshape(B_total * S, D)
        xt_vec = xt_truth.reshape(B_total * S, D)

        real_mat = vector_to_matrix(xt_vec, mean_mat, std_mat)
        recon_mat = vector_to_matrix(xt_recon_vec, mean_mat, std_mat)

        frob = torch.norm(recon_mat - real_mat, dim=(1, 2))
        s_indices = torch.arange(S).repeat_interleave(B_total)
        frob_records.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "s_idx": s_indices.numpy(),
                    "frob_norm": frob.numpy(),
                }
            )
        )

    if frob_records:
        frob_df = pd.concat(frob_records, ignore_index=True)
        frob_df.to_csv(save_dir / f"{split_name}_frob2.csv", index=False)
        fig = px.box(frob_df, x="s_idx", y="frob_norm", color="s_idx", points="all", title=f"Frobenius norm: {split_name}")
        fig.write_html(save_dir / f"{split_name}_frob2.html")


def main(args):
    device = torch.device(args.device)

    # Load configs
    with open("../config/metadata.yaml", "r") as file:
        master_config = yaml.safe_load(file)

    with open("../config/vae_config.yaml", "r") as file:
        vae_config = yaml.safe_load(file)

    data = load_data(metadata_path="../config/metadata.yaml")

    # Prepare splits
    FC20_train, FC20_val, FC20_test = data["FC"]["train"], data["FC"]["val"], data["FC"]["test"]
    FC3_train, FC3_val, FC3_test = data["FC3"]["train"], data["FC3"]["val"], data["FC3"]["test"]
    SC_train, SC_val, SC_test = data["SC"]["train"], data["SC"]["val"], data["SC"]["test"]
    Cov_train, Cov_val, Cov_test = data["Cov"]["train"], data["Cov"]["val"], data["Cov"]["test"]

    EPS = 1e-8
    mean_FC = FC20_train.mean(0, keepdim=True)
    std_FC = FC20_train.std(0, keepdim=True) + EPS

    FC20_train_scaled = (FC20_train - mean_FC) / std_FC
    FC3_train_scaled = (FC3_train - mean_FC.unsqueeze(1)) / std_FC.unsqueeze(1)

    FC20_val_scaled = (FC20_val - mean_FC) / std_FC
    FC3_val_scaled = (FC3_val - mean_FC.unsqueeze(1)) / std_FC.unsqueeze(1)

    FC20_test_scaled = (FC20_test - mean_FC) / std_FC
    FC3_test_scaled = (FC3_test - mean_FC.unsqueeze(1)) / std_FC.unsqueeze(1)

    save_dir = Path("/data/benjamin_project/diffusion_models/experiments/no_mean/latent_data")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model
    vae = vae_unet(im_channels=1, model_config=vae_config["VAE_params"]).to(device)
    vae.load_state_dict(torch.load(args.ckpt, map_location=device))
    vae.eval()

    # Mean/std tensors for de-scaling (on CPU for plotting)
    mean_mat = torch.as_tensor(mean_FC, dtype=torch.float32).squeeze(0)
    std_mat = torch.as_tensor(std_FC, dtype=torch.float32).squeeze(0)

    splits = {
        "train": (FC20_train_scaled, SC_train, FC3_train_scaled, Cov_train),
        "val": (FC20_val_scaled, SC_val, FC3_val_scaled, Cov_val),
        "test": (FC20_test_scaled, SC_test, FC3_test_scaled, Cov_test),
    }

    for split_name, split_data in splits.items():
        loader = build_loader(split_data, device, batch_size=args.batch_size)
        process_split(split_name, loader, vae, device, mean_mat, std_mat, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export VAE latents, reconstructions, and Frobenius norm plots.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt", default="/data/benjamin_project/diffusion_models/experiments/no_mean/vae_models/VAE2.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(args)
