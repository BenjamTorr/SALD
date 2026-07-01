import yaml
import os
import subprocess
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def create_and_save_splits(metadata, save_path="config/splits_master.npz", seed=42, siemens_only=True):
    """
    Generate train/val/test indices only for SIEMENS scans and save them once.
    """
    # Filter metadata for SIEMENS manufacturer
    if siemens_only:
        siemens_mask = metadata["mri_info_manufacturer"] == "SIEMENS"
    else:
        siemens_mask = np.ones(len(metadata), dtype=bool)  # Include all if not filtering
    siemens_idx = np.where(siemens_mask)[0]

    # Ensure reproducibility and proper indexing
    rng = np.random.RandomState(seed)
    
    # First split: train / temp
    train_idx, temp_idx = train_test_split(
        siemens_idx, test_size=0.2, random_state=rng
    )

    # Second split: temp -> val / test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=rng
    )

    # Save indices
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, train=train_idx, val=val_idx, test=test_idx)

    print(f"Splits saved at {save_path}")
    print(f"Total SIEMENS samples: {len(siemens_idx)} | train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    return train_idx, val_idx, test_idx



def load_splits(path="config/splits_master.npz"):
    """
    Load stored splits for reproducibility.
    """
    splits = np.load(path, allow_pickle=True)
    return splits["train"], splits["val"], splits["test"]


if __name__ == "__main__":
    metadata_path = "/home/ibta/benjamin/diffusion_experiments/Diffusion_FCSC51015/diffusion-models-project/experiments/no_mean/config/metadata_full.yaml"
    with open(metadata_path, "r") as file:
        master_config = yaml.safe_load(file)
    
    MAIN_DATA_DIR = master_config['master_data_dir']

    # Save full environment (all packages, exact versions)
    subprocess.run(f"conda env export > {master_config['environment_file']}.yml", shell=True)

    # Save pip requirements
    subprocess.run(f"pip freeze > {master_config['environment_file']}.txt", shell=True)

    _, _, _ = create_and_save_splits(metadata = torch.load(MAIN_DATA_DIR + 'metadata.pt', weights_only=False), 
                                    save_path= master_config['split_file'],
                                    seed = master_config['seed'], siemens_only=False)
