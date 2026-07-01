import torch
import pandas as pd
import yaml
import numpy as np
import os
from tqdm import tqdm

def load_data(
    metadata_path="metadata.yaml",
    latent_paths=None
):
    """
    Load Schaefer dataset (functional, structural, latent, covariates, and splits).

    Parameters
    ----------
    metadata_path : str
        Path to the metadata YAML file containing configuration (e.g., master_data_dir, split_file, trait_variable).
    latent_paths : dict, optional
        Dictionary with keys '20', '7', '3' containing paths to latent data (.pt files).
        Defaults to the Siemens experiment paths used by Benjamin Torres.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - FC_20, FC_7, FC_3, SC (full tensors)
        - Z_20, Z_7, Z_3 (latent tensors)
        - train/val/test splits for all (FC, Z, SC, Covariates, metadata, masks, targets)
        - variable_name (target column name)
    """



    # --- Load configuration ---
    with open(metadata_path, "r") as file:
        master_config = yaml.safe_load(file)

    MAIN_DATA_DIR = master_config['master_data_dir']
    split_file = master_config['split_file']
    variable_name = master_config['trait_variable']

    # --- Set random seeds ---
    torch.manual_seed(master_config['seed'])
    np.random.seed(master_config['seed'])
    torch.cuda.manual_seed_all(master_config['seed'])


    # --- Load functional and structural data ---
    FC_20 = torch.load(MAIN_DATA_DIR + 'FC_20.pt')
    FC_1 = torch.load(MAIN_DATA_DIR + 'FC_1_segments.pt')
    FC_2 = torch.load(MAIN_DATA_DIR + 'FC_2_segments.pt')
    FC_3  = torch.load(MAIN_DATA_DIR + 'FC_3_segments.pt')
    FC_4 = torch.load(MAIN_DATA_DIR + 'FC_4_segments.pt')
    FC_5 = torch.load(MAIN_DATA_DIR + 'FC_5_segments.pt')
    FC_6 = torch.load(MAIN_DATA_DIR + 'FC_6_segments.pt')
    FC_7 = torch.load(MAIN_DATA_DIR + 'FC_7_segments.pt')
    FC_8 = torch.load(MAIN_DATA_DIR + 'FC_8_segments.pt')
    FC_9 = torch.load(MAIN_DATA_DIR + 'FC_9_segments.pt')
    FC_10 = torch.load(MAIN_DATA_DIR + 'FC_10_segments.pt')
    
    SC    = torch.load(MAIN_DATA_DIR + 'SC.pt')

    # --- Load latent data ---
    if latent_paths is not None:
        Z_20 = torch.load(latent_paths['20'])
        Z_3  = torch.load(latent_paths['3'])
    else:
        Z_20 = Z_7 = Z_3 = None


    # --- Load metadata and covariates ---
    metadata = torch.load(MAIN_DATA_DIR + 'metadata.pt', weights_only=False)
    Covariates = metadata[['interview_age', 'sex']].values

    # --- Load or create splits ---
    splits = np.load(split_file, allow_pickle=True)
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']

    # --- Apply splits ---
    def split_tensor(tensor):
        return tensor[train_idx], tensor[val_idx], tensor[test_idx]

    FC_train, FC_val, FC_test = split_tensor(FC_20)
    FC_1_train, FC_1_val, FC_1_test = split_tensor(FC_1)
    FC_2_train, FC_2_val, FC_2_test = split_tensor(FC_2)
    FC3_train, FC3_val, FC3_test = split_tensor(FC_3)
    FC_4_train, FC_4_val, FC_4_test = split_tensor(FC_4)
    FC_5_train, FC_5_val, FC_5_test = split_tensor(FC_5)
    FC_6_train, FC_6_val, FC_6_test = split_tensor(FC_6)
    FC_7_train, FC_7_val, FC_7_test = split_tensor(FC_7)
    FC_8_train, FC_8_val, FC_8_test = split_tensor(FC_8)
    FC_9_train, FC_9_val, FC_9_test = split_tensor(FC_9)
    FC_10_train, FC_10_val, FC_10_test = split_tensor(FC_10)
    SC_train, SC_val, SC_test = split_tensor(SC)

    Z_20_train, Z_20_val, Z_20_test = split_tensor(Z_20) if latent_paths is not None else (None, None, None)
    Z_7_train, Z_7_val, Z_7_test = split_tensor(Z_7) if latent_paths is not None else (None, None, None)
    Z_3_train, Z_3_val, Z_3_test = split_tensor(Z_3) if latent_paths is not None else (None, None, None)

    Cov_train, Cov_val, Cov_test = Covariates[train_idx], Covariates[val_idx], Covariates[test_idx]
    meta_train, meta_val, meta_test = metadata.iloc[train_idx], metadata.iloc[val_idx], metadata.iloc[test_idx]

    # --- Masks and targets ---
    m_train = meta_train[variable_name].notna().values
    m_val = meta_val[variable_name].notna().values
    m_test = meta_test[variable_name].notna().values

    y_train = torch.tensor(meta_train[variable_name].values)
    y_val = torch.tensor(meta_val[variable_name].values)
    y_test = torch.tensor(meta_test[variable_name].values)

    return {
        'FC': {'train': FC_train, 'val': FC_val, 'test': FC_test},
        'FC3': {'train': FC3_train, 'val': FC3_val, 'test': FC3_test},
        'FC_1': {'train': FC_1_train, 'val': FC_1_val, 'test': FC_1_test},
        'FC_2': {'train': FC_2_train, 'val': FC_2_val, 'test': FC_2_test},
        'FC_4': {'train': FC_4_train, 'val': FC_4_val, 'test': FC_4_test},
        'FC_5': {'train': FC_5_train, 'val': FC_5_val, 'test': FC_5_test},
        'FC_6': {'train': FC_6_train, 'val': FC_6_val, 'test': FC_6_test},
        'FC_7': {'train': FC_7_train, 'val': FC_7_val, 'test': FC_7_test},
        'FC_8': {'train': FC_8_train, 'val': FC_8_val, 'test': FC_8_test},
        'FC_9': {'train': FC_9_train, 'val': FC_9_val, 'test': FC_9_test},
        'FC_10': {'train': FC_10_train, 'val': FC_10_val, 'test': FC_10_test},
        'SC': {'train': SC_train, 'val': SC_val, 'test': SC_test},
        'Z': {
            '20': {'train': Z_20_train, 'val': Z_20_val, 'test': Z_20_test},
            '3': {'train': Z_3_train, 'val': Z_3_val, 'test': Z_3_test},
        },
        'Cov': {'train': Cov_train, 'val': Cov_val, 'test': Cov_test},
        'meta': {'train': meta_train, 'val': meta_val, 'test': meta_test},
        'mask': {'train': m_train, 'val': m_val, 'test': m_test},
        'target': {'train': y_train, 'val': y_val, 'test': y_test},
        'variable_name': variable_name,
        'master_config': master_config,
    }


def load_data_non_overlap(
    metadata_path="metadata.yaml",
    data_dir=None,
    latent_paths=None
):
    """
    Load non-overlapping Schaefer dataset
    (functional connectivity, structural connectivity, metadata, covariates, and splits).

    Expected files inside data_dir:
        FC_1min.pt
        FC_2min.pt
        FC_3min.pt
        FC_4min.pt
        FC_5min.pt
        FC_15min.pt
        SC.pt
        metadata.pt

    Parameters
    ----------
    metadata_path : str
        Path to metadata YAML config file.
    data_dir : str, optional
        Directory containing the .pt files above.
        If None, uses master_config['master_data_dir'].
    latent_paths : dict, optional
        Optional latent paths, for example:
            {
                '15': 'path/to/Z_15.pt',
                '5': 'path/to/Z_5.pt',
                '3': 'path/to/Z_3.pt',
                ...
            }

    Returns
    -------
    dict
        Dictionary with full tensors and train/val/test splits.
    """

    # --- Load configuration ---
    with open(metadata_path, "r") as file:
        master_config = yaml.safe_load(file)

    if data_dir is None:
        data_dir = master_config["master_data_dir"]

    split_file = master_config["split_file"]
    variable_name = master_config["trait_variable"]

    # Make sure path ends correctly
    if not data_dir.endswith(os.sep):
        data_dir += os.sep

    # --- Set random seeds ---
    seed = master_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Load functional and structural data ---
    FC_1 = torch.load(os.path.join(data_dir, "FC_1min.pt"))
    FC_2 = torch.load(os.path.join(data_dir, "FC_2min.pt"))
    FC_3 = torch.load(os.path.join(data_dir, "FC_3min.pt"))
    FC_4 = torch.load(os.path.join(data_dir, "FC_4min.pt"))
    FC_5 = torch.load(os.path.join(data_dir, "FC_5min.pt"))
    FC_15 = torch.load(os.path.join(data_dir, "FC_15min.pt"))
    SC = torch.load(os.path.join(data_dir, "SC.pt"))

    # --- Load latent data if provided ---
    Z = {}
    if latent_paths is not None:
        for key, path in latent_paths.items():
            Z[key] = torch.load(path)
    else:
        Z = None

    # --- Load metadata and covariates ---
    metadata = torch.load(os.path.join(data_dir, "metadata.pt"), weights_only=False)
    Covariates = metadata[["interview_age", "sex"]].values

    # --- Load splits ---
    splits = np.load(split_file, allow_pickle=True)
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]

    # --- Helper for splitting ---
    def split_tensor(tensor):
        return tensor[train_idx], tensor[val_idx], tensor[test_idx]

    # --- Apply splits ---
    FC_1_train, FC_1_val, FC_1_test = split_tensor(FC_1)
    FC_2_train, FC_2_val, FC_2_test = split_tensor(FC_2)
    FC_3_train, FC_3_val, FC_3_test = split_tensor(FC_3)
    FC_4_train, FC_4_val, FC_4_test = split_tensor(FC_4)
    FC_5_train, FC_5_val, FC_5_test = split_tensor(FC_5)
    FC_15_train, FC_15_val, FC_15_test = split_tensor(FC_15)
    SC_train, SC_val, SC_test = split_tensor(SC)

    if Z is not None:
        Z_split = {
            key: {
                "train": split_tensor(z_tensor)[0],
                "val": split_tensor(z_tensor)[1],
                "test": split_tensor(z_tensor)[2],
            }
            for key, z_tensor in Z.items()
        }
    else:
        Z_split = None

    Cov_train, Cov_val, Cov_test = (
        Covariates[train_idx],
        Covariates[val_idx],
        Covariates[test_idx],
    )

    meta_train = metadata.iloc[train_idx]
    meta_val = metadata.iloc[val_idx]
    meta_test = metadata.iloc[test_idx]

    # --- Masks and targets ---
    m_train = meta_train[variable_name].notna().values
    m_val = meta_val[variable_name].notna().values
    m_test = meta_test[variable_name].notna().values

    y_train = torch.tensor(meta_train[variable_name].values)
    y_val = torch.tensor(meta_val[variable_name].values)
    y_test = torch.tensor(meta_test[variable_name].values)

    return {
        "FC_1": {"train": FC_1_train, "val": FC_1_val, "test": FC_1_test},
        "FC_2": {"train": FC_2_train, "val": FC_2_val, "test": FC_2_test},
        "FC_3": {"train": FC_3_train, "val": FC_3_val, "test": FC_3_test},
        "FC_4": {"train": FC_4_train, "val": FC_4_val, "test": FC_4_test},
        "FC_5": {"train": FC_5_train, "val": FC_5_val, "test": FC_5_test},
        "FC_15": {"train": FC_15_train, "val": FC_15_val, "test": FC_15_test},
        "SC": {"train": SC_train, "val": SC_val, "test": SC_test},
        "Z": Z_split,
        "Cov": {"train": Cov_train, "val": Cov_val, "test": Cov_test},
        "meta": {"train": meta_train, "val": meta_val, "test": meta_test},
        "mask": {"train": m_train, "val": m_val, "test": m_test},
        "target": {"train": y_train, "val": y_val, "test": y_test},
        "variable_name": variable_name,
        "master_config": master_config,
    }
