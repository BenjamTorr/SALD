import torch
import pandas as pd
import yaml
import numpy as np 
from data.load_data import load_data
import random
import os
from vae.unet_vae import vae_unet
import sys
from pathlib import Path
from data.loaders import FC_SCVectorDataset
from torch.utils.data import DataLoader
from utils.preprocessing.transformations import upper_elements_to_symmetric_matrix_no_chan
import plotly.express as px

sys.path.append("/home/ibta/.../experiments/no_mean")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaled = False
time_short = 3

print("\n" + "=" * 60)
print("EXPERIMENT CONFIG SUMMARY")
print("=" * 60)
print(f"time_short (minutes) : {time_short}")
print(f"Using scaled data     : {scaled}")
print(f"FC short window       : FC_{time_short}")
print(f"FC long window        : FC20")
print(f"Device                : {device}")
print("=" * 60 + "\n")


with open("../config/metadata.yaml", "r") as file:
    master_config = yaml.safe_load(file)

with open("../config/vae_config.yaml", "r") as file:
    vae_config = yaml.safe_load(file)

# --- Set fixed global seed ---
SEED = master_config['seed'] # you can change this to any integer
EPS = 1e-8

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def vector_to_matrix(vec, mean_mat, std_mat):
    """
    vec: (N, num_upper)
    returns: (N, L, L) with mean/std added back.
    """
    mat = upper_elements_to_symmetric_matrix_no_chan(vec)
    return mat * std_mat + mean_mat

def process_split(split_name, loader, vae, device, mean_mat, std_mat, save_dir, type_data):
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
        torch.save(out["x0_embeddings"], save_dir / f"{split_name}_x0_embeddings_{type_data}.pt")
    if out["x0_recons"] is not None:
        torch.save(out["x0_recons"], save_dir / f"{split_name}_x0_recons_{type_data}.pt")
    if out["xt_embeddings"] is not None:
        torch.save(out["xt_embeddings"], save_dir / f"{split_name}_xt_embeddings_{type_data}.pt")
    if out["xt_recons"] is not None:
        torch.save(out["xt_recons"], save_dir / f"{split_name}_xt_recon_{type_data}.pt")

    # Frobenius norms per time step using stored reconstructions
    if out["xt_recons"] is not None:
        xt_recon = out["xt_recons"]  # (B_total, S, C, D)
        B_total, S, _, D = xt_recon.shape
        xt_recon_vec = xt_recon.reshape(B_total * S, D)
        xt_vec = xt_truth.reshape(B_total * S, D)

        real_mat = vector_to_matrix(xt_vec, mean_mat, std_mat)
        recon_mat = vector_to_matrix(xt_recon_vec, mean_mat, std_mat)

        frob = torch.norm(recon_mat - real_mat, dim=(1, 2))
        s_indices =  torch.arange(S).repeat(B_total)

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
        frob_df.to_csv(save_dir / f"{split_name}_frob2.csv", index=False)
        fig = px.box(frob_df, x="s_idx", y="frob_norm", color="s_idx", points="all", title=f"Frobenius norm: {split_name}")
        fig.write_html(save_dir / f"{split_name}_frob2.html")

MAIN_DATA_DIR = master_config['master_data_dir']

data = load_data(metadata_path='../config/metadata.yaml')

# --- Extract the components you need ---
FC20_train, FC20_val, FC20_test = data['FC']['train'], data['FC']['val'], data['FC']['test']
FC3_train, FC3_val, FC3_test = data['FC3']['train'], data['FC3']['val'], data['FC3']['test']
FC_1_train, FC_1_val, FC_1_test = data['FC_1']['train'], data['FC_1']['val'], data['FC_1']['test']
FC_2_train, FC_2_val, FC_2_test = data['FC_2']['train'], data['FC_2']['val'], data['FC_2']['test']
FC_4_train, FC_4_val, FC_4_test = data['FC_4']['train'], data['FC_4']['val'], data['FC_4']['test']
FC_5_train, FC_5_val, FC_5_test = data['FC_5']['train'], data['FC_5']['val'], data['FC_5']['test']
FC_6_train, FC_6_val, FC_6_test = data['FC_6']['train'], data['FC_6']['val'], data['FC_6']['test']
FC_7_train, FC_7_val, FC_7_test = data['FC_7']['train'], data['FC_7']['val'], data['FC_7']['test']
SC_train,   SC_val,   SC_test   = data['SC']['train'],      data['SC']['val'],      data['SC']['test']
Cov_train,  Cov_val,  Cov_test  = data['Cov']['train'],     data['Cov']['val'],     data['Cov']['test']
y_train,    y_val,    y_test    = data['target']['train'],  data['target']['val'],  data['target']['test']


mean_FC= FC20_train.mean(0, keepdim=True)
std_FC = FC20_train.std(0, keepdim=True) + EPS



if time_short == 1:
    FC_t_train = FC_1_train
    FC_t_val = FC_1_val
    FC_t_test = FC_1_test
elif time_short == 2:
    FC_t_train = FC_2_train
    FC_t_val = FC_2_val
    FC_t_test = FC_2_test
elif time_short == 3:
    FC_t_train = FC3_train
    FC_t_val = FC3_val
    FC_t_test = FC3_test
elif time_short == 4:
    FC_t_train = FC_4_train
    FC_t_val = FC_4_val
    FC_t_test = FC_4_test
elif time_short == 5:
    FC_t_train = FC_5_train
    FC_t_val = FC_5_val
    FC_t_test = FC_5_test
elif time_short == 6:
    FC_t_train = FC_6_train
    FC_t_val = FC_6_val
    FC_t_test = FC_6_test
else:
    FC_t_train = FC_7_train
    FC_t_val = FC_7_val
    FC_t_test = FC_7_test


FC20_train_scaled = (FC20_train - mean_FC) / std_FC
FC_t_train_scaled = (FC_t_train - mean_FC.unsqueeze(1)) / std_FC.unsqueeze(1)

FC20_val_scaled = (FC20_val - mean_FC) / std_FC
FC_t_val_scaled = (FC_t_val - mean_FC.unsqueeze(1)) / std_FC.unsqueeze(1)

FC20_test_scaled = (FC20_test - mean_FC) / std_FC
FC_t_test_scaled = (FC_t_test - mean_FC.unsqueeze(1)) / std_FC.unsqueeze(1)

vae_network = vae_unet(im_channels=1, model_config = vae_config['VAE_params']).to(device)


if scaled:
    print("Using scaled data loaders")
    pin_memory = device.type == "cuda"
    training_loader = DataLoader(FC_SCVectorDataset(FC20_train_scaled, SC_train, FC_t_train_scaled, Cov_train, age_dim = 126, 
                                                log_transform = False, shape = (-1, 1, 100, 100)), batch_size=32, shuffle =True, pin_memory=pin_memory)
    validation_loader = DataLoader(FC_SCVectorDataset(FC20_val_scaled, SC_val, FC_t_val_scaled, Cov_val, age_dim = 126, 
                                                log_transform = False, shape = (-1, 1, 100, 100)), batch_size=32, shuffle =False, pin_memory=pin_memory)
    test_loader = DataLoader(FC_SCVectorDataset(FC20_test_scaled, SC_test, FC_t_test_scaled, Cov_test, age_dim = 126, 
                                                log_transform = False, shape = (-1, 1, 100, 100)), batch_size=32, shuffle =False, pin_memory=pin_memory)
else:
    print("Using unscaled data loaders")
    pin_memory = device.type == "cuda"
    training_loader = DataLoader(FC_SCVectorDataset(FC20_train, SC_train, FC_t_train, Cov_train, age_dim = 126, 
                                                log_transform = False, shape = (-1, 1, 100, 100)), batch_size=32, shuffle =True, pin_memory=pin_memory)
    validation_loader = DataLoader(FC_SCVectorDataset(FC20_val, SC_val, FC_t_val, Cov_val, age_dim = 126, 
                                                log_transform = False, shape = (-1, 1, 100, 100)), batch_size=32, shuffle =False, pin_memory=pin_memory)
    test_loader = DataLoader(FC_SCVectorDataset(FC20_test, SC_test, FC_t_test, Cov_test, age_dim = 126, 
                                                log_transform = False, shape = (-1, 1, 100, 100)), batch_size=32, shuffle =False, pin_memory=pin_memory)
    

optimizer = torch.optim.Adam(vae_network.parameters(), lr=1e-4)

model_store_path = Path('/data/benjamin_project/diffusion_models/experiments/no_mean/vae_models')
type_data = 'scaled' if scaled else 'unscaled'



vae_network.train_vae_extended(loader = training_loader, 
                               loader_val = validation_loader, 
                               n_epochs=500, 
                               optim = optimizer, 
                               device = device, 
                               beta=float(vae_config['VAE_params']['beta']),  
                               patience=25, use_scheduler = True, accumulation_steps = 2,  store_path = str(model_store_path / f'VAE_{time_short}_{type_data}.pt'))

process_split('train', training_loader, vae_network, device, mean_FC, std_FC, 
              save_dir = Path("/data/benjamin_project/diffusion_models/experiments/no_mean/latent_data") / f"{time_short}min", type_data =type_data)

process_split('val', validation_loader, vae_network, device, mean_FC, std_FC, 
              save_dir = Path("/data/benjamin_project/diffusion_models/experiments/no_mean/latent_data") / f"{time_short}min", type_data =type_data)

process_split('test', test_loader, vae_network, device, mean_FC, std_FC, 
              save_dir = Path("/data/benjamin_project/diffusion_models/experiments/no_mean/latent_data") / f"{time_short}min", type_data =type_data)
