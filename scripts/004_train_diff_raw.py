import torch
import pandas as pd
import yaml
import numpy as np 
from data.load_data import load_data
import random
import os
from vae.unet_vae import vae_unet
from diffusion.ddpm_graph import ddpm_graph
from diffusion.ddpm import ddpm
from diffusion.dit_cat import dit_cat
from diffusion.dit_FiLM import dit_film
from diffusion.graph_encoder import SCGraphModel1D
from data.loaders import FC_SCGraphDataset, FC_SC_vec_Dataset
from data.loaders import custom_collate_fn
from torch.utils.data import DataLoader
from utils.preprocessing.transformations import gaussian_resample
import sys

##### preconfig  #######
name = 'ddpm_5min_graph'
time_short = 5

use_sc_resample = True
device_numb = 0

use_cond_sc = 1
use_cond_3 = 1
use_cond_cov = 1

model_method = 'graph' #graph

print(
    "===== CONFIG SUMMARY =====\n"
    f"Name:            {name}\n"
    f"SC Resample:     {use_sc_resample}\n"
    f"Device Number:   {device_numb}\n"
    f"Use cond SC:     {use_cond_sc}\n"
    f"Use cond FC3:    {use_cond_3}\n"
    f"Use covariates:  {use_cond_cov}\n"
    f"Model method:    {model_method}\n"
    "=========================="
)

##########



sys.path.append("/home/ibta/.../experiments/no_mean")
device = torch.device(f"cuda:{device_numb}" if torch.cuda.is_available() else "cpu")

with open("../config/metadata.yaml", "r") as file:
    master_config = yaml.safe_load(file)

with open("../config/diffusion_config.yaml", "r") as file:
    diffusion_config = yaml.safe_load(file)


# --- Set fixed global seed ---
SEED = master_config['seed'] # you can change this to any integer
EPS = 1e-8
MAIN_MODEL_DIR = '/data/benjamin_project/diffusion_models/experiments/no_mean/diffusion_models/'

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MAIN_DATA_DIR = master_config['master_data_dir']
LATENT_DATA_DIR = '/data/benjamin_project/diffusion_models/experiments/no_mean/latent_data/5min/'

data = load_data(metadata_path='../config/metadata.yaml')

# --- Extract the components you need ---
FC20_train, FC20_val, FC20_test = data['FC']['train'], data['FC']['val'], data['FC']['test']
FC_1_train, FC_1_val, FC_1_test = data['FC_1']['train'], data['FC_1']['val'], data['FC_1']['test']
FC_2_train, FC_2_val, FC_2_test = data['FC_2']['train'], data['FC_2']['val'], data['FC_2']['test']
FC3_train, FC3_val, FC3_test = data['FC3']['train'], data['FC3']['val'], data['FC3']['test']
FC_4_train, FC_4_val, FC_4_test = data['FC_4']['train'], data['FC_4']['val'], data['FC_4']['test']
FC_5_train, FC_5_val, FC_5_test = data['FC_5']['train'], data['FC_5']['val'], data['FC_5']['test']
FC_6_train, FC_6_val, FC_6_test = data['FC_6']['train'], data['FC_6']['val'], data['FC_6']['test']
FC_7_train, FC_7_val, FC_7_test = data['FC_7']['train'], data['FC_7']['val'], data['FC_7']['test']
SC_train,   SC_val,   SC_test   = data['SC']['train'],      data['SC']['val'],      data['SC']['test']
Cov_train,  Cov_val,  Cov_test  = data['Cov']['train'],     data['Cov']['val'],     data['Cov']['test']
y_train,    y_val,    y_test    = data['target']['train'],  data['target']['val'],  data['target']['test']

Z20_train, Z20_val, Z20_test = torch.load(LATENT_DATA_DIR + 'train_x0_embeddings_scaled.pt'), torch.load(LATENT_DATA_DIR + 'val_x0_embeddings_scaled.pt'), torch.load(LATENT_DATA_DIR + 'test_x0_embeddings_scaled.pt')
Z_t_train, Z_t_val, Z_t_test = torch.load(LATENT_DATA_DIR + 'train_xt_embeddings_scaled.pt'), torch.load(LATENT_DATA_DIR + 'val_xt_embeddings_scaled.pt'), torch.load(LATENT_DATA_DIR + 'test_xt_embeddings_scaled.pt')

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

mean_FC= FC20_train.mean(0, keepdim=True)
std_FC = FC20_train.std(0, keepdim=True) + EPS

FC20_train_scaled = (FC20_train - mean_FC) / std_FC
FC_t_train_scaled = (FC_t_train - mean_FC.unsqueeze(1)) / std_FC.unsqueeze(1)

FC20_val_scaled = (FC20_val - mean_FC) / std_FC
FC_t_val_scaled = (FC_t_val - mean_FC.unsqueeze(1)) / std_FC.unsqueeze(1)

FC20_test_scaled = (FC20_test - mean_FC) / std_FC
FC_t_test_scaled = (FC_t_test - mean_FC.unsqueeze(1)) / std_FC.unsqueeze(1)



### Estabilize SC


if use_sc_resample:
    SC_train = gaussian_resample(SC_train, seed = SEED)
    SC_val = gaussian_resample(SC_val, seed = SEED)
    SC_test = gaussian_resample(SC_test, seed = SEED)


if use_cond_cov == 0:
    Cov_train = np.zeros_like(Cov_train)
    Cov_test = np.zeros_like(Cov_test)
    Cov_val = np.zeros_like(Cov_val)


## Graph Loaders
pin_memory = device.type == "cuda"

training_loader = DataLoader(FC_SCGraphDataset(Z20_train,
                                                SC_train * use_cond_sc,
                                                Z_t_train * use_cond_3, 
                                                Cov_train, 
                                                y_train,age_dim = 126, 
                                               transform_sc = (not use_sc_resample), shape = (-1, 1, 100, 100)), batch_size=32, shuffle =True, collate_fn= custom_collate_fn, pin_memory=pin_memory)
validation_loader = DataLoader(FC_SCGraphDataset(Z20_val, 
                                                 SC_val * use_cond_sc, 
                                                 Z_t_val * use_cond_3, 
                                                 Cov_val,
                                                   y_val, age_dim = 126, 
                                               transform_sc =(not use_sc_resample), shape = (-1, 1, 100, 100)), batch_size=32, shuffle =False, collate_fn= custom_collate_fn, pin_memory=pin_memory)
test_loader = DataLoader(FC_SCGraphDataset(Z20_test, 
                                           SC_test * use_cond_sc, 
                                           Z_t_test * use_cond_3, 
                                           Cov_test,
                                             y_test, age_dim = 126, 
                                               transform_sc = (not use_sc_resample), shape = (-1, 1, 100, 100)), batch_size=32, shuffle =False, collate_fn= custom_collate_fn, pin_memory=pin_memory)

## FiLM loader

training_loaderfm = DataLoader(FC_SC_vec_Dataset(Z20_train, 
                                                 SC_train * use_cond_sc, 
                                                 Z_t_train * use_cond_3, 
                                                 Cov_train,
                                                   y_train,age_dim = 126, 
                                               transform_sc = (not use_sc_resample), shape = (-1, 1, 100, 100)), batch_size=32, shuffle =True, pin_memory=pin_memory)
validation_loaderfm = DataLoader(FC_SC_vec_Dataset(Z20_val, 
                                                   SC_val * use_cond_sc, 
                                                   Z_t_val * use_cond_3, 
                                                   Cov_val, 
                                                   y_val, age_dim = 126, 
                                               transform_sc = (not use_sc_resample), shape = (-1, 1, 100, 100)), batch_size=32, shuffle =False, pin_memory=pin_memory)
test_loaderfm = DataLoader(FC_SC_vec_Dataset(Z20_test, 
                                             SC_test * use_cond_sc, 
                                             Z_t_test * use_cond_3, 
                                             Cov_test, 
                                             y_test, age_dim = 126, 
                                               transform_sc = (not use_sc_resample), shape = (-1, 1, 100, 100)), batch_size=32, shuffle =False, pin_memory=pin_memory)



#### FiLM version
if model_method == 'fm':
    dit_fm = dit_film(seq_len = diffusion_config['DIT_config_cat']['seq_len'], seq_channels=diffusion_config['DIT_config_cat']['seq_channels'], config = diffusion_config['DIT_config_film']).to(device)
    ddpm_fm = ddpm(network = dit_fm, n_steps = diffusion_config['DDPM_config']['n_steps'], min_beta = diffusion_config['DDPM_config']['min_beta'],
                max_beta=diffusion_config['DDPM_config']['max_beta'], schedule='linear', device = device, 
                vector_cl = (diffusion_config['DDPM_config']['vector_c'], diffusion_config['DDPM_config']['vector_l'])).to(device)
    optimizer = torch.optim.Adam(ddpm_fm.parameters(), lr=1e-4)
    ddpm_fm.train_ddpm_amp_timestep(loader = training_loaderfm, loader_val = validation_loaderfm, n_epochs = 500,
                        optimizer= optimizer, patience = 25, accumulation_steps = 1, use_scheduler= True, debug = False, 
                        store_path = MAIN_MODEL_DIR + name + '_fm.pt')



### Graph version  ####
if model_method == 'graph':
    graph_enc = SCGraphModel1D(args = diffusion_config['Graph_encoder_config'])
    dit_cross = dit_cat(seq_len = diffusion_config['DIT_config_cat']['seq_len'], seq_channels=diffusion_config['DIT_config_cat']['seq_channels'], config = diffusion_config['DIT_config_cat']).to(device)
    ddpm_g = ddpm_graph(network = dit_cross, GraphEncoder= graph_enc, n_steps = diffusion_config['DDPM_config']['n_steps'], min_beta = diffusion_config['DDPM_config']['min_beta'],
                max_beta=diffusion_config['DDPM_config']['max_beta'], schedule='linear', device = device, 
                vector_cl = (diffusion_config['DDPM_config']['vector_c'], diffusion_config['DDPM_config']['vector_l'])).to(device)

    optimizer = torch.optim.Adam(ddpm_g.parameters(), lr=1e-4)
    ddpm_g.train_ddpm_amp(loader = training_loader, loader_val = validation_loader, n_epochs = 500,
                        optimizer= optimizer, patience = 25, accumulation_steps = 1, use_scheduler= True, debug = False, 
                        store_path = MAIN_MODEL_DIR + name + '_graphV2.pt')
