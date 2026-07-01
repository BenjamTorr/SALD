import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import plotly.graph_objects as go
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import DataParallel as GeoDataParallel
from torch.utils.checkpoint import checkpoint
import random
import os
import torch
import math
from collections import defaultdict
from contextlib import nullcontext

def cosine_beta_schedule(n_steps, s=0.008, device="cpu"):
    steps = n_steps + 1
    x = torch.linspace(0, n_steps, steps, device=device) / n_steps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize so that alphā(0)=1

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)  # numerical stability



class ddpm_graph(nn.Module):
    def __init__(self, network, GraphEncoder, n_steps=200, min_beta=10 ** -4, max_beta=0.02, schedule = 'linear', device=None, vector_cl=(1, 5151)):
        super(ddpm_graph, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.vector_cl = vector_cl
        self.c = vector_cl[0]
        self.l = vector_cl[1]
        self.network = network.to(device)
        self.graph_encoder = GraphEncoder.to(device)
        if schedule == 'linear':
            self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        else:
            self.betas = cosine_beta_schedule(n_steps, s=0.008, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).clamp(min=1e-8)
        self.scheduler = None


    def construct_x0(self, noisy, t, eta_theta):
        n, _, _ = noisy.shape
        a_t_bar = self.alpha_bars[t].to(self.device)
        x0 = (1 / a_t_bar.sqrt()).reshape(n, 1, 1) * noisy - ((1 - a_t_bar) / (a_t_bar)).sqrt().reshape(n, 1, 1) * eta_theta
        return x0

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, l = x0.shape
        a_bar = self.alpha_bars[t].to(self.device)

        if eta is None:
            eta = torch.randn(n, c, l).to(self.device)
        if eta.device != self.device:
            eta.to(self.device)
            
        noisy = a_bar.sqrt().reshape(n, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1) * eta
        return noisy

    def backward(self, x, t, cond1, cond2, cov_cond):
        input = torch.cat((x, cond2), dim=1) #
        return self.network(input, t, cov_cond, cond1)


    def sample_repeated_chunked(self, cond1_data, cond2, cov_cond, n=1, chunk_size=32, amp = True):
        self.network.eval()
        with torch.no_grad():
            cond1_data = cond1_data.to(self.device)
            cond2 = cond2.to(self.device)
            cov_cond = cov_cond.to(self.device)
            if amp:
                with torch.cuda.amp.autocast():
                    cond1 = self.graph_encoder(cond1_data.x.float(), cond1_data.edge_index, edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())
            else:
                cond1 = self.graph_encoder(cond1_data.x.float(), cond1_data.edge_index, edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())

            B, C, L = cond1.shape
            B2, C2, L2 = cond2.shape
            device = cond1.device

            cond1_rep_full = cond1.unsqueeze(1).repeat(1, n, 1, 1).reshape(B * n, C, L)
            cond2_rep_full = cond2.unsqueeze(1).repeat(1, n, 1, 1).reshape(B2 * n, C2, L2)
            cov_cond = cov_cond.unsqueeze(1).repeat(1, n, 1).reshape(B * n, cov_cond.shape[1])

            total_samples = B * n
            x_final = torch.zeros(total_samples, self.c, self.l, device=device)

            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                cond1_chunk = cond1_rep_full[start:end]
                cond2_chunk = cond2_rep_full[start:end]
                cov_cond_chunk = cov_cond[start:end]
                x = torch.randn(end - start, self.c, self.l, device=device)

                for idx, t in enumerate(tqdm(list(range(self.n_steps))[::-1], desc=f"Sampling [{start}:{end}]", total=self.n_steps)):
                    t_tensor = (t * torch.ones(end - start, device=device)).long()
                    if amp:
                        with torch.cuda.amp.autocast():
                            eta_theta = self.backward(x, t_tensor, cond1_chunk, cond2_chunk, cov_cond_chunk)
                    else:
                        eta_theta = self.backward(x, t_tensor, cond1_chunk, cond2_chunk, cov_cond_chunk)

                    alpha_t = self.alphas[t]
                    alpha_t_bar = self.alpha_bars[t]

                    x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                    if t > 0:
                        z = torch.randn(end - start, self.c, self.l, device=device)
                        beta_t = self.betas[t]
                        prev_alpha_t_bar = self.alpha_bars[t - 1]
                        beta_tilda_t = ((1 - prev_alpha_t_bar) / (1 - alpha_t_bar)) * beta_t
                        sigma_t = beta_tilda_t.sqrt()
                        x = x + sigma_t * z

                x_final[start:end] = x.float()

            return x_final.reshape(B2, n, self.c, self.l)

    def sample_repeated_chunked_ddim(self, cond1_data, cond2, cov_cond, denoising_steps=1000, eta=0,
        n=1,
        chunk_size=32,
        amp=True,
        precision="bf16",
        grad=False,
        slice_index=None,
    ):
        self.network.eval()
        self.graph_encoder.eval()

        device = self.device
        cond1_data = cond1_data.to(device)
        cond2 = cond2.to(device)
        cov_cond = cov_cond.to(device)
        if precision not in {"fp16", "bf16"}:
            raise ValueError(f"Unsupported precision='{precision}'. Use 'fp16' or 'bf16'.")
        amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)) if (amp and str(device).startswith("cuda")) else (lambda: nullcontext())
        steps = torch.linspace(0, self.n_steps - 1, denoising_steps, device=device).round().long()

        with torch.set_grad_enabled(grad):
            # Encode cond1
            if amp:
                with amp_ctx():
                    cond1 = self.graph_encoder(cond1_data.x.float(), cond1_data.edge_index, edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())
            else:
                cond1 = self.graph_encoder(cond1_data.x.float(), cond1_data.edge_index, edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())

            B, C, L = cond1.shape
            B2, S, C2, L2 = cond2.shape

            if slice_index is not None:
                if not (0 <= slice_index < S):
                    raise ValueError(f"slice_index={slice_index} out of range for S={S}")
                cond2 = cond2[:, slice_index : slice_index + 1]
                S_eff = 1
            else:
                S_eff = S

            # Step 1: repeat cond1 and cov S times (one copy per slice)
            cond1_S = cond1.repeat_interleave(S_eff, dim=0)      # (B*S_eff, C, L)
            cov_S   = cov_cond.repeat_interleave(S_eff, dim=0)   # (B*S_eff, D)

            # Step 2: repeat those n times
            cond1_rep_full = cond1_S.unsqueeze(1).repeat(1, n, 1, 1)\
                                            .reshape(B * S_eff * n, C, L)

            cov_cond   = cov_S.unsqueeze(1).repeat(1, n, 1)\
                                            .reshape(B * S_eff * n, cov_cond.shape[1])

            # Step 3: cond2 already has S slices → flatten first
            cond2_flat = cond2.reshape(B2 * S_eff, C2, L2)

            # Step 4: repeat cond2 n times (per slice)
            cond2_rep_full = cond2_flat.unsqueeze(1).repeat(1, n, 1, 1)\
                                                .reshape(B2 * S_eff * n, C2, L2)

            total_samples = B2 * n * S_eff
            x_final = torch.zeros(total_samples, self.c, self.l, device=device)

            # Process in chunks
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                cond1_chunk = cond1_rep_full[start:end]
                cond2_chunk = cond2_rep_full[start:end]
                cov_cond_chunk = cov_cond[start:end]

                # Initialize noise
                x = torch.randn(end - start, self.c, self.l, device=device)

                # Reverse timesteps including t=0
                reversed_steps = steps.flip(0)

                for idx, t in enumerate(tqdm(reversed_steps, desc=f"Sampling [{start}:{end}]", total=len(reversed_steps))):
                    t_tensor = t * torch.ones(end - start, device=device, dtype=torch.long)

                    # Predict noise at current step
                    if amp:
                        with amp_ctx():
                            eta_theta = self.backward(x, t_tensor, cond1_chunk, cond2_chunk, cov_cond_chunk)
                    else:
                        eta_theta = self.backward(x, t_tensor, cond1_chunk, cond2_chunk, cov_cond_chunk)

                    alpha_t_bar = self.alpha_bars[t].to(device)

                    if idx < len(reversed_steps) - 1:
                        # Standard DDIM update for t>0
                        s_prev = reversed_steps[idx + 1]
                        alpha_t_bar_prev = self.alpha_bars[s_prev].to(device)

                        # Predict x0
                        x0 = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()

                        # Compute stochastic noise scale
                        sigma_t = eta * ((1 - alpha_t_bar_prev) / (1 - alpha_t_bar)).sqrt() * (1 - alpha_t_bar / alpha_t_bar_prev).sqrt()

                        # Update x
                        x = alpha_t_bar_prev.sqrt() * x0 + ((1 - alpha_t_bar_prev - sigma_t**2).clamp(min=1e-12)).sqrt() * eta_theta
                        z = torch.randn_like(x) if sigma_t.item() > 0 else torch.zeros_like(x)
                        x = x + sigma_t * z
                    else:
                        # Final step t=0, deterministic
                        x0 = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()
                        x = x0

                x_final[start:end] = x.float()

        return x_final.reshape(B2, S_eff, n, self.c, self.l)

    def train_ddpm_amp(self, loader, loader_val, n_epochs, optimizer, patience= 10, accumulation_steps = 1, use_scheduler = False, debug = False, store_path="models/ddpm_Graph_cond_model.pt"):
        mse = nn.MSELoss()
        #mse = nn.L1Loss(reduction='mean')
        best_loss = float("inf")
        n_steps = self.n_steps
        best_epoch = 0

        losses = []
        val_losses = []

        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss over Epochs')
        no_improvement_counter = 0

        if use_scheduler:
            self.scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            mode='min',
                            factor=0.5,       # Reduce LR by half
                            patience= 6,       # Wait 5 epochs before reducing
                            threshold=1e-4,   # Minimal significant improvement
                            cooldown=10,       # Wait 2 epochs after LR reduction
                            min_lr=1e-6)


        scaler = torch.amp.GradScaler('cuda')
        optimizer.zero_grad()
        for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
            grad_stats = {"network": [], "graph": []}
            self.network.train()
            self.graph_encoder.train()
            epoch_loss = 0.0
            for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
                # Loading data
                
                with torch.amp.autocast('cuda'):
                    x0, cond1_data, cond2, cov_cond, _ = batch
                    x0 = x0.to(self.device)
                    cov_cond = cov_cond.to(self.device).float()
                    cond1_data = cond1_data.to(self.device)
                    cond1_x = cond1_data.x.to(self.device).float()
                    cond1_edge_index = cond1_data.edge_index.to(self.device)

                    cond1 = self.graph_encoder(cond1_x, cond1_edge_index,edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())
                    B, S, C, _ = cond2.shape
                    s_idx = random.randint(0, S - 1)
                    cond2 = cond2[:, s_idx].to(self.device).reshape(B,C,-1)
                    n = len(x0)

                    # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
                    eta = torch.randn_like(x0).to(self.device)
                    t = torch.randint(0, self.n_steps, (n,)).to(self.device)

                    # Computing the noisy matrix based on x0 and the time-step (forward process)
                    noisy_imgs = self.forward(x0, t, eta)

                    eta_theta = self.backward(noisy_imgs, t, cond1, cond2, cov_cond)

                    # Optimizing the MSE between the noise plugged and the predicted noise 
                    #loss = mse(eta_theta, eta)
                    loss = mse(eta_theta, eta) / accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if debug:
                    # inside training loop per step
                    if step % accumulation_steps == 0:  # after backward+step
                        g_net, g_graph = 0.0, 0.0
                        for p in self.network.parameters():
                            if p.grad is not None:
                                g_net += p.grad.data.norm(2).item() ** 2
                        g_net = g_net ** 0.5

                        for p in self.graph_encoder.parameters():
                            if p.grad is not None:
                                g_graph += p.grad.data.norm(2).item() ** 2
                        g_graph = g_graph ** 0.5

                        grad_stats["network"].append(g_net)
                        grad_stats["graph"].append(g_graph)

                epoch_loss += loss.item() * len(x0) / len(loader.dataset)

            if debug:
                # at end of epoch
                mean_net = sum(grad_stats["network"]) / len(grad_stats["network"])
                mean_graph = sum(grad_stats["graph"]) / len(grad_stats["graph"])
                print(f"[Epoch {epoch}] GradNorm Net: {mean_net:.2f}, GradNorm GE: {mean_graph:.2f}")

            ## adding validation
            self.network.eval()
            self.graph_encoder.eval()
            epoch_val_loss = 0

            with torch.no_grad():
                for _, batch in enumerate(tqdm(loader_val, leave=False, desc=f"Validation Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
                    # Loading data
                    with torch.amp.autocast('cuda'):
                        x0, cond1_data, cond2, cov_cond, _  = batch
                        x0 = x0.to(self.device)
                        cov_cond = cov_cond.to(self.device).float()
                        cond1_x = cond1_data.x.to(self.device).float()
                        cond1_data = cond1_data.to(self.device)
                        cond1_edge_index = cond1_data.edge_index.to(self.device)
                        cond1 = self.graph_encoder(cond1_x, cond1_edge_index, edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())
                        B, S, C, _ = cond2.shape
                        s_idx = random.randint(0, S - 1)
                        cond2 = cond2[:, s_idx].to(self.device).reshape(B,C,-1)
                        n = len(x0)

                        # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
                        eta = torch.randn_like(x0).to(self.device)
                        t = torch.randint(0, self.n_steps, (n,)).to(self.device)

                        # Computing the noisy matrix based on x0 and the time-step (forward process)
                        noisy_imgs = self.forward(x0, t, eta)
                        eta_theta = self.backward(noisy_imgs, t, cond1, cond2, cov_cond)

                        # Optimizing the MSE between the noise plugged and the predicted noise
                        val_loss = mse(eta_theta, eta)
                    epoch_val_loss += val_loss.item() * len(x0) / len(loader_val.dataset)
            
            val_losses.append(epoch_val_loss)

            if use_scheduler:
                self.scheduler.step(epoch_val_loss)

            # Storing the model
            if best_loss > epoch_val_loss:
                best_epoch = epoch + 1
                best_loss = epoch_val_loss
                no_improvement_counter = 0
                torch.save(self.state_dict(), store_path)
            else:
                no_improvement_counter += 1

            if (epoch) % 20 == 0:
                torch.save(self.state_dict(), store_path[:-4] + f"_epoch{epoch + 1}.pth")

            # Early stopping
            if no_improvement_counter >= patience:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

            best_loss_message = f"Best validaton loss at epoch {best_epoch} with loss {best_loss:.4f}"
            current_val_loss_message = f"Current training loss: {epoch_loss:.4f}, Validation loss: {epoch_val_loss:.4f}"

            # Plotting
            ax.clear()
            losses.append(epoch_loss)
            clear_output(wait=True)

            max_y = np.percentile(np.concatenate([losses, val_losses]), 90)
            min_y = np.min(np.concatenate([losses, val_losses]))
            ax.plot(losses, label='Loss', color='blue')
            ax.plot(val_losses, label="Validation Loss", color="red")
            ax.legend(loc="upper right")
            ax.set_ylim(bottom=min_y, top=max_y)
            display(fig)

            print(best_loss_message)
            print(current_val_loss_message)
            curve_path = os.path.splitext(store_path)[0] + "_training_curve.pt"
            # Save as a dictionary
            torch.save(
                {
                    "train_loss": losses,
                    "val_loss": val_losses
                },
                curve_path
            )


        plt.show()

    def train_ddpm_amp_timestep(self, loader, loader_val, n_epochs, optimizer, patience=10, accumulation_steps=1, use_scheduler=False, debug=False, store_path="models/ddpm_Graph_cond_model.pt"):
        """AMP training with per-timestep average loss logging."""
        mse = nn.MSELoss()
        best_loss = float("inf")
        best_epoch = 0
        losses, val_losses = [], []
        no_improvement_counter = 0

        train_timestep_records = []
        val_timestep_records = []
        timestep_curve_path = os.path.splitext(store_path)[0] + "_timestep_losses.pt"

        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss over Epochs")

        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=6,
                threshold=1e-4,
                cooldown=10,
                min_lr=1e-6,
            )

        scaler = torch.amp.GradScaler("cuda")
        optimizer.zero_grad()

        for epoch in tqdm(range(n_epochs), desc="Training progress", colour="#00ff00"):
            grad_stats = {"network": [], "graph": []}
            self.network.train()
            self.graph_encoder.train()
            epoch_loss = 0.0
            train_loss_sum, train_loss_count = defaultdict(float), defaultdict(int)

            for step, batch in enumerate(
                tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")
            ):
                with torch.amp.autocast("cuda"):
                    x0, cond1_data, cond2, cov_cond, _ = batch
                    x0 = x0.to(self.device)
                    cov_cond = cov_cond.to(self.device).float()
                    cond1_x = cond1_data.x.to(self.device).float()
                    cond1_edge_index = cond1_data.edge_index.to(self.device)
                    cond1_data = cond1_data.to(self.device)

                    cond1 = self.graph_encoder(cond1_x, cond1_edge_index, edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())
                    B, S, C, _ = cond2.shape
                    s_idx = random.randint(0, S - 1)
                    cond2 = cond2[:, s_idx].to(self.device).reshape(B, C, -1)
                    n = len(x0)

                    eta = torch.randn_like(x0).to(self.device)
                    t = torch.randint(0, self.n_steps, (n,)).to(self.device)

                    noisy_imgs = self.forward(x0, t, eta)
                    eta_theta = self.backward(noisy_imgs, t, cond1, cond2, cov_cond)

                    per_sample_loss = (eta_theta - eta).pow(2).flatten(start_dim=1).mean(dim=1)
                    loss = per_sample_loss.mean() / accumulation_steps

                scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if debug and step % accumulation_steps == 0:
                    g_net, g_graph = 0.0, 0.0
                    for p in self.network.parameters():
                        if p.grad is not None:
                            g_net += p.grad.data.norm(2).item() ** 2
                    g_net = g_net ** 0.5

                    for p in self.graph_encoder.parameters():
                        if p.grad is not None:
                            g_graph += p.grad.data.norm(2).item() ** 2
                    g_graph = g_graph ** 0.5

                    grad_stats["network"].append(g_net)
                    grad_stats["graph"].append(g_graph)

                epoch_loss += loss.item() * len(x0) / len(loader.dataset)

                t_cpu = t.detach().cpu()
                per_sample_loss_cpu = per_sample_loss.detach().cpu()
                for sample_loss, t_val in zip(per_sample_loss_cpu, t_cpu):
                    train_loss_sum[int(t_val)] += sample_loss.item()
                    train_loss_count[int(t_val)] += 1

            if debug and grad_stats["network"]:
                mean_net = sum(grad_stats["network"]) / len(grad_stats["network"])
                mean_graph = sum(grad_stats["graph"]) / len(grad_stats["graph"])
                print(f"[Epoch {epoch}] GradNorm Net: {mean_net:.2f}, GradNorm GE: {mean_graph:.2f}")

            self.network.eval()
            self.graph_encoder.eval()
            epoch_val_loss = 0.0
            val_loss_sum, val_loss_count = defaultdict(float), defaultdict(int)

            with torch.no_grad():
                for _, batch in enumerate(
                    tqdm(loader_val, leave=False, desc=f"Validation Epoch {epoch + 1}/{n_epochs}", colour="#005500")
                ):
                    with torch.amp.autocast("cuda"):
                        x0, cond1_data, cond2, cov_cond, _ = batch
                        x0 = x0.to(self.device)
                        cov_cond = cov_cond.to(self.device).float()
                        cond1_x = cond1_data.x.to(self.device).float()
                        cond1_edge_index = cond1_data.edge_index.to(self.device)
                        cond1_data = cond1_data.to(self.device)
                        cond1 = self.graph_encoder(cond1_x, cond1_edge_index, edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())
                        B, S, C, _ = cond2.shape
                        s_idx = random.randint(0, S - 1)
                        cond2 = cond2[:, s_idx].to(self.device).reshape(B, C, -1)
                        n = len(x0)

                        eta = torch.randn_like(x0).to(self.device)
                        t = torch.randint(0, self.n_steps, (n,)).to(self.device)

                        noisy_imgs = self.forward(x0, t, eta)
                        eta_theta = self.backward(noisy_imgs, t, cond1, cond2, cov_cond)

                        per_sample_loss = (eta_theta - eta).pow(2).flatten(start_dim=1).mean(dim=1)
                        val_loss = per_sample_loss.mean()

                    epoch_val_loss += val_loss.item() * len(x0) / len(loader_val.dataset)

                    t_cpu = t.detach().cpu()
                    per_sample_loss_cpu = per_sample_loss.detach().cpu()
                    for sample_loss, t_val in zip(per_sample_loss_cpu, t_cpu):
                        val_loss_sum[int(t_val)] += sample_loss.item()
                        val_loss_count[int(t_val)] += 1

            for t_val, total in train_loss_sum.items():
                avg_loss = total / max(1, train_loss_count[t_val])
                train_timestep_records.append({"epoch": epoch + 1, "t": int(t_val), "avg_loss": avg_loss})

            for t_val, total in val_loss_sum.items():
                avg_loss = total / max(1, val_loss_count[t_val])
                val_timestep_records.append({"epoch": epoch + 1, "t": int(t_val), "avg_loss": avg_loss})

            val_losses.append(epoch_val_loss)

            if use_scheduler:
                self.scheduler.step(epoch_val_loss)

            if best_loss > epoch_val_loss:
                best_epoch = epoch + 1
                best_loss = epoch_val_loss
                no_improvement_counter = 0
                torch.save(self.state_dict(), store_path)
            else:
                no_improvement_counter += 1

            if epoch % 20 == 0:
                torch.save(self.state_dict(), store_path[:-4] + f"_epoch{epoch + 1}.pth")

            if no_improvement_counter >= patience:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch + 1} (no improvement for {patience} epochs).")
                break

            best_loss_message = f"Best validaton loss at epoch {best_epoch} with loss {best_loss:.4f}"
            current_val_loss_message = f"Current training loss: {epoch_loss:.4f}, Validation loss: {epoch_val_loss:.4f}"

            ax.clear()
            losses.append(epoch_loss)
            clear_output(wait=True)

            max_y = np.percentile(np.concatenate([losses, val_losses]), 90)
            min_y = np.min(np.concatenate([losses, val_losses]))
            ax.plot(losses, label="Loss", color="blue")
            ax.plot(val_losses, label="Validation Loss", color="red")
            ax.legend(loc="upper right")
            ax.set_ylim(bottom=min_y, top=max_y)
            display(fig)

            print(best_loss_message)
            print(current_val_loss_message)
            curve_path = os.path.splitext(store_path)[0] + "_training_curve.pt"
            torch.save({"train_loss": losses, "val_loss": val_losses}, curve_path)
            torch.save(
                {
                    "train_timestep": train_timestep_records,
                    "val_timestep": val_timestep_records,
                },
                timestep_curve_path,
            )
        plt.show()
        
    def fine_tune_DRAFT_prediction(
        self,
        loader,
        loader_val,
        n_epochs,
        optimizer,
        guide_model,
        decoder,
        denoising_steps=1000,
        K=5,
        patience=10,
        accumulation_steps=1,
        lambd=0.05,
        include_diff_loss = True, 
        L1 = False,
        warmup_iters=200,
        LV = True,
        n_rep = 2,
        m = 1,
        steps_log = 200,                  
        use_scheduler = True,
        evaluate_baseline = True,
        store_path="fine_tuning/models/feature.pt",
        target_norm: dict | None = None,
        use_fc20_as_target: bool = False,
    ):
        print(f'\nWorking on model saved on {store_path}\n')
        # ==========================================================
        # 🔹 Freeze auxiliary networks
        # ==========================================================
        guide_model.eval()
        decoder.eval()
        for p in guide_model.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False
        mse = nn.MSELoss()
        if L1:
            mse = nn.L1Loss()

        if target_norm is None:
            raise ValueError("target_norm must be provided with keys 'mean' and 'std'.")
        y_mean = target_norm["mean"].to(self.device)
        y_std = target_norm["std"].to(self.device)

        
        best_loss, best_step = float("inf"), 0

        losses, val_losses = [], []
        no_improvement_counter = 0

        # ==========================================================
        # 🔹 Learning-rate scheduler: linear warmup → cosine annealing
        # ==========================================================

        steps_per_epoch = len(loader)
        updates_per_epoch = steps_per_epoch // accumulation_steps
        total_updates = n_epochs * updates_per_epoch

        base_lr = optimizer.param_groups[0]['lr']
        eta_min = base_lr / 25

        # You want 4 cosine restart cycles
        num_cycles = 4
        T_0 = total_updates // num_cycles  # optimizer steps per cycle

        # Warmup is always applied; cosine annealing is optional.
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_iters
        )
        scheduler = warmup
        if use_scheduler:
            cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=1,
                eta_min=eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters]
            )

        scaler = torch.amp.GradScaler("cuda")
        optimizer.zero_grad()

        # ==========================================================
        # 🔹 Baseline: evaluate with LoRA weights = 0
        # ==========================================================

        best_loss = float("inf")
        if evaluate_baseline:
            print("\n🔍 Evaluating baseline (LoRA weights = 0)...")
            lora_params = []
            for name, param in self.named_parameters():
                if "lora" in name.lower():
                    lora_params.append((param, param.data.clone()))
                    param.data.zero_()

            self.network.eval()
            self.graph_encoder.eval()
            baseline_loss, baseline_feat_loss_norm, baseline_feat_loss_raw = 0.0, 0.0, 0.0
            
            with torch.no_grad():
                for step, batch in enumerate(tqdm(loader_val, desc="Baseline eval", colour="#ffaa00", leave=False)):
                    with torch.amp.autocast("cuda"):
                        real, cond1_data, cond2, cov_cond, target = batch
                        real = real.to(self.device).float()
                        cov_cond = cov_cond.to(self.device).float()
                        cond1_x = cond1_data.x.to(self.device).float()
                        cond1_edge_index = cond1_data.edge_index.to(self.device, dtype=torch.long)
                        cond1_edge_attr = cond1_data.edge_attr.to(self.device).float()
                        cond1_data = cond1_data.to(self.device)
                        cond1 = self.graph_encoder(cond1_x, cond1_edge_index,edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())
                        cond2 = cond2.to(self.device)
                        B, S, C, _ = cond2.shape
                        s_idx = random.randint(0, S - 1)
                        cond2 = cond2[:, s_idx].to(self.device).reshape(B, C, -1)
                        target = target.to(self.device).float()
                        target_z = (target - y_mean) / y_std
                        n = len(cov_cond)
                        
                        loss_diff = torch.tensor(0.0, device=self.device)
                        if include_diff_loss:
                            encoder_output = decoder.encode(real)[1].detach()
                            z, _ = torch.chunk(encoder_output, 2, dim=1)

                            eta = torch.randn_like(z)
                            t = torch.randint(0, self.n_steps, (n,), device=self.device)
                            noisy = self.forward(z, t, eta)
                            eta_theta = self.backward(noisy, t, cond1, cond2, cov_cond)
                            loss_diff = mse(eta_theta, eta)  
                        
                        cov_cond_exp = cov_cond.repeat_interleave(m, dim=0)
                        cond1_exp = cond1.repeat_interleave(m, dim=0)
                        cond2_exp = cond2.repeat_interleave(m, dim=0)

                        # --- deterministic DDIM sample ---
                        steps = torch.linspace(0, self.n_steps - 1, denoising_steps, device=self.device).round().long()
                        x = torch.randn(n * m , self.c, self.l, device=self.device)
                        reversed_steps = steps.flip(0)

                        for idx, t in enumerate(reversed_steps):
                            t_tensor = t * torch.ones(n * m, device=self.device, dtype=torch.long)
                            eta_theta = self.backward(x, t_tensor, cond1_exp, cond2_exp, cov_cond_exp)
                            alpha_t_bar = self.alpha_bars[t].to(self.device)
                            if idx < len(reversed_steps) - 1:
                                s_prev = reversed_steps[idx + 1]
                                alpha_t_bar_prev = self.alpha_bars[s_prev].to(self.device)
                                x0 = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()
                                x = alpha_t_bar_prev.sqrt() * x0 + ((1 - alpha_t_bar_prev).clamp(min=1e-12)).sqrt() * eta_theta
                                
                            else:
                                x0 = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()
                                x = x0

                        x_decoded = decoder.decode(x)
                        feat_pred = guide_model(x_decoded.view(n, m, 1, 4950).mean(dim = 1))
                        feat_pred_norm = (feat_pred - y_mean) / y_std
                        feat_loss_norm = mse(target_z, feat_pred_norm)
                        feat_loss_raw = mse(target, feat_pred)

                        total_loss = loss_diff + lambd * feat_loss_norm
                        baseline_loss += total_loss.item() * n / len(loader_val.dataset)
                        baseline_feat_loss_norm += feat_loss_norm.item() * n / len(loader_val.dataset)
                        baseline_feat_loss_raw += feat_loss_raw.item() * n / len(loader_val.dataset)
            
            # Restore LoRA weights
            for param, saved_data in lora_params:
                param.data = saved_data.clone()
            
            print(
                f"✅ Baseline total loss (norm): {baseline_loss:.6f}, "
                f"feature loss norm: {baseline_feat_loss_norm:.6f}, "
                f"feature loss raw: {baseline_feat_loss_raw:.6f}\n"
            )
        # Save a fallback checkpoint so later loading does not fail even if
        # fine-tuning never improves beyond baseline.
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), store_path)
        # ==========================================================
        # 🔹 Training loop
        # ==========================================================
        fig, ax = plt.subplots()
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss over Epochs")

        reversed_steps = torch.linspace(0, self.n_steps - 1, denoising_steps, device=self.device).round().long().flip(0)
        best_loss = float("inf")
        step_count = 0
        step_loss = 0.0
        step_loss_count = 0
        for epoch in tqdm(range(n_epochs), desc="Training progress", colour="#00ff00"):
            self.network.train()
            self.graph_encoder.train()
            epoch_loss = 0.0
            for vvstep, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch+1}/{n_epochs}", colour="#005500")):
                self.network.train()
                self.graph_encoder.train()
                with torch.amp.autocast("cuda"):
                    real_FC, cond1_data, cond2, cov_cond, target = batch
                    real_FC = real_FC.to(self.device).float()
                    cov_cond = cov_cond.to(self.device).float()
                    cond1_x = cond1_data.x.to(self.device).float()
                    cond1_edge_index = cond1_data.edge_index.to(self.device)
                    cond1_edge_attr = cond1_data.edge_attr.to(self.device).float()
                    cond1_data = cond1_data.to(self.device)
                    cond1 = self.graph_encoder(cond1_x, cond1_edge_index,edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())
                    cond2 = cond2.to(self.device)
                    B, S, C, _ = cond2.shape
                    s_idx = random.randint(0, S - 1)
                    cond2 = cond2[:, s_idx].to(self.device).reshape(B, C, -1)
                    target = target.to(self.device).float()
                    target_z = (target - y_mean) / y_std


                    n = len(cov_cond)

                    loss_diff = torch.tensor(0.0, device=self.device)
                    if include_diff_loss:
                        encoder_output = decoder.encode(real_FC)[1].detach()
                        z, _ = torch.chunk(encoder_output, 2, dim=1)

                        eta = torch.randn_like(z)
                        t = torch.randint(0, self.n_steps, (n,), device=self.device)
                        noisy = self.forward(z, t, eta)
                        eta_theta = self.backward(noisy, t, cond1, cond2, cov_cond)
                        loss_diff = mse(eta_theta, eta) 

                    cov_cond_exp = cov_cond.repeat_interleave(m, dim=0)
                    cond1_exp = cond1.repeat_interleave(m, dim=0)
                    cond2_exp = cond2.repeat_interleave(m, dim=0)

                    # --- partial DDIM sampling ---
                    x = torch.randn(n * m, self.c, self.l, device=self.device)
                    with torch.no_grad():
                        for idx, tidx in enumerate(reversed_steps[:-K]):
                            t_tensor = tidx * torch.ones(n * m, device=self.device, dtype=torch.long)
                            eta_theta = self.backward(x, t_tensor, cond1_exp, cond2_exp, cov_cond_exp)
                            alpha_t_bar = self.alpha_bars[tidx]
                            s_prev = reversed_steps[idx + 1]
                            alpha_t_bar_prev = self.alpha_bars[s_prev]
                            x0 = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()
                            x = alpha_t_bar_prev.sqrt() * x0 + ((1 - alpha_t_bar_prev).clamp(min=1e-12)).sqrt() * eta_theta

                    x = x.detach().requires_grad_(True)
                    for idx, tidx in enumerate(reversed_steps[-K:]):
                        t_tensor = tidx * torch.ones(n * m, device=self.device, dtype=torch.long)
                        eta_theta = checkpoint(self.backward, x, t_tensor, cond1_exp, cond2_exp, cov_cond_exp, use_reentrant=False)
                        #eta_theta = self.backward(x, t_tensor, cond1, cond2, cov_cond)
                        alpha_t_bar = self.alpha_bars[tidx]
                        if idx < len(reversed_steps[-K:]) - 1:
                            s_prev = reversed_steps[idx + 1]
                            alpha_t_bar_prev = self.alpha_bars[s_prev]
                            x0 = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()
                            x = alpha_t_bar_prev.sqrt() * x0 + ((1 - alpha_t_bar_prev).clamp(min=1e-12)).sqrt() * eta_theta
                        else:
                            x = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()
                    # --- feature reconstruction loss ---

                    x_decoded = decoder.decode(x)
                    feat_pred = guide_model(x_decoded.view(n, m, 1, 4950).mean(dim = 1))
                    feat_pred_norm = (feat_pred - y_mean) / y_std
                    loss_feat = mse(target_z, feat_pred_norm)

                    loss_LV = torch.tensor(0.0, device=self.device)
                    if LV and n_rep > 0:
                        x0_det = x.detach()
                        for _ in range(n_rep):
                            # go one step behind 
                            noise = torch.randn_like(x0_det)
                            x1 = alpha_t_bar.sqrt() * x0_det + (1 - alpha_t_bar).sqrt() * noise
                            eta_theta = checkpoint(self.backward, x1, t_tensor, cond1_exp, cond2_exp, cov_cond_exp, use_reentrant=False)
                            x_new = (x1 - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()
                            x_new_decoded = decoder.decode(x_new)
                            x_new_decoded_mean = x_new_decoded.view(n, m, 1, 4950).mean(dim = 1)
                            feat_pred = guide_model(x_new_decoded_mean)
                            feat_pred_norm = (feat_pred - y_mean) / y_std
                            loss_LV += mse(target_z, feat_pred_norm)

                    loss = (loss_diff + lambd * (loss_feat + loss_LV) / (n_rep + 1)) / accumulation_steps

                # --- backward + optimizer step ---
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * n * accumulation_steps / len(loader.dataset)
                step_loss += loss.item() * accumulation_steps 
                step_loss_count += 1

                if (vvstep + 1) % accumulation_steps == 0 or (vvstep + 1) == len(loader):
                    step_count += 1
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    #print(f"[DEBUG] Optimizer step {step_count} triggered at batch {vvstep+1}")

                    if step_count  % steps_log == 0:
                        #print(f"[DEBUG] Validation triggered {step_count} triggered at steps_log {steps_log}")
                        # ======================================================
                        # 🔹 Validation
                        # ======================================================
                        self.network.eval()
                        self.graph_encoder.eval()
                        val_loss, val_feat_loss, val_feat_loss_raw = 0.0, 0.0, 0.0

                        with torch.no_grad():
                            for vstep, batch in enumerate(tqdm(loader_val, leave=False, desc="Validation", colour="#005500")):
                                with torch.amp.autocast("cuda"):
                                    real_FC, cond1_data, cond2, cov_cond, target = batch
                                    real_FC = real_FC.to(self.device).float()
                                    cov_cond = cov_cond.to(self.device).float()
                                    cond1_x = cond1_data.x.to(self.device).float()
                                    cond1_edge_index = cond1_data.edge_index.to(self.device)
                                    cond1_edge_attr = cond1_data.edge_attr.to(self.device).float()
                                    cond1_data = cond1_data.to(self.device)
                                    cond1 = self.graph_encoder(cond1_x, cond1_edge_index,edge_weight=cond1_data.edge_weight.float(), edge_attr=cond1_data.edge_attr.float())
                                    cond2 = cond2.to(self.device)
                                    B, S, C, _ = cond2.shape
                                    s_idx = random.randint(0, S - 1)
                                    cond2 = cond2[:, s_idx].to(self.device).reshape(B, C, -1)
                                    target = target.to(self.device).float()
                                    target_z = (target - y_mean) / y_std
                                    n = len(cov_cond)

                                    loss_diff = torch.tensor(0.0, device=self.device)
                                    if include_diff_loss:
                                        encoder_output = decoder.encode(real_FC)[1].detach()
                                        z, _ = torch.chunk(encoder_output, 2, dim=1)

                                        eta = torch.randn_like(z)
                                        t = torch.randint(0, self.n_steps, (n,), device=self.device)
                                        noisy = self.forward(z, t, eta)
                                        eta_theta = self.backward(noisy, t, cond1, cond2, cov_cond)
                                        loss_diff = mse(eta_theta, eta) 

                                    cov_cond_exp = cov_cond.repeat_interleave(m, dim=0)
                                    cond1_exp = cond1.repeat_interleave(m, dim=0)
                                    cond2_exp = cond2.repeat_interleave(m, dim=0)

                                    # --- deterministic DDIM sample ---
                            
                                    x = torch.randn(n * m , self.c, self.l, device=self.device)
                                    for idx, t in enumerate(reversed_steps):
                                        t_tensor = t * torch.ones(n * m, device=self.device, dtype=torch.long)
                                        eta_theta = self.backward(x, t_tensor, cond1_exp, cond2_exp, cov_cond_exp)
                                        alpha_t_bar = self.alpha_bars[t].to(self.device)
                                        if idx < len(reversed_steps) - 1:
                                            s_prev = reversed_steps[idx + 1]
                                            alpha_t_bar_prev = self.alpha_bars[s_prev].to(self.device)
                                            x0 = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()
                                            x = alpha_t_bar_prev.sqrt() * x0 + ((1 - alpha_t_bar_prev).clamp(min=1e-12)).sqrt() * eta_theta 
                                        else:
                                            x0 = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()
                                            x = x0

                                    x_decoded = decoder.decode(x)
                                    feat_pred = guide_model(x_decoded.view(n, m, 1, 4950).mean(dim = 1))
                                    feat_pred_norm = (feat_pred - y_mean) / y_std
                                    loss_feat_norm = mse(target_z, feat_pred_norm)
                                    loss_feat_raw = mse(target, feat_pred)
                                    loss_total = loss_diff + lambd * loss_feat_norm

                                    val_loss += loss_total.item() * n / len(loader_val.dataset)
                                    val_feat_loss += loss_feat_norm.item() * n / len(loader_val.dataset)
                                    val_feat_loss_raw += loss_feat_raw.item() * n / len(loader_val.dataset)
                        # ======================================================
                        # 🔹 Early stopping & logging
                        # ======================================================
                        if val_feat_loss < best_loss:
                            best_loss, best_step = val_feat_loss, step_count 
                            no_improvement_counter = 0
                            torch.save(self.state_dict(), store_path)
                        else:
                            no_improvement_counter += 1

                        if step_count % (5 * steps_log) == 0:
                            torch.save(self.state_dict(), store_path[:-3] + f"_step{step_count + 1}.pt")
                        if no_improvement_counter >= patience:
                            print(f"⏹️ Early stopping after {no_improvement_counter} validation intervals ({step_count} steps total).")
                            break

                        current_lr = optimizer.param_groups[0]['lr']
                        print(
                            f"Step {step_count:5d} | LR={current_lr:.2e} | "
                            f"Train={step_loss / step_loss_count:.4f} | Val={val_loss:.4f} | "
                            f"Val_feat_norm={val_feat_loss:.4f} | Val_feat_raw={val_feat_loss_raw:.4f} | "
                            f"Best (norm)={best_loss:.4f} (step {best_step})"
                        )

                        # --- Plot & print ---
                        losses.append(step_loss / step_loss_count)
                        val_losses.append(val_loss)
                        clear_output(wait=True)
                        ax.clear()
                        ax.plot(losses, label="Train", color="blue")
                        ax.plot(val_losses, label="Val", color="red")
                        ax.legend(loc="upper right")

                        # 🔹 Save (overwrite) the figure with model identifier
                        save_dir = "scripts/plots"
                        os.makedirs(save_dir, exist_ok=True)

                        # Extract everything before .pt or .pth (no extension)
                        filename = os.path.basename(store_path)
                        if filename.endswith(".pt") or filename.endswith(".pth"):
                            model_id = filename.rsplit(".", 1)[0]  # remove last extension only
                        else:
                            model_id = filename  # fallback if extension missing

                        save_plot_path = os.path.join(save_dir, f"loss_plot_{model_id}_graph.png")
                        fig.savefig(save_plot_path, dpi=150, bbox_inches="tight")
                        step_loss = 0.0
                        step_loss_count = 0


        
