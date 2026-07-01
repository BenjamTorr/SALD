from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random

# Loss function
def kl_div( mu, logvar):
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    return kl_div

def sample_slice(S, p0=0.7):
    if random.random() < p0:
        return 0
    return random.randint(1, S - 1)

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock1D(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels, cross_attn=False, context_dim=None, 
                 down_kernel_size=4,down_stride=2, down_padding=1):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
        
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv1d(out_channels, out_channels,
                                          down_kernel_size, down_stride, down_padding) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            if self.attn:
                # Attention block of Unet
                batch_size, channels, L = out.shape
                in_attn = out.reshape(batch_size, channels, L)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, L)
                out = out + out_attn
            
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, L = out.shape
                in_attn = out.reshape(batch_size, channels, L)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, L)
                out = out + out_attn
            
        # Downsample
        out = self.down_sample_conv(out)
        return out


class MidBlock1D(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels, cross_attn=None, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels)
             for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )
    
    def forward(self, x, t_emb=None, context=None):
        out = x
        
        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, L = out.shape
            in_attn = out.reshape(batch_size, channels, L)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, L)
            out = out + out_attn
            
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, L = out.shape
                in_attn = out.reshape(batch_size, channels, L)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, L)
                out = out + out_attn
                
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        
        return out


class UpBlock1D(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels,
                 up_kernel_size=4, up_stride=2, up_padding=1, output_padding=0):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.output_padding = output_padding
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose1d(in_channels, in_channels,
                                                 up_kernel_size, up_stride, up_padding, output_padding) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down=None, t_emb=None):
        # Upsample
        x = self.up_sample_conv(x)
        
        # Concat with Downblock output
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Self Attention
            if self.attn:
                batch_size, channels, L = out.shape
                in_attn = out.reshape(batch_size, channels, L)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, L)
                out = out + out_attn
        return out

class vae_unet(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        self.conv_kernel_size = model_config['conv_kernel_size']
        self.conv_kernel_stride = model_config['conv_kernel_strides']
        self.transpose_kernel_size = model_config['transpose_kernel_size']
        self.transpose_kernel_stride = model_config['transpose_kernel_strides']

        self.output_padding = [0 for _ in range(len(self.down_sample))]
        if 'output_padding' in model_config:
            self.output_padding = model_config['output_padding']
        
        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['attn_down']
        
        # Latent Dimension
        self.z_channels = model_config['z_channels']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        
        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))
        
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv1d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock1D(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels,
                                                 down_kernel_size=self.conv_kernel_size[i],
                                                 down_stride=self.conv_kernel_stride[i],
                                                 down_padding=1))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock1D(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv1d(self.down_channels[-1], 2*self.z_channels, kernel_size=3, padding=1)
        
        # Latent Dimension is 2*Latent because we are predicting mean & variance
        self.pre_quant_conv = nn.Conv1d(2*self.z_channels, 2*self.z_channels, kernel_size=1)
        ####################################################
        
        
        ##################### Decoder ######################
        self.post_quant_conv = nn.Conv1d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv1d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=1)
        
        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock1D(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock1D(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i - 1],
                                               norm_channels=self.norm_channels,
                                               up_kernel_size=self.transpose_kernel_size[i - 1],
                                               up_stride=self.transpose_kernel_stride[i - 1],
                                               up_padding=1, output_padding=self.output_padding[i - 1]))
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv1d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
    
    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        return sample, out
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, encoder_output = self.encode(x)
        out = self.decode(z)
        return out, encoder_output

    def get_embeddings_and_reconstructions(self, loader, device):
        """
        Extract latent samples (z) and reconstructions for every element in the
        provided loader. Works with the same dataset structure as training
        (x0, _, xt, _), but processes all time steps in xt instead of sampling
        a single s_idx. Returns a dict of concatenated tensors on CPU.
        """
        self.eval()

        x0_embeddings, x0_recons = [], []
        xt_embeddings, xt_recons = [], []

        amp_context = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Embedding+recon extraction", leave=False):
                x0, _, xt, _ = batch
                x0 = x0.to(device, non_blocking=True)
                xt = xt.to(device, non_blocking=True)

                with amp_context:
                    # Current static sample
                    _, encoder_out_x0 = self.encode(x0)
                    mean_x0, _ = torch.chunk(encoder_out_x0, 2, dim=1)
                    recon_x0 = self.decode(mean_x0)

                    # All dynamic samples (no random s_idx)
                    B, S, D = xt.shape
                    xt_all = xt.reshape(B * S, 1, D)
                    _, encoder_out_xt = self.encode(xt_all)
                    mean_xt, _ = torch.chunk(encoder_out_xt, 2, dim=1)
                    recon_xt = self.decode(mean_xt)

                x0_embeddings.append(mean_x0.detach().cpu())
                x0_recons.append(recon_x0.detach().cpu())

                xt_embeddings.append(mean_xt.detach().cpu().reshape(B, S, *mean_xt.shape[1:]))
                xt_recons.append(recon_xt.detach().cpu().reshape(B, S, *recon_xt.shape[1:]))

        return {
            "x0_embeddings": torch.cat(x0_embeddings, dim=0) if x0_embeddings else None,
            "x0_recons": torch.cat(x0_recons, dim=0) if x0_recons else None,
            "xt_embeddings": torch.cat(xt_embeddings, dim=0) if xt_embeddings else None,
            "xt_recons": torch.cat(xt_recons, dim=0) if xt_recons else None,
        }

    def train_vae_extended(
        self,
        loader,
        loader_val,
        n_epochs,
        optim,
        device,
        beta=1e-6,
        patience=10,
        accumulation_steps=2,
        use_scheduler=True,
        store_path="encoders/models/vae_unet.pth",
        plot_title=None,
        plot_filename=None,
    ):
        best_loss = float("inf")
        best_epoch = 0
        no_improvement_counter = 0

        losses = []
        val_losses = []

        title = plot_title or "Training Loss over Epochs"
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        
        if use_scheduler:
            self.scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            optim,
                                                            mode='min',
                                                            factor=0.5,       # Reduce LR by half
                                                            patience=5,       # Wait 5 epochs before reducing
                                                            threshold=1e-4,   # Minimal significant improvement
                                                            cooldown=2,       # Wait 2 epochs after LR reduction
                                                            min_lr=1e-6)
        else:
            self.scheduler = None
        amp_enabled = device.type == "cuda"
        amp_dtype = torch.bfloat16
        optim.zero_grad()
        for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
            epoch_loss = 0.0
            self.train()

            for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
                amp_context = (
                    torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with amp_context:
                    x0, _, xt, _ = batch
                    x0 = x0.to(device, non_blocking=True)
                    xt = xt.to(device, non_blocking=True)

                    B, S, _ = xt.shape

                    recon_loss = 0.0
                    kl_loss = 0.0

                    out, encoder_output = self.forward(x0)
                    mu, logvar = torch.chunk(encoder_output, 2, dim=1)
                    kl_loss += kl_div(mu, logvar)
                    recon_loss += F.mse_loss(x0, out)


                    s_idx = sample_slice(S, p0=0.85)
                    xt_single = xt[:, s_idx, :].reshape(B, 1, -1)

                    out, encoder_output = self.forward(xt_single)
                    mu, logvar = torch.chunk(encoder_output, 2, dim=1)
                    kl_loss += kl_div(mu, logvar)
                    recon_loss += F.mse_loss(xt_single, out)

                    kl_loss /= 2
                    recon_loss /= 2
                    true_loss = (recon_loss + beta * kl_loss) 
                    loss = (recon_loss + beta * kl_loss) / accumulation_steps
                loss.backward()
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                
                epoch_loss += true_loss.item() * len(x0) / len(loader.dataset)

            ## Validation
            self.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for step, batch in enumerate(tqdm(loader_val, leave=False, desc=f"Validation Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
                    amp_context = (
                        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                        if amp_enabled
                        else nullcontext()
                    )
                    with amp_context:
                        x0, _, xt, _  = batch
                        x0 = x0.to(device, non_blocking=True)
                        xt = xt.to(device, non_blocking=True)

                        out, _ = self.forward(x0)

                        B, S, _ = xt.shape
                        s_idx = 0
                        xt_single = xt[:, s_idx, :].reshape(B, 1, -1)
                        out3, _ = self.forward(xt_single)


                        val_loss = (F.mse_loss(out, x0) + F.mse_loss(out3, xt_single)) / 2
                    epoch_val_loss += val_loss.item() * len(x0) / len(loader_val.dataset)
            if use_scheduler:
                self.scheduler.step(epoch_val_loss)

            val_losses.append(epoch_val_loss)

            if epoch % 10 == 0:
                torch.save(self.state_dict(), store_path[:-4] + f"_epoch{epoch}.pth")

            # Save best model and reset patience
            if (best_loss - epoch_val_loss) > 1e-4:
                best_epoch = epoch + 1
                best_loss = epoch_val_loss
                no_improvement_counter = 0
                torch.save(self.state_dict(), store_path)
            else:
                no_improvement_counter += 1

            # Early stopping
            if no_improvement_counter >= patience:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

            best_loss_message = f"Best loss at epoch {best_epoch} with loss {best_loss:.4f}"
            current_val_loss_message = f"Current epoch {epoch+1} with val loss {epoch_val_loss:.4f}"
            current_training_loss_message = f"Current epoch {epoch+1} with training loss {epoch_loss:.4f}"
            # Plotting
            ax.clear()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(title)
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
            print(current_training_loss_message)
        torch.save(self.state_dict(), store_path[:-4] + f"_epoch{epoch}.pth")
        if plot_filename:
            plot_path = Path(plot_filename)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
