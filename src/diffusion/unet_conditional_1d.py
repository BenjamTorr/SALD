import torch
import torch.nn as nn

"""
Conditioned 1D U-Net that mirrors the input/output contract of the DiT models:
    forward(x, t, class_encoding, cond1=None) -> noise prediction with shape (B, out_channels, L)

Design choices:
- Operates on (batch, channels, length) sequences (e.g., 8 channels x 618 tokens).
- Time embedding is combined with a class embedding (age + sex).
- Structural connectivity (cond1) can be injected through cross-attention
  either in all blocks or only in the mid blocks (configurable).
"""


def get_time_embedding(time_steps, temb_dim):
    """
    Sinusoidal timestep embedding identical to the DiT variants.
    Args:
        time_steps: 1D tensor of shape (B,)
        temb_dim: even dimension of the embedding
    Returns:
        (B, temb_dim) tensor
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    factor = 10000 ** (
        torch.arange(0, temb_dim // 2, device=time_steps.device, dtype=torch.float32)
        / (temb_dim // 2)
    )
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


def _compute_lengths(seq_len, down_sample_flags, kernel=4, stride=2, padding=1):
    """Track sequence lengths after each down-sampling step."""
    lengths = [seq_len]
    length = seq_len
    for flag in down_sample_flags:
        if flag:
            length = ((length + 2 * padding - (kernel - 1) - 1) // stride) + 1
        lengths.append(length)
    return lengths


def _compute_output_paddings(lengths, kernel=4, stride=2, padding=1):
    """
    Compute output_padding values for ConvTranspose so that we invert the recorded lengths.
    `lengths` should be the result of `_compute_lengths`.
    """
    pads = []
    for idx in range(len(lengths) - 1, 0, -1):
        target = lengths[idx - 1]
        cur = lengths[idx]
        base = (cur - 1) * stride - 2 * padding + kernel
        pads.append(target - base)
    return pads


def _suggest_groups(channels: int) -> int:
    """Pick the largest typical group count that cleanly divides channels."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class DownBlock(nn.Module):
    """
    Residual + (self + cross) attention + optional downsample.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        cond_channels=None,
        down_sample=True,
        num_heads=4,
        num_layers=1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.cond_channels = cond_channels

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(_suggest_groups(in_channels if i == 0 else out_channels),
                                 in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
                for _ in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(_suggest_groups(out_channels), out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.self_attn_norms = nn.ModuleList(
            [nn.LayerNorm(out_channels) for _ in range(num_layers)]
        )
        self.self_attns = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )

        if cond_channels is not None:
            self.cross_proj = nn.ModuleList(
                [nn.Conv1d(cond_channels, out_channels, kernel_size=1) for _ in range(num_layers)]
            )
            self.cross_attn_norms = nn.ModuleList(
                [nn.LayerNorm(out_channels) for _ in range(num_layers)]
            )
            self.cross_attns = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
            )
        else:
            self.cross_proj = self.cross_attn_norms = self.cross_attns = None

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        self.down_sample_conv = (
            nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if self.down_sample
            else nn.Identity()
        )

    def forward(self, x, t_emb, context=None):
        out = x
        for i in range(self.num_layers):
            res_in = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](res_in)

            # self-attention over sequence dimension
            tokens = out.transpose(1, 2)  # (B, L, C)
            tokens = self.self_attn_norms[i](tokens)
            attn_out, _ = self.self_attns[i](tokens, tokens, tokens)
            out = out + attn_out.transpose(1, 2)

            # cross-attention with SC if provided
            if context is not None and self.cross_attns is not None:
                ctx = self.cross_proj[i](context).transpose(1, 2)  # (B, Lc, C)
                q = self.cross_attn_norms[i](out.transpose(1, 2))
                cross_out, _ = self.cross_attns[i](q, ctx, ctx)
                out = out + cross_out.transpose(1, 2)

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    """
    (resnet + self-attn + cross-attn + resnet) repeated.
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, cond_channels=None, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(_suggest_groups(in_channels if i == 0 else out_channels),
                                 in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers + 1)]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(_suggest_groups(out_channels), out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

        self.self_attn_norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(num_layers)])
        self.self_attns = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )

        if cond_channels is not None:
            self.cross_proj = nn.ModuleList(
                [nn.Conv1d(cond_channels, out_channels, kernel_size=1) for _ in range(num_layers)]
            )
            self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(num_layers)])
            self.cross_attns = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
            )
        else:
            self.cross_proj = self.cross_attn_norms = self.cross_attns = None

    def forward(self, x, t_emb, context=None):
        out = x
        # first resnet
        res_in = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](res_in)

        for i in range(self.num_layers):
            tokens = out.transpose(1, 2)
            tokens = self.self_attn_norms[i](tokens)
            attn_out, _ = self.self_attns[i](tokens, tokens, tokens)
            out = out + attn_out.transpose(1, 2)

            if context is not None and self.cross_attns is not None:
                ctx = self.cross_proj[i](context).transpose(1, 2)
                q = self.cross_attn_norms[i](out.transpose(1, 2))
                cross_out, _ = self.cross_attns[i](q, ctx, ctx)
                out = out + cross_out.transpose(1, 2)

            res_in = out
            out = self.resnet_conv_first[i + 1](out)
            out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](res_in)
        return out


class UpBlock(nn.Module):
    """
    Upsample -> concat skip -> resnet + (self + cross) attention.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        t_emb_dim,
        cond_channels=None,
        up_sample=True,
        num_heads=4,
        num_layers=1,
        output_padding=0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.cond_channels = cond_channels

        self.up_sample_conv = (
            nn.ConvTranspose1d(
                in_channels,
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=output_padding,
            )
            if up_sample
            else nn.Identity()
        )

        merged_channels = in_channels + skip_channels
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(_suggest_groups(merged_channels if i == 0 else out_channels),
                                 merged_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(merged_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1),
                )
                for i in range(num_layers)
            ]
        )

        self.t_emb_layers = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers)]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(_suggest_groups(out_channels), out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.self_attn_norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(num_layers)])
        self.self_attns = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )

        if cond_channels is not None:
            self.cross_proj = nn.ModuleList(
                [nn.Conv1d(cond_channels, out_channels, kernel_size=1) for _ in range(num_layers)]
            )
            self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(num_layers)])
            self.cross_attns = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
            )
        else:
            self.cross_proj = self.cross_attn_norms = self.cross_attns = None

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv1d(merged_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, skip, t_emb, context=None):
        x = self.up_sample_conv(x)
        out = torch.cat([x, skip], dim=1)

        for i in range(self.num_layers):
            res_in = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](res_in)

            tokens = out.transpose(1, 2)
            tokens = self.self_attn_norms[i](tokens)
            attn_out, _ = self.self_attns[i](tokens, tokens, tokens)
            out = out + attn_out.transpose(1, 2)

            if context is not None and self.cross_attns is not None:
                ctx = self.cross_proj[i](context).transpose(1, 2)
                q = self.cross_attn_norms[i](out.transpose(1, 2))
                cross_out, _ = self.cross_attns[i](q, ctx, ctx)
                out = out + cross_out.transpose(1, 2)

        return out


class ConditionalUNet1D(nn.Module):
    """
    1D U-Net with class embedding (age/sex) + cross-attention to SC.
    Compatible with ddpm.forward usage in this repo:
        network(input, t, cov_cond, cond1)
    """

    def __init__(self, model_config):
        super().__init__()

        im_channels = model_config["im_channels"]  # e.g., 8 (x + cond2 concatenated)
        self.out_channels = model_config.get("out_channels", im_channels // 2)
        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]
        self.t_emb_dim = model_config["time_emb_dim"]
        self.class_emb_dim = model_config["class_emb_dim"]
        self.cond_channels = model_config.get("cond_channels", None)
        self.down_sample = model_config["down_sample"]
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]
        self.num_heads = model_config.get("num_heads", 4)
        self.seq_len = model_config.get("seq_len", None)
        self.use_cross_attn_down_up = model_config.get("cross_attn_in_down_up", False)
        # If cross-attn is enabled in down/up, exclude the outermost resolutions by default.
        self.cross_attn_exclude_outer = model_config.get("cross_attn_exclude_outer", True)

        assert self.mid_channels[0] == self.down_channels[-1], "mid_channels[0] must match bottleneck channels"
        assert self.mid_channels[-1] == self.down_channels[-2], "mid_channels[-1] must match penultimate down channels"
        assert len(self.down_sample) == len(self.down_channels) - 1, "down_sample len must match down blocks"

        # Output padding to exactly recover the input length
        if "output_padding" in model_config:
            self.output_padding = model_config["output_padding"]
        else:
            if self.seq_len is None:
                raise ValueError("seq_len required to auto-compute output padding")
            lengths = _compute_lengths(self.seq_len, self.down_sample)
            self.output_padding = _compute_output_paddings(lengths)

        # Initial projections
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )
        self.c_proj = nn.Sequential(
            nn.Linear(self.class_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.conv_in = nn.Conv1d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList(
            [
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    self.t_emb_dim,
                    cond_channels=self.cond_channels,
                    down_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_down_layers,
                )
                for i in range(len(self.down_channels) - 1)
            ]
        )

        self.mids = nn.ModuleList(
            [
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    self.t_emb_dim,
                    cond_channels=self.cond_channels,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                )
                for i in range(len(self.mid_channels) - 1)
            ]
        )

        # up_sample flags mirror down_sample
        self.up_sample = list(reversed(self.down_sample))
        self.ups = nn.ModuleList([])

        # compute channel flow for up path
        in_ch = self.mid_channels[-1]
        for idx, i in enumerate(reversed(range(len(self.down_channels) - 1))):
            skip_ch = self.down_channels[i]
            out_ch = self.out_channels if i == 0 else self.down_channels[i - 1]
            self.ups.append(
                UpBlock(
                    in_ch,
                    skip_ch,
                    out_ch,
                    self.t_emb_dim,
                    cond_channels=self.cond_channels,
                    up_sample=self.up_sample[idx],
                    num_heads=self.num_heads,
                    num_layers=self.num_up_layers,
                    output_padding=self.output_padding[idx],
                )
            )
            in_ch = out_ch

        self.norm_out = nn.GroupNorm(4, self.out_channels)
        self.conv_out = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, class_encoding, cond1=None):
        """
        Args:
            x: (B, im_channels, L)
            t: (B,) timesteps
            class_encoding: (B, class_emb_dim) age + sex embedding
            cond1: (B, cond_channels, Lc) structural connectivity (SC) for cross-attn
        """
        out = self.conv_in(x)

        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb) + self.c_proj(class_encoding)

        down_outs = []

        for down in self.downs:
            down_outs.append(out)
            if self.use_cross_attn_down_up:
                idx = len(down_outs) - 1
                total = len(self.downs)
                use_ctx = not (self.cross_attn_exclude_outer and (idx == 0 or idx == total - 1))
                ctx = cond1 if use_ctx else None
            else:
                ctx = None
            out = down(out, t_emb, ctx)

        for mid in self.mids:
            out = mid(out, t_emb, cond1)

        for idx, up in enumerate(self.ups):
            skip = down_outs.pop()
            if self.use_cross_attn_down_up:
                total = len(self.ups)
                use_ctx = not (self.cross_attn_exclude_outer and (idx == total - 1 or idx == 0 and total == 1))
                ctx = cond1 if use_ctx else None
            else:
                ctx = None
            out = up(out, skip, t_emb, ctx)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out


class ConditionalUNet1DGraph(nn.Module):
    """
    Wrapper that injects a graph encoder (e.g., SCGraphModel1D) before the UNet.
    Forward signature mirrors ddpm_graph.backward:
        forward(x, t, class_encoding, graph_data) -> (B, out_channels, L)
    """

    def __init__(self, model_config, graph_encoder):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.unet = ConditionalUNet1D(model_config)

    def forward(self, x, t, class_encoding, graph_data):
        # graph_data is expected to be a torch_geometric Data/Batch with fields:
        #   x (node features), edge_index, edge_weight (optional), edge_attr (optional)
        cond1 = self.graph_encoder(
            graph_data.x.float(),
            graph_data.edge_index,
            edge_weight=getattr(graph_data, "edge_weight", None),
            edge_attr=getattr(graph_data, "edge_attr", None),
        )
        return self.unet(x, t, class_encoding, cond1)
