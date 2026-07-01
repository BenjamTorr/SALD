import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, orig_linear, r=4, alpha=1.0):
        super().__init__()
        self.orig_linear = orig_linear
        self.r = r
        self.alpha = alpha

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(r, orig_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(orig_linear.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = alpha / r

    def forward(self, x):
        return self.orig_linear(x) + self.scaling * (x @ self.lora_A.T) @ self.lora_B.T

def apply_lora(module, r=4, alpha=1.0, target_classes=(nn.Linear,)):
    for name, child in module.named_children():
        if isinstance(child, target_classes):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            apply_lora(child, r=r, alpha=alpha, target_classes=target_classes)


def apply_lora_ditwcat(module: nn.Module, *, dims: dict | None = None, include_mlp: bool = False, include_ln: bool = True):
    """
    Recursively applies LoRA to:
      - Attention.qkv_proj
      - CrossAttention.q_proj
      - CrossAttention.kv_proj
      - Attention.output_proj[0]
      - CrossAttention.output_proj[0]
      - Final DiT projection ``proj_out`` (nn.Linear)
      - Feedforward layers (both mlp_block[0], mlp_block[2]) if include_mlp=True
      - AdaLN Linear layer if include_ln=True

    Args:
        module (nn.Module): The root module (e.g., your DiT1DWCAT model)
        r (int): LoRA rank
        alpha (float): LoRA scaling factor
        include_mlp (bool): Whether to also apply LoRA to MLP feedforward layers
    """
    dims = dims or {}
    # Default proj_out dims to output_proj if not explicitly provided
    proj_out_dims = dims.get("proj_out", dims.get("output_proj", None))

    for name, child in module.named_children():

        # ---- 1️⃣ QKV projections ----
        if isinstance(child, nn.Linear) and (
            "qkv_proj" in name
            or "q_proj" in name
            or "kv_proj" in name
        ):
            setattr(module, name, LoRALinear(child, r=dims['attn_proj']['r'], alpha=dims['attn_proj']['alpha']))
            continue

        # ---- 2️⃣ Output projections (first Linear inside Sequential) ----
        if isinstance(child, nn.Sequential) and "output_proj" in name:
            if isinstance(child[0], nn.Linear):
                child[0] = LoRALinear(child[0], r=dims['output_proj']['r'], alpha=dims['output_proj']['alpha'])
            continue

        # ---- 2️⃣b Final projection back to model dimension ----
        if name == "proj_out" and isinstance(child, nn.Linear) and proj_out_dims is not None:
            setattr(module, name, LoRALinear(child, r=proj_out_dims['r'], alpha=proj_out_dims['alpha']))
            continue

        # ---- 3️⃣ Optional: Feedforward block (first Linear in mlp_block) ----
        if include_mlp and isinstance(child, nn.Sequential) and "mlp_block" in name:
            if isinstance(child[0], nn.Linear):
                child[0] = LoRALinear(child[0], r=dims['mlp_block']['r'], alpha=dims['mlp_block']['alpha'])
                if len(child) > 2 and isinstance(child[2], nn.Linear):
                    child[2] = LoRALinear(child[2], r=dims['mlp_block']['r'], alpha=dims['mlp_block']['alpha'])
            continue

        # ---- 4️⃣ Target the AdaLN Linear layer ----
        if include_ln and "adaptive_norm_layer" in name and isinstance(child, nn.Sequential):
            # child[0] is SiLU, child[1] is the Linear layer
            if len(child) > 1 and isinstance(child[1], nn.Linear):
                # Using a smaller rank for stability in conditioning
                child[1] = LoRALinear(child[1], r=dims['adaln']['r'], alpha=dims['adaln']['alpha']) 
            continue


        # ---- 4️⃣ Otherwise recurse ----
        apply_lora_ditwcat(child, dims = dims, include_mlp=include_mlp, include_ln=include_ln)
    return module

def list_lora_layers(model):
    print("── LoRA layers in model ─────────────────────────────")
    total_lora = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"[LoRA] {name:60s} | in={module.lora_A.shape[1]} | out={module.lora_B.shape[0]} | rank={module.r} | alpha={module.alpha}")
            total_lora += 1
    print(f"Total LoRA-injected layers: {total_lora}")
    print("─────────────────────────────────────────────────────")



def apply_lora_ditwcat_deprecated(module: nn.Module, *, r: int = 4,
    alpha: float = 1.0,  include_mlp: bool = False):
    """
    Recursively applies LoRA to:
      - Attention.qkv_proj
      - CrossAttention.q_proj
      - CrossAttention.kv_proj
      - Attention.output_proj[0]
      - CrossAttention.output_proj[0]
      - Final DiT projection ``proj_out`` (nn.Linear)
      - (Optional) Feedforward layers (mlp_block[0]) if include_mlp=True

    Args:
        module (nn.Module): The root module (e.g., your DiT1DWCAT model)
        r (int): LoRA rank
        alpha (float): LoRA scaling factor
        include_mlp (bool): Whether to also apply LoRA to MLP feedforward layers
    """

    for name, child in module.named_children():

        # ---- 1️⃣ QKV projections ----
        if isinstance(child, nn.Linear) and (
            "qkv_proj" in name
            or "q_proj" in name
            or "kv_proj" in name
        ):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            continue

        # ---- 2️⃣ Output projections (first Linear inside Sequential) ----
        if isinstance(child, nn.Sequential) and "output_proj" in name:
            if isinstance(child[0], nn.Linear):
                child[0] = LoRALinear(child[0], r=r, alpha=alpha)
            continue

        # ---- 2️⃣b Final projection back to model dimension ----
        if name == "proj_out" and isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            continue

        # ---- 3️⃣ Optional: Feedforward block (first Linear in mlp_block) ----
        if include_mlp and isinstance(child, nn.Sequential) and "mlp_block" in name:
            if isinstance(child[0], nn.Linear):
                child[0] = LoRALinear(child[0], r=r, alpha=alpha)
            continue

        # ---- 4️⃣ Otherwise recurse ----
        apply_lora_ditwcat(child, r=r, alpha=alpha, include_mlp=include_mlp)
