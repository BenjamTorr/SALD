import torch
import math


def encode_age_gender(x, age_dim=8):
    B = len(x)
    cond = torch.zeros(B, age_dim + 2, dtype=torch.float32)
    for i, row in enumerate(x):
        age = float(row[0])
        sex = row[1]

        # Age embedding
        age_emb = []
        for j in range(age_dim):
            freq = 1.0 / (10000 ** (j / age_dim))
            if j % 2 == 0:
                age_emb.append(math.sin(age * freq))
            else:
                age_emb.append(math.cos(age * freq))
        cond[i, :age_dim] = torch.tensor(age_emb, dtype=torch.float32)

        # Sex one-hot
        cond[i, age_dim:] = torch.tensor([1.0, 0.0] if sex == "M" else [0.0, 1.0])
    return cond

def get_upper_diagonal_elements(x):
    """
    Extracts upper diagonal elements (excluding the diagonal) from a (B, C, L, L) tensor.
    
    Args:
        x (torch.Tensor): A tensor of shape (B, C, L, L)
    
    Returns:
        torch.Tensor: A tensor of shape (B, C, num_upper_elements) containing the upper diagonal elements
    """
    B, C, L, _ = x.shape
    # Create mask for upper triangle excluding the diagonal
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1).to(x.device)  # (L, L)
    # Apply the mask
    x_masked = x[:, :, mask]  # (B, C, num_upper_elements)
    return x_masked


def upper_elements_to_symmetric_matrix_no_chan(x):
    """
    Converts a (B,  num_upper_elements) tensor into a symmetric (B, L, L) matrix
    with diagonal elements set to 1 and upper/lower triangle from x.
    
    Args:
        x (torch.Tensor): A tensor of shape (B, C, num_upper_elements)
    
    Returns:
        torch.Tensor: A symmetric tensor of shape (B, C, L, L)
    """
    B, num_upper = x.shape
    mat = upper_elements_to_symmetric_matrix(x.unsqueeze(1)).squeeze(1)
    return mat

def get_upper_diagonal_elements_no_chan(x):
    """
    Extracts upper diagonal elements (excluding the diagonal) from a (B, C, L, L) tensor.
    
    Args:
        x (torch.Tensor): A tensor of shape (B, C, L, L)
    
    Returns:
        torch.Tensor: A tensor of shape (B, C, num_upper_elements) containing the upper diagonal elements
    """
    B, L, _ = x.shape
    vec = get_upper_diagonal_elements(x.unsqueeze(1)).squeeze(1)
    return vec


def upper_elements_to_symmetric_matrix(x):
    """
    Converts a (B, C, num_upper_elements) tensor into a symmetric (B, C, L, L) matrix
    with diagonal elements set to 1 and upper/lower triangle from x.
    
    Args:
        x (torch.Tensor): A tensor of shape (B, C, num_upper_elements)
    
    Returns:
        torch.Tensor: A symmetric tensor of shape (B, C, L, L)
    """
    B, C, num_upper = x.shape
    # Solve for L: L = (1 + sqrt(1 + 8*num_upper)) / 2
    L = int((1 + (1 + 8 * num_upper) ** 0.5) / 2)
    assert L * (L - 1) // 2 == num_upper, "Input size is not consistent with a valid L"

    # Create empty matrix
    mat = torch.zeros(B, C, L, L, device=x.device, dtype=x.dtype)

    # Get upper triangle indices (excluding diagonal)
    triu_indices = torch.triu_indices(L, L, offset=1)

    # Fill upper triangle
    mat[:, :, triu_indices[0], triu_indices[1]] = x

    # Mirror to lower triangle
    mat[:, :, triu_indices[1], triu_indices[0]] = x

    # Set diagonal to 1
    diag_idx = torch.arange(L, device=x.device)
    mat[:, :, diag_idx, diag_idx] = 1.0

    return mat

def vector_to_matrix(vec, mean_mat, std_mat):
    """
    vec: (N, num_upper)
    returns: (N, L, L) with mean/std added back.
    """
    mat = upper_elements_to_symmetric_matrix_no_chan(vec)
    return mat * std_mat + mean_mat

def gaussian_resample(
    sc,
    seed=0,
    rescale_mean=0.5,
    rescale_std=0.1,
):
    """
    Gaussian rank-based resampling ONLY for NONZERO edges.
    Zero edges remain zero (sparsity preserved).
    Rank among positive edges preserved.

    Parameters
    ----------
    sc : torch.Tensor
        Shape (L, L) or (B, L, L)
    seed : int
        Random seed for reproducibility.
    keep_diagonal : bool
        Keep diagonal values untouched.
    rescale_mean : float
        Desired mean after resampling positive edges.
    rescale_std : float
        Desired std after resampling positive edges.

    Returns
    -------
    torch.Tensor
        SC with positive edges Gaussianized, zeros preserved.
    """
    two_dim = False
    # Ensure batch dimension
    if sc.dim() == 2:
        sc = sc.unsqueeze(0)
        two_dim = True


    B, L, _ = sc.shape
    device = sc.device
    dtype = sc.dtype

    # RNG
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    out = torch.zeros_like(sc)

    for b in range(B):

        mat = sc[b]

        # Mask for nonzero edges
        nonzero_mask = (mat != 0)
        # Extract NONZERO values
        vals = mat[nonzero_mask]
        N = vals.numel()

        if N > 0:
            # Generate Gaussian samples
            gaussian = torch.randn(N, device=device, generator=g)

            # Sort both arrays
            sorted_idx = torch.argsort(vals)
            sorted_gauss = torch.sort(gaussian).values

            # Replace sorted vals with sorted Gaussian samples
            new_vals = torch.empty_like(vals)
            new_vals[sorted_idx] = sorted_gauss

            # Rescale
            new_vals = new_vals * rescale_std + rescale_mean
        else:
            new_vals = torch.tensor([], device=device, dtype=dtype)

        # Create new matrix
        new_mat = mat.clone()
        new_mat[nonzero_mask] = new_vals

        out[b] = new_mat

    # Remove batch dimension if input was 2D
    return out[0] if two_dim  else out

def unscale_generated_corr(gen_scaled, mean_FC, std_FC, diag_eps=1e-8):
    """
    gen_scaled: (B, S, R, L, L) in z-scored space
    mean_FC, std_FC: (1, L, L)
    """
    device = gen_scaled.device
    L = gen_scaled.shape[-1]
    idx = torch.arange(L, device=device)

    # Work on a copy
    gen_scaled = gen_scaled.clone()

    # 1) Diagonal in scaled space must be 0
    # since mean_diag = 1 exactly
    gen_scaled[..., idx, idx] = 0.0

    # 2) Broadcast mean/std to (1,1,1,L,L)
    mean = mean_FC.unsqueeze(1).unsqueeze(1)
    std  = std_FC.unsqueeze(1).unsqueeze(1)

    # 3) Defensively patch std (handles degenerate diagonal)
    std = torch.nan_to_num(std, nan=diag_eps, posinf=diag_eps, neginf=diag_eps)
    std = torch.clamp(std, min=diag_eps)

    # 4) Unscale
    gen_unscaled = gen_scaled * std + mean

    # 5) Enforce exact correlation diagonal
    gen_unscaled[..., idx, idx] = 1.0

    return gen_unscaled