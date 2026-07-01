import torch
import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.subplots as sp


# ---------- Metrics (per-subject) ----------

def frobenius_distance(real, gen):
    """Frobenius distance per subject."""
    return torch.linalg.norm(real - gen, dim=(-2, -1))  # (B,)


def correlation_distance(real, gen):
    """Correlation between upper-triangular entries per subject."""
    N = real.shape[-1]
    triu_idx = torch.triu_indices(N, N, offset=1)
    r = real[:, triu_idx[0], triu_idx[1]]
    g = gen[:, triu_idx[0], triu_idx[1]]
    out = []
    for b in range(r.shape[0]):
        corr = torch.corrcoef(torch.stack([r[b], g[b]]))[0,1].item()
        out.append(corr)
    return torch.tensor(out)


def spectral_similarity(real, gen):
    """L2 distance between eigenvalue spectra per subject."""
    out = []
    for b in range(real.shape[0]):
        eig_r = eigh(real[b].cpu().numpy(), eigvals_only=True)
        eig_g = eigh(gen[b].cpu().numpy(), eigvals_only=True)
        out.append(np.linalg.norm(np.sort(eig_r) - np.sort(eig_g)))
    return torch.tensor(out)


def hub_preservation(real, gen, topk=5):
    """Jaccard similarity of top-k hub nodes per subject."""
    out = []
    for b in range(real.shape[0]):
        deg_r = real[b].abs().sum(0).cpu().numpy()
        hubs_r = set(np.argsort(deg_r)[-topk:])
        deg_g = gen[b].abs().sum(0).cpu().numpy()
        hubs_g = set(np.argsort(deg_g)[-topk:])
        inter = len(hubs_r & hubs_g)
        union = len(hubs_r | hubs_g)
        out.append(inter/union if union>0 else 0)
    return torch.tensor(out)


# ------ multi metrics comparisons -----

def frobenius_distance_multi(real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Frobenius distance between real and multiple predicted matrices.

    Parameters
    ----------
    real : torch.Tensor
        Shape (B, N, N)
    pred : torch.Tensor
        Shape (B, S, N, N)

    Returns
    -------
    torch.Tensor
        Shape (B, S), Frobenius norm per subject per estimator
    """
    if real.dim() != 3:
        raise ValueError(f"`real` must have shape (B, N, N). Got {real.shape}")
    if pred.dim() != 4:
        raise ValueError(f"`pred` must have shape (B, S, N, N). Got {pred.shape}")
    if real.shape[0] != pred.shape[0]:
        raise ValueError("Batch size B must match between real and pred.")
    if real.shape[-1] != pred.shape[-1] or real.shape[-2] != pred.shape[-2]:
        raise ValueError("Matrix dimensions N x N must match.")

    # Expand real to (B, 1, N, N) so it broadcasts over S
    diff = pred - real.unsqueeze(1)  # (B, S, N, N)

    # Frobenius norm over matrix dimensions
    return torch.linalg.norm(diff, dim=(-2, -1))  # (B, S)

def correlation_distance_multi(real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Correlation between upper-triangular entries per subject and per estimator.

    Parameters
    ----------
    real : torch.Tensor
        Shape (B, N, N)
    pred : torch.Tensor
        Shape (B, S, N, N)

    Returns
    -------
    torch.Tensor
        Shape (B, S), Pearson correlation per subject per estimator
    """
    if real.dim() != 3:
        raise ValueError(f"`real` must have shape (B, N, N). Got {real.shape}")
    if pred.dim() != 4:
        raise ValueError(f"`pred` must have shape (B, S, N, N). Got {pred.shape}")
    if real.shape[0] != pred.shape[0]:
        raise ValueError("Batch size B must match.")
    if real.shape[-1] != pred.shape[-1] or real.shape[-2] != pred.shape[-2]:
        raise ValueError("Matrix dimensions N x N must match.")

    B, N, _ = real.shape
    _, S, _, _ = pred.shape

    # Upper-triangular indices
    triu_idx = torch.triu_indices(N, N, offset=1, device=real.device)

    # Extract upper triangles
    r = real[:, triu_idx[0], triu_idx[1]]          # (B, L)
    g = pred[:, :, triu_idx[0], triu_idx[1]]       # (B, S, L)

    # Center
    r_mean = r.mean(dim=-1, keepdim=True)          # (B, 1)
    g_mean = g.mean(dim=-1, keepdim=True)          # (B, S, 1)

    r_centered = r - r_mean                        # (B, L)
    g_centered = g - g_mean                        # (B, S, L)

    # Covariance
    cov = (r_centered.unsqueeze(1) * g_centered).sum(dim=-1)  # (B, S)

    # Standard deviations
    r_std = torch.sqrt((r_centered ** 2).sum(dim=-1))         # (B,)
    g_std = torch.sqrt((g_centered ** 2).sum(dim=-1))         # (B, S)

    # Pearson correlation
    corr = cov / (r_std.unsqueeze(1) * g_std + 1e-8)          # (B, S)

    return corr



# -------- Plotting results ----------- 

# plot for visual reconstruction quality

def plot_fc20_plus_slices(
    fc20,          # (B, N, N)
    fc3_slices,    # (B, S, N, N)
    gen_slices,    # (B, S, N, N)
    *,
    subject_idx: int = 0,                 # alias for b
    max_cols: int | None = None,          # optionally cap number of slices shown
    slice_indices: list[int] | None = None,  # optionally choose specific slices
    row_titles=("True 20-min FC (target)", "3-min slices (inputs)", "Diffusion-generated slices"),
    col_title_fmt="Slice {i}",
    # ---- Color scale controls (shared across ALL panels) ----
    fixed_scale: bool = True,
    zmin: float = -1.0,
    zmax: float = 1.0,
    colorscale: str = "RdBu",
    zmid: float = 0.0,
    show_colorbar: bool = True,
    height_per_row: int = 260,
    width_per_col: int = 220,
):
    """
    Layout:
      Row 1: FC_20 target repeated across columns (same matrix shown in each column)
      Row 2: S slices of FC_3 for subject_idx
      Row 3: S slices of FC_diffusion for subject_idx

    Shapes (as requested):
      - fc20:       (B, N, N)
      - fc3_slices: (B, S, N, N)
      - gen_slices: (B, S, N, N)

    Color scale:
      - Uses ONE shared RdBu scale with zmid=0.
      - By default fixed to [-1, 1] for immediate interpretability.
      - If fixed_scale=False, scale is computed from the selected subject’s matrices
        but still shared across all panels (symmetric around 0).

    Returns:
      fig, (FC20_b, FC3_b, GEN_b)
      where FC3_b and GEN_b are (S_shown, N, N).
    """

    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        raise TypeError(f"Unsupported input type: {type(x)}. Use numpy arrays or torch tensors.")

    fc20 = np.asarray(to_numpy(fc20))
    fc3_slices = np.asarray(to_numpy(fc3_slices))
    gen_slices = np.asarray(to_numpy(gen_slices))

    # Strict shape validation
    if fc20.ndim != 3:
        raise ValueError(f"fc20 must have shape (B,N,N). Got {fc20.shape}.")
    if fc3_slices.ndim != 4:
        raise ValueError(f"fc3_slices must have shape (B,S,N,N). Got {fc3_slices.shape}.")
    if gen_slices.ndim != 4:
        raise ValueError(f"gen_slices must have shape (B,S,N,N). Got {gen_slices.shape}.")

    B, N1, N2 = fc20.shape
    if N1 != N2:
        raise ValueError(f"fc20 must be square per batch. Got (B,{N1},{N2}).")
    if subject_idx < 0 or subject_idx >= B:
        raise IndexError(f"subject_idx={subject_idx} out of range for B={B}.")

    B3, S3, N3, N4 = fc3_slices.shape
    Bg, Sg, Ng, Nh = gen_slices.shape

    if B3 != B or Bg != B:
        raise ValueError(f"Batch mismatch: fc20 B={B}, fc3_slices B={B3}, gen_slices B={Bg}.")
    if (N3, N4) != (N1, N2) or (Ng, Nh) != (N1, N2):
        raise ValueError(
            f"Matrix size mismatch: fc20 is {(N1,N2)}, fc3_slices is {(N3,N4)}, gen_slices is {(Ng,Nh)}."
        )
    if S3 != Sg:
        raise ValueError(f"Slice count mismatch: fc3_slices S={S3}, gen_slices S={Sg}.")

    # Slice selection
    S = S3
    if slice_indices is None:
        idx = list(range(S))
    else:
        idx = list(slice_indices)
        bad = [i for i in idx if i < 0 or i >= S]
        if bad:
            raise IndexError(f"slice_indices contains out-of-range indices (valid 0..{S-1}): {bad}")

    if max_cols is not None:
        if max_cols <= 0:
            raise ValueError("max_cols must be a positive integer.")
        idx = idx[:max_cols]

    S_show = len(idx)
    if S_show == 0:
        raise ValueError("No slices selected to display (S_show=0).")

    FC20_b = fc20[subject_idx]            # (N,N)
    FC3_b  = fc3_slices[subject_idx, idx] # (S_show,N,N)
    GEN_b  = gen_slices[subject_idx, idx] # (S_show,N,N)

    # Shared scale across ALL panels
    if fixed_scale:
        lo, hi = float(zmin), float(zmax)
        if lo >= hi:
            raise ValueError(f"Invalid fixed scale: zmin={lo} must be < zmax={hi}.")
    else:
        # Symmetric around 0 for diverging scale (good practice for correlations)
        m = np.nanmax(np.abs(np.concatenate([FC20_b[None, ...], FC3_b, GEN_b], axis=0)))
        m = float(m) if np.isfinite(m) else 1.0
        lo, hi = -m, m

    # Build subplot grid: 3 rows x S_show cols
    fig = make_subplots(
        rows=3,
        cols=S_show,
        subplot_titles=[col_title_fmt.format(i=j) for j in idx],
        vertical_spacing=0.06,
        horizontal_spacing=0.03,
    )

    # Row 1: repeat target across all columns
    for c in range(1, S_show + 1):
        fig.add_trace(
            go.Heatmap(
                z=FC20_b,
                colorscale=colorscale,
                zmin=lo,
                zmax=hi,
                zmid=zmid,
                showscale=False,
            ),
            row=1, col=c
        )

    # Row 2: FC3 slices
    for c in range(1, S_show + 1):
        fig.add_trace(
            go.Heatmap(
                z=FC3_b[c - 1],
                colorscale=colorscale,
                zmin=lo,
                zmax=hi,
                zmid=zmid,
                showscale=False,
            ),
            row=2, col=c
        )

    # Row 3: GEN slices (attach ONE shared colorbar on last column)
    for c in range(1, S_show + 1):
        fig.add_trace(
            go.Heatmap(
                z=GEN_b[c - 1],
                colorscale=colorscale,
                zmin=lo,
                zmax=hi,
                zmid=zmid,
                showscale=(show_colorbar and c == S_show),
                colorbar=dict(
                    title="FC",
                    len=0.85,
                    y=0.5,
                    thickness=14,
                ) if (show_colorbar and c == S_show) else None,
            ),
            row=3, col=c
        )

    # Clean axes everywhere
    for r in (1, 2, 3):
        for c in range(1, S_show + 1):
            fig.update_xaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False, row=r, col=c)
            fig.update_yaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False, row=r, col=c)

    # Row labels on the left (paper coords)
    fig.add_annotation(x=-0.01, y=1.00, xref="paper", yref="paper",
                       text=row_titles[0], showarrow=False, xanchor="right", yanchor="top")
    fig.add_annotation(x=-0.01, y=0.64, xref="paper", yref="paper",
                       text=row_titles[1], showarrow=False, xanchor="right", yanchor="top")
    fig.add_annotation(x=-0.01, y=0.28, xref="paper", yref="paper",
                       text=row_titles[2], showarrow=False, xanchor="right", yanchor="top")

    fig.update_layout(
        template="plotly_white",
        height=height_per_row * 3 + 80,
        width=width_per_col * S_show + 90,
        margin=dict(l=110, r=20, t=70, b=20),
    )

    return fig, (FC20_b, FC3_b, GEN_b)


def plot_metric_boxplots(
    metrics: dict,
    *,
    title: str = "Metric comparison (lower is better)",
    y_label: str = "Value",
    show_points: str | bool = "outliers",   # False | "outliers" | "all" | "suspectedoutliers"
    log_y: bool = False,
    sort_by: str | None = "median",         # None | "median" | "mean"
    lower_is_better: bool = True,
    width: int = 1100,
    height: int = 450,
):
    """
    Compare multiple metric arrays via side-by-side boxplots.

    Parameters
    ----------
    metrics:
        Dict[str, array_like]
        Each value can be any shape; it will be flattened to 1D.
        Example:
          {
            "FM (mean)": fm_frob,          # (B,) or (B,S) etc.
            "Graph (mean)": graph_frob,
            "Baseline": baseline_frob,
          }

    Returns
    -------
    fig : plotly.graph_objects.Figure

    Notes
    -----
    - Non-finite values (nan/inf) are dropped.
    - Box shows median and IQR; whiskers follow Plotly default (1.5*IQR).
    """

    if not isinstance(metrics, dict) or len(metrics) == 0:
        raise ValueError("metrics must be a non-empty dict: {name: array}.")

    def _clean_1d(x, name: str) -> np.ndarray:
        a = np.asarray(x).reshape(-1)
        a = a[np.isfinite(a)]
        if a.size == 0:
            raise ValueError(f"Metric '{name}' has no finite values after cleaning.")
        return a

    cleaned = {name: _clean_1d(arr, name) for name, arr in metrics.items()}

    # Optional sorting
    names = list(cleaned.keys())
    if sort_by in {"median", "mean"}:
        key_fn = (np.median if sort_by == "median" else np.mean)
        names = sorted(names, key=lambda n: float(key_fn(cleaned[n])), reverse=not lower_is_better)
    elif sort_by is None:
        pass
    else:
        raise ValueError("sort_by must be one of: None, 'median', 'mean'.")

    fig = go.Figure()

    for name in names:
        fig.add_trace(
            go.Box(
                y=cleaned[name],
                name=name,
                boxmean=False,            # shows mean and std marker/line; remove if you prefer
                boxpoints=show_points,   # points overlay
                jitter=0.25,             # slight horizontal jitter when points shown
                pointpos=0,              # centered
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(l=70, r=20, t=60, b=80),
        xaxis=dict(title="", tickangle=-20),
        yaxis=dict(title=y_label, type="log" if log_y else "linear", zeroline=False),
    )

    return fig

from sklearn.metrics import r2_score

def safe_corrcoef(a, b):
    """Safe Pearson correlation: returns 0 if either input has zero variance."""
    a, b = np.asarray(a), np.asarray(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]

def compare_regression_methods_shared_truth(y_true, preds_dict, n_bins=30, color_map=None):
    """
    Compare multiple regression models visually and numerically, assuming the same y_true.

    Parameters
    ----------
    y_true : array-like
        Ground truth values, shared across all models.
    preds_dict : dict
        Dictionary like:
        {
            "Model A": y_pred_A,
            "Model B": y_pred_B,
            ...
        }
    n_bins : int
        Number of bins for histograms.
    color_map : dict or None
        Optional mapping {model_name: color_hex}. If None, uses the project plotting
        defaults for the main series names and a qualitative fallback for others.
    """
    y_true = np.asarray(y_true).reshape(-1)

    # Keep model order from input dict.
    model_names = list(preds_dict.keys())
    n_models = len(model_names)

    # Normalize predictions to numpy arrays once.
    preds_np = {name: np.asarray(preds_dict[name]).reshape(-1) for name in model_names}

    # Default palette aligned with notebook plot colors.
    default_colors = {
        "Real scans (20 min)": "#3949AB",  # Indigo
        "Real scans (short)": "#9E9E9E",   # Gray
        "Conditional DDPM": "#FB8C00",     # Orange
        "DraftDFC": "#D81B60",             # Magenta
    }
    if color_map is None:
        color_map = default_colors.copy()
    else:
        merged = default_colors.copy()
        merged.update(color_map)
        color_map = merged

    # Assign fallback colors for any series not in the explicit map.
    fallback_palette = px.colors.qualitative.Plotly
    for i, name in enumerate(model_names):
        if name not in color_map:
            color_map[name] = fallback_palette[i % len(fallback_palette)]
    
    # ---------- Add baseline ----------
    y_mean = np.mean(y_true)
    baseline_pred = np.full_like(y_true, y_mean)
    #preds_dict = {"Mean baseline": baseline_pred, **preds_dict}
    model_names = list(preds_np.keys())
    n_models = len(model_names)

    # ---------- Metrics ----------
    metrics = {}
    for name, y_pred in preds_np.items():
        r2 = r2_score(y_true, y_pred)
        corr = safe_corrcoef(y_true, y_pred)
        metrics[name] = {"R2": r2, "Corr": corr}

    # ---------- Global axis limits ----------
    all_pred = np.concatenate(list(preds_np.values()))
    min_val = float(min(y_true.min(), all_pred.min()))
    max_val = float(max(y_true.max(), all_pred.max()))

    # Shared residual y-limits across models (symmetric around 0)
    all_residuals = np.concatenate([pred - y_true for pred in preds_np.values()])
    if all_residuals.size > 0:
        max_abs_resid = float(np.max(np.abs(all_residuals)))
        residual_min, residual_max = -max_abs_resid, max_abs_resid
    else:
        residual_min, residual_max = -1.0, 1.0

    # Shared histogram scales (x and y) across models
    hist_range = (min_val, max_val)
    max_hist_count = 0
    for pred in preds_np.values():
        true_counts, _ = np.histogram(y_true, bins=n_bins, range=hist_range)
        pred_counts, _ = np.histogram(pred, bins=n_bins, range=hist_range)
        max_hist_count = max(max_hist_count, int(true_counts.max()), int(pred_counts.max()))
    hist_ymax = float(max_hist_count) * 1.05 if max_hist_count > 0 else 1.0

    # ---------- Subplots layout ----------
    fig = make_subplots(
        rows=3, cols=n_models,
        subplot_titles=[f"{name}" for name in model_names] * 3,
        specs=[[{"type": "scatter"}]*n_models,
               [{"type": "scatter"}]*n_models,
               [{"type": "xy"}]*n_models],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # ---------- Row 1: True vs Predicted ----------
    for j, (name, y_pred) in enumerate(preds_np.items(), 1):
        model_color = color_map[name]
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred,
                mode='markers', marker=dict(size=5, opacity=0.6, color=model_color),
                name=name, showlegend=False
            ),
            row=1, col=j
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=1, col=j
        )
        fig.update_xaxes(title_text="True", range=[min_val, max_val], row=1, col=j)
        fig.update_yaxes(title_text="Predicted", range=[min_val, max_val], row=1, col=j)

    # ---------- Row 2: True vs Residuals ----------
    for j, (name, y_pred) in enumerate(preds_np.items(), 1):
        model_color = color_map[name]
        residuals = y_pred - y_true
        fig.add_trace(
            go.Scatter(
                x=y_true, y=residuals,
                mode='markers', marker=dict(size=5, opacity=0.6, color=model_color),
                showlegend=False
            ),
            row=2, col=j
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[0, 0],
                mode='lines', line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=2, col=j
        )
        fig.update_xaxes(title_text="True", range=[min_val, max_val], row=2, col=j)
        fig.update_yaxes(title_text="Residuals", range=[residual_min, residual_max], row=2, col=j)

    # ---------- Row 3: Histogram comparison ----------
    for j, (name, y_pred) in enumerate(preds_np.items(), 1):
        model_color = color_map[name]
        fig.add_trace(
            go.Histogram(
                x=y_true,
                nbinsx=n_bins,
                name=f"{name} True",
                opacity=0.35,
                marker_color="#BDBDBD",
                xbins=dict(start=min_val, end=max_val, size=(max_val - min_val) / n_bins if max_val > min_val else 1),
            ),
            row=3, col=j
        )
        fig.add_trace(
            go.Histogram(
                x=y_pred,
                nbinsx=n_bins,
                name=f"{name} Pred",
                opacity=0.5,
                marker_color=model_color,
                xbins=dict(start=min_val, end=max_val, size=(max_val - min_val) / n_bins if max_val > min_val else 1),
            ),
            row=3, col=j
        )
        fig.update_xaxes(title_text="Value", range=[min_val, max_val], row=3, col=j)
        fig.update_yaxes(title_text="Count", range=[0, hist_ymax], row=3, col=j)

    fig.update_layout(
        height=1000, width=350 * n_models,
        title="Regression Comparison: Shared True Values Across Methods",
        barmode="overlay",
        template="plotly_white"
    )
    fig.show()

    # ---------- Summary Metrics Bar Plot ----------
    bar_fig = go.Figure()

    for model in model_names:
        bar_fig.add_trace(
            go.Bar(
                x=["R²", "Correlation"],
                y=[metrics[model]["R2"], metrics[model]["Corr"]],
                name=model,
                marker_color=color_map[model],
            )
        )

    bar_fig.update_layout(
        title="R² and Correlation (Shared Ground Truth)",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode="group",
        template="plotly_white",
        height=400
    )
    bar_fig.show()

    return metrics
