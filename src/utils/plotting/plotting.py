import torch
import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.subplots as sp

def plot_overlapping_histograms(tensor_list, names, title = 'Metric', nbins=30):
    """
    Plots overlapping histograms of torch tensors using Plotly.
    
    Parameters:
        tensor_list (list of torch.Tensor): Each tensor should be shape (B,).
        names (list of str): Names corresponding to each tensor.
        nbins (int): Number of bins for the histograms.
    """
    if len(tensor_list) != len(names):
        raise ValueError("tensor_list and names must have the same length")
    
    fig = go.Figure()
    
    for tensor, name in zip(tensor_list, names):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)} for {name}")
        
        data = tensor.detach().cpu().numpy()  # Convert to NumPy for Plotly
        
        fig.add_trace(go.Histogram(
            x=data,
            name=name,
            nbinsx=nbins,
            opacity=0.6
        ))
    
    # Overlay histograms
    fig.update_layout(
        barmode='overlay',
        title=title,
        xaxis_title="Value",
        yaxis_title="Count"
    )
    
    fig.show()


def plot_three_matrices(
    A, B, C, 
    vmin=None, vmax=None, 
    titles=("Matrix A", "Matrix B", "Matrix C")
):
    """
    Plot 3 matrices side-by-side using Plotly heatmaps (RdBu colorscale).

    Parameters
    ----------
    A, B, C : 2D numpy arrays or torch tensors
        The matrices to visualize.
    vmin, vmax : float, optional
        Color scale limits. If None, uses global min and max across all matrices.
    titles : tuple of str
        Titles for each subplot.
    """

    # Convert torch tensors if needed
    if hasattr(A, "detach"): A = A.detach().cpu().numpy()
    if hasattr(B, "detach"): B = B.detach().cpu().numpy()
    if hasattr(C, "detach"): C = C.detach().cpu().numpy()

    # Default vmin/vmax
    all_vals = np.concatenate([A.flatten(), B.flatten(), C.flatten()])
    if vmin is None:
        vmin = all_vals.min()
    if vmax is None:
        vmax = all_vals.max()

    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)

    mats = [A, B, C]
    for i, M in enumerate(mats):
        fig.add_trace(
            go.Heatmap(
                z=M,
                colorscale="RdBu",
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="Value") if i == 2 else None  # one colorbar
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        width=1500,
        height=500,
        showlegend=False,
    )
    
    fig.show()
