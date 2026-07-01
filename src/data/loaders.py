import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from utils.preprocessing.transformations import get_upper_diagonal_elements, encode_age_gender 

def _ensure_4d_fc_tensor(x):
    """
    Normalize FC-like tensors to (B, C, L, L).
    Accepts:
      - (B, L, L)   -> (B, 1, L, L)
      - (B, C, L, L)-> unchanged
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected FC tensor with 3 or 4 dims, got shape={tuple(x.shape)}")


def _ensure_sliced_conditioning(x):
    """
    Normalize conditioning tensors to include explicit slice axis:
      - (B, C, L)      -> (B, 1, C, L)
      - (B, S, C, L)   -> unchanged
      - anything else  -> unchanged
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


class BaseDataset(Dataset):
    """
    All datasets inherit device logic + conversion utilities.
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def to_device(self, x):
        """
        Move tensor to the dataset's device.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x.to(self.device)


# ---------------------------------------------------------
# 1) FC_SCVectorDataset
# ---------------------------------------------------------

class FC_SCVectorDataset(BaseDataset):
    """
    FC vector input, SC vector conditioning.
    """

    def __init__(self, FC_matrices, SC_matrices, FCt_matrices,
                 Covariates, age_dim=126, log_transform=True,
                 shape=(-1, 1, 100, 100), device="cpu"):

        super().__init__(device=device)

        # --- FC / SC reshape ---
        FC = torch.as_tensor(FC_matrices.reshape(shape), dtype=torch.float32)
        SC = torch.as_tensor(SC_matrices.reshape(shape), dtype=torch.float32)
        FCt = _ensure_4d_fc_tensor(FCt_matrices)

        # Symmetrize + Log Transform SC
        SC = SC.transpose(2, 3) + SC
        if log_transform:
            SC = torch.log1p(SC)

        # Vectorize
        self.fc_vectorize = get_upper_diagonal_elements(FC)
        self.sc_vectorize = get_upper_diagonal_elements(SC)
        self.fct_vectorize = get_upper_diagonal_elements(FCt)

        # Covariates
        self.cond_representation = encode_age_gender(Covariates, age_dim)

    def __len__(self):
        return len(self.fc_vectorize)

    def __getitem__(self, idx):
        return (
            self.fc_vectorize[idx],
            self.sc_vectorize[idx],
            self.fct_vectorize[idx],
            self.cond_representation[idx],
        )


# ---------------------------------------------------------
# 2) FC_SCGraphDataset
# ---------------------------------------------------------

class FC_SCGraphDataset(BaseDataset):
    """
    FC vector input + SC graph + target FC.
    """

    def __init__(self, fc_vectorized, SC_matrices, fct_vectorized,
                 Covariates, target, age_dim=8,
                 shape=(-1, 1, 100, 100), threshold=0.0,
                 transform_sc=True, device="cpu"):

        super().__init__(device=device)

        assert len(fc_vectorized) == len(SC_matrices)
        assert len(fc_vectorized) == len(target)

        self.threshold = threshold
        self.shape = shape

        self.fc_vectorized = torch.as_tensor(fc_vectorized, dtype=torch.float32)
        self.fct_vectorized = _ensure_sliced_conditioning(fct_vectorized)

        # --- SC Preprocess ---
        SC = torch.as_tensor(SC_matrices.reshape(shape), dtype=torch.float32)
        SC = SC.transpose(2, 3) + SC
        if transform_sc:
            SC = torch.log1p(SC)
        self.SC_matrices = SC

        # Covariates + target
        self.cond_representation = encode_age_gender(Covariates, age_dim)
        self.target = torch.as_tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.fc_vectorized)

    def __getitem__(self, idx):

        x = self.fc_vectorized[idx]
        xt = self.fct_vectorized[idx]

        # --- SC Graph ---
        sc_mat = self.SC_matrices[idx].reshape(self.shape[2], self.shape[3])

        # Edge list
        edge_index = (sc_mat > self.threshold).nonzero(as_tuple=False).t().contiguous()
        edge_weight = sc_mat[edge_index[0], edge_index[1]]
        edge_attr = edge_weight.unsqueeze(-1)

        # Node features
        N = sc_mat.size(0)
        x_graph = torch.eye(N, dtype=torch.float32)

        graph = Data(
            x=x_graph,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr
        )

        return (
            x,
            graph,
            xt,
            self.cond_representation[idx],
            self.target[idx]
        )

class FC_SC_vec_Dataset(BaseDataset):
    """
    FC vector input + SC vec + target FC.
    """

    def __init__(self, fc_vectorized, SC_matrices, fct_vectorized,
                 Covariates, target, age_dim=8,
                 shape=(-1, 1, 100, 100), threshold=0.0,
                 transform_sc=True, device="cpu"):

        super().__init__(device=device)

        assert len(fc_vectorized) == len(SC_matrices)
        assert len(fc_vectorized) == len(target)

        self.threshold = threshold
        self.shape = shape

        self.fc_vectorized = torch.as_tensor(fc_vectorized, dtype=torch.float32)
        self.fct_vectorized = fct_vectorized

        # --- SC Preprocess ---
        SC = torch.as_tensor(SC_matrices.reshape(shape), dtype=torch.float32)
        SC = SC.transpose(2, 3) + SC
        if transform_sc:
            SC = torch.log1p(SC)
        self.SC_matrices = get_upper_diagonal_elements(SC).squeeze(1)

        # Covariates + target
        self.cond_representation = encode_age_gender(Covariates, age_dim)
        self.target = torch.as_tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.fc_vectorized)

    def __getitem__(self, idx):

        x = self.fc_vectorized[idx]
        xt = self.fct_vectorized[idx]

        # --- SC Graph ---
        sc_mat = self.SC_matrices[idx]

        return (
            x,
            sc_mat,
            xt,
            self.cond_representation[idx],
            self.target[idx]
        )


# ---------------------------------------------------------
# 3) FC_FCDataset (dynamic FC)
# ---------------------------------------------------------

class FC_FCDataset(BaseDataset):
    """
    FC input + dynamic-FC + target FC.
    """

    def __init__(self, fc_vectorized, fcd_vectorized, fct_vectorized,
                 Covariates, target, age_dim=126, device="cpu"):

        super().__init__(device=device)

        assert len(fc_vectorized) == len(fcd_vectorized)
        assert len(fc_vectorized) == len(fct_vectorized)
        assert len(fc_vectorized) == len(target)

        self.fc_vectorized = torch.as_tensor(fc_vectorized, dtype=torch.float32)
        self.fcd_vectorized = torch.as_tensor(fcd_vectorized, dtype=torch.float32)
        self.fct_vectorized = torch.as_tensor(fct_vectorized, dtype=torch.float32)

        self.cond_representation = encode_age_gender(Covariates, age_dim)
        self.target = torch.as_tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.fc_vectorized)

    def __getitem__(self, idx):
        return (
            self.fc_vectorized[idx],
            self.fcd_vectorized[idx],
            self.fct_vectorized[idx],
            self.cond_representation[idx],
            self.target[idx]
        )


# ---------------------------------------------------------
# 4) FC_SCGraphDatasetV2 (normalization-focused)
# ---------------------------------------------------------

class FC_SCGraphDatasetV2(BaseDataset):
    """
    Same as FC_SCGraphDataset but with FC normalization and SC preserved.
    """

    def __init__(self, fc_vectorized, SC_matrices, fct_vectorized,
                 Covariates, target, age_dim=8,
                 shape=(-1, 1, 68, 68), threshold=0.0,
                 normalize_fc=True, device="cpu"):

        super().__init__(device=device)

        assert len(fc_vectorized) == len(SC_matrices)
        assert len(fc_vectorized) == len(target)

        self.shape = shape
        self.threshold = threshold

        # FC tensors
        fc = torch.as_tensor(fc_vectorized, dtype=torch.float32)
        fct = torch.as_tensor(fct_vectorized, dtype=torch.float32)

        # Normalize FC only
        if normalize_fc:
            mean = fc.mean(0, keepdim=True)
            std = fc.std(0, keepdim=True) + 1e-8
            fc = (fc - mean) / std

            mean_t = fct.mean(0, keepdim=True)
            std_t = fct.std(0, keepdim=True) + 1e-8
            fct = (fct - mean_t) / std_t

        self.fc_vectorized = fc
        self.fct_vectorized = fct

        # --- SC Preprocessing (log transform only) ---
        SC = torch.as_tensor(SC_matrices.reshape(shape), dtype=torch.float32)
        SC = SC.transpose(2, 3) + SC
        SC = torch.log1p(SC)
        self.SC_matrices = SC

        # Covariates + target
        self.cond_representation = encode_age_gender(Covariates, age_dim)
        self.target = torch.as_tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.fc_vectorized)

    def __getitem__(self, idx):

        x = self.fc_vectorized[idx]
        xt = self.fct_vectorized[idx]

        # Graph
        sc_mat = self.SC_matrices[idx].reshape(self.shape[2], self.shape[3])

        edge_index = (sc_mat > self.threshold).nonzero(as_tuple=False).t()
        edge_attr = sc_mat[edge_index[0], edge_index[1]].unsqueeze(-1)

        N = sc_mat.size(0)
        x_graph = torch.eye(N, dtype=torch.float32)

        graph = Data(
            x=x_graph,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        return (
            x,
            graph,
            xt,
            self.cond_representation[idx],
            self.target[idx]
        )


# ---------------------------------------------------------
# Collate function
# ---------------------------------------------------------

def custom_collate_fn(batch):
    """
    Collates: (x, graph, xt, cond, target)
    """

    X_list, graph_list, Xt_list, cond_list, target_list = zip(*batch)

    X_batch = torch.stack(X_list)
    Xt_batch = torch.stack(Xt_list)
    cond_batch = torch.stack(cond_list)
    target_batch = torch.stack(target_list)

    graph_batch = Batch.from_data_list(graph_list)

    return X_batch, graph_batch, Xt_batch, cond_batch, target_batch
