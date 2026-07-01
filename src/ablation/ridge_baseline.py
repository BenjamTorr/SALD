import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from utils.preprocessing.transformations import (
    get_upper_diagonal_elements_no_chan,
    upper_elements_to_symmetric_matrix_no_chan,
)


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    """(B, ...) -> (B, D)."""
    if x.dim() < 2:
        raise ValueError(f"Expected batched tensor with shape (B, ...), got {tuple(x.shape)}")
    if x.dim() == 2:
        return x
    return x.reshape(x.size(0), -1)


def get_sc_features(sc_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Returns SC features as (B, D_sc) for vector/tensor SC inputs.
    Compatible with FC_SCVectorDataset batches from the VAE pipeline.
    """
    if not isinstance(sc_batch, torch.Tensor):
        raise ValueError(
            f"Expected SC batch as Tensor from VAE loader, got {type(sc_batch).__name__}"
        )

    sc = sc_batch.to(device).float()
    if sc.dim() == 1:
        return sc.unsqueeze(0)
    return _flatten_batch(sc)


class RidgeLatentExact(nn.Module):
    """
    Closed-form ridge regression in observation space.

    Default mapping:
    (FC_t, SC, covariates) -> FC_20

    Expected VAE loader batch:
      (x, sc, xt, cov)
    with optional fallback support for:
      (x, sc, xt, cov, target)
    """

    def __init__(
        self,
        ridge_grid=None,
        device=None,
        plot=True,
        save_path=None,
        use_fct=True,
        use_sc=True,
        use_cov=True,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        if ridge_grid is None:
            ridge_grid = [1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100]
        self.ridge_grid = ridge_grid
        self.plot = plot
        self.save_path = save_path
        self.use_fct = use_fct
        self.use_sc = use_sc
        self.use_cov = use_cov
        self.best_lambda = None
        self.W = None
        self.bias = None
        self.target_shape = None

    def _build_inputs(self, xt, sc_batch, cov):
        parts = []
        if self.use_fct:
            parts.append(_flatten_batch(xt.to(self.device).float()))
        if self.use_sc:
            parts.append(get_sc_features(sc_batch, self.device))
        if self.use_cov:
            parts.append(_flatten_batch(cov.to(self.device).float()))
        if not parts:
            raise ValueError("At least one input block must be enabled (use_fct/use_sc/use_cov).")
        return torch.cat(parts, dim=1)

    def _build_target(self, x):
        return _flatten_batch(x.to(self.device).float())

    def _accumulate(self, loader):
        Xs, ys = [], []
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                x, sc_batch, xt, cov = batch
            elif isinstance(batch, (tuple, list)) and len(batch) == 5:
                x, sc_batch, xt, cov, _ = batch
            elif isinstance(batch, dict):
                x = batch["x"]
                sc_batch = batch["sc"]
                xt = batch["xt"]
                cov = batch["cov"]
            else:
                raise ValueError(
                    "Loader batch must be (x, sc, xt, cov), (x, sc, xt, cov, target), "
                    "or dict with keys x/sc/xt/cov."
                )

            if self.target_shape is None:
                self.target_shape = tuple(x.shape[1:])

            Xs.append(self._build_inputs(xt, sc_batch, cov))
            ys.append(self._build_target(x))
        return torch.cat(Xs), torch.cat(ys)

    @torch.no_grad()
    def fit(self, train_loader, val_loader):
        X_train, y_train = self._accumulate(train_loader)
        X_val, y_val = self._accumulate(val_loader)

        X_mean = X_train.mean(0, keepdim=True)
        y_mean = y_train.mean(0, keepdim=True)
        Xc = X_train - X_mean
        yc = y_train - y_mean

        val_losses = {}

        for lam in self.ridge_grid:
            XtX = Xc.T @ Xc
            I = torch.eye(XtX.size(0), device=self.device)
            beta = torch.linalg.solve(XtX + lam * I, Xc.T @ yc)
            bias = y_mean - X_mean @ beta

            preds_val = X_val @ beta + bias
            val_loss = F.mse_loss(preds_val, y_val).item()
            val_losses[lam] = val_loss

        best_lambda = min(val_losses, key=val_losses.get)
        self.best_lambda = best_lambda

        XtX = Xc.T @ Xc
        I = torch.eye(XtX.size(0), device=self.device)
        self.W = torch.linalg.solve(XtX + best_lambda * I, Xc.T @ yc)
        self.bias = y_mean - X_mean @ self.W

        print(
            f"[Exact Ridge Observation] λ={best_lambda:.3g}, Validation MSE={val_losses[best_lambda]:.6f}"
        )

        if self.plot:
            self._plot_val_curve(val_losses)

        return best_lambda

    def _plot_val_curve(self, val_losses):
        plt.figure(figsize=(6, 4))
        plt.semilogx(list(val_losses.keys()), list(val_losses.values()), marker="o", linewidth=2)
        best_lam = self.best_lambda
        plt.axvline(best_lam, color="r", linestyle="--", label=f"Best λ={best_lam:.3g}")
        plt.xlabel("Ridge penalty λ (log scale)")
        plt.ylabel("Validation MSE ↓")
        plt.title("Ridge Validation Curve")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        if self.save_path:
            plt.savefig(self.save_path, bbox_inches="tight", dpi=150)
        plt.show(block=False)
        plt.close()

    @torch.no_grad()
    def predict(self, loader):
        X_all, y_all = self._accumulate(loader)
        preds = X_all @ self.W + self.bias
        if self.target_shape is None:
            return preds, y_all
        return preds.reshape(-1, *self.target_shape), y_all.reshape(-1, *self.target_shape)


# Backward-compatible alias for older references.
RidgeObservationExact = RidgeLatentExact


def _time_to_key(t: int) -> str:
    if t == 3:
        return "FC3"
    return f"FC_{t}"


def _slice0_fc(fc_t: torch.Tensor) -> torch.Tensor:
    """Return slice-0 matrix per subject as (B, N, N)."""
    if fc_t.dim() == 4:
        return fc_t[:, 0].float()
    if fc_t.dim() == 3:
        return fc_t.float()
    raise ValueError(f"Expected FC tensor with shape (B,S,N,N) or (B,N,N), got {tuple(fc_t.shape)}")


def _cov_age_sex_onehot(cov) -> torch.Tensor:
    """
    cov expected like load_data['Cov'][split]: [age, sex].
    Output columns: [age_numeric, sex_is_male, sex_is_female]
    """
    if isinstance(cov, torch.Tensor):
        cov = cov.detach().cpu().numpy()

    age_vals = []
    sex_vals = []
    for row in cov:
        age_vals.append(float(row[0]))
        sex_raw = str(row[1]).strip().upper()
        is_male = 1.0 if sex_raw in {"M", "MALE", "1", "TRUE"} else 0.0
        sex_vals.append([is_male, 1.0 - is_male])

    age = torch.tensor(age_vals, dtype=torch.float32).unsqueeze(1)
    sex = torch.tensor(sex_vals, dtype=torch.float32)
    return torch.cat([age, sex], dim=1)


def _sc_features(sc: torch.Tensor) -> torch.Tensor:
    if sc.dim() == 3 and sc.size(-1) == sc.size(-2):
        return get_upper_diagonal_elements_no_chan(sc.float())
    if sc.dim() == 2:
        return sc.float()
    return sc.reshape(sc.size(0), -1).float()


@torch.no_grad()
def _fit_ridge_from_splits(
    X_train,
    y_train,
    X_val,
    y_val,
    ridge_grid,
    device,
    show_progress: bool = False,
    progress_label: str = "ridge",
):
    X_train = X_train.to(device).float()
    y_train = y_train.to(device).float()
    X_val = X_val.to(device).float()
    y_val = y_val.to(device).float()

    X_mean = X_train.mean(0, keepdim=True)
    y_mean = y_train.mean(0, keepdim=True)
    Xc = X_train - X_mean
    yc = y_train - y_mean

    XtX = Xc.T @ Xc
    I = torch.eye(XtX.size(0), device=device)

    best_lam = None
    best_loss = float("inf")
    val_losses = {}
    lam_iter = ridge_grid
    if show_progress:
        lam_iter = tqdm(ridge_grid, desc=progress_label, leave=False)
    for lam in lam_iter:
        W = torch.linalg.solve(XtX + lam * I, Xc.T @ yc)
        b = y_mean - X_mean @ W
        val_loss = F.mse_loss(X_val @ W + b, y_val).item()
        val_losses[float(lam)] = float(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_lam = lam

    W = torch.linalg.solve(XtX + best_lam * I, Xc.T @ yc)
    b = y_mean - X_mean @ W
    return W, b, best_lam, best_loss, val_losses


def fit_ridge_per_timestamp_slice0(
    data: dict,
    ridge_grid=None,
    device=None,
    plot_curves: bool = True,
    plot_size=(3.0, 2.2),
    show_progress: bool = True,
):
    """
    Fit one ridge model for each timestamp t in [1..10], using only slice 0 of FC_t.

    Features per subject:
      concat([flatten(FC_t[:,0]), flatten(SC), cov=[age_numeric, sex_one_hot]])
    Target:
      FC20 upper-triangle vector.

    Returns:
      {
        "pred_matrices": {t: {"train": (B,N,N), "val": (B,N,N), "test": (B,N,N)}},
        "models": {t: {"W": ..., "bias": ..., "best_lambda": ..., "val_mse": ...}},
      }
    """
    if ridge_grid is None:
        ridge_grid = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    device = device or torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)

    # FC20 target vectors
    y_train = get_upper_diagonal_elements_no_chan(data["FC"]["train"].float())
    y_val = get_upper_diagonal_elements_no_chan(data["FC"]["val"].float())
    # Shared extra features: SC + covariates
    sc_train = _sc_features(data["SC"]["train"])
    sc_val = _sc_features(data["SC"]["val"])
    sc_test = _sc_features(data["SC"]["test"])

    cov_train = _cov_age_sex_onehot(data["Cov"]["train"])
    cov_val = _cov_age_sex_onehot(data["Cov"]["val"])
    cov_test = _cov_age_sex_onehot(data["Cov"]["test"])

    out_preds = {}
    out_models = {}

    t_iter = range(1, 11)
    if show_progress:
        t_iter = tqdm(t_iter, desc="Timestamps", leave=True)

    for t in t_iter:
        key = _time_to_key(t)
        if key not in data:
            raise KeyError(f"Missing time key '{key}' in input data.")

        xt_train = _slice0_fc(data[key]["train"]).reshape(data[key]["train"].shape[0], -1).float()
        xt_val = _slice0_fc(data[key]["val"]).reshape(data[key]["val"].shape[0], -1).float()
        xt_test = _slice0_fc(data[key]["test"]).reshape(data[key]["test"].shape[0], -1).float()

        X_train = torch.cat([xt_train, sc_train, cov_train], dim=1)
        X_val = torch.cat([xt_val, sc_val, cov_val], dim=1)
        X_test = torch.cat([xt_test, sc_test, cov_test], dim=1)

        W, b, best_lam, best_loss, val_losses = _fit_ridge_from_splits(
            X_train,
            y_train,
            X_val,
            y_val,
            ridge_grid,
            device,
            show_progress=show_progress,
            progress_label=f"t={t} lambdas",
        )

        pred_train_vec = (X_train.to(device) @ W + b).cpu()
        pred_val_vec = (X_val.to(device) @ W + b).cpu()
        pred_test_vec = (X_test.to(device) @ W + b).cpu()

        if plot_curves:
            xs = list(val_losses.keys())
            ys = [val_losses[k] for k in xs]
            plt.figure(figsize=plot_size)
            plt.semilogx(xs, ys, marker="o", linewidth=1.2, markersize=2.5)
            plt.axvline(float(best_lam), color="r", linestyle="--", linewidth=1.0)
            plt.title(f"t={t} val MSE", fontsize=9)
            plt.xlabel("lambda", fontsize=8)
            plt.ylabel("MSE", fontsize=8)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.grid(True, which="both", ls="--", alpha=0.35)
            plt.tight_layout()
            plt.show(block=False)
            plt.close()

        out_preds[t] = {
            "train": upper_elements_to_symmetric_matrix_no_chan(pred_train_vec),
            "val": upper_elements_to_symmetric_matrix_no_chan(pred_val_vec),
            "test": upper_elements_to_symmetric_matrix_no_chan(pred_test_vec),
        }
        out_models[t] = {
            "W": W.detach().cpu(),
            "bias": b.detach().cpu(),
            "best_lambda": float(best_lam),
            "val_mse": float(best_loss),
            "val_curve": val_losses,
        }

    return {"pred_matrices": out_preds, "models": out_models}
