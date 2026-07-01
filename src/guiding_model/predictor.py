import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Helpers
# -----------------------------
def upper_triangle_from_fc(FC: torch.Tensor, offset: int = 1) -> torch.Tensor:
    """
    FC: (B, N, N)
    Returns: (B, L) where L = N*(N-1)/2 (if offset=1) using upper triangle.
    """
    if FC.dim() != 3 or FC.size(-1) != FC.size(-2):
        raise ValueError(f"FC must be (B, N, N). Got {tuple(FC.shape)}")

    B, N, _ = FC.shape
    iu = torch.triu_indices(N, N, offset=offset, device=FC.device)
    return FC[:, iu[0], iu[1]]  # (B, L)

def _row_corr_normalize(X, eps=1e-8):
    # Normalize along the feature axis so both (B, D) and (B, 1, D) inputs work.
    Xc = X - X.mean(dim=-1, keepdim=True)
    return Xc / (Xc.norm(dim=-1, keepdim=True) + eps)

# -----------------------------
# Closed-form Ridge regression: FC -> scalar
# -----------------------------
class RidgeRegression(nn.Module):
    """
    Closed-form ridge regression mapping subject FC (B,100,100) to a scalar target (B,1).
    Uses upper-triangular FC features by default.
    """

    def __init__(
        self,
        ridge_grid=None,
        device=None,
        plot=True,
        save_path=None,
        use_upper_triangle=True,
        include_diagonal=False,
        standardize_y=False,
        corr_ker = False,
    ):
        
        super().__init__()
        self.device = device or torch.device("cpu")
        self.corr_ker = corr_ker
        if ridge_grid is None:
            ridge_grid = [1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100]
        self.ridge_grid = ridge_grid

        self.plot = plot
        self.save_path = save_path

        self.use_upper_triangle = use_upper_triangle
        self.include_diagonal = include_diagonal
        self.standardize_y = standardize_y

        self.best_lambda = None
        self.W = None        # (D, 1)
        self.bias = None     # (1, 1)

        # Stored for consistent preprocessing in predict()
        self.X_mean = None
        self.y_mean = None
        self.y_std = None

    def _build_inputs(self, FC: torch.Tensor) -> torch.Tensor:
        """
        FC: (B, N, N)
        Returns X: (B, D)
        """
        FC = FC.to(self.device).float()
        if self.use_upper_triangle:
            offset = 0 if self.include_diagonal else 1
            X = upper_triangle_from_fc(FC, offset=offset)
        else:
            # full flatten (includes redundant symmetry)
            X = FC.reshape(FC.size(0), -1)
        return X

    def _build_target(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B,) or (B,1)
        Returns: (B,1)
        """
        y = y.to(self.device).float()
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if y.dim() != 2 or y.size(1) != 1:
            raise ValueError(f"Target must be (B,) or (B,1). Got {tuple(y.shape)}")
        return y
    
    """
    def _accumulate(self, loader):
        
        #Expects each batch as (FC, y) OR dict-like with keys "FC" and "y".
        
        Xs, ys = [], []
        for batch in loader:
            if len(batch) == 5:
                FC, _, _, _, y = batch
            elif isinstance(batch, dict):
                FC, y = batch["FC"], batch["y"]
            else:
                raise ValueError(
                    "Loader batch must be (FC, y) or {'FC': FC, 'y': y}."
                )

            Xs.append(self._build_inputs(FC))
            ys.append(self._build_target(y))

        return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)
    """

    def _accumulate(
        self,
        loader,
        vae: torch.nn.Module = None,
        use_vae_noise_aug: bool = False,
        n_latent_samples: int = 1,
        device = 'cpu'
    ):
        """
        Expects each batch as:
        - (FC, y), OR
        - 5-tuple (FC, _, _, _, y), OR
        - dict with keys {"FC","y"}.

        If use_vae_noise_aug=True:
        - FC is passed through the provided VAE encoder to obtain (mean, logvar)
        - We draw n_latent_samples latent samples z ~ N(mean, exp(logvar))
        - Decode each z, obtain reconstructed FCs
        - Build X from each reconstructed FC and average X across samples
            (noise-invariant / "blinded" features)

        Shapes:
        - FC from loader: (B,100,100)
        - VAE expects: (B,1,100,100)  -> we unsqueeze(1)
        - VAE decode output: (B,1,100,100) -> we squeeze(1) before ridge features
        """
        self.device = device
        if use_vae_noise_aug:
            if vae is None:
                raise ValueError("use_vae_noise_aug=True requires `vae`.")
            if int(n_latent_samples) < 1:
                raise ValueError("n_latent_samples must be >= 1.")
            vae = vae.to(self.device)

        Xs, ys = [], []
        if use_vae_noise_aug:
            vae.eval()
        for batch in tqdm(loader):
            if isinstance(batch, (tuple, list)) and len(batch) == 5:
                FC, _, _, _, y = batch
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                FC, y = batch
            elif isinstance(batch, dict):
                FC, y = batch["FC"], batch["y"]
            else:
                raise ValueError("Loader batch must be (FC,y), (FC,_,_,_,y), or {'FC':...,'y':...}.")

            y_t = self._build_target(y)  # (B,1)

            if not use_vae_noise_aug:
                X_batch = self._build_inputs(FC)
                Xs.append(X_batch)
                ys.append(y_t)
                continue

            # --- VAE augmentation path (encode -> sample latent -> decode)
            FC = FC.to(self.device).float()     # (B,100,100)
            offset = 0 if self.include_diagonal else 1
            X = upper_triangle_from_fc(FC, offset=offset)
            x = X.unsqueeze(1)                 # (B,1,4950)

            # closed-form ridge: no need for grads through VAE
            vae_was_training = vae.training

            with torch.no_grad():
                # Use the *specific* VAE API you provided
                # encode(x) returns: (sample_z, encoder_output)
                # encoder_output = out BEFORE chunk, so chunk -> mean, logvar
                _, encoder_output = vae.encode(x)
                mean, logvar = torch.chunk(encoder_output, 2, dim=1)
                std = torch.exp(0.5 * logvar)

                X_mc = []
                for _ in range(int(n_latent_samples)):
                    eps = torch.randn_like(std)
                    z = mean + std * eps
                    recon = vae.decode(z)        # (B,1,4950)
                    FC_recon = recon.squeeze(1)  # (B,4950)
                    X_mc.append(FC_recon)  # (B,D)

                # Average features over latent samples -> (B,D)
                X_batch = torch.stack(X_mc, dim=0).mean(dim=0)

            if vae_was_training:
                vae.train()

            Xs.append(X_batch)
            ys.append(y_t)

        return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)


    @torch.no_grad()
    def fit(self, train_loader, val_loader, 
        vae: torch.nn.Module = None,
        use_vae_noise_aug: bool = False,
        n_latent_samples: int = 1, device = 'cpu'):
        X_train, y_train = self._accumulate(train_loader, vae, use_vae_noise_aug, n_latent_samples, device = device)
        X_val, y_val = self._accumulate(val_loader, vae, use_vae_noise_aug, n_latent_samples, device = device)

        if self.corr_ker:
            X_train = _row_corr_normalize(X_train)
            X_val = _row_corr_normalize(X_val)

        # Center X and y (ridge with intercept via centering)
        self.X_mean = X_train.mean(0, keepdim=True)
        self.y_mean = y_train.mean(0, keepdim=True)

        if self.corr_ker:
            Xc = X_train - self.X_mean # already row-normalized
        else:
            Xc = X_train - self.X_mean

        yc = y_train - self.y_mean

        # Optional: scale y (sometimes stabilizes when y magnitude is large)
        if self.standardize_y:
            self.y_std = yc.std(0, keepdim=True).clamp_min(1e-12)
            yc = yc / self.y_std
        else:
            self.y_std = torch.ones_like(self.y_mean)

        val_losses = {}

        # Precompute XtX once (same for all lambdas)
        XtX = Xc.T @ Xc
        D = XtX.size(0)
        I = torch.eye(D, device=self.device)

        for lam in self.ridge_grid:
            beta = torch.linalg.solve(XtX + lam * I, Xc.T @ yc)  # (D,1)

            # bias in standardized-y space, then apply to val predictions similarly
            bias_std = (self.y_mean / self.y_std) - (self.X_mean @ beta)

            preds_val_std = (X_val @ beta) + bias_std
            preds_val = preds_val_std * self.y_std  # back to original scale

            val_losses[lam] = F.mse_loss(preds_val, y_val).item()

        self.best_lambda = min(val_losses, key=val_losses.get)

        # Refit with best lambda
        lam = self.best_lambda
        self.W = torch.linalg.solve(XtX + lam * I, Xc.T @ yc)
        bias_std = (self.y_mean / self.y_std) - (self.X_mean @ self.W)
        self.bias = bias_std  # store in standardized-y space

        print(f"[Exact Ridge FC->Scalar] λ={lam:.3g}, Validation MSE={val_losses[lam]:.6f}")

        if self.plot:
            self._plot_val_curve(val_losses)

        return self.best_lambda

    def _plot_val_curve(self, val_losses):
        plt.figure(figsize=(6, 4))
        xs = list(val_losses.keys())
        ys = list(val_losses.values())
        plt.semilogx(xs, ys, marker="o", linewidth=2)
        plt.axvline(self.best_lambda, linestyle="--", label=f"Best λ={self.best_lambda:.3g}")
        plt.xlabel("Ridge penalty λ (log scale)")
        plt.ylabel("Validation MSE ↓")
        plt.title("Ridge Validation Curve (FC → Scalar)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        if self.save_path:
            plt.savefig(self.save_path, bbox_inches="tight", dpi=150)
        plt.show()

    @torch.no_grad()
    def predict(self, loader, return_y=True, 
                vae: torch.nn.Module = None,
                use_vae_noise_aug: bool = False,
                n_latent_samples: int = 1,
                device = 'cpu'):
        """
        Returns:
          preds: (B,1)
          y (optional): (B,1)
        """
        X_all, y_all = self._accumulate(loader, vae, use_vae_noise_aug, n_latent_samples, device = device)
        if self.corr_ker:
            X_all = _row_corr_normalize(X_all)
        preds_std = (X_all @ self.W) + self.bias
        preds = preds_std * self.y_std

        if return_y:
            return preds, y_all
        return preds

class LinearRegression(nn.Module):
    def __init__(self, in_features, beta_vector=None, intercept=None, freeze=False, corr_ker=False):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=True)  # include intercept (bias)
        self.corr_ker = corr_ker
        # If coefficients are provided, set them
        with torch.no_grad():
            if beta_vector is not None:
                # Ensure shape (1, p)
                self.linear.weight.copy_(beta_vector.reshape(1, -1))
            if intercept is not None:
                # Ensure scalar tensor
                self.linear.bias.copy_(torch.tensor([intercept], dtype=torch.float32))
        
        # Optionally freeze all parameters
        if freeze:
            for param in self.linear.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.corr_ker:
            x = _row_corr_normalize(x)
        return self.linear(x).reshape(-1)

    @torch.no_grad()
    def predict_from_fc_tensor(self, x: torch.Tensor, offset: int = 1) -> torch.Tensor:
        """
        x can be:
        - (B, N, N)      -> returns (B,)
        - (B, S, N, N)   -> returns (B, S)

        Uses upper triangle features (offset=1 by default) then feeds into the linear model.
        """
        if x.dim() == 3:
            # (B, N, N) -> (B, U)
            feats = upper_triangle_from_fc(x, offset=offset)  # (B, U)

            # Sanity check: model input size must match extracted feature length
            expected = self.linear.in_features
            if feats.size(-1) != expected:
                raise ValueError(
                    f"Feature length mismatch. Got {feats.size(-1)} from FC, "
                    f"but model expects in_features={expected}."
                )

            # Model returns (B,)
            return self.forward(feats)

        elif x.dim() == 4:
            # (B, S, N, N) -> reshape to (B*S, N, N)
            if x.size(-1) != x.size(-2):
                raise ValueError(f"x must be square in last two dims. Got {tuple(x.shape)}")

            B, S, N, _ = x.shape
            x_flat = x.reshape(B * S, N, N)  # (B*S, N, N)

            # (B*S, U)
            feats_flat = upper_triangle_from_fc(x_flat, offset=offset)

            expected = self.linear.in_features
            if feats_flat.size(-1) != expected:
                raise ValueError(
                    f"Feature length mismatch. Got {feats_flat.size(-1)} from FC, "
                    f"but model expects in_features={expected}."
                )

            # Predict per slice: (B*S,)
            preds_flat = self.forward(feats_flat)

            # Back to (B, S)
            return preds_flat.reshape(B, S)

        else:
            raise ValueError(
                f"x must have shape (B,N,N) or (B,S,N,N). Got {tuple(x.shape)}"
            )



# -----------------------------
# Example expected loader batches
# -----------------------------
# for FC, y in train_loader:
#     FC: torch.Tensor of shape (B,100,100)
#     y:  torch.Tensor of shape (B,) or (B,1)
#
# model = RidgeFCToScalarExact(device=torch.device("cuda"), plot=True)
# model.fit(train_loader, val_loader)
# preds, y = model.predict(test_loader)
