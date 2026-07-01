# SALD

Code for **"Short-to-Long Functional Connectivity Transfer via Structure-Aware Latent Diffusion"**.

A VAE encodes functional connectivity (FC) matrices into a latent space, a structure-aware
diffusion model (FiLM-conditioned DiT, conditioned on structural connectivity, a short FC window,
and covariates such as age/sex) is trained on those latents to transfer short-window FC into
long-window FC, and the diffusion model is optionally LoRA-fine-tuned against a kernel ridge-regression
trait predictor.

See [docs/CREDITS.md](docs/CREDITS.md) for third-party code this project builds on.

## Repository layout

```
config/     YAML configs (metadata, VAE, diffusion, LoRA)
scripts/    Entry points — see "Scripts" below
src/        Installable package (data loading, VAE, diffusion, fine-tuning, guiding model, utils)
docs/       Pipeline notes and credits
```

`src/` subpackages:
- `data/` — `load_data.py` (loads FC/SC/covariates/splits from `master_data_dir`), `loaders.py` (PyTorch `Dataset`s)
- `vae/` — `unet_vae.py`, the FC20 ↔ latent VAE
- `diffusion/` — `ddpm.py` (noise scheduler + train/sample/fine-tune loops), `dit_FiLM.py` (the FM network)
- `fine_tuning/` — `LoRA.py`, LoRA injection for the diffusion network
- `guiding_model/` — `predictor.py`, the ridge/linear trait predictor used both as a training signal and for DRaFT-style fine-tuning
- `utils/` — preprocessing, plotting, evaluation, reproducibility helpers
- `ablation/` — currently unused (not wired into any script)

## Setup

1. Create the environment from the exported spec (adjust for your platform if needed):
   ```bash
   conda env create -f no_mean.yml
   # or: pip install -r no_mean.txt
   ```
2. Install this package in editable mode so `data`, `vae`, `diffusion`, etc. are importable:
   ```bash
   pip install -e .
   ```
3. All scripts assume they're run **from inside `scripts/`** (configs are loaded via relative
   paths like `../config/metadata.yaml`):
   ```bash
   cd scripts
   ```

## Data assumptions

- `config/metadata.yaml` / `metadata_full.yaml` point `master_data_dir` at a shared cluster path
  (`/data/benjamin_project/diffusion_models/experiments/slices/`) containing the raw FC/SC tensors.
  This data is not part of the repo.
- `split_file` in those same configs currently points at `.npz` split-index files living in the
  sibling `no_mean` experiment's `config/` directory rather than a copy inside SALD. If you move
  off that machine, regenerate or copy those split files first.

## Reproducing results

### Option A — full pipeline (recommended)

`scripts/run_full_experiment.py` runs the entire workflow end-to-end from a single `CONFIG` dict
at the top of the file (no CLI flags): VAE training → ridge predictors → fm diffusion training →
raw sampling → LoRA fine-tuning (base + VAE predictors) → sampling of the fine-tuned models.

```bash
cd scripts
python run_full_experiment.py
```

Edit `CONFIG` in-place first (`experiment_id`, `time_short`, `finetune.run_name` — currently
`config_21`, matching `config/lora_config.yaml`). See [docs/full_pipeline.md](docs/full_pipeline.md)
for the full field-by-field breakdown.

### Option B — step-by-step (legacy scripts)

Run in order, editing each script's in-file config block first:

1. `001_train_vae.py` → 2. `002_export_latents_and_recons.py` → 3. `003_train_diffusion.py`
   (or `004_train_diff_raw.py`) → 4. `005_sample_diff_raw.py` → 5. `006_fine_tune_ddpm.py` →
   6. `007_sampling_finetuned.py`

These predate `run_full_experiment.py` and write outputs under paths that currently point into
the sibling `no_mean` project's directories — check the `CONFIG`/path constants at the top of each
before running.

## Scripts

| Script | Purpose |
|---|---|
| `001_train_vae.py` | Train the FC20 ↔ latent VAE; exports training-curve plots. |
| `002_export_latents_and_recons.py` | Run a trained VAE over train/val/test to export latent embeddings and reconstructions (CLI: `--ckpt`, `--device`, `--batch-size`). |
| `003_train_diffusion.py` | Train the fm diffusion model on exported latents; CLI-driven (`--epochs`, `--lr`, `--no-sc`, etc.). |
| `004_train_diff_raw.py` | Same as above but config-at-top-of-file style (no CLI), standalone script. |
| `005_sample_diff_raw.py` | Sample FC matrices from a trained (non-fine-tuned) diffusion checkpoint and decode with the VAE. |
| `006_fine_tune_ddpm.py` | LoRA fine-tune a diffusion checkpoint against a ridge/linear trait predictor (DRaFT-style). |
| `007_sampling_finetuned.py` | Sample FC matrices from a LoRA-fine-tuned diffusion checkpoint. |
| `run_full_experiment.py` | End-to-end pipeline (see Option A above). Also defines `ExperimentPaths`, `build_diffusion_model`, and the train/sample/fine-tune functions reused by the rerun scripts below. |
| `rerun_finetune_and_sampling.py` | Reuses a prior run's VAE/latents/diffusion checkpoint and re-runs only fine-tuning + sampling into a fresh run folder. See `rerun_finetune_and_sampling.md` for full CLI docs. |
| `rerun_from_vae.py` | Reuses only a prior run's VAE checkpoint; re-exports latents and retrains diffusion + fine-tuning + sampling from there. |
| `resample_existing_run.py` | Re-samples (overwrites) outputs of an existing run in-place using its stored checkpoints, without creating a new run directory. |

## Config files

| File | Used by |
|---|---|
| `metadata.yaml` / `metadata_full.yaml` | Data location, split file, trait variable, seed. `run_full_experiment.py` defaults to `metadata_full.yaml`; the numbered scripts default to `metadata.yaml`. |
| `vae_config.yaml` | VAE architecture (`VAE_params`). |
| `diffusion_config.yaml` | `DDPM_config` (noise schedule) + `DIT_config_film` (fm network) + `DIT_config_cat` (shared seq length/channels, legacy name). |
| `lora_config.yaml` | Named LoRA fine-tuning configs; `config_21` is the one currently used, plus `config_21_ablation` / `config_21_KL_ablation` variants. |

## Credits

Portions of this codebase adapt code from other repositories — see
[docs/CREDITS.md](docs/CREDITS.md).
