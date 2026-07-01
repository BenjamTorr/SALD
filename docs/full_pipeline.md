# Full Experiment Pipeline (no\_mean)

This script (`scripts/run_full_experiment.py`) runs the entire workflow end‑to‑end:

1. Train VAE on FC20 ↦ latent, export latents/recons/Frobenius stats.
2. Train two ridge predictors (base, VAE‑noise augmented) from FC → target.
3. Train diffusion model (graph or fm) on exported latents.
4. Optional raw sampling from the trained diffusion model.
5. LoRA fine‑tune the diffusion model twice (base + VAE predictors).
6. Sample both fine‑tuned models and save decoded FCs.

## How to run

All settings live in the `CONFIG` dictionary at the top of `scripts/run_full_experiment.py`.
Edit the values in-place, then execute:

```bash
python scripts/run_full_experiment.py
```

No CLI flags are required.

### Key CONFIG fields

- `experiment_id`: tag for the run; used in output folder name.
- `time_short`: FC_t window (minutes, 1–7) used across VAE, diffusion, ridge.
- `model_type`: `"graph"` or `"fm"` diffusion.
- `standardize`: True → subtract mean/divide std of FC20 before training.
- `use_sc`, `use_fct`, `use_cov`, `use_resample`: conditioning / SC options.
- `device`, `seed`, `num_workers`.
- Paths: `output_root` (default `/data/benjamin_project/diffusion_models/experiments/times`).

Sub-blocks:

- `vae`: `epochs`, `batch_size`, `lr`, `patience`, `accumulation_steps`.
- `diffusion`: `epochs`, `lr`, `patience`, `accumulation_steps`, `use_scheduler`, `batch_size`.
- `sampling`: `denoising_steps`, `eta`, `n_samples_per_subject`, `chunk_size`,
  `decode_batch_size`, `batch_size`, `split`.
- `ridge`: `ridge_grid`, `n_latent_samples`, `plot`, `batch_size`.
- `finetune`: `run_name` (key in `config/lora_config.yaml`), `epochs`, `patience`.
- Flags: `skip_sampling`, `skip_finetune`.

## Outputs

Each run creates `/data/benjamin_project/diffusion_models/experiments/times/<experiment_id>_<time>_<timestamp>/`
with subfolders:

- `vae/` — VAE checkpoint.
- `latents/<time>/` — exported latents, reconstructions, Frobenius metrics.
- `diffusion/<time>/<model>/` — diffusion checkpoint.
- `generated/<time>/<model>/` — raw diffusion samples (if enabled).
- `prediction_models/<time>/` — ridge→linear models (`base`, `vae`).
- `finetuned_models/<time>/<model>/` — LoRA fine‑tuned checkpoints (base + vae).
- `finetuned_data/<time>/<model>/` — decoded FC samples from fine‑tuned models.
- `config_snapshot.yaml` — full config + metadata used for the run.

## Notes / Tips

- GPU is strongly recommended (CUDA autocast is used in training/sampling).
- `standardize=False` keeps original space; latents, ridge, and diffusion all follow that choice.
- `use_resample=True` applies Gaussian SC resampling consistently (training, sampling, finetune).
- Two predictors are always trained:  
  - `base`: deterministic ridge.  
  - `vae`: ridge with VAE noise augmentation (latent sampling).
- Fine‑tuning is executed for both predictors; generated files are suffixed with `base` or `vae`.
- Adjust `time_short` to switch FC_t window; latents and downstream steps are recomputed automatically.
