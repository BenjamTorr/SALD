## rerun_finetune_and_sampling.py

Replays the fine-tuning and sampling stages of a previous diffusion experiment, reusing its trained artifacts (VAE, latent exports, ridge predictors, diffusion checkpoint) while writing all new outputs into a fresh run directory. This is useful after the `transform_sc` / resampling fix to regenerate downstream results without retraining everything from scratch.

### What it does
- Loads the prior run’s `config_snapshot.yaml` to rebuild the configuration (including `scale_tag` and `time_tag`).
- Copies required checkpoints and latents from the old run into a new run folder.
- Re-runs LoRA fine-tuning for both ridge predictors (base and VAE-aug).
- Samples both the raw diffusion model and the fine-tuned variants on val/test splits, saving outputs under the new run root.

### Arguments
- `--prev-run` **(required)**: Path to the previous experiment root (the folder that contains `config_snapshot.yaml`, `vae/`, `latents/`, `diffusion/`, `prediction_models/`).
- `--output-root` *(optional)*: Override for where the new run folder is created. Defaults to the `output_root` stored in the prior config snapshot.
- `--suffix` *(optional, default: `resamplefix`)*: Text inserted into the new folder name before the timestamp to keep reruns distinguishable.
- `--device` *(optional)*: Override the device string from the previous config (e.g., `cuda:0`, `cuda:1`, `cpu`).
- `--finetune-epochs` *(optional)*: Override `cfg.finetune.epochs` for this rerun.

### Example
```bash
python scripts/rerun_finetune_and_sampling.py \
  --prev-run /data/benjamin_project/diffusion_models/experiments/times/6min_fm_20240110-123456 \
  --suffix resamplefix \
  --output-root /data/benjamin_project/diffusion_models/experiments/times
```
New artifacts land in a directory named like `<prev_run_name>_resamplefix_<timestamp>` under the chosen output root.
