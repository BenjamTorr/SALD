"""Re-run fine-tuning and sampling using a previous experiment's checkpoints.

This script is intended for the resample/transform_sc fix: it reuses the VAE,
latent exports, and diffusion model from an earlier run, then
repeats the fine-tuning and sampling stages under a fresh output directory so
artifacts don't get mixed.
"""

import argparse
import datetime
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import diffusion.ddpm as ddpm_module

from run_full_experiment import (
    DEFAULT_OUTPUT_ROOT,
    ExperimentPaths,
    build_diffusion_model,
    compact_experiment_tag,
    finetune_diffusion,
    parse_predictor_spec,
    train_ridge_models,
    sample_diffusion,
    sample_finetuned,
    set_seed,
    snapshot_config,
    to_ns,
    prepare_fc_sets,
)
from data.load_data import load_data


def metadata_suffix(metadata_path: str) -> str:
    stem = Path(metadata_path).stem.lower()
    return "_full" if stem.endswith("full") else ""


def load_previous_config(prev_root: Path):
    cfg_path = prev_root / "config_snapshot.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find {cfg_path}; ensure you point to a previous run root.")
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_ns = to_ns(cfg_dict)

    # ensure expected convenience fields exist
    if not hasattr(cfg_ns, "scale_tag"):
        cfg_ns.scale_tag = "scaled" if cfg_ns.standardize else "raw"
    if not hasattr(cfg_ns, "time_tag"):
        cfg_ns.time_tag = f"{cfg_ns.time_short}min"
    if not hasattr(cfg_ns, "metadata_path"):
        cfg_ns.metadata_path = "../config/metadata.yaml"

    # backfill newer finetune flags for older snapshots
    if hasattr(cfg_ns, "finetune"):
        if not hasattr(cfg_ns.finetune, "use_scheduler"):
            cfg_ns.finetune.use_scheduler = True
        if not hasattr(cfg_ns.finetune, "evaluate_baseline"):
            cfg_ns.finetune.evaluate_baseline = True
        if not hasattr(cfg_ns.finetune, "debug_lora_grads"):
            cfg_ns.finetune.debug_lora_grads = False
    if not hasattr(cfg_ns, "ridge"):
        cfg_ns.ridge = SimpleNamespace()
    if not hasattr(cfg_ns.ridge, "batch_size"):
        cfg_ns.ridge.batch_size = 256
    if not hasattr(cfg_ns.ridge, "n_latent_samples"):
        cfg_ns.ridge.n_latent_samples = 10
    if not hasattr(cfg_ns.ridge, "plot"):
        cfg_ns.ridge.plot = True
    if not hasattr(cfg_ns.ridge, "ridge_grid"):
        cfg_ns.ridge.ridge_grid = [float(x) for x in torch.logspace(-2, 2, steps=100)]
    if not hasattr(cfg_ns, "sampling"):
        cfg_ns.sampling = SimpleNamespace()
    if not hasattr(cfg_ns.sampling, "slice_index"):
        cfg_ns.sampling.slice_index = 0
    return cfg_ns


def _raw_generated_path(paths: ExperimentPaths, cfg, split: str) -> Path:
    return paths.generated / f"FC_gen_{cfg.model_type}_{paths.time_tag}_{cfg.scale_tag}_{split}.pt"


def copy_prior_artifacts(prev_paths: ExperimentPaths, new_paths: ExperimentPaths, cfg):
    """Copy VAE, latents, diffusion, and existing raw samples."""

    new_paths.make_dirs()

    # VAE
    vae_name = f"vae_{prev_paths.time_tag}_{cfg.scale_tag}.pt"
    shutil.copy2(prev_paths.vae / vae_name, new_paths.vae / vae_name)

    # Latents (entire time-tag folder)
    shutil.copytree(prev_paths.latents, new_paths.latents, dirs_exist_ok=True)

    # Diffusion checkpoint
    diff_ckpt = "ddpm_fm.pt" if cfg.model_type == "fm" else "ddpm_graph.pt"
    shutil.copy2(prev_paths.diffusion / diff_ckpt, new_paths.diffusion / diff_ckpt)

    # Resampled SC (if present)
    prev_resample_dir = prev_paths.root / "resampled_sc"
    if prev_resample_dir.exists():
        shutil.copytree(prev_resample_dir, new_paths.root / "resampled_sc", dirs_exist_ok=True)

    copied_raw_splits = []
    for split in ["val", "test"]:
        src = _raw_generated_path(prev_paths, cfg, split)
        dst = _raw_generated_path(new_paths, cfg, split)
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_raw_splits.append(split)

    return copied_raw_splits


def build_new_run_root(prev_root: Path, cfg, output_root: Path | None, suffix: str):
    base_root = output_root or Path(getattr(cfg, "output_root", DEFAULT_OUTPUT_ROOT))
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    md_suffix = metadata_suffix(cfg.metadata_path)
    run_tag = compact_experiment_tag(getattr(cfg, "experiment_id", "rerun"))
    suffix_tag = compact_experiment_tag(suffix, max_len=12)
    run_name = f"{run_tag}_{suffix_tag}_{stamp}" if suffix_tag else f"{run_tag}_{stamp}"
    legacy_run_name = f"{prev_root.name}_{suffix}{md_suffix}_{stamp}"
    return base_root / run_name, stamp, legacy_run_name


def load_diffusion_from_ckpt(cfg, paths: ExperimentPaths, device):
    with open("../config/diffusion_config.yaml", "r") as f:
        diffusion_config = yaml.safe_load(f)

    model, ckpt_name = build_diffusion_model(cfg.model_type, diffusion_config, device)
    ckpt_path = paths.diffusion / ckpt_name
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Reuse prior experiment artifacts and rerun fine-tune + sampling.")
    parser.add_argument("--prev-run", required=True, help="Path to the previous experiment root folder.")
    parser.add_argument("--output-root", default=None, help="Optional override for the new output root.")
    parser.add_argument("--device", default=None, help="Optional override for the device string (e.g., cuda:0).")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional LoRA run config key override (cfg.finetune.run_name from config/lora_config.yaml).",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=None,
        help="Optional override for finetune epochs (cfg.finetune.epochs).",
    )
    parser.add_argument(
        "--finetune-use-scheduler",
        dest="finetune_use_scheduler",
        action="store_true",
        default=None,
        help="Override: enable scheduler during finetune (cfg.finetune.use_scheduler). Add --no-finetune-use-scheduler to disable.",
    )
    parser.add_argument(
        "--no-finetune-use-scheduler",
        dest="finetune_use_scheduler",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--finetune-eval-baseline",
        dest="finetune_eval_baseline",
        action="store_true",
        default=None,
        help="Override: run baseline (LoRA=0) eval before finetune (cfg.finetune.evaluate_baseline). Add --no-finetune-eval-baseline to skip.",
    )
    parser.add_argument(
        "--no-finetune-eval-baseline",
        dest="finetune_eval_baseline",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--finetune-debug-lora-grads",
        dest="finetune_debug_lora_grads",
        action="store_true",
        default=None,
        help="Override: print LoRA grad norms every optimizer step (cfg.finetune.debug_lora_grads). Add --no-finetune-debug-lora-grads to silence.",
    )
    parser.add_argument(
        "--no-finetune-debug-lora-grads",
        dest="finetune_debug_lora_grads",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--suffix",
        default="resamplefix",
        help="Suffix appended to the new run folder name (timestamp is added automatically).",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only fine-tune/sample with the base guide model (skip VAE guide model).",
    )
    parser.add_argument(
        "--predictors",
        default="bvc",
        help="Compact predictor selector: b=base, v=vae, c=corr (examples: bvc, b, c, bv).",
    )
    parser.add_argument("--ridge-grid-min-exp", type=float, default=-2.0, help="Ridge grid min exponent for 10**x.")
    parser.add_argument("--ridge-grid-max-exp", type=float, default=5.0, help="Ridge grid max exponent for 10**x.")
    parser.add_argument("--ridge-grid-steps", type=int, default=100, help="Number of ridge-grid points.")
    args = parser.parse_args()

    prev_root = Path(args.prev_run).expanduser().resolve()
    cfg = load_previous_config(prev_root)

    # Optional overrides (non-essential so we allow changes)
    if args.device is not None:
        cfg.device = args.device
    if args.finetune_epochs is not None and hasattr(cfg, "finetune"):
        cfg.finetune.epochs = args.finetune_epochs
    if hasattr(cfg, "finetune"):
        if args.run_name is not None:
            cfg.finetune.run_name = args.run_name
        if args.finetune_use_scheduler is not None:
            cfg.finetune.use_scheduler = args.finetune_use_scheduler
        if args.finetune_eval_baseline is not None:
            cfg.finetune.evaluate_baseline = args.finetune_eval_baseline
        if args.finetune_debug_lora_grads is not None:
            cfg.finetune.debug_lora_grads = args.finetune_debug_lora_grads
    cfg.ridge.ridge_grid = [
        float(x)
        for x in torch.logspace(args.ridge_grid_min_exp, args.ridge_grid_max_exp, steps=args.ridge_grid_steps)
    ]
    selected_predictors = ["base"] if args.base_only else parse_predictor_spec(args.predictors)

    # Force downstream steps to run
    cfg.skip_sampling = False
    cfg.skip_finetune = False

    time_tag = getattr(cfg, "time_tag", f"{cfg.time_short}min")

    # Create path helpers for old and new runs
    new_root, stamp, legacy_run_name = build_new_run_root(
        prev_root, cfg, Path(args.output_root) if args.output_root else None, args.suffix
    )
    paths_prev = ExperimentPaths(root=prev_root, time_tag=time_tag, model_type=cfg.model_type)
    paths_new = ExperimentPaths(root=new_root, time_tag=time_tag, model_type=cfg.model_type)

    copied_raw_splits = copy_prior_artifacts(paths_prev, paths_new, cfg)

    snapshot_config(
        cfg,
        paths_new,
        meta={
            "prev_run": str(prev_root),
            "run_stamp": stamp,
            "scale_tag": cfg.scale_tag,
            "time_tag": time_tag,
            "legacy_run_name": legacy_run_name,
            "finetune_run_name": getattr(cfg.finetune, "run_name", None) if hasattr(cfg, "finetune") else None,
        },
    )

    device = torch.device(cfg.device)
    print(f"Using ddpm module: {ddpm_module.__file__}")
    if not torch.cuda.is_available():
        print("⚠️ CUDA not detected; training/fine-tuning will run on CPU.")

    set_seed(cfg.seed)

    data = load_data(metadata_path=cfg.metadata_path)
    fc20_sets, fct_sets, mean_fc, std_fc = prepare_fc_sets(data, cfg.time_short, cfg.standardize)
    sc_size = data["SC"]["train"].shape[-1]
    sc_shape = (-1, 1, sc_size, sc_size)

    # Load diffusion model from copied checkpoint for raw sampling
    model = load_diffusion_from_ckpt(cfg, paths_new, device)

    print("\n[1/4] Refitting ridge predictors (base, vae, corr)...")
    ridge_ckpts = train_ridge_models(
        cfg,
        data,
        paths_new,
        device,
        sc_size,
        fc20_sets,
        fct_sets,
        mean_fc,
        std_fc,
        predictor_keys=selected_predictors,
    )
    predictor_keys = list(ridge_ckpts.keys())

    print(f"\nNew run root: {paths_new.root}")
    print(f"Reusing artifacts from: {paths_prev.root}")
    print(f"Metadata source: {cfg.metadata_path}")
    if hasattr(cfg, "finetune"):
        print(f"LoRA run_name: {cfg.finetune.run_name}")
    if copied_raw_splits:
        print(f"Copied existing raw samples from previous run for: {', '.join(copied_raw_splits)}")
    if args.base_only:
        print("Base-only mode enabled: fine-tuning/sampling only with predictor 'base'.")

    # Fine-tuning
    if args.base_only:
        print("\n[2/4] Fine-tuning diffusion with LoRA (base predictor only)...")
    else:
        print("\n[2/4] Fine-tuning diffusion with LoRA (base + VAE + corr predictors)...")
    for predictor_type in predictor_keys:
        pred_path = ridge_ckpts[predictor_type]
        print(f"   - Fine-tune with predictor '{predictor_type}'")
        finetune_diffusion(cfg, data, paths_new, device, sc_size, predictor_path=pred_path, predictor_type=predictor_type, fc20_sets=fc20_sets)

    # Raw diffusion sampling
    print("\n[3/4] Sampling raw diffusion (val & test)...")
    for split in ["val", "test"]:
        if split in copied_raw_splits:
            out_path = _raw_generated_path(paths_new, cfg, split)
            print(f"   - Raw samples [{split}] already copied -> {out_path}")
            continue
        out_path = sample_diffusion(cfg, model, data, paths_new, device, sc_shape, split=split)
        print(f"   - Raw samples [{split}] -> {out_path}")

    # Fine-tuned sampling
    if args.base_only:
        print("\n[4/4] Sampling fine-tuned models (val & test, base predictor only)...")
    else:
        print("\n[4/4] Sampling fine-tuned models (val & test, base + VAE + corr predictors)...")
    for predictor_type in predictor_keys:
        for split in ["val", "test"]:
            out_path = sample_finetuned(
                cfg, data, paths_new, device, sc_size, predictor_type=predictor_type, split=split
            )
            print(f"   - Finetuned '{predictor_type}' [{split}] -> {out_path}")

    print("\nDone. All artifacts saved under:", paths_new.root)


if __name__ == "__main__":
    main()
