"""Rerun pipeline starting from an existing trained VAE checkpoint.

This script creates a fresh run directory, copies the trained VAE checkpoint
from a previous run, re-exports latent tensors (to avoid stale/corrupted
latents), then retrains downstream stages and re-samples outputs.
"""

import argparse
import datetime
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import diffusion.ddpm as ddpm_module

from data.load_data import load_data
from run_full_experiment import (
    DEFAULT_OUTPUT_ROOT,
    ExperimentPaths,
    compact_experiment_tag,
    export_latents,
    finetune_diffusion,
    load_trained_vae,
    prepare_fc_sets,
    sample_diffusion,
    sample_finetuned,
    set_seed,
    snapshot_config,
    to_ns,
    parse_predictor_spec,
    train_diffusion,
    train_ridge_models,
    build_vae_loaders_export,
)


def metadata_suffix(metadata_path: str) -> str:
    stem = Path(metadata_path).stem.lower()
    return "_full" if stem.endswith("full") else ""


def load_previous_config(prev_root: Path):
    cfg_path = prev_root / "config_snapshot.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find {cfg_path}; ensure --prev-run points to a valid run root.")

    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = to_ns(cfg_dict)

    if not hasattr(cfg, "scale_tag"):
        cfg.scale_tag = "scaled" if cfg.standardize else "raw"
    if not hasattr(cfg, "time_tag"):
        cfg.time_tag = f"{cfg.time_short}min"
    if not hasattr(cfg, "metadata_path"):
        cfg.metadata_path = "../config/metadata.yaml"
    if not hasattr(cfg, "use_sc"):
        cfg.use_sc = True
    if not hasattr(cfg, "use_fct"):
        cfg.use_fct = True
    if not hasattr(cfg, "use_cov"):
        cfg.use_cov = True
    if hasattr(cfg, "finetune"):
        if not hasattr(cfg.finetune, "use_scheduler"):
            cfg.finetune.use_scheduler = True
        if not hasattr(cfg.finetune, "evaluate_baseline"):
            cfg.finetune.evaluate_baseline = True
        if not hasattr(cfg.finetune, "debug_lora_grads"):
            cfg.finetune.debug_lora_grads = False
    if not hasattr(cfg, "ridge"):
        cfg.ridge = SimpleNamespace()
    if not hasattr(cfg.ridge, "batch_size"):
        cfg.ridge.batch_size = 256
    if not hasattr(cfg.ridge, "n_latent_samples"):
        cfg.ridge.n_latent_samples = 10
    if not hasattr(cfg.ridge, "plot"):
        cfg.ridge.plot = True
    if not hasattr(cfg.ridge, "ridge_grid"):
        cfg.ridge.ridge_grid = [float(x) for x in torch.logspace(-2, 2, steps=100)]
    if not hasattr(cfg, "sampling"):
        cfg.sampling = SimpleNamespace()
    if not hasattr(cfg.sampling, "slice_index"):
        cfg.sampling.slice_index = 0
    return cfg


def build_new_run_root(prev_root: Path, cfg, output_root: Path | None, suffix: str):
    base_root = output_root or Path(getattr(cfg, "output_root", DEFAULT_OUTPUT_ROOT))
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    md_suffix = metadata_suffix(cfg.metadata_path)
    run_tag = compact_experiment_tag(getattr(cfg, "experiment_id", "rerun"))
    suffix_tag = compact_experiment_tag(suffix, max_len=12)
    run_name = f"{run_tag}_{suffix_tag}_{stamp}" if suffix_tag else f"{run_tag}_{stamp}"
    legacy_run_name = f"{prev_root.name}_{suffix}{md_suffix}_{stamp}"
    return base_root / run_name, stamp, legacy_run_name


def copy_vae_checkpoint(cfg, paths_prev: ExperimentPaths, paths_new: ExperimentPaths):
    vae_name = f"vae_{paths_prev.time_tag}_{cfg.scale_tag}.pt"
    src = paths_prev.vae / vae_name
    if not src.exists():
        raise FileNotFoundError(f"Missing VAE checkpoint: {src}")
    shutil.copy2(src, paths_new.vae / vae_name)
    return paths_new.vae / vae_name


def export_latents_from_existing_vae(cfg, data, paths: ExperimentPaths, device, sc_size, fc20_sets, fct_sets, mean_fc, std_fc):
    vae = load_trained_vae(cfg, paths, device)
    mean_mat = torch.as_tensor(mean_fc, dtype=torch.float32).squeeze(0)
    std_mat = torch.as_tensor(std_fc, dtype=torch.float32).squeeze(0)

    loaders_export = build_vae_loaders_export(
        fc20_sets,
        fct_sets,
        (data["SC"]["train"], data["SC"]["val"], data["SC"]["test"]),
        (data["Cov"]["train"], data["Cov"]["val"], data["Cov"]["test"]),
        device,
        cfg.vae.batch_size,
        sc_size,
    )

    for split_name, loader in zip(["train", "val", "test"], loaders_export):
        export_latents(split_name, loader, vae, device, mean_mat, std_mat, paths.latents, cfg.scale_tag)


def main():
    parser = argparse.ArgumentParser(description="Rerun experiment from trained VAE (recompute latents + rerun downstream).")
    parser.add_argument("--prev-run", required=True, help="Path to previous experiment root.")
    parser.add_argument("--output-root", default=None, help="Optional new output root.")
    parser.add_argument("--device", default=None, help="Optional device override, e.g., cuda:0 or cpu.")
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
        default="fromvae",
        help="Suffix appended to new run folder name (timestamp added automatically).",
    )
    parser.add_argument(
        "--predictors",
        default="bvc",
        help="Compact predictor selector: b=base, v=vae, c=corr (examples: bvc, b, c, bv).",
    )
    parser.add_argument("--ridge-grid-min-exp", type=float, default=-2.0, help="Ridge grid min exponent for 10**x.")
    parser.add_argument("--ridge-grid-max-exp", type=float, default=5.0, help="Ridge grid max exponent for 10**x.")
    parser.add_argument("--ridge-grid-steps", type=int, default=100, help="Number of ridge-grid points.")
    parser.add_argument(
        "--use-fc-short",
        dest="use_fct",
        action="store_true",
        default=None,
        help="Override: include FC-short (x_t) conditioning. Add --no-fc-short to remove it for ablation.",
    )
    parser.add_argument(
        "--no-fc-short",
        dest="use_fct",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use-sc",
        dest="use_sc",
        action="store_true",
        default=None,
        help="Override: include SC conditioning. Add --no-sc to remove it for ablation.",
    )
    parser.add_argument(
        "--no-sc",
        dest="use_sc",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use-covariates",
        dest="use_cov",
        action="store_true",
        default=None,
        help="Override: include covariates conditioning. Add --no-covariates to remove it for ablation.",
    )
    parser.add_argument(
        "--no-covariates",
        dest="use_cov",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    prev_root = Path(args.prev_run).expanduser().resolve()
    cfg = load_previous_config(prev_root)
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
    if args.use_fct is not None:
        cfg.use_fct = args.use_fct
    if args.use_sc is not None:
        cfg.use_sc = args.use_sc
    if args.use_cov is not None:
        cfg.use_cov = args.use_cov
    cfg.ridge.ridge_grid = [
        float(x)
        for x in torch.logspace(args.ridge_grid_min_exp, args.ridge_grid_max_exp, steps=args.ridge_grid_steps)
    ]
    selected_predictors = parse_predictor_spec(args.predictors)

    cfg.skip_sampling = False
    cfg.skip_finetune = False

    time_tag = getattr(cfg, "time_tag", f"{cfg.time_short}min")
    new_root, stamp, legacy_run_name = build_new_run_root(
        prev_root, cfg, Path(args.output_root) if args.output_root else None, args.suffix
    )
    paths_prev = ExperimentPaths(root=prev_root, time_tag=time_tag, model_type=cfg.model_type)
    paths_new = ExperimentPaths(root=new_root, time_tag=time_tag, model_type=cfg.model_type)
    paths_new.make_dirs()

    vae_ckpt = copy_vae_checkpoint(cfg, paths_prev, paths_new)
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
        print("WARNING: CUDA not detected; downstream training will run on CPU.")

    set_seed(cfg.seed)
    data = load_data(metadata_path=cfg.metadata_path)
    sc_size = data["SC"]["train"].shape[-1]
    fc20_sets, fct_sets, mean_fc, std_fc = prepare_fc_sets(data, cfg.time_short, cfg.standardize)

    print(f"\nNew run root: {paths_new.root}")
    print(f"Using VAE checkpoint from previous run: {vae_ckpt}")
    print(f"Metadata source: {cfg.metadata_path}")
    print(f"Conditioning ablation: use_fct={cfg.use_fct}, use_sc={cfg.use_sc}, use_cov={cfg.use_cov}")
    if hasattr(cfg, "finetune"):
        print(f"LoRA run_name: {cfg.finetune.run_name}")

    print("\n[1/5] Re-exporting latents from existing VAE (no cached latents reused)...")
    export_latents_from_existing_vae(cfg, data, paths_new, device, sc_size, fc20_sets, fct_sets, mean_fc, std_fc)
    print(f"[1/5] Latents saved under: {paths_new.latents}")

    print("\n[2/5] Refitting ridge predictors (base, vae, corr)...")
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
    print(f"[2/5] Ridge checkpoints -> {ridge_ckpts}")

    print("\n[3/5] Training diffusion model from regenerated latents...")
    model, diff_ckpt, sc_shape = train_diffusion(cfg, data, paths_new, device, sc_size)
    print(f"[3/5] Diffusion saved -> {diff_ckpt}")

    print(f"\n[4/5] Fine-tuning diffusion with LoRA ({', '.join(selected_predictors)} predictors)...")
    for predictor_type, pred_path in ridge_ckpts.items():
        print(f"   - Fine-tune with predictor '{predictor_type}'")
        finetune_diffusion(
            cfg,
            data,
            paths_new,
            device,
            sc_size,
            predictor_path=pred_path,
            predictor_type=predictor_type,
            fc20_sets=fc20_sets,
        )

    print("\n[5/5] Sampling raw + fine-tuned models on val/test...")
    for split in ["val", "test"]:
        out_path = sample_diffusion(cfg, model, data, paths_new, device, sc_shape, split=split)
        print(f"   - Raw samples [{split}] -> {out_path}")
    for predictor_type in ridge_ckpts.keys():
        for split in ["val", "test"]:
            out_path = sample_finetuned(
                cfg, data, paths_new, device, sc_size, predictor_type=predictor_type, split=split
            )
            print(f"   - Finetuned '{predictor_type}' [{split}] -> {out_path}")

    print("\nDone. All artifacts saved under:", paths_new.root)


if __name__ == "__main__":
    main()
