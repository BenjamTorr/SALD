"""Resample an existing run in-place using stored checkpoints.

This script does NOT create a new run directory. It reuses:
- diffusion checkpoint under the provided run root (for raw samples)
- finetuned checkpoint(s) under the same run root (for finetuned samples)

and overwrites sample outputs in:
- generated/<time>/<model>/
- finetuned_data/<time>/<model>/
"""

import argparse
from pathlib import Path

import torch

from data.load_data import load_data
from rerun_finetune_and_sampling import load_previous_config
from run_full_experiment import (
    ExperimentPaths,
    build_diffusion_model,
    sample_diffusion,
    sample_finetuned,
    set_seed,
)


def _resolve_predictors(paths: ExperimentPaths, cfg, mode: str):
    all_candidates = ["base", "vae"]
    existing = []
    missing = []

    for ptype in all_candidates:
        ckpt = paths.finetuned_models / (
            f"{cfg.finetune.run_name}_finetuned_{cfg.model_type}_{ptype}_{paths.time_tag}.pt"
        )
        if ckpt.exists():
            existing.append(ptype)
        else:
            missing.append(ptype)

    if mode == "auto":
        requested = existing
    elif mode == "all":
        requested = all_candidates
    elif mode in {"base", "vae"}:
        requested = [mode]
    else:
        raise ValueError(f"Unsupported predictor mode: {mode}")

    selected = [ptype for ptype in requested if ptype in existing]
    skipped = [ptype for ptype in requested if ptype not in existing]
    return selected, skipped, existing, missing


def main():
    parser = argparse.ArgumentParser(
        description="Resample raw + finetuned outputs for an existing run root (in-place)."
    )
    parser.add_argument("--run-root", required=True, help="Path to existing run root.")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda:0")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Optional physical GPU index to force single-GPU execution (overrides --device).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
        help="Splits to sample.",
    )
    parser.add_argument(
        "--predictors",
        default="auto",
        choices=["auto", "base", "vae", "all"],
        help="Which finetuned predictor branch(es) to sample.",
    )
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw diffusion sampling.")
    parser.add_argument("--skip-finetuned", action="store_true", help="Skip finetuned sampling.")

    # Optional sampling overrides
    parser.add_argument("--precision", default=None, choices=["bf16", "fp16"], help="Override cfg.sampling.precision.")
    parser.add_argument("--denoising-steps", type=int, default=None, help="Override cfg.sampling.denoising_steps.")
    parser.add_argument("--eta", type=float, default=None, help="Override cfg.sampling.eta.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override cfg.sampling.n_samples_per_subject.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override cfg.sampling.chunk_size.")
    parser.add_argument("--decode-batch-size", type=int, default=None, help="Override cfg.sampling.decode_batch_size.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override cfg.sampling.batch_size.")
    parser.add_argument("--slice-index", type=int, default=None, help="Override cfg.sampling.slice_index (use 0 for first slice).")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    cfg = load_previous_config(run_root)

    if args.gpu_id is not None:
        cfg.device = f"cuda:{args.gpu_id}"
    elif args.device is not None:
        cfg.device = args.device

    # Backfill precision if missing in older snapshots
    if not hasattr(cfg.sampling, "precision"):
        cfg.sampling.precision = "bf16"

    # Apply sampling overrides
    if args.precision is not None:
        cfg.sampling.precision = args.precision
    if args.denoising_steps is not None:
        cfg.sampling.denoising_steps = args.denoising_steps
    if args.eta is not None:
        cfg.sampling.eta = args.eta
    if args.n_samples is not None:
        cfg.sampling.n_samples_per_subject = args.n_samples
    if args.chunk_size is not None:
        cfg.sampling.chunk_size = args.chunk_size
    if args.decode_batch_size is not None:
        cfg.sampling.decode_batch_size = args.decode_batch_size
    if args.batch_size is not None:
        cfg.sampling.batch_size = args.batch_size
    if args.slice_index is not None:
        cfg.sampling.slice_index = args.slice_index

    time_tag = getattr(cfg, "time_tag", f"{cfg.time_short}min")
    paths = ExperimentPaths(root=run_root, time_tag=time_tag, model_type=cfg.model_type)
    paths.make_dirs()

    device = torch.device(cfg.device)
    if not torch.cuda.is_available():
        print("⚠️ CUDA not detected; sampling will run on CPU.")
    elif device.type == "cuda":
        torch.cuda.set_device(device)

    set_seed(cfg.seed)
    data = load_data(metadata_path="../config/metadata.yaml")
    sc_size = data["SC"]["train"].shape[-1]
    sc_shape = (-1, 1, sc_size, sc_size)

    print("\n=== In-place Resampling ===")
    print("run_root              :", paths.root)
    print("model_type            :", cfg.model_type)
    print("time_tag              :", time_tag)
    print("scale_tag             :", cfg.scale_tag)
    print("device                :", device)
    print("gpu_id arg            :", args.gpu_id)
    print("splits                :", args.splits)
    print("sampling.precision    :", cfg.sampling.precision)
    print("sampling.denoising    :", cfg.sampling.denoising_steps)
    print("sampling.eta          :", cfg.sampling.eta)
    print("sampling.n_samples    :", cfg.sampling.n_samples_per_subject)
    print("sampling.chunk_size   :", cfg.sampling.chunk_size)
    print("sampling.decode_batch :", cfg.sampling.decode_batch_size)
    print("sampling.batch_size   :", cfg.sampling.batch_size)
    print("sampling.slice_index  :", getattr(cfg.sampling, "slice_index", 0))

    if not args.skip_raw:
        print("\n[1/2] Resampling raw diffusion outputs...")
        with open("../config/diffusion_config.yaml", "r") as f:
            import yaml

            diffusion_config = yaml.safe_load(f)
        model, ckpt_name = build_diffusion_model(cfg.model_type, diffusion_config, device)
        ckpt_path = paths.diffusion / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing diffusion checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        model.eval()

        for split in args.splits:
            out_path = sample_diffusion(cfg, model, data, paths, device, sc_shape, split=split)
            print(f"   - raw [{split}] -> {out_path}")
    else:
        print("\n[1/2] Skipped raw diffusion sampling.")

    if not args.skip_finetuned:
        predictor_types, skipped_requested, existing, missing = _resolve_predictors(
            paths, cfg, args.predictors
        )
        print(f"   - available finetuned checkpoints: {existing if existing else 'none'}")
        if skipped_requested:
            print(f"   - requested but missing: {skipped_requested}")
        elif missing:
            print(f"   - missing branches (not requested): {missing}")

        if not predictor_types:
            print("\n[2/2] No finetuned checkpoints found for requested predictor mode; nothing to sample.")
        else:
            print(f"\n[2/2] Resampling finetuned outputs for predictors: {predictor_types}")
            for ptype in predictor_types:
                for split in args.splits:
                    out_path = sample_finetuned(
                        cfg, data, paths, device, sc_size, predictor_type=ptype, split=split
                    )
                    print(f"   - finetuned '{ptype}' [{split}] -> {out_path}")
    else:
        print("\n[2/2] Skipped finetuned sampling.")

    print("\nDone. Existing sample files in this run were regenerated in-place.")


if __name__ == "__main__":
    main()
