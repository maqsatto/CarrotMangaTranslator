#!/usr/bin/env python3
"""Standalone Anime/Manga LaMa CUDA smoke test.

This script deliberately does not import Electron or the application runtime.
It proves that one 512x512 RGB crop and a same-size binary mask can be
inpainted with the TorchScript model on CUDA before the backend is wired into
CarrotMangaTranslator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_rect(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("rect width/height must be positive")
    return x, y, w, h


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--crop", required=True, type=parse_rect)
    parser.add_argument("--mask", type=parse_rect)
    parser.add_argument(
        "--mask-file",
        type=Path,
        help="optional 1-channel same-size mask; overrides --mask",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            os.environ.get(
                "LAMA_MODEL",
                "inpaint-runtime/models/anime-manga-big-lama.pt",
            )
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inpaint-runtime/out"),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to this Python environment")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")

    size = args.size
    crop_x, crop_y, crop_w, crop_h = args.crop
    if crop_w != size or crop_h != size:
        raise SystemExit(
            f"crop must be exactly {size}x{size}, got {crop_w}x{crop_h}"
        )
    if args.mask_file is None and args.mask is None:
        raise SystemExit("provide --mask or --mask-file")
    if args.mask is not None:
        mask_x, mask_y, mask_w, mask_h = args.mask
        if mask_x < 0 or mask_y < 0:
            raise SystemExit("mask must start inside the crop")
        if mask_x + mask_w > size or mask_y + mask_h > size:
            raise SystemExit("mask must fit inside the crop")

    source = Image.open(args.source).convert("RGB")
    if crop_x < 0 or crop_y < 0:
        raise SystemExit("crop must start inside the source")
    if crop_x + size > source.width or crop_y + size > source.height:
        raise SystemExit("crop must fit inside the source")

    crop = source.crop((crop_x, crop_y, crop_x + size, crop_y + size))
    if args.mask_file is not None:
        mask = Image.open(args.mask_file).convert("L")
        if mask.size != (size, size):
            raise SystemExit(
                f"mask file must be {size}x{size}, got {mask.size}"
            )
        mask_array = np.asarray(mask, dtype=np.uint8)
        mask = Image.fromarray(mask_array, mode="L")
    else:
        mask_array = np.zeros((size, size), dtype=np.uint8)
        mask_array[mask_y : mask_y + mask_h, mask_x : mask_x + mask_w] = 255
        mask = Image.fromarray(mask_array, mode="L")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"lama-{size}"
    crop_path = out_dir / f"{stem}-crop.png"
    mask_path = out_dir / f"{stem}-mask.png"
    raw_path = out_dir / f"{stem}-raw.png"
    composite_path = out_dir / f"{stem}-composite.png"
    metrics_path = out_dir / f"{stem}-metrics.json"
    crop.save(crop_path)
    mask.save(mask_path)

    os.environ["LAMA_MODEL"] = str(args.model.resolve())
    if not np.isin(mask_array, [0, 255]).all():
        raise SystemExit("mask must be binary: 0 or 255")

    # Import after the CLI validation so a missing optional package produces a
    # clear model/package error rather than an import traceback first.
    from simple_lama_inpainting import SimpleLama

    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    load_start = time.perf_counter()
    simple_lama = SimpleLama(device=device)
    load_seconds = time.perf_counter() - load_start

    for _ in range(max(0, args.warmup)):
        simple_lama(crop, mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    durations: list[float] = []
    result: Image.Image | None = None
    for _ in range(max(1, args.runs)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = simple_lama(crop, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)

    if result is None:
        raise SystemExit("no inference result")

    result = result.convert("RGB")
    if result.size != (size, size):
        result = result.resize((size, size), Image.Resampling.LANCZOS)

    # The standalone check also demonstrates the production compositing rule:
    # only masked pixels come from the model; every other pixel stays original.
    original_array = np.asarray(crop, dtype=np.uint8)
    result_array = np.asarray(result, dtype=np.uint8)
    mask_bool = mask_array > 0
    composite = original_array.copy()
    composite[mask_bool] = result_array[mask_bool]

    result.save(raw_path)
    Image.fromarray(composite, mode="RGB").save(composite_path)

    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    metrics = {
        "model": str(args.model),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "crop": [crop_x, crop_y, size, size],
        "mask": str(args.mask_file)
        if args.mask_file is not None
        else [mask_x, mask_y, mask_w, mask_h],
        "dtype": "float32",
        "batch_size": 1,
        "model_load_seconds": round(load_seconds, 4),
        "warmup_runs": max(0, args.warmup),
        "runs": len(durations),
        "run_seconds": [round(value, 4) for value in durations],
        "mean_seconds": round(sum(durations) / len(durations), 4),
        "min_seconds": round(min(durations), 4),
        "max_seconds": round(max(durations), 4),
        "peak_vram_allocated_mb": round(peak_allocated, 1),
        "peak_vram_reserved_mb": round(peak_reserved, 1),
        "outputs": {
            "crop": str(crop_path),
            "mask": str(mask_path),
            "raw": str(raw_path),
            "composite": str(composite_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
