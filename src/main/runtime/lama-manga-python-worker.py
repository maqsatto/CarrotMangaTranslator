#!/usr/bin/env python3
"""Anime/Manga LaMa inpainting worker for CarrotMangaTranslator (Linux).

The parent sends JSON Lines requests with page-sized input/mask/output paths
and the already-computed text windows. This worker never sends a full webtoon
page to the model: it crops a bounded local square around each window, keeps
surrounding context, and composites only the requested mask pixels back into
the page. The model is loaded once for the lifetime of the worker; the parent
disposes the worker after the inpainting stage, which also releases CUDA.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def resolve_crop_rect(
    window: tuple[int, int, int, int],
    page_width: int,
    page_height: int,
    crop_size: int,
    context: int,
) -> tuple[int, int, int, int]:
    """Return a bounded local crop, never the full page for a small window."""
    x0, y0, x1, y1 = window
    x0 = clamp(x0, 0, page_width)
    x1 = clamp(x1, x0, page_width)
    y0 = clamp(y0, 0, page_height)
    y1 = clamp(y1, y0, page_height)
    content_width = x1 - x0
    content_height = y1 - y0
    side = max(
        crop_size,
        content_width + 2 * context,
        content_height + 2 * context,
    )
    side = max(1, side)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    left = clamp(center_x - side // 2, 0, max(0, page_width - side))
    top = clamp(center_y - side // 2, 0, max(0, page_height - side))
    right = min(page_width, left + side)
    bottom = min(page_height, top + side)
    return left, top, right, bottom


def decode_windows(
    raw: object,
    mask_array: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                continue
            try:
                x0, y0, x1, y1 = (int(value) for value in item)
            except (TypeError, ValueError):
                continue
            if x1 > x0 and y1 > y0:
                windows.append((x0, y0, x1, y1))
    if windows:
        return windows

    # Defensive fallback: use the mask's bounding box, not the full page.
    ys, xs = np.nonzero(mask_array > 0)
    if xs.size == 0:
        return []
    return [
        (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
    ]


def inpaint_request(
    simple_lama,
    request: dict,
    crop_size: int,
    context: int,
) -> int:
    input_path = Path(request["input"])
    mask_path = Path(request["mask"])
    output_path = Path(request["output"])

    original = Image.open(input_path).convert("RGBA")
    page_width, page_height = original.size
    rgb_page = original.convert("RGB")
    mask_page = Image.open(mask_path).convert("L")
    if mask_page.size != (page_width, page_height):
        raise ValueError(
            f"mask size {mask_page.size} does not match input {original.size}"
        )
    mask_array = np.asarray(mask_page, dtype=np.uint8)
    windows = decode_windows(request.get("windows"), mask_array)
    if not windows:
        raise ValueError("request has no text windows or mask pixels")

    output_array = np.asarray(rgb_page, dtype=np.uint8).copy()
    covered = np.zeros_like(mask_array, dtype=bool)
    processed = 0
    # Fixed-size inference with context around each core. Large windows are
    # tiled at original resolution rather than shrinking the whole page.
    core_size = max(1, crop_size - 2 * context)
    for x0, y0, x1, y1 in windows:
        x0, x1 = clamp(x0, 0, page_width), clamp(x1, 0, page_width)
        y0, y1 = clamp(y0, 0, page_height), clamp(y1, 0, page_height)
        for core_top in range(y0, y1, core_size):
            for core_left in range(x0, x1, core_size):
                core_right = min(x1, core_left + core_size)
                core_bottom = min(y1, core_top + core_size)
                left, top, right, bottom = resolve_crop_rect(
                    (core_left, core_top, core_right, core_bottom),
                    page_width, page_height, crop_size, context,
                )
                local_mask_array = mask_array[top:bottom, left:right]
                keep = np.zeros_like(local_mask_array, dtype=bool)
                keep[core_top-top:core_bottom-top,
                     core_left-left:core_right-left] = True
                keep &= (local_mask_array > 0) & ~covered[top:bottom, left:right]
                if not np.any(keep):
                    continue
                crop = rgb_page.crop((left, top, right, bottom))
                local_mask = Image.fromarray(
                    np.where(local_mask_array > 0, 255, 0).astype(np.uint8),
                )
                raw_result = simple_lama(crop, local_mask)
                if raw_result.size != crop.size:
                    raise ValueError("LaMa changed crop dimensions")
                result_array = np.asarray(raw_result.convert("RGB"), dtype=np.uint8)
                output_array[top:bottom, left:right][keep] = result_array[keep]
                covered[top:bottom, left:right][keep] = True
                processed += 1

    output_page = Image.fromarray(output_array)
    output_rgba = output_page.convert("RGBA")
    output_rgba.putalpha(original.getchannel("A"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rgba.save(output_path)
    return processed


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to the LaMa worker")
    if not args.model.is_file():
        raise SystemExit(f"LaMa model not found: {args.model}")

    # simple-lama uses LAMA_MODEL to avoid its default download path.
    os.environ["LAMA_MODEL"] = str(args.model)
    from simple_lama_inpainting import SimpleLama

    device = torch.device(args.device)
    load_start = time.perf_counter()
    simple_lama = SimpleLama(device=device)
    simple_lama.model.eval().float()
    load_seconds = time.perf_counter() - load_start
    print(
        json.dumps(
            {
                "type": "ready",
                "device": str(device),
                "load_ms": round(load_seconds * 1000, 1),
            }
        ),
        file=sys.stderr,
        flush=True,
    )

    if args.warmup > 0:
        warm_image = Image.new("RGB", (args.crop_size, args.crop_size), (255, 255, 255))
        warm_mask = Image.new("L", (args.crop_size, args.crop_size), 0)
        warm_mask.paste(255, (32, 32, args.crop_size - 32, args.crop_size - 32))
        simple_lama(warm_image, warm_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    json.dumps({"id": "", "ok": False, "error": f"invalid json: {exc}"}),
                    flush=True,
                )
                continue
            if request.get("type") == "shutdown":
                break
            if request.get("type") != "inpaint":
                print(
                    json.dumps(
                        {
                            "id": request.get("id", ""),
                            "ok": False,
                            "error": f"unsupported request type: {request.get('type')}",
                        }
                    ),
                    flush=True,
                )
                continue
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            try:
                processed = inpaint_request(
                    simple_lama,
                    request,
                    args.crop_size,
                    args.context,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                print(
                    json.dumps(
                        {
                            "id": request.get("id", ""),
                            "ok": True,
                            "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1048576, 1) if device.type == "cuda" else 0,
                            "elapsed_ms": round(
                                (time.perf_counter() - started) * 1000, 1
                            ),
                            "windows": processed,
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - worker protocol reports errors
                print(
                    json.dumps(
                        {
                            "id": request.get("id", ""),
                            "ok": False,
                            "error": str(exc),
                            "elapsed_ms": round(
                                (time.perf_counter() - started) * 1000, 1
                            ),
                        }
                    ),
                    flush=True,
                )
    finally:
        del simple_lama
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
