#!/usr/bin/env python3
"""Check that clean/ images match processed/ images inside the field of view.

For every frame in NIGHT/processed/*.png, the file with the same name in
NIGHT/clean/ is loaded and both are compared inside the circular field of
view (radius = 0.44 * min(height, width), centered). The text annotations in
processed/ are drawn in the corners, outside this circle, so the two images
must differ there by at most --tol gray levels (default 1).

Frames that fail (difference above tolerance, missing clean file, or size
mismatch) are listed. Exit code is 1 if any frame fails, 0 otherwise.

Usage:
    python util/check_clean.py                 # every util/*/ night with processed/
    python util/check_clean.py util/CA_20251022 util/BJL_20250722
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FOV_RADIUS_FRAC = 0.44


def fov_mask(shape: tuple[int, int], radius_frac: float = FOV_RADIUS_FRAC) -> np.ndarray:
    """Boolean mask of the centered circular field of view."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2)
    return r <= radius_frac * min(h, w)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.int16)


def compare_frame(clean_path: Path, proc_path: Path, tol: int = 1,
                  radius_frac: float = FOV_RADIUS_FRAC) -> str | None:
    """Return None if the frame passes, otherwise a short failure reason."""
    if not clean_path.is_file():
        return "missing clean file"
    try:
        clean = load_gray(clean_path)
        proc = load_gray(proc_path)
    except Exception as exc:  # unreadable or corrupt PNG
        return f"read error: {exc}"
    if clean.shape != proc.shape:
        return f"size mismatch: clean {clean.shape} vs processed {proc.shape}"
    mask = fov_mask(clean.shape, radius_frac)
    diff = np.abs(clean - proc)[mask]
    max_diff = int(diff.max()) if diff.size else 0
    if max_diff > tol:
        n_bad = int((diff > tol).sum())
        return f"max diff {max_diff} gray levels ({n_bad} px > {tol})"
    return None


def check_night(night_dir: Path, tol: int = 1,
                radius_frac: float = FOV_RADIUS_FRAC) -> tuple[int, list[tuple[Path, str]]]:
    """Compare all frames of one night. Returns (n_checked, failures)."""
    proc_dir = night_dir / "processed"
    clean_dir = night_dir / "clean"
    failures: list[tuple[Path, str]] = []
    frames = sorted(proc_dir.glob("*.png"))
    for proc_path in frames:
        reason = compare_frame(clean_dir / proc_path.name, proc_path, tol, radius_frac)
        if reason is not None:
            failures.append((proc_path, reason))
    return len(frames), failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nights", nargs="*", type=Path,
                        help="night folders (default: every util/*/ with processed/)")
    parser.add_argument("--tol", type=int, default=1,
                        help="maximum allowed gray-level difference (default 1)")
    parser.add_argument("--radius", type=float, default=FOV_RADIUS_FRAC,
                        help="field-of-view radius as fraction of min(h, w) (default 0.44)")
    args = parser.parse_args(argv)

    nights = args.nights
    if not nights:
        util_dir = Path(__file__).resolve().parent
        nights = sorted(p for p in util_dir.iterdir() if (p / "processed").is_dir())
    if not nights:
        print("No night folders with processed/ found.", file=sys.stderr)
        return 1

    all_failures: list[tuple[Path, str]] = []
    total = 0
    for night in nights:
        if not (night / "processed").is_dir():
            print(f"{night.name}: no processed/ folder", file=sys.stderr)
            all_failures.append((night, "no processed/ folder"))
            continue
        n, failures = check_night(night, args.tol, args.radius)
        total += n
        all_failures.extend(failures)
        status = "OK" if not failures else f"{len(failures)} FAILED"
        print(f"{night.name}: {n} frames checked, {status}")

    if all_failures:
        print(f"\n{len(all_failures)} frame(s) failed:")
        for path, reason in all_failures:
            print(f"  {path}: {reason}")
        return 1
    print(f"\nAll {total} frames match within {args.tol} gray level(s) inside the FOV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
