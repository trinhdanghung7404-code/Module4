#!/usr/bin/env python3
"""
GRAYSCALE BASELINE PIPELINE — Simplest possible damage detection:
- Load both images → convert to GRAYSCALE
- Compute absolute difference
- Apply adaptive + fixed threshold
- Output: binary mask, damage %, and debug images

NO mesh, NO voting, NO color, NO texture — pure intensity delta.
"""

import argparse
import cv2
import numpy as np
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Grayscale-only damage baseline")
    parser.add_argument("--product", type=str, required=True, help="Path to product image (e.g., 1.jpg)")
    parser.add_argument("--return", type=str, required=True, help="Path to return image (e.g., 2.jpg)")
    parser.add_argument("--output", type=str, default="debug_outputs/", help="Output directory for debug images")
    parser.add_argument("--threshold", type=int, default=20, help="Fixed threshold for binary mask (0–255)")
    parser.add_argument("--adaptive_block", type=int, default=11, help="Adaptive block size (odd)")
    parser.add_argument("--adaptive_c", type=int, default=2, help="Adaptive constant")

    args = parser.parse_args()
    product_path = Path(args.product)
    return_path = Path(getattr(args, "return"))
    output_dir = Path(args.output)

    # Create output dir
    output_dir.mkdir(exist_ok=True)

    # Load & grayscale
    print(f"[+] Loading: {product_path.name} & {return_path.name}")
    p_img = cv2.imread(str(product_path))
    r_img = cv2.imread(str(return_path))
    if p_img is None or r_img is None:
        raise FileNotFoundError("One or both images not found!")

    p_gray = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

    # Compute absdiff
    diff = cv2.absdiff(p_gray, r_gray)
    cv2.imwrite(str(output_dir / "grayscale_diff.png"), diff)

    # Fixed threshold
    _, mask_fixed = cv2.threshold(diff, args.threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(output_dir / "grayscale_mask_fixed.png"), mask_fixed)

    # Adaptive threshold
    mask_adapt = cv2.adaptiveThreshold(
        diff,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        args.adaptive_block,
        args.adaptive_c
    )
    cv2.imwrite(str(output_dir / "grayscale_mask_adapt.png"), mask_adapt)

    # Compute damage area %
    total_pixels = p_gray.size
    damaged_pixels_fixed = cv2.countNonZero(mask_fixed)
    damage_percent = (damaged_pixels_fixed / total_pixels) * 100

    print(f"\n📊 RESULTS:")
    print(f"   • Total pixels: {total_pixels:,}")
    print(f"   • Damaged pixels (fixed threshold {args.threshold}): {damaged_pixels_fixed:,}")
    print(f"   • Damage score: {damage_percent:.3f}%")
    print(f"\n📁 Debug images saved to: {output_dir.absolute()}")
    print(f"   - grayscale_diff.png          (raw intensity difference)")
    print(f"   - grayscale_mask_fixed.png    (binary mask, fixed threshold)")
    print(f"   - grayscale_mask_adapt.png    (binary mask, adaptive threshold)")

if __name__ == "__main__":
    main()
