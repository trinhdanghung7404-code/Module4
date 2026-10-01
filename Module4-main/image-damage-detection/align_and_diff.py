import cv2
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--product', required=True, help='Path to product image')
    parser.add_argument('--return', dest='return_img', required=True, help='Path to return image')
    parser.add_argument('--threshold', type=int, default=5)
    parser.add_argument('--output', default='aligned_outputs')
    args = parser.parse_args()

    # Load grayscale
    img1 = cv2.imread(args.product, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.return_img, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one or both images")

    # Estimate translation via phase correlation
    shift, response = cv2.phaseCorrelate(img1.astype(np.float32), img2.astype(np.float32))
    dx, dy = shift
    print(f'[+] Estimated shift: dx={dx:.3f}, dy={dy:.3f}')

    # Build affine transform (translation only)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Compute diff
    diff = cv2.absdiff(img1, aligned)
    mask = (diff >= args.threshold).astype(np.uint8) * 255

    # Stats
    total_pixels = img1.size
    damaged_pixels = cv2.countNonZero(mask)
    score = (damaged_pixels / total_pixels) * 100

    print(f'\n📊 ALIGNED RESULTS:')
    print(f'   • Total pixels: {total_pixels:,}')
    print(f'   • Damaged pixels (threshold {args.threshold}): {damaged_pixels:,}')
    print(f'   • Damage score: {score:.3f}%')

    # Save
    os.makedirs(args.output, exist_ok=True)
    cv2.imwrite(os.path.join(args.output, 'aligned_2.jpg'), aligned)
    cv2.imwrite(os.path.join(args.output, 'aligned_diff.png'), diff)
    cv2.imwrite(os.path.join(args.output, 'aligned_mask.png'), mask)
    print(f'\n📁 Aligned outputs saved to: {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()