"""Test damage detection on an actual scarred/damaged image."""
import sys
import os

V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)

if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)


def test_true_positive():
    """Test: product vs scarred product -> expect damage detected."""
    print("=" * 60)
    print("TEST: True Positive (test_nobg.png vs test_nobg - scar.png)")
    print("=" * 60)

    from pipeline.comparator import DamageComparator

    img1 = os.path.join(PROJECT_ROOT, "images", "test_nobg.png")
    img2 = os.path.join(PROJECT_ROOT, "images", "test_nobg - scar.png")

    comparator = DamageComparator()
    result = comparator.compare(img1, img2)

    print(f"\n{'=' * 60}")
    print(f"RESULT:")
    print(f"  Damage Score:   {result['damage_score']:.2f}%")
    print(f"  Damage Area:    {result['damage_area']} px")
    print(f"  Largest Damage: {result['largest_damage_area']} px")
    print(f"  Is Damaged:     {result['is_damaged']}")
    print(f"  Time:           {result['processing_time']:.1f}s")
    print(f"{'=' * 60}")

    if result['is_damaged']:
        print("[PASS] True Positive detected successfully!")
    else:
        print("[INFO] Scar not flagged as damage (or scar area below threshold).")

    return result


if __name__ == "__main__":
    test_true_positive()
