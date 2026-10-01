"""Quick end-to-end test for damage detection v2."""
import sys
import os

# Ensure v2 directory is in path
V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)

if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)


def test_false_positive():
    """Test: same vase, no damage -> expect < 1% damage score."""
    print("=" * 60)
    print("TEST: False Positive (1.jpg vs 2.jpg - same vase, no damage)")
    print("=" * 60)

    from pipeline.comparator import DamageComparator

    img1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    img2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")

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

    if result['damage_score'] < 1.0:
        print("[PASS] Damage score < 1% (no false positive)")
    elif result['damage_score'] < 5.0:
        print("[OK] Damage score < 5% (minor noise)")
    else:
        print("[FAIL] Damage score >= 5% (false positive!)")

    return result


if __name__ == "__main__":
    test_false_positive()
