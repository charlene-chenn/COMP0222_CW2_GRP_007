#!/usr/bin/env python3
"""
sanity_check.py
---------------
Validates the CSV files produced by extract_bag_to_csv.py.

Checks:
  1. File exists and is non-empty
  2. Row / column counts
  3. Null / NaN counts
  4. Range validity (within [range_min, range_max])
  5. Timestamp monotonicity
  6. Basic statistics (min, max, mean of ranges)
"""

import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def check_file_basics(df: pd.DataFrame, path: Path):
    """Check row/column counts and print summary."""
    print(f"\n{'═' * 60}")
    print(f"  File: {path.name}")
    print(f"{'═' * 60}")
    rows, cols = df.shape
    print(f"  Rows:    {rows}")
    print(f"  Columns: {cols}")

    if rows == 0:
        print(f"  {FAIL} — File contains no data rows")
        return False
    print(f"  {PASS} — File is non-empty")
    return True


def check_nulls(df: pd.DataFrame):
    """Check for null / NaN values."""
    total_nulls = df.isnull().sum().sum()
    if total_nulls == 0:
        print(f"  {PASS} — No null/NaN values")
    else:
        null_cols = df.isnull().sum()
        bad = null_cols[null_cols > 0]
        print(f"  {WARN} — {total_nulls} null/NaN value(s) in {len(bad)} column(s):")
        for col, count in bad.items():
            print(f"         {col}: {count} nulls")
    return total_nulls == 0


def check_range_validity(df: pd.DataFrame):
    """Check that range values fall within [range_min, range_max].
    
    LiDAR convention: 0.0 means 'no return' (beam didn't hit anything).
    These are excluded from the validity check.
    """
    range_cols = [c for c in df.columns if c.startswith("range_") and c != "range_min" and c != "range_max"]

    if not range_cols:
        print(f"  {WARN} — No range columns found, skipping range validity check")
        return True

    if "range_min" not in df.columns or "range_max" not in df.columns:
        print(f"  {WARN} — range_min / range_max columns missing, skipping validity check")
        return True

    r_min = df["range_min"].iloc[0]
    r_max = df["range_max"].iloc[0]
    print(f"  Expected range: [{r_min:.4f}, {r_max:.4f}]")

    all_ranges = df[range_cols].values.flatten()
    total = len(all_ranges)

    # 0.0 = no return (LiDAR convention)
    no_return = (all_ranges == 0.0).sum()
    no_return_pct = 100.0 * no_return / total
    print(f"  No-return beams (0.0): {no_return}/{total} ({no_return_pct:.1f}%)")

    # Only check values that are actual returns (> 0 and finite)
    valid = all_ranges[(all_ranges > 0) & np.isfinite(all_ranges)]

    if len(valid) == 0:
        print(f"  {WARN} — No valid range returns found")
        return False

    out_of_range = ((valid < r_min) | (valid > r_max)).sum()
    valid_total = len(valid)
    pct = 100.0 * out_of_range / valid_total

    if out_of_range == 0:
        print(f"  {PASS} — All {valid_total} valid returns within [{r_min:.4f}, {r_max:.4f}]")
    else:
        print(f"  {WARN} — {out_of_range}/{valid_total} ({pct:.1f}%) valid returns outside bounds")

    return out_of_range == 0


def check_timestamps(df: pd.DataFrame):
    """Check that timestamps are monotonically non-decreasing."""
    if "timestamp" not in df.columns:
        if "bag_time_ns" in df.columns:
            ts = df["bag_time_ns"]
        else:
            print(f"  {WARN} — No timestamp column found")
            return True
    else:
        ts = df["timestamp"]

    if len(ts) < 2:
        print(f"  {PASS} — Only 1 row, timestamp monotonicity trivially satisfied")
        return True

    diffs = ts.diff().dropna()
    non_monotone = (diffs < 0).sum()

    if non_monotone == 0:
        dt = ts.iloc[-1] - ts.iloc[0]
        print(f"  {PASS} — Timestamps monotonically increasing (span: {dt:.3f}s)")
    else:
        print(f"  {FAIL} — {non_monotone} timestamp(s) are out-of-order")

    return non_monotone == 0


def check_statistics(df: pd.DataFrame):
    """Print basic statistics of range values (excluding 0.0 no-return beams)."""
    range_cols = [c for c in df.columns if c.startswith("range_") and c != "range_min" and c != "range_max"]

    if not range_cols:
        return

    all_ranges = df[range_cols].values.flatten()
    # Only include actual returns (> 0 and finite)
    valid = all_ranges[(all_ranges > 0) & np.isfinite(all_ranges)]

    if len(valid) == 0:
        print(f"  Statistics: no valid range returns")
        return

    print(f"\n  📊 Range Statistics ({len(valid)} valid returns, excluding no-return beams):")
    print(f"     Min:    {np.min(valid):.4f}")
    print(f"     Max:    {np.max(valid):.4f}")
    print(f"     Mean:   {np.mean(valid):.4f}")
    print(f"     Median: {np.median(valid):.4f}")
    print(f"     Std:    {np.std(valid):.4f}")
    inf_count = np.isinf(all_ranges).sum()
    nan_count = np.isnan(all_ranges).sum()
    if inf_count or nan_count:
        print(f"     Inf:    {inf_count}   NaN: {nan_count}")


def main():
    csv_files = sorted(DATA_DIR.glob("*_data.csv"))

    if not csv_files:
        print("⚠  No *_data.csv files found.")
        print("   Run extract_bag_to_csv.py first.")
        sys.exit(0)

    print(f"Found {len(csv_files)} CSV file(s) to check.\n")

    all_pass = True

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"\n{FAIL} — Could not read {csv_path.name}: {e}")
            all_pass = False
            continue

        ok = check_file_basics(df, csv_path)
        if not ok:
            all_pass = False
            continue

        all_pass &= check_nulls(df)
        all_pass &= check_range_validity(df)
        all_pass &= check_timestamps(df)
        check_statistics(df)

    print(f"\n{'═' * 60}")
    if all_pass:
        print(f"  Overall: {PASS} — All checks passed")
    else:
        print(f"  Overall: {WARN} — Some checks had warnings (see above)")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
