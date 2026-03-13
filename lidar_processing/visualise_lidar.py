#!/usr/bin/env python3
"""
visualise_lidar.py
------------------
Generates three visualisations from the CSV produced by extract_bag_to_csv.py:

  1. Polar scatter   — single scan (range vs angle)
  2. Cartesian scatter — single scan as (x, y)
  3. Range heatmap   — all scans over time (scan idx × beam idx, colour = range)

Saves PNGs and shows interactively.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "figs"


def find_csv():
    """Find the first *_data.csv that has range columns (i.e. LaserScan data)."""
    for csv_path in sorted(DATA_DIR.glob("*_data.csv")):
        df = pd.read_csv(csv_path)
        range_cols = [c for c in df.columns if c.startswith("range_") and c not in ("range_min", "range_max")]
        if range_cols:
            return csv_path, df, range_cols
    return None, None, None


def get_angles_and_ranges(df: pd.DataFrame, range_cols: list, scan_idx: int = 0):
    """Extract angles and ranges for a given scan index."""
    row = df.iloc[scan_idx]

    angle_cols = [c for c in df.columns if c.startswith("angle_") and c != "angle_min" and c != "angle_max" and c != "angle_increment"]
    if angle_cols:
        angles = row[angle_cols].values.astype(float)
    else:
        # Reconstruct from angle_min / angle_increment
        a_min = row["angle_min"]
        a_inc = row["angle_increment"]
        n = len(range_cols)
        angles = np.array([a_min + i * a_inc for i in range(n)])

    ranges = row[range_cols].values.astype(float)
    return angles, ranges


def plot_polar(angles, ranges, scan_idx, out_path):
    """Polar scatter plot of a single scan."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

    # Mask invalid ranges
    valid = np.isfinite(ranges) & (ranges > 0)
    ax.scatter(angles[valid], ranges[valid], c=ranges[valid], cmap="viridis",
               s=3, alpha=0.8)

    ax.set_title(f"LiDAR Scan (Polar) — Scan #{scan_idx}", pad=20, fontsize=14)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓  Saved {out_path.name}")


def plot_cartesian(angles, ranges, scan_idx, out_path):
    """Cartesian scatter plot of a single scan."""
    valid = np.isfinite(ranges) & (ranges > 0)
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])

    fig, ax = plt.subplots(figsize=(9, 9))
    scatter = ax.scatter(x, y, c=ranges[valid], cmap="plasma", s=4, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(f"LiDAR Scan (Cartesian) — Scan #{scan_idx}", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark sensor origin
    ax.plot(0, 0, "r+", markersize=15, markeredgewidth=2, label="Sensor")
    ax.legend(fontsize=10)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Range (m)", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓  Saved {out_path.name}")


def plot_heatmap(df: pd.DataFrame, range_cols: list, out_path):
    """Heatmap of all scans: rows = scan index, cols = beam index, colour = range."""
    data = df[range_cols].values.astype(float)

    # Replace inf and 0.0 (no-return beams) with NaN for better colour mapping
    data[~np.isfinite(data)] = np.nan
    data[data == 0.0] = np.nan

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", cmap="inferno", interpolation="nearest")

    ax.set_xlabel("Beam Index", fontsize=12)
    ax.set_ylabel("Scan Index (time →)", fontsize=12)
    ax.set_title("LiDAR Range Heatmap (all scans)", fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Range (m)", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓  Saved {out_path.name}")


def main():
    csv_path, df, range_cols = find_csv()

    if csv_path is None:
        print("⚠  No CSV with range data found.")
        print("   Run extract_bag_to_csv.py first (needs a bag with LaserScan data).")
        sys.exit(0)

    print(f"Using: {csv_path.name}  ({len(df)} scans, {len(range_cols)} beams)\n")

    # Pick first and middle scan for the single-scan plots
    scan_idx = len(df) // 2  # middle scan for a representative view

    angles, ranges = get_angles_and_ranges(df, range_cols, scan_idx)

    # 1. Polar scatter
    plot_polar(angles, ranges, scan_idx, OUTPUT_DIR / "lidar_polar.png")

    # 2. Cartesian scatter
    plot_cartesian(angles, ranges, scan_idx, OUTPUT_DIR / "lidar_cartesian.png")

    # 3. Range heatmap
    plot_heatmap(df, range_cols, OUTPUT_DIR / "lidar_heatmap.png")

    print("\nDone! ✓")
    print("Showing plots interactively (close windows to exit)...")
    plt.show()


if __name__ == "__main__":
    main()
