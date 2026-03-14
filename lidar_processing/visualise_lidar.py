#!/usr/bin/env python3
"""
visualise_lidar.py
------------------
Data-verification visualisations for LiDAR bag data before SLAM.

Per dataset generates:
  1. All-scans overlay   — every scan stacked in Cartesian (x, y) to show
                           the aggregate environment shape
  2. Time-lapse grid     — 6 evenly-spaced scans across the recording
                           so you can see how the view changes over time
  3. Scan rate plot      — timestamp deltas to verify consistent FPS
  4. Range heatmap       — all scans over time (beam index vs scan index)

Saves PNGs to figs/.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "figs"


def get_angles(df: pd.DataFrame, range_cols: list, scan_idx: int = 0):
    """Get the angle array for a given scan."""
    row = df.iloc[scan_idx]
    angle_cols = [c for c in df.columns if c.startswith("angle_")
                  and c not in ("angle_min", "angle_max", "angle_increment")]
    if angle_cols:
        return row[angle_cols].values.astype(float)
    else:
        a_min = row["angle_min"]
        a_inc = row["angle_increment"]
        return np.array([a_min + i * a_inc for i in range(len(range_cols))])


# ──────────────────────────────────────────────────────────
# Plot 1: All-scans overlay (dense Cartesian point cloud)
# ──────────────────────────────────────────────────────────
def plot_all_scans_overlay(df, range_cols, out_path):
    """Overlay every scan in Cartesian coords to see the full environment."""
    angles = get_angles(df, range_cols, 0)  # angles are the same for all scans

    all_ranges = df[range_cols].values.astype(float)  # (N_scans, N_beams)
    # Broadcast angles to all scans
    angles_2d = np.broadcast_to(angles, all_ranges.shape)

    # Mask no-return and invalid
    valid = (all_ranges > 0) & np.isfinite(all_ranges)

    x = (all_ranges * np.cos(angles_2d))[valid]
    y = (all_ranges * np.sin(angles_2d))[valid]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y, s=0.3, alpha=0.15, c="steelblue", edgecolors="none")
    ax.plot(0, 0, "r+", markersize=20, markeredgewidth=3, label="Sensor origin",
            zorder=10)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(f"All Scans Overlay ({len(df)} scans, {valid.sum():,} points)",
                 fontsize=14)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved {out_path.name}")


# ──────────────────────────────────────────────────────────
# Plot 2: Time-lapse grid (6 scans spread across recording)
# ──────────────────────────────────────────────────────────
def plot_timelapse(df, range_cols, out_path):
    """Show 6 scans evenly spread across time to verify data changes."""
    n_scans = len(df)
    indices = np.linspace(0, n_scans - 1, 6, dtype=int)
    angles = get_angles(df, range_cols, 0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, scan_idx in enumerate(indices):
        ax = axes[i]
        row = df.iloc[scan_idx]
        ranges = row[range_cols].values.astype(float)

        valid = (ranges > 0) & np.isfinite(ranges)
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])

        # Timestamp for title
        if "timestamp" in df.columns:
            t0 = df["timestamp"].iloc[0]
            t = row["timestamp"] - t0
            time_label = f"t = {t:.1f}s"
        else:
            time_label = f"scan {scan_idx}"

        ax.scatter(x, y, s=2, alpha=0.7, c="steelblue", edgecolors="none")
        ax.plot(0, 0, "r+", markersize=12, markeredgewidth=2)
        ax.set_aspect("equal")
        ax.set_title(f"Scan #{scan_idx}  ({time_label})", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)

    fig.suptitle("Time-Lapse: 6 Scans Across Recording", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved {out_path.name}")


# ──────────────────────────────────────────────────────────
# Plot 3: Scan rate (FPS verification)
# ──────────────────────────────────────────────────────────
def plot_scan_rate(df, out_path):
    """Plot inter-scan time deltas to verify consistent scan rate."""
    if "timestamp" not in df.columns:
        print("  ⚠  No timestamp column — skipping scan rate plot")
        return

    ts = df["timestamp"].values
    deltas = np.diff(ts) * 1000  # convert to ms

    fps = 1000.0 / deltas  # instantaneous FPS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Delta plot
    ax1.plot(deltas, linewidth=0.5, color="steelblue")
    ax1.axhline(np.median(deltas), color="red", linestyle="--", linewidth=1,
                label=f"Median: {np.median(deltas):.1f} ms")
    ax1.set_ylabel("Inter-scan Δt (ms)", fontsize=12)
    ax1.set_title("Scan Rate Consistency", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # FPS plot
    ax2.plot(fps, linewidth=0.5, color="darkorange")
    ax2.axhline(np.median(fps), color="red", linestyle="--", linewidth=1,
                label=f"Median: {np.median(fps):.1f} FPS")
    ax2.set_xlabel("Scan Index", fontsize=12)
    ax2.set_ylabel("Instantaneous FPS", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Summary stats in text box
    duration = ts[-1] - ts[0]
    stats = (f"Duration: {duration:.1f}s  |  "
             f"Scans: {len(ts)}  |  "
             f"Avg FPS: {len(ts)/duration:.1f}  |  "
             f"Δt range: [{deltas.min():.1f}, {deltas.max():.1f}] ms")
    fig.text(0.5, -0.02, stats, ha="center", fontsize=10,
             style="italic", color="gray")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved {out_path.name}")


# ──────────────────────────────────────────────────────────
# Plot 4: Range heatmap
# ──────────────────────────────────────────────────────────
def plot_heatmap(df, range_cols, out_path):
    """Heatmap of all scans: rows = scan index, cols = beam index."""
    data = df[range_cols].values.astype(float)

    # Mask no-return (0.0) and inf
    data[~np.isfinite(data)] = np.nan
    data[data == 0.0] = np.nan

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", cmap="inferno", interpolation="nearest")

    ax.set_xlabel("Beam Index", fontsize=12)
    ax.set_ylabel("Scan Index (time →)", fontsize=12)
    ax.set_title("Range Heatmap (all scans)", fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Range (m)", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved {out_path.name}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def find_all_csvs():
    """Find all *_data.csv files that have range columns."""
    results = []
    for csv_path in sorted(DATA_DIR.glob("*_data.csv")):
        df = pd.read_csv(csv_path)
        range_cols = [c for c in df.columns
                      if c.startswith("range_") and c not in ("range_min", "range_max")]
        if range_cols:
            results.append((csv_path, df, range_cols))
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = find_all_csvs()
    if not datasets:
        print("⚠  No CSV with range data found.")
        print("   Run extract_bag_to_csv.py first.")
        sys.exit(0)

    print(f"Found {len(datasets)} dataset(s).\n")

    n_figs = 0
    for csv_path, df, range_cols in datasets:
        label = csv_path.stem.replace("_data", "")

        print(f"{'─' * 60}")
        print(f"  {csv_path.name}  ({len(df)} scans, {len(range_cols)} beams)")
        print(f"{'─' * 60}")

        # 1. Dense overlay of all scans
        plot_all_scans_overlay(df, range_cols, OUTPUT_DIR / f"{label}_overlay.png")

        # 2. Time-lapse grid
        plot_timelapse(df, range_cols, OUTPUT_DIR / f"{label}_timelapse.png")

        # 3. Scan rate / FPS
        plot_scan_rate(df, OUTPUT_DIR / f"{label}_scanrate.png")

        # 4. Range heatmap
        plot_heatmap(df, range_cols, OUTPUT_DIR / f"{label}_heatmap.png")

        n_figs += 4
        print()

    print(f"All done! ✓  ({n_figs} figures saved to figs/)")


if __name__ == "__main__":
    main()
