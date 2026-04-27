#!/usr/bin/env python3
"""Convenience runner for EVO trajectory comparison in this coursework repo.

Usage:
  python3 run_evo.py help
  python3 run_evo.py runplaygroundlong
  python3 run_evo.py run <ref_tum> <est_tum> [results_dir]
  python3 run_evo.py make_colmap_tum <colmap_images_txt> <rgb_txt> [out_tum]
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PLAYGROUND_LONG = SCRIPT_DIR / "playground_long_SLAM"
COLMAP_PLAYGROUND_LONG_IMAGES = SCRIPT_DIR / "playground_long_colmap" / "txt_export" / "images.txt"
RGB_PLAYGROUND_LONG = DATASET_PLAYGROUND_LONG / "rgb.txt"
EST_PLAYGROUND_LONG = DATASET_PLAYGROUND_LONG / "orb_slam_results.txt"
REF_PLAYGROUND_LONG = DATASET_PLAYGROUND_LONG / "colmap_ref_tum.txt"
RESULTS_PLAYGROUND_LONG = DATASET_PLAYGROUND_LONG / "evo"

DATASET_LOBBY = SCRIPT_DIR / "lobby_SLAM"
COLMAP_LOBBY_IMAGES = SCRIPT_DIR / "lobby_colmap" / "colmap_output" / "sparse" / "1_txt" / "images.txt"
RGB_LOBBY = DATASET_LOBBY / "rgb.txt"
EST_LOBBY = DATASET_LOBBY / "orb_slam_results.txt"
REF_LOBBY = DATASET_LOBBY / "colmap_ref_tum.txt"
RESULTS_LOBBY = DATASET_LOBBY / "evo"

TMAX_DIFF = 0.05


def die(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def script_or_die(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path

    scripts_dir = Path(sysconfig_path("scripts"))
    candidate = scripts_dir / name
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)

    die(f"{name} not found. Install evo first.")


def sysconfig_path(key: str) -> str:
    import sysconfig

    return sysconfig.get_path(key) or ""


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def make_colmap_tum(colmap_images_txt: Path, rgb_txt: Path, out_tum: Path) -> None:
    if not colmap_images_txt.exists():
        die(f"COLMAP images.txt not found: {colmap_images_txt}")
    if not rgb_txt.exists():
        die(f"rgb.txt not found: {rgb_txt}")

    name_to_ts: dict[str, float] = {}
    with rgb_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ts = float(parts[0])
            base = os.path.basename(parts[1])
            name_to_ts[base] = ts

    def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
        return [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]

    def transpose(m: list[list[float]]) -> list[list[float]]:
        return [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]

    def mat_vec_mul(m: list[list[float]], v: list[float]) -> list[float]:
        return [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]

    rows: list[tuple[float, float, float, float, float, float, float, float]] = []
    with colmap_images_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10 or not parts[0].isdigit():
                continue
            image_name = os.path.basename(parts[9])
            if image_name not in name_to_ts:
                continue

            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])
            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])

            r_cw = quat_to_rot(qw, qx, qy, qz)
            r_wc = transpose(r_cw)
            c_w = mat_vec_mul(r_wc, [-tx, -ty, -tz])

            ts = name_to_ts[image_name]
            rows.append((ts, c_w[0], c_w[1], c_w[2], -qx, -qy, -qz, qw))

    rows.sort(key=lambda row: row[0])
    out_tum.parent.mkdir(parents=True, exist_ok=True)
    with out_tum.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write("{:.6f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(*row))

    print(f"Wrote {len(rows)} poses to {out_tum}")


def run_evo_compare(
    ref_tum: Path,
    est_tum: Path,
    results_dir: Path,
    ref_label: str = "COLMAP",
    est_label: str = "ORB-SLAM",
    file_prefix: str = "",
) -> None:
    if not ref_tum.exists():
        die(f"Reference trajectory not found: {ref_tum}")
    if not est_tum.exists():
        die(f"Estimated trajectory not found: {est_tum}")

    results_dir.mkdir(parents=True, exist_ok=True)

    evo_ape = script_or_die("evo_ape")
    evo_rpe = script_or_die("evo_rpe")
    evo_traj = shutil.which("evo_traj") or str(Path(sysconfig_path("scripts")) / "evo_traj")
    has_traj = Path(evo_traj).exists() and os.access(evo_traj, os.X_OK)

    # evo uses the filename stem as the legend label, so create temp
    # symlinks with the desired names to get clean labels in plots.
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="evo_labels_"))
    ref_link = tmp / f"{ref_label}.txt"
    est_link = tmp / f"{est_label}.txt"
    ref_link.symlink_to(ref_tum.resolve())
    est_link.symlink_to(est_tum.resolve())

    try:
        if has_traj:
            print("Running EVO trajectory overlay...")
            run_cmd([
                evo_traj,
                "tum",
                "--ref",
                str(ref_link),
                str(ref_link),
                str(est_link),
                "--sync",
                "--t_max_diff",
                str(TMAX_DIFF),
                "--align",
                "--correct_scale",
                "--no_warnings",
                "--save_plot",
                str(results_dir / f"{file_prefix}traj_overlay.png"),
            ])
        else:
            print("Skipping trajectory overlay (evo_traj unavailable).")

        print("Running EVO APE...")
        run_cmd([
            evo_ape,
            "tum",
            str(ref_link),
            str(est_link),
            "--t_max_diff",
            str(TMAX_DIFF),
            "--align",
            "--correct_scale",
            "--no_warnings",
            "--save_plot",
            str(results_dir / f"{file_prefix}ape.png"),
            "--save_results",
            str(results_dir / f"{file_prefix}ape.zip"),
        ])

        print("Running EVO RPE...")
        run_cmd([
            evo_rpe,
            "tum",
            str(ref_link),
            str(est_link),
            "--t_max_diff",
            str(TMAX_DIFF),
            "--align",
            "--correct_scale",
            "--no_warnings",
            "--save_plot",
            str(results_dir / f"{file_prefix}rpe.png"),
            "--save_results",
            str(results_dir / f"{file_prefix}rpe.zip"),
        ])
    finally:
        # Clean up temp symlinks
        ref_link.unlink(missing_ok=True)
        est_link.unlink(missing_ok=True)
        tmp.rmdir()

    print(f"Done. EVO outputs saved to: {results_dir}")


def print_help() -> None:
    print(
        """run_evo.py - quick EVO runner for this repo

Commands:
  help
    Show this message.

  runplaygroundlong
    Build TUM reference from COLMAP + rgb timestamps, then run EVO against:
      ref: camera/playground_long_SLAM/colmap_ref_tum.txt
      est: camera/playground_long_SLAM/orb_slam_results.txt
      out: camera/playground_long_SLAM/evo

  runlobby
    Build TUM reference from COLMAP + rgb timestamps, then run EVO against:
      ref: camera/lobby_SLAM/colmap_ref_tum.txt
      est: camera/lobby_SLAM/orb_slam_results.txt
      out: camera/lobby_SLAM/evo

  run <ref_tum> <est_tum> [results_dir]
    Run EVO comparison on custom TUM trajectories.

  make_colmap_tum <colmap_images_txt> <rgb_txt> [out_tum]
    Convert COLMAP images.txt into TUM trajectory using rgb.txt timestamps.

Examples:
  python3 camera/run_evo.py runplaygroundlong
  python3 camera/run_evo.py run \
    /abs/path/ref.txt \
    /abs/path/est.txt \
    /abs/path/evo_results
"""
    )


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "help"

    if cmd in {"help", "-h", "--help"}:
        print_help()
        return 0

    if cmd == "runplaygroundlong":
        if not COLMAP_PLAYGROUND_LONG_IMAGES.exists():
            die(f"Missing COLMAP images export: {COLMAP_PLAYGROUND_LONG_IMAGES}")
        if not RGB_PLAYGROUND_LONG.exists():
            die(f"Missing rgb.txt: {RGB_PLAYGROUND_LONG}")
        if not EST_PLAYGROUND_LONG.exists():
            die(f"Missing ORB trajectory: {EST_PLAYGROUND_LONG}")
        make_colmap_tum(COLMAP_PLAYGROUND_LONG_IMAGES, RGB_PLAYGROUND_LONG, REF_PLAYGROUND_LONG)
        run_evo_compare(REF_PLAYGROUND_LONG, EST_PLAYGROUND_LONG, RESULTS_PLAYGROUND_LONG, file_prefix="outdoor_")
        return 0

    if cmd == "runlobby":
        if not COLMAP_LOBBY_IMAGES.exists():
            die(f"Missing COLMAP images export: {COLMAP_LOBBY_IMAGES}")
        if not RGB_LOBBY.exists():
            die(f"Missing rgb.txt: {RGB_LOBBY}")
        if not EST_LOBBY.exists():
            die(f"Missing ORB trajectory: {EST_LOBBY}")
        make_colmap_tum(COLMAP_LOBBY_IMAGES, RGB_LOBBY, REF_LOBBY)
        run_evo_compare(REF_LOBBY, EST_LOBBY, RESULTS_LOBBY, file_prefix="lobby_")
        return 0

    if cmd == "make_colmap_tum":
        if len(argv) < 4:
            die(f"Usage: {argv[0]} make_colmap_tum <colmap_images_txt> <rgb_txt> [out_tum]")
        out_tum = Path(argv[4]) if len(argv) > 4 else SCRIPT_DIR / "colmap_ref_tum.txt"
        make_colmap_tum(Path(argv[2]), Path(argv[3]), out_tum)
        return 0

    if cmd == "run":
        if len(argv) < 4:
            die(f"Usage: {argv[0]} run <ref_tum> <est_tum> [results_dir]")
        results_dir = Path(argv[4]) if len(argv) > 4 else SCRIPT_DIR / "evo_results"
        run_evo_compare(Path(argv[2]), Path(argv[3]), results_dir)
        return 0

    die(f"Unknown command: {cmd} (use: {argv[0]} help)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))