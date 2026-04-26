#!/usr/bin/env python3
"""
Generate TUM dataset format from extracted frames.

Usage:
    python3 generate_tum_format.py <frames_dir> <output_dir> [fps]

Example:
    python3 generate_tum_format.py camera/clean_frames/playground_longer camera/playground_SLAM 29.997
"""

import os
import sys
import argparse
from pathlib import Path


def generate_tum_format(frames_dir, output_dir, fps=29.997):
    """
    Generate TUM format structure with rgb.txt timestamp file.
    
    Args:
        frames_dir: Directory containing frame_*.jpg files
        output_dir: Directory where to save rgb.txt and create rgb/ symlink
        fps: Frames per second for timestamp calculation
    """
    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir).resolve()
    rgb_dir = output_dir / "rgb"
    rgb_txt = output_dir / "rgb.txt"
    
    # Validate input directory
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sorted list of frames
    frames = sorted([f for f in frames_dir.iterdir() if f.name.startswith('frame_') and f.suffix.lower() == '.jpg'])
    
    if not frames:
        print(f"Error: No frame_*.jpg files found in {frames_dir}")
        sys.exit(1)
    
    num_frames = len(frames)
    duration = num_frames / fps
    
    print(f"Found {num_frames} frames")
    print(f"FPS: {fps}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Frames directory: {frames_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create symlink to frames directory as 'rgb'
    if rgb_dir.exists() or rgb_dir.is_symlink():
        if rgb_dir.is_symlink():
            rgb_dir.unlink()
        else:
            print(f"Error: {rgb_dir} already exists and is not a symlink")
            sys.exit(1)
    
    os.symlink(frames_dir, rgb_dir)
    print(f"Created symlink: {rgb_dir} -> {frames_dir}")
    
    # Generate rgb.txt with timestamps
    with open(rgb_txt, 'w') as f:
        for i, frame_path in enumerate(frames):
            timestamp = i / fps
            f.write(f"{timestamp:.6f} rgb/{frame_path.name}\n")
    
    print(f"Generated {rgb_txt} with {num_frames} entries")
    print("\nTUM format ready for ORB-SLAM2!")
    print(f"Run ORB-SLAM2 with: mono_tum ORBvoc.txt camera_intrinsics.yaml {output_dir}/rgb.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TUM dataset format from extracted frames")
    parser.add_argument("frames_dir", help="Directory containing frame_*.jpg files")
    parser.add_argument("output_dir", help="Directory where to save rgb.txt and create rgb/ symlink")
    parser.add_argument("--fps", type=float, default=29.997, help="Frames per second (default: 29.997)")
    
    args = parser.parse_args()
    
    generate_tum_format(args.frames_dir, args.output_dir, args.fps)
