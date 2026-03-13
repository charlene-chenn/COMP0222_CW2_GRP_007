#!/usr/bin/env python3
"""
extract_bag_to_csv.py
---------------------
Reads lidar2.bag (ROS1 bag) and extracts all LaserScan messages into a CSV.
For other message types it extracts all primitive fields.

Output: one CSV per topic, e.g. scan_data.csv
"""

import sys
import os
import csv
import math
import numpy as np
from pathlib import Path

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BAG_FILE = Path(__file__).parent / "data" / "lidar2.bag"
OUTPUT_DIR = Path(__file__).parent / "data"


def sanitise_topic_name(topic: str) -> str:
    """Convert a ROS topic name like /scan to a safe filename."""
    return topic.strip("/").replace("/", "_") or "unknown_topic"


def extract_laser_scan(reader, typestore, conn):
    """Extract all LaserScan messages from a single connection/topic."""
    rows = []
    for c, timestamp, rawdata in reader.messages(connections=[conn]):
        msg = typestore.deserialize_ros1(rawdata, c.msgtype)

        # Header timestamp
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec if hasattr(msg.header.stamp, "nanosec") else msg.header.stamp.nsec
        stamp = sec + nsec * 1e-9

        # Compute angles for each beam
        num_beams = len(msg.ranges)
        angles = [msg.angle_min + i * msg.angle_increment for i in range(num_beams)]

        # Build one row per message
        row = {
            "timestamp": stamp,
            "bag_time_ns": timestamp,
            "frame_id": msg.header.frame_id,
            "angle_min": msg.angle_min,
            "angle_max": msg.angle_max,
            "angle_increment": msg.angle_increment,
            "time_increment": msg.time_increment,
            "scan_time": msg.scan_time,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
            "num_beams": num_beams,
        }

        # Add individual range & intensity columns
        for i, (angle, r) in enumerate(zip(angles, msg.ranges)):
            row[f"angle_{i}"] = angle
            row[f"range_{i}"] = r

        if len(msg.intensities) > 0:
            for i, intensity in enumerate(msg.intensities):
                row[f"intensity_{i}"] = intensity

        rows.append(row)

    return rows


def extract_generic(reader, typestore, conn):
    """Extract primitive fields from any message type."""
    rows = []
    for c, timestamp, rawdata in reader.messages(connections=[conn]):
        msg = typestore.deserialize_ros1(rawdata, c.msgtype)
        row = {"bag_time_ns": timestamp}

        # Walk the message fields
        for field_name in msg.__dataclass_fields__:
            val = getattr(msg, field_name)
            if isinstance(val, (int, float, str, bool)):
                row[field_name] = val
            elif isinstance(val, np.ndarray):
                for i, v in enumerate(val.flat):
                    row[f"{field_name}_{i}"] = v
            else:
                row[field_name] = str(val)

        rows.append(row)

    return rows


def write_csv(rows: list[dict], path: Path):
    """Write a list of row dicts to a CSV file."""
    if not rows:
        print(f"  ⚠  No data to write for {path.name}")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓  Wrote {len(rows)} rows → {path.name}")


def main():
    if not BAG_FILE.exists():
        print(f"ERROR: Bag file not found: {BAG_FILE}")
        sys.exit(1)

    print(f"Opening {BAG_FILE.name} ({BAG_FILE.stat().st_size} bytes)")
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(BAG_FILE) as reader:
        connections = list(reader.connections)

        if not connections:
            print("\n⚠  No topics found in the bag file.")
            print("   The bag may be empty or corrupted.")
            print("   Replace lidar2.bag with a valid file and re-run.")
            sys.exit(0)

        print(f"Found {len(connections)} topic(s):\n")
        for conn in connections:
            print(f"  • {conn.topic}  [{conn.msgtype}]  ({conn.msgcount} msgs)")

        print()

        for conn in connections:
            topic_name = sanitise_topic_name(conn.topic)
            out_path = OUTPUT_DIR / f"{topic_name}_data.csv"

            print(f"Extracting: {conn.topic}")

            if "LaserScan" in conn.msgtype:
                rows = extract_laser_scan(reader, typestore, conn)
            else:
                rows = extract_generic(reader, typestore, conn)

            write_csv(rows, out_path)

    print("\nDone! ✓")


if __name__ == "__main__":
    main()
