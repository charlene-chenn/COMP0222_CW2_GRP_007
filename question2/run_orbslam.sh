#!/usr/bin/env bash
# Convenience runner for ORB-SLAM2 in this coursework repo.
#
# Usage:
#   ./run_orbslam.sh help
#   ./run_orbslam.sh runplaygroundlong
#   ./run_orbslam.sh runplayground
#   ./run_orbslam.sh runclassroom
#   ./run_orbslam.sh runlobby
#   ./run_orbslam.sh run <yaml> <dataset_dir> [output_file]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MONO_TUM="/Users/omarahmed/orb_slam2_build/Install/bin/mono_tum"

# Preset datasets/calibrations used in this project.
YAML_PLAYGROUND_LONG="$SCRIPT_DIR/COLMAP/playground_long_colmap/camera_intrinsics.yaml"
YAML_PLAYGROUND="$SCRIPT_DIR/COLMAP/playground_colmap/camera_intrinsics.yaml"
YAML_CLASSROOM="$SCRIPT_DIR/COLMAP/classroom_colmap/camera_intrinsics.yaml"
YAML_LOBBY="$SCRIPT_DIR/COLMAP/lobby_colmap/colmap_output/camera_intrinsics.yaml"

DATASET_PLAYGROUND_LONG="$SCRIPT_DIR/SLAM/playground_long_SLAM"
DATASET_PLAYGROUND="$SCRIPT_DIR/SLAM/playground_SLAM"
DATASET_CLASSROOM="$SCRIPT_DIR/SLAM/classroom_SLAM"
DATASET_LOBBY="$SCRIPT_DIR/SLAM/lobby_SLAM"

OUT_PLAYGROUND_LONG="$DATASET_PLAYGROUND_LONG/orb_slam_results.txt"
OUT_PLAYGROUND="$DATASET_PLAYGROUND/orb_slam_results.txt"
OUT_CLASSROOM="$DATASET_CLASSROOM/orb_slam_results.txt"
OUT_LOBBY="$DATASET_LOBBY/orb_slam_results.txt"

die() {
	echo "Error: $*" >&2
	exit 1
}

check_prereqs() {
	[ -x "$MONO_TUM" ] || die "mono_tum not found/executable at: $MONO_TUM"
}

run_orb() {
	local yaml="$1"
	local dataset_dir="$2"
	local output_file="$3"

	[ -f "$yaml" ] || die "YAML not found: $yaml"
	[ -d "$dataset_dir" ] || die "Dataset directory not found: $dataset_dir"
	[ -f "$dataset_dir/rgb.txt" ] || die "Missing rgb.txt in: $dataset_dir"
	[ -e "$dataset_dir/rgb" ] || die "Missing rgb/ in: $dataset_dir"

	mkdir -p "$(dirname "$output_file")"

	echo "Running ORB-SLAM2 mono_tum"
	echo "  YAML:    $yaml"
	echo "  Dataset: $dataset_dir"
	echo "  Output:  $output_file"

	cd "$ROOT_DIR"
	"$MONO_TUM" "$yaml" "$dataset_dir" "$output_file"
}

print_help() {
	cat <<'EOF'
run_orbslam.sh - quick runner for this repo

Commands:
	help
		Show this message.

	runplaygroundlong
		Run playground_long dataset:
		YAML:    question2/playground_long_colmap/camera_intrinsics.yaml
		Dataset: question2/playground_long_SLAM
		Output:  question2/playground_long_SLAM/orb_slam_results.txt

	runplayground
		Run playground dataset:
		YAML:    question2/playground_colmap/camera_intrinsics.yaml
		Dataset: question2/playground_SLAM
		Output:  question2/playground_SLAM/orb_slam_results.txt

	runclassroom
		Run classroom dataset:
		YAML:    question2/classroom_colmap/camera_intrinsics.yaml
		Dataset: question2/classroom_SLAM
		Output:  question2/classroom_SLAM/orb_slam_results.txt

	runlobby
		Run lobby dataset:
		YAML:    question2/lobby_SLAM/camera_intrinsics.yaml
		Dataset: question2/lobby_SLAM
		Output:  question2/lobby_SLAM/orb_slam_results.txt

	run <yaml> <dataset_dir> [output_file]
		Fully custom run.
		If output_file is omitted, uses: <dataset_dir>/orb_slam_results.txt

Examples:
	./question2/run_orbslam.sh runplaygroundlong
	./question2/run_orbslam.sh runplayground
	./question2/run_orbslam.sh runclassroom
	./question2/run_orbslam.sh runlobby
EOF
}

main() {
	check_prereqs

	local cmd="${1:-help}"
	case "$cmd" in
		help|-h|--help)
			print_help
			;;
		runplaygroundlong)
			run_orb "$YAML_PLAYGROUND_LONG" "$DATASET_PLAYGROUND_LONG" "$OUT_PLAYGROUND_LONG"
			;;
		runplayground)
			run_orb "$YAML_PLAYGROUND" "$DATASET_PLAYGROUND" "$OUT_PLAYGROUND"
			;;
		runclassroom)
			run_orb "$YAML_CLASSROOM" "$DATASET_CLASSROOM" "$OUT_CLASSROOM"
			;;
		runlobby)
			run_orb "$YAML_LOBBY" "$DATASET_LOBBY" "$OUT_LOBBY"
			;;
		run)
			[ "$#" -ge 3 ] || die "Usage: $0 run <yaml> <dataset_dir> [output_file]"
			local yaml="$2"
			local dataset_dir="$3"
			local out="${4:-$dataset_dir/orb_slam_results.txt}"
			run_orb "$yaml" "$dataset_dir" "$out"
			;;
		*)
			die "Unknown command: $cmd (use: $0 help)"
			;;
	esac
}

main "$@"
