import cv2
import os
import argparse

def extract_frames(video_path, output_dir, prefix="frame", start_idx=0):
	"""
	Extract frames from a .mov video file.
	
	Args:
		video_path: Path to the .mov file
		output_dir: Directory to save extracted frames
		prefix: Prefix for the output image filenames
		start_idx: Starting index for frame numbering
	"""
	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)
	
	# Open the video file
	video = cv2.VideoCapture(video_path)
	
	if not video.isOpened():
		print(f"Error: Could not open video file {video_path}")
		return
	
	frame_idx = start_idx
	while True:
		# Read next frame
		ret, frame = video.read()
		
		if not ret:
			break
		
		# Construct output filename
		output_path = os.path.join(output_dir, f"{prefix}_{frame_idx:04d}.jpg")
		
		# Save frame as JPEG
		cv2.imwrite(output_path, frame)
		
		frame_idx += 1
		
		# Optional: print progress
		if frame_idx % 100 == 0:
			print(f"Extracted {frame_idx} frames...")
	
	video.release()
	print(f"Done! Extracted {frame_idx - start_idx} frames to {output_dir}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Extract frames from .mov video file")
	parser.add_argument("video", help="Path to input .mov file")
	parser.add_argument("-o", "--output", default="frames", help="Output directory (default: frames)")
	parser.add_argument("-p", "--prefix", default="frame", help="Output filename prefix (default: frame)")
	parser.add_argument("-s", "--start", type=int, default=0, help="Starting frame index (default: 0)")
	
	args = parser.parse_args()
	
	extract_frames(args.video, args.output, args.prefix, args.start)