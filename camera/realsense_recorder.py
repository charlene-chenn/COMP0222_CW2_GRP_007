"""
Record from Intel RealSense D455 camera and extract visual features.
"""

import pyrealsense2 as rs
import cv2
import numpy as np
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path

class RealsenseRecorder:
    def __init__(self, output_dir='data', framerate=30):
        """
        Initialize RealSense recorder.
        
        Args:
            output_dir: Base directory to save recordings (default: data)
            framerate: Camera framerate in Hz (default 30)
        """
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = None  # Will be set when recording starts
        self.framerate = framerate
        
        # Feature detector (ORB - good for SLAM)
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Setup RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams with specified framerate
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.framerate)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.framerate)
        
        # Optionally enable IMU if available
        try:
            self.config.enable_stream(rs.stream.accel)
            self.config.enable_stream(rs.stream.gyro)
            self.has_imu = True
        except:
            self.has_imu = False
        
        # Get device and enable recording to bag file
        self.profile = None
        
    def start_recording(self, bag_path=None):
        """Start recording and feature extraction."""
        # Create timestamped subdirectory for this recording session
        session_name = f"realsense_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = self.base_output_dir / session_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if bag_path is None:
            bag_path = self.output_dir / "recording.bag"
        
        # Enable recording to bag file
        if bag_path:
            self.config.enable_record_to_file(str(bag_path))
            print(f"Recording to: {bag_path}")
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        self.device = self.profile.get_device()
        
        # Get camera intrinsics
        color_profile = self.profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        depth_profile = self.profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
        # Create CSV for feature metadata
        self.csv_path = self.output_dir / "features.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['frame_id', 'timestamp', 'num_features', 'depth_mean', 'depth_std'])
        
        print(f"Camera intrinsics (color): {color_intrinsics}")
        print(f"Camera intrinsics (depth): {depth_intrinsics}")
        
        return color_intrinsics, depth_intrinsics
    
    def extract_features(self, frame):
        """Extract ORB features from frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def record_and_extract(self, duration=10, display=True):
        """
        Record for specified duration and extract features from each frame.
        
        Args:
            duration: Recording duration in seconds
            display: Whether to display frames with features
        """
        color_intrinsics, depth_intrinsics = self.start_recording()
        
        frame_id = 0
        start_time = time.time()
        
        # Aligners for depth-to-color
        align = rs.align(rs.stream.color)
        
        try:
            while (time.time() - start_time) < duration:
                frames = self.pipeline.wait_for_frames()
                
                # Align depth to color
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asarray(color_frame.get_data())
                depth_image = np.asarray(depth_frame.get_data())
                
                # Extract features
                keypoints, descriptors = self.extract_features(color_image)
                
                # Save frame with keypoints
                frame_with_kp = cv2.drawKeypoints(color_image, keypoints, None, color=(0, 255, 0))
                frame_path = self.output_dir / f"frame_{frame_id:06d}.png"
                cv2.imwrite(str(frame_path), frame_with_kp)
                
                # Save descriptors if features found
                if descriptors is not None:
                    desc_path = self.output_dir / f"descriptors_{frame_id:06d}.npy"
                    np.save(desc_path, descriptors)
                
                # Save keypoint data
                kp_path = self.output_dir / f"keypoints_{frame_id:06d}.npy"
                kp_data = np.array([(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in keypoints])
                np.save(kp_path, kp_data)
                
                # Log feature info
                depth_mean = np.nanmean(depth_image[depth_image > 0])
                depth_std = np.nanstd(depth_image[depth_image > 0])
                
                self.csv_writer.writerow([
                    frame_id,
                    frames.get_timestamp(),
                    len(keypoints),
                    depth_mean,
                    depth_std
                ])
                
                # Display
                if display:
                    print(f"Frame {frame_id}: {len(keypoints)} features, "
                          f"depth: {depth_mean:.2f}±{depth_std:.2f} mm")
                    
                    if len(keypoints) > 0:
                        cv2.imshow('Features', frame_with_kp)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                frame_id += 1
        
        finally:
            self.stop_recording()
            if display:
                cv2.destroyAllWindows()
            print(f"\nRecording complete: {frame_id} frames recorded")
    
    def stop_recording(self):
        """Stop recording and cleanup."""
        if self.csv_file:
            self.csv_file.close()
        self.pipeline.stop()
        print(f"Features saved to: {self.output_dir}")
        print(f"CSV log: {self.csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record from RealSense D455 and extract features')
    parser.add_argument('--duration', type=int, default=30,
                        help='Recording duration in seconds (default: 30)')
    parser.add_argument('--framerate', type=int, default=30,
                        help='Camera framerate in Hz (default: 30, options: 6, 15, 30, 60)')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory for recorded data (default: data)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable live display during recording')
    
    args = parser.parse_args()
    
    print(f"RealSense D455 Recorder")
    print(f"  Duration: {args.duration} seconds")
    print(f"  Framerate: {args.framerate} Hz")
    print(f"  Output: {args.output}")
    print(f"  Display: {'Disabled' if args.no_display else 'Enabled'}")
    print()
    
    recorder = RealsenseRecorder(output_dir=args.output, framerate=args.framerate)
    recorder.record_and_extract(duration=args.duration, display=not args.no_display)
