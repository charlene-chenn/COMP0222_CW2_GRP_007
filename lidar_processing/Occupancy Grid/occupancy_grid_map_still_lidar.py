import os
import sys
import math
import pygame
import numpy as np
from rplidar_driver import LidarDriver
import platform
import argparse

# --- Configuration ---
PORT_NAME = '' # Change to 'COM3' on Windows
BAUD_RATE = 256000
TIMEOUT = 1

# Display Settings
WINDOW_SIZE = (800, 800)
MAP_SIZE = (1600, 1600)
LIDAR_RADIUS_MM = 4000
MIN_DISTANCE_MM = 300 # Ignore anything closer than 30cm

# --- PROBABILISTIC CONSTANTS (TUNED) ---
# Start Map Color: (128, 128, 128) -> Gray/Unknown

# Activity 1
# 1. CONFIDENCE_FREE (The "Eraser")
# How fast does Gray turn to White?
# (15, 15, 15) means it turns white in roughly 8-10 hits. 
# Increasing this makes the robot "trust" empty space faster.
CONFIDENCE_FREE = (15, 15, 15)

# 2. CONFIDENCE_OCCUPIED (The "Pen")
# How fast does Gray turn to Black?
# (50, 50, 50) means it turns black in roughly 2-3 hits.
# Obstacles should appear faster than they disappear.
CONFIDENCE_OCCUPIED = (50, 50, 50)


# Detect OS
os_name = platform.system()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Lab 09 - Still Lidar Occupancy Grid")
parser.add_argument('--mode', type=str, default='replay', choices=['live', 'replay'], help='Operation mode')
parser.add_argument('--file', type=str, default='../../Lab_08_-_Point_Cloud/Solution/indoor_lidar_scan_data.json', help='File to replay')
args = parser.parse_args()

# 2. Assign Port based on OS
if os_name == 'Windows':
    PORT_NAME = 'COM8'                # Windows default
elif os_name == 'Darwin':             # macOS
    PORT_NAME = '/dev/tty.usbserial-120' # Actual port found in Lab 08
else:                                 # Linux / Ubuntu
    PORT_NAME = '/dev/ttyUSB0'        # Linux default

print(f"Detected {os_name}. Using mode: {args.mode}")


def run_probabilistic_mapping():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("RPLIDAR - Probabilistic Grid (Gray=Unknown, White=Free, Black=Wall)")

    # 1. Create the Map Surface
    map_surface = pygame.Surface(MAP_SIZE)
    map_surface.fill((128, 128, 128)) # Start Middle Gray (Unknown)

    center_x = MAP_SIZE[0] // 2
    center_y = MAP_SIZE[1] // 2

    center_x = MAP_SIZE[0] // 2
    center_y = MAP_SIZE[1] // 2

    # Initialize Driver
    driver = LidarDriver(mode=args.mode, filename=args.file)
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    print(f"Driver initialized in {args.mode} mode.")

    try:
        # Iterate scans from the driver
        for scan in driver.iter_scans():
            
            # Exit Handler
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: raise KeyboardInterrupt

                    if event.key == pygame.K_r:
                        print("RESETTING MAP...")
                        # Clear Visuals
                        map_surface.fill((128, 128, 128))


            # --- Layers Setup ---
            # We create temporary layers to batch the drawing operations
            flash_surface = pygame.Surface(MAP_SIZE)
            flash_surface.fill((0,0,0)) # Black adds nothing in BLEND_ADD
            
            hits_surface = pygame.Surface(MAP_SIZE)
            hits_surface.fill((0,0,0)) # Black substracts nothing in BLEND_SUB

            # --- Process Scan ---
            for (_, angle, distance) in scan:
                 if distance > MIN_DISTANCE_MM:
                    angle_rad = math.radians(angle)
                    
                    # Calculate endpoint
                    px = int(center_x + (distance * math.cos(angle_rad) / 10.0))
                    py = int(center_y + (distance * math.sin(angle_rad) / 10.0))
                    
                    if 0 <= px < MAP_SIZE[0] and 0 <= py < MAP_SIZE[1]:
                        
                        # 1. FREE SPACE (Brighten)
                        # We draw a line from center to point.
                        # Thickness=3 ensures we don't leave tiny gray gaps between rays.
                        pygame.draw.line(flash_surface, CONFIDENCE_FREE, (center_x, center_y), (px, py), 3)
                        
                        # 2. OBSTACLES (Darken)
                        # We draw a circle at the hit point.
                        pygame.draw.circle(hits_surface, CONFIDENCE_OCCUPIED, (px, py), 4)
            
            # --- Apply Updates to Map ---
            
            # A. Add White (Free Space)
            # This makes the gray (128) move toward 255
            map_surface.blit(flash_surface, (0,0), special_flags=pygame.BLEND_ADD)
            
            # B. Subtract White (Occupied Space)
            # This makes the gray (128) move toward 0
            # Note: We do this AFTER adding, so walls override free space if they overlap
            map_surface.blit(hits_surface, (0,0), special_flags=pygame.BLEND_SUB)

            # --- Render to Screen ---
            offset_x = (WINDOW_SIZE[0] - MAP_SIZE[0]) // 2
            offset_y = (WINDOW_SIZE[1] - MAP_SIZE[1]) // 2
            
            screen.fill((50, 50, 50)) # Dark Grey background for the window
            screen.blit(map_surface, (offset_x, offset_y))
            
            # Draw Robot Center
            pygame.draw.circle(screen, (255, 0, 0), (WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2), 5)

            # Draw UI
            fps = int(clock.get_fps())
            text = font.render(f"FPS: {fps} | Points: {len(scan)} | [R] to Reset", True, (255, 255, 255))
            screen.blit(text, (10, 10))

            pygame.display.flip()
            clock.tick(30) # 30 FPS is plenty for Lidar visualization

        # --- Replay Finished - Wait for User to Close ---
        print("Replay finished. Close the window to exit.")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
            clock.tick(10)

    except KeyboardInterrupt:
        print("Stopping...")
    except RPLidarException as e:
        print(f"Lidar Error: {e}")
    finally:
        driver.disconnect()
        pygame.quit()

if __name__ == '__main__':
    run_probabilistic_mapping()