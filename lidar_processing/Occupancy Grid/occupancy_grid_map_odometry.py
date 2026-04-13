import os
import sys
import math
import pygame
import numpy as np
from rplidar_driver import LidarDriver
import platform
import argparse

# --- Configuration ---
PORT_NAME = ''
BAUD_RATE = 256000
TIMEOUT = 1

# Display & Map Settings
WINDOW_SIZE = (800, 800)
MAP_DIM = 1200
CELL_SIZE_MM = 10
LIDAR_RADIUS_MM = 4000
MIN_DISTANCE_MM = 300 # Ignore anything closer than 30cm (e.g. chassis or hand)

# SLAM / Probability Constants
CONFIDENCE_FREE = (10, 10, 10)
CONFIDENCE_OCCUPIED = (50, 50, 50)

# Blind Spot (User Location) - 90 degrees behind sensor
CUT_ANGLE_MIN = 135.0
CUT_ANGLE_MAX = 225.0
ENABLE_BLIND_SPOT = True # Set to True to ignore user body behind sensor

# Detect OS
os_name = platform.system()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Lab 09 - Lidar SLAM: Fixed Map + Trajectory")
parser.add_argument('--mode', type=str, default='replay', choices=['live', 'replay'], help='Operation mode')
parser.add_argument('--file', type=str, default='../../Lab_08_-_Point_Cloud/Solution/indoor_lidar_scan_data.json', help='File to replay')
parser.add_argument('--iterations', type=int, default=10, help='ICP iterations')
args = parser.parse_args()

# 2. Assign Port based on OS
if os_name == 'Windows':
    PORT_NAME = 'COM8'                
elif os_name == 'Darwin':             
    PORT_NAME = '/dev/tty.usbserial-120' 
else:                                 
    PORT_NAME = '/dev/ttyUSB0'        

print(f"Detected {os_name}. Mode: {args.mode}")


class PoseEstimator:
    def __init__(self, map_dim, cell_size_mm):
        self.map_w = map_dim
        self.map_h = map_dim
        self.cell_size = cell_size_mm
        self.reset()

    def reset(self):
        """Resets the robot to the center of the map."""
        self.x = (self.map_w * self.cell_size) / 2
        self.y = (self.map_h * self.cell_size) / 2
        self.theta = 0.0        

    def get_pose(self):
        return self.x, self.y, self.theta

    def get_pixel_pos(self):
        return self.x / self.cell_size, self.y / self.cell_size

    def optimize_pose(self, scan_points, grid_map, iterations=10):
        if len(scan_points) == 0: return

        scan_arr = np.array(scan_points)
        angles, dists = scan_arr[:, 0], scan_arr[:, 1]
        local_x, local_y = dists * np.cos(angles), dists * np.sin(angles)

        step_xy, step_th = 40, np.radians(1.0) # Start wider

        for _ in range(iterations):
            best_score = -float('inf')
            best_pose = (self.x, self.y, self.theta)
            found_better = False

            for dth in [-step_th, 0, step_th]:
                th = self.theta + dth
                cos_th, sin_th = np.cos(th), np.sin(th)
                for dx in [-step_xy, 0, step_xy]:
                    for dy in [-step_xy, 0, step_xy]:
                        tx, ty = self.x + dx, self.y + dy
                        gx = ((local_x * cos_th - local_y * sin_th) + tx) / self.cell_size
                        gy = ((local_x * sin_th + local_y * cos_th) + ty) / self.cell_size
                        ix, iy = gx.astype(int), gy.astype(int)

                        mask = (ix >= 0) & (ix < self.map_w) & (iy >= 0) & (iy < self.map_h)
                        if np.sum(mask) > 5:
                            score = np.mean(grid_map[ix[mask], iy[mask]])
                            if score > best_score:
                                best_score, best_pose, found_better = score, (tx, ty, th), True
            
            if found_better:
                self.x, self.y, self.theta = best_pose
                step_xy *= 0.9
                step_th *= 0.9
            else: break

def run_fixed_map_slam():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Lidar SLAM: WASD=Pan, QE=Zoom, Space=Reset")

    view_surface = pygame.Surface((MAP_DIM, MAP_DIM))
    view_surface.fill((128, 128, 128))
    occupancy_grid = np.full((MAP_DIM, MAP_DIM), 0.5, dtype=np.float32)
    trajectory = []

    estimator = PoseEstimator(MAP_DIM, CELL_SIZE_MM)
    driver = LidarDriver(mode=args.mode, filename=args.file)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    # Camera settings
    v_zoom, v_pan_x, v_pan_y = 0.6, 0, 0
    replay_done = False

    print(f"Driver initialized in {args.mode} mode.")

    try:
        iterator = driver.iter_scans()
        while True:
            # 1. Process Input
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: raise KeyboardInterrupt
                    if event.key == pygame.K_r:
                        view_surface.fill((128, 128, 128))
                        occupancy_grid.fill(0.5)
                        trajectory.clear()
                        estimator.reset()
                    if event.key == pygame.K_SPACE:
                        v_zoom, v_pan_x, v_pan_y = 0.6, 0, 0

            keys = pygame.key.get_pressed()
            spd = 15 / v_zoom
            if keys[pygame.K_w]: v_pan_y += spd
            if keys[pygame.K_s]: v_pan_y -= spd
            if keys[pygame.K_a]: v_pan_x += spd
            if keys[pygame.K_d]: v_pan_x -= spd
            if keys[pygame.K_q]: v_zoom *= 1.05
            if keys[pygame.K_e]: v_zoom *= 0.95
            v_zoom = max(0.1, min(10.0, v_zoom))

            # 2. Process Next Scan (if available)
            if not replay_done:
                try:
                    scan = next(iterator)
                    valid = []
                    for (_, a, d) in scan:
                        if ENABLE_BLIND_SPOT and (CUT_ANGLE_MIN <= a <= CUT_ANGLE_MAX): continue
                        if d > MIN_DISTANCE_MM: valid.append((math.radians(a), d))
                    
                    if valid:
                        estimator.optimize_pose(valid, occupancy_grid, iterations=args.iterations)
                        cx, cy, ct = estimator.get_pose()
                        cpx, cpy = estimator.get_pixel_pos()
                        trajectory.append((cpx, cpy))

                        # Draw to Map Surface
                        f_surf = pygame.Surface((MAP_DIM, MAP_DIM)); f_surf.fill((0,0,0))
                        h_surf = pygame.Surface((MAP_DIM, MAP_DIM)); h_surf.fill((0,0,0))
                        
                        cos_t, sin_t = math.cos(ct), math.sin(ct)
                        m_hx, m_hy = [], []

                        for (ar, d) in valid:
                            gx, gy = (d*math.cos(ar)*cos_t - d*math.sin(ar)*sin_t) + cx, (d*math.cos(ar)*sin_t + d*math.sin(ar)*cos_t) + cy
                            px, py = int(gx/CELL_SIZE_MM), int(gy/CELL_SIZE_MM)
                            if 0 <= px < MAP_DIM and 0 <= py < MAP_DIM:
                                pygame.draw.line(f_surf, CONFIDENCE_FREE, (int(cpx), int(cpy)), (px, py), 2)
                                pygame.draw.circle(h_surf, CONFIDENCE_OCCUPIED, (px, py), 2)
                                m_hx.append(px); m_hy.append(py)

                        view_surface.blit(f_surf, (0,0), special_flags=pygame.BLEND_ADD)
                        view_surface.blit(h_surf, (0,0), special_flags=pygame.BLEND_SUB)
                        if m_hx: occupancy_grid[np.array(m_hx), np.array(m_hy)] = np.minimum(1.0, occupancy_grid[np.array(m_hx), np.array(m_hy)] + 0.2)
                except StopIteration:
                    print("Replay finished. Inspect the map or close window."); replay_done = True

            # 3. Render
            screen.fill((50, 50, 50))
            sz = int(MAP_DIM * v_zoom)
            if sz > 0:
                s_map = pygame.transform.scale(view_surface, (sz, sz))
                mx = (WINDOW_SIZE[0]//2) - (sz//2) + (v_pan_x * v_zoom)
                my = (WINDOW_SIZE[1]//2) - (sz//2) + (v_pan_y * v_zoom)
                screen.blit(s_map, (mx, my))

                # World to screen helper
                def w2s(wx, wy): return int(mx + wx * v_zoom), int(my + wy * v_zoom)

                cpx, cpy = estimator.get_pixel_pos()
                rx, ry = w2s(cpx, cpy)
                cx, cy, ct = estimator.get_pose()

                if len(trajectory) > 1:
                    pygame.draw.lines(screen, (0, 150, 255), False, [w2s(p[0], p[1]) for p in trajectory], 2)
                
                # Heading & Robot
                pygame.draw.line(screen, (0, 0, 255), (rx, ry), (rx + 30*v_zoom*math.cos(ct), ry + 30*v_zoom*math.sin(ct)), 2)
                pygame.draw.circle(screen, (255, 0, 0), (rx, ry), int(5*v_zoom))

                if ENABLE_BLIND_SPOT:
                    for a_deg in [CUT_ANGLE_MIN, CUT_ANGLE_MAX]:
                        pygame.draw.line(screen, (255, 255, 0), (rx, ry), (rx + 30*v_zoom*math.cos(ct+math.radians(a_deg)), ry + 30*v_zoom*math.sin(ct+math.radians(a_deg))), 1)

            txt = font.render(f"FPS: {int(clock.get_fps())} | Zoom: {v_zoom:.2f} | {'FINISHED' if replay_done else 'RUNNING'}", True, (0, 255, 0))
            screen.blit(txt, (10, 10))
            pygame.display.flip()
            clock.tick(60 if not replay_done else 30)

    except KeyboardInterrupt: print("Stopping...")
    finally:
        driver.disconnect()
        pygame.quit()

if __name__ == '__main__':
    run_fixed_map_slam()