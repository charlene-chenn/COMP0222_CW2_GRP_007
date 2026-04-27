import csv
import json
import math
import sys
import os
import glob

def convert_csv_to_json(csv_path, json_path):
    print(f"Converting {csv_path} to {json_path}...")
    
    with open(csv_path, 'r') as f_in, open(json_path, 'w') as f_out:
        reader = csv.DictReader(f_in)
        
        scan_count = 0
        for row in reader:
            try:
                num_beams = int(float(row['num_beams']))
            except (ValueError, KeyError):
                # Fallback if num_beams isn't correctly parsed
                num_beams = 230
                
            scan_data = []
            for i in range(num_beams):
                angle_key = f'angle_{i}'
                range_key = f'range_{i}'
                intensity_key = f'intensity_{i}'
                
                # Check if keys exist
                if angle_key not in row or range_key not in row or intensity_key not in row:
                    continue
                    
                angle_str = row[angle_key]
                range_str = row[range_key]
                intensity_str = row[intensity_key]
                
                if not angle_str or not range_str or not intensity_str:
                    continue
                    
                angle_rad = float(angle_str)
                range_m = float(range_str)
                intensity = float(intensity_str)
                
                # Scale the range to millimeters 
                distance_mm = range_m * 1000.0
                
                # Convert angle from radians to 0-360 degrees
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0:
                    angle_deg += 360.0
                    
                # The RPLiDAR scripts expect [quality, angle_deg, distance_mm]
                scan_data.append([int(intensity), round(angle_deg, 4), round(distance_mm, 2)])
                
            # Write one JSON array per line to mimic the iter_scans() json lines dump
            json.dump(scan_data, f_out)
            f_out.write('\n')
            scan_count += 1
            
    print(f"  -> Converted {scan_count} scans.")

def main():
    # If arguments are passed, process those specific files
    if len(sys.argv) > 1:
        for csv_file in sys.argv[1:]:
            json_file = os.path.splitext(csv_file)[0] + '.json'
            convert_csv_to_json(csv_file, json_file)
    else:
        # Otherwise, process all CSVs in the current directory matching '*.csv'
        csv_files = glob.glob('*.csv')
        if not csv_files:
            print("No matching CSV files ('*.csv') found.")
            return
            
        for csv_file in csv_files:
            json_file = os.path.splitext(csv_file)[0] + '.json'
            convert_csv_to_json(csv_file, json_file)

if __name__ == '__main__':
    main()
