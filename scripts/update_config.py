import json
import numpy as np
import os

def update_config():
    with open('mean_std.json', 'r') as f:
        stats = json.load(f)
        
    config_path = 'scripts/config.py'
    with open(config_path, 'r') as f:
        lines = f.readlines()
        
    # Find start and end of the section to replace
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if "def get_normalization_constants():" in line:
            start_idx = i
        if "LIPS_MEAN, LIPS_STD, LEFT_HANDS_MEAN, LEFT_HANDS_STD, POSE_MEAN, POSE_STD = get_normalization_constants()" in line:
            end_idx = i
            break
            
    if start_idx == -1 or end_idx == -1:
        print("Could not find the section to replace in config.py")
        return

    # Construct new content
    new_lines = []
    new_lines.append("# Calculated Normalization Constants\n")
    
    for key, value in stats.items():
        # Format as np.array
        # Value is a list of lists
        arr_str = str(value)
        new_lines.append(f"{key} = np.array({arr_str}, dtype=np.float32)\n")
        
    # Replace lines
    # We replace from start_idx to end_idx (inclusive)
    final_lines = lines[:start_idx] + new_lines + lines[end_idx+1:]
    
    with open(config_path, 'w') as f:
        f.writelines(final_lines)
        
    print("Updated config.py successfully.")

if __name__ == "__main__":
    update_config()
