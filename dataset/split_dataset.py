import os
import shutil

import sys

# Ensure root directory is in sys.path so we can import config
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
sys.path.append(workspace_dir)

from ActionConditionedLSTM.config import (
    DATASET_DIR,
    CATEGORIZED_DATASET_DIR as OUTPUT_BASE_DIR,
    TARGET_ACTIONS
)

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return

    print("Creating category folders...")
    for action_name in TARGET_ACTIONS.values():
        folder_path = os.path.join(OUTPUT_BASE_DIR, action_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"  -> Created/Exists: {folder_path}")

    print("\nScanning and sorting .npy files...")
    
    count_dict = {name: 0 for name in TARGET_ACTIONS.values()}
    ignored_count = 0

    for filename in os.listdir(DATASET_DIR):
        if not filename.endswith(".npy"):
            continue

        # Extract the action code from the filename, e.g., P01G01R01F0001T0064A0101.npy -> 0101
        # Filename structure ends with A[code].npy
        name_no_ext = filename.split('.')[0]
        if 'A' in name_no_ext:
            action_code = name_no_ext.split('A')[-1]
            
            if action_code in TARGET_ACTIONS:
                action_name = TARGET_ACTIONS[action_code]
                src_path = os.path.join(DATASET_DIR, filename)
                dst_path = os.path.join(OUTPUT_BASE_DIR, action_name, filename)
                
                # Copy the file to the categorized folder
                shutil.copy2(src_path, dst_path)
                count_dict[action_name] += 1
            else:
                ignored_count += 1
        else:
            ignored_count += 1

    print("\nSort Summary:")
    for action_name, count in count_dict.items():
        print(f"  {action_name}: {count} files")
    print(f"\nIgnored {ignored_count} files (not belonging to the {len(TARGET_ACTIONS)} target categories).")
    print(f"\nAll {len(TARGET_ACTIONS)} categories have been separated into: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()
