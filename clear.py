#!/usr/bin/env python3
"""
clear.py

Deletes all files and folders inside the `plots`, `data`, and `pred_analysis` directories.
"""
import os
import shutil

def clear_directory(dir_path):
    """
    Remove all files and subdirectories in the given directory.
    """
    if not os.path.isdir(dir_path):
        return
    for entry in os.listdir(dir_path):
        entry_path = os.path.join(dir_path, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.unlink(entry_path)
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
        except Exception as e:
            print(f"Failed to delete {entry_path}: {e}")

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    folders = ["plots", "data", "metrics"]
    for folder in folders:
        path = os.path.join(root, folder)
        print(f"Clearing contents of {path}...")
        clear_directory(path)
    print("All specified directories have been cleared.")

if __name__ == "__main__":
    main()
