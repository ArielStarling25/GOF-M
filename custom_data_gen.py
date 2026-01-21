# training scripts for the TNT datasets
import os
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

# NOTE: Current implementation is for windows environment, not linux

DRY_RUN = False

# INPUT_DIR = os.path.join("custom_datasets", "Panther")
INPUT_DIR = "D:/Masters-Alt/GOF-M/custom_datasets/Panther"

COLMAP_DIR = "D:/Masters-Alt/COLMAP/COLMAP.bat"

MAGICK_DIR = "D:/Masters-Alt/ImageMagick/magick.exe"

def run_script():
    cmd = f"python convert.py -s {INPUT_DIR} --colmap_executable {COLMAP_DIR} --magick_executable {MAGICK_DIR} --resize"
    print(cmd)
    if not DRY_RUN:
        os.system(cmd)

run_script()