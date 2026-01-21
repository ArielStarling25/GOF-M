# training scripts for the TNT datasets
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

dry_run = False

def run_script():
    cmd = f"python3 convert.py "
    print(cmd)
    if not dry_run:
        os.system(cmd)