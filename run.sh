#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./run.sh [dtu|mip|nerf|tnt]"
    exit 1
fi

DATASET=$1

# Execute the corresponding python script based on the input
case $DATASET in
    "dtu")
        echo "Starting DTU run..."
        python3 scripts/run_dtu.py
        ;;
    "mip")
        echo "Starting Mip-NeRF 360 run..."
        python3 scripts/run_mipnerf360.py
        ;;
    "nerf")
        echo "Starting NeRF run..."
        python3 scripts/run_nerf_synthetic.py 
        ;;
    "tnt")
        echo "Starting Tanks & Temples run..."
        python3 scripts/run_tnt.py  
        ;;
    *)
        echo "Error: Invalid argument '$DATASET'."
        echo "Please use one of: dtu, mip, nerf, tnt"
        exit 1
        ;;
esac