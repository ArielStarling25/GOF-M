#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
from os import makedirs
from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from datetime import datetime

def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_ply(dataset: ModelParams, iteration: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        # Loading the scene automatically populates 'gaussians' with the checkpoint data
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # Define the output directory and filename
        export_dir = os.path.join(dataset.model_path, "exported_models")
        makedirs(export_dir, exist_ok=True)
        
        output_path = os.path.join(export_dir, f"gaussians_iter_{scene.loaded_iter}_{get_current_timestamp()}.ply")
        
        # Export the Gaussians using the standard 3DGS save method
        print(f"Exporting 3D Gaussians to: {output_path}...")
        gaussians.save_fused_ply(output_path)
        print("Extraction complete. This .ply file is ready for standard 3DGS viewers.")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Extract 3D Gaussians to a portable PLY format")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Extracting model from " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    extract_ply(model.extract(args), args.iteration)