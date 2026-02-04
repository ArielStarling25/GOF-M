import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, integrate
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
from datetime import datetime

def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@torch.no_grad()
def evaluate_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=False):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            if return_color:
                color_integrated = ret["color_integrated"]    
                final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated, final_color)
            final_alpha = torch.min(final_alpha, alpha_integrated)
            
        alpha = 1 - final_alpha
    if return_color:
        return alpha, final_color
    return alpha

@torch.no_grad()
def evaluate_alpha2(points, views, gaussians, pipeline, background, kernel_size, return_color=False):
    # Initialize final_alpha (tracks minimum transmittance/transparency)
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    
    # Counter for how many views think this point is "background"
    # This acts as a voting system to handle noise
    background_votes = torch.zeros((points.shape[0]), dtype=torch.int32, device="cuda")

    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            # 1. Standard Opacity Integration (Unchanged)
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            
            if return_color:
                color_integrated = ret["color_integrated"]    
                final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated, final_color)
            
            final_alpha = torch.min(final_alpha, alpha_integrated)
            
            # 2. Binary Mask (Visual Hull) Logic - CORRECTED
            if view.gt_alpha_mask is not None:
                # Transform points to Homogeneous Clip Space
                full_proj = view.full_proj_transform.to(points.device)
                points_hom = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
                p_proj = points_hom @ full_proj
                
                # Convert to NDC (Normalized Device Coordinates)
                p_w = p_proj[:, 3:4]
                p_ndc = p_proj[:, :3] / (p_w + 1e-6)

                # Check Frustum Bounds
                valid_z = p_w[:, 0] > 0.0
                valid_x = (p_ndc[:, 0] >= -1.0) & (p_ndc[:, 0] <= 1.0)
                valid_y = (p_ndc[:, 1] >= -1.0) & (p_ndc[:, 1] <= 1.0)
                in_frustum = valid_z & valid_x & valid_y

                # Convert NDC to Pixel Coordinates
                H, W = view.image_height, view.image_width
                
                # X coordinate: -1 (Left) -> 0, +1 (Right) -> W
                u = ((p_ndc[:, 0] + 1) * W - 1) * 0.5
                
                # Y coordinate: FIX IS HERE
                # OpenGL NDC Y: -1 (Bottom), +1 (Top)
                # Image Pixel Y: 0 (Top), H (Bottom)
                # We must flip Y so +1 maps to 0
                v = ((1.0 - p_ndc[:, 1]) * H - 1) * 0.5
                
                u = u.long()
                v = v.long()
                
                # Clamp to avoid indexing crashes
                u = torch.clamp(u, 0, W - 1)
                v = torch.clamp(v, 0, H - 1)

                # Sample the Mask
                mask = view.gt_alpha_mask.to(points.device)
                if len(mask.shape) == 3:
                    mask_values = mask[0, v, u]
                else:
                    mask_values = mask[v, u]

                # Identify Background Points
                # If point is in frustum AND mask is black (0), it's background
                is_background = (mask_values < 0.5) & in_frustum
                
                # Increment the "strike" counter for these points
                background_votes[is_background] += 1

        # Finalize Alpha
        alpha = 1 - final_alpha

        # --- APPLY FILTER ---
        # "Strict" mode (TSDF style): If even 1 view sees background, delete it.
        # "Robust" mode: If > 2 views see background, delete it (handles mask noise).
        vote_threshold = 1 # Set to 0 for strict TSDF behavior, 2 or 3 for safer/noisier masks
        
        # Force alpha to 0 if enough cameras agree it's background
        alpha[background_votes > vote_threshold] = 0.0

    if return_color:
        return alpha, final_color
    return alpha

@torch.no_grad()
def evaluate_alpha3(points, views, gaussians, pipeline, background, kernel_size, return_color=False):
    # final_alpha tracks the ACCUMULATED OPACITY/OCCUPANCY
    # initialized to 1 (Fully Occupied/Solid) for the 'min' operation to work
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    
    # Track which points have been seen by at least one camera
    seen_mask = torch.zeros((points.shape[0]), dtype=torch.bool, device="cuda")
    
    # Voting for Visual Hull (0 = Inside Object, High = Background)
    background_votes = torch.zeros((points.shape[0]), dtype=torch.int32, device="cuda")

    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            # 1. Standard Opacity Integration
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            
            if return_color:
                color_integrated = ret["color_integrated"]    
                final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated, final_color)
            
            # Space Carving: Keep the 'most transparent' observation
            final_alpha = torch.min(final_alpha, alpha_integrated)
            
            # 2. Visual Hull / Frustum Check
            if view.gt_alpha_mask is not None:
                full_proj = view.full_proj_transform.to(points.device)
                points_hom = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
                p_proj = points_hom @ full_proj
                
                # NDC Conversion
                p_w = p_proj[:, 3:4]
                p_ndc = p_proj[:, :3] / (p_w + 1e-6)

                # Frustum Validity Check
                valid_z = p_w[:, 0] > 0.0
                valid_x = (p_ndc[:, 0] >= -1.0) & (p_ndc[:, 0] <= 1.0)
                valid_y = (p_ndc[:, 1] >= -1.0) & (p_ndc[:, 1] <= 1.0)
                in_frustum = valid_z & valid_x & valid_y
                
                # Update 'Seen' mask
                seen_mask = seen_mask | in_frustum

                # Project to Pixel Coords
                H, W = view.image_height, view.image_width
                u = ((p_ndc[:, 0] + 1) * W - 1) * 0.5
                v = ((1.0 - p_ndc[:, 1]) * H - 1) * 0.5 # Corrected Y-Flip
                
                u = torch.clamp(u.long(), 0, W - 1)
                v = torch.clamp(v.long(), 0, H - 1)

                # Sample Mask
                mask = view.gt_alpha_mask.to(points.device)
                if len(mask.shape) == 3:
                    mask_values = mask[0, v, u]
                else:
                    mask_values = mask[v, u]

                # Identify Background (Points inside frustum but on black mask pixels)
                is_background = (mask_values < 0.5) & in_frustum
                background_votes[is_background] += 1

    # Convert to Transmittance/Vacancy (1 = Empty, 0 = Solid)
    # Because alpha_to_sdf = alpha - 0.5. 
    # If alpha=1 -> SDF=0.5 (Outside). If alpha=0 -> SDF=-0.5 (Inside).
    alpha = 1 - final_alpha
    
    # Points never seen by any camera default to "Solid" in the original logic.
    # We must force them to "Empty" (alpha = 1.0).
    alpha[~seen_mask] = 1.0
    
    # Points voted as background should be "Empty" (alpha = 1.0).
    # (Previously set to 0.0 which made them Solid/Shroud).
    vote_threshold = 1 # Require 2 views to agree it's background to reduce noise
    alpha[background_votes > vote_threshold] = 1.0

    if return_color:
        return alpha, final_color
    return alpha

@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, filter_mesh : bool, texture_mesh : bool, near : float, far : float):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fusion")

    makedirs(render_path, exist_ok=True)
    
    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points(views, near, far)
    # load cell if exists
    if os.path.exists(os.path.join(render_path, "cells.pt")):
        print("load existing cells")
        cells = torch.load(os.path.join(render_path, "cells.pt"))
    else:
        # create cell and save cells
        print("create cells and save")
        cells = cpp.triangulate(points)
        # we should filter the cell if it is larger than the gaussians
        torch.save(cells, os.path.join(render_path, "cells.pt"))
    
    # evaluate alpha
    alpha = evaluate_alpha2(points, views, gaussians, pipeline, background, kernel_size)

    vertices = points.cuda()[None]
    tets = cells.cuda().long()

    print(vertices.shape, tets.shape, alpha.shape)
    def alpha_to_sdf(alpha):    
        sdf = alpha - 0.5
        sdf = sdf[None]
        return sdf
    
    sdf = alpha_to_sdf(alpha)
    
    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()
    
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    
    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    
    n_binary_steps = 8
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        alpha = evaluate_alpha2(mid_points, views, gaussians, pipeline, background, kernel_size)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
    
        points = (left_points + right_points) / 2
        if step not in [7]:
            continue
        
        if texture_mesh:
            _, color = evaluate_alpha2(points, views, gaussians, pipeline, background, kernel_size, return_color=True)
            vertex_colors=(color.cpu().numpy() * 255).astype(np.uint8)
        else:
            vertex_colors=None
        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
        
        # filter
        if filter_mesh:
            mask = (distance <= scale).cpu().numpy()
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)
        
        mesh.export(os.path.join(render_path, f"mesh_binary_search_{step}_{get_current_timestamp()}.ply"))

    # linear interpolation
    # right_sdf *= -1
    # points = (left_points * left_sdf + right_points * right_sdf) / (left_sdf + right_sdf)
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces)
    # mesh.export(os.path.join(render_path, f"mesh_binary_search_interp.ply"))

def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams, filter_mesh : bool, texture_mesh : bool, near : float, far : float):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        
        cams = scene.getTrainCameras()
        print("Starting Marching Cubes + Tetrahedra + Binary Search")
        marching_tetrahedra_with_binary_search(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size, filter_mesh, texture_mesh, near, far)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("--near", default=0.02, type=float)
    parser.add_argument("--far", default=1e6, type=float)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), args.filter_mesh, args.texture_mesh, args.near, args.far)