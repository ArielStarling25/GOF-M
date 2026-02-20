import torch
import matplotlib.pyplot as plt
import numpy as np

# Path to your depth file
depth_file_path = "datasets/dtu/scan37/depths/0000.pt"

try:
    # Load the dictionary
    data_dict = torch.load(depth_file_path, map_location="cpu")
    
    print(f"File loaded. It contains a {type(data_dict)}.")
    print("-" * 30)
    print("Available Keys:")
    
    # Loop through keys to show what they are
    for key, value in data_dict.items():
        # Check if the value is a tensor to print its shape
        if torch.is_tensor(value):
            print(f"Key: '{key}' | Type: Tensor | Shape: {value.shape} | Dtype: {value.dtype}")
        else:
            print(f"Key: '{key}' | Type: {type(value)} | Value: {value}")
except Exception as e:
    print(f"Error: {e}")

# DTU is typically 1200 (height) x 1600 (width)
HEIGHT, WIDTH = 1162, 1554

try:
    data = torch.load(depth_file_path, map_location="cpu")
except Exception as e:
    print(e)

# 1. Extract data (convert to numpy if they aren't already)
coords = data['coord']  # Shape: (N, 2) -> (x, y)
depths = data['depth']  # Shape: (N,)
weights = data['weight'] # Shape: (N,)

# 2. Create an empty depth map (initialize with 0)
depth_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

# 3. Create a mask to filter low-confidence points (Optional but recommended)
# This keeps points with weight > 0.1 (tune this threshold based on your data)
valid_mask = weights > 0.000000000000000000000000000001

filtered_coords = coords[valid_mask]
filtered_depths = depths[valid_mask]

# 4. Round coordinates to nearest integer to use as array indices
# coords are usually (x, y) -> (width_index, height_index)
# Numpy expects [row, col] -> [y, x]
pixel_x = np.round(filtered_coords[:, 0]).astype(int)
pixel_y = np.round(filtered_coords[:, 1]).astype(int)

# 5. Boundary check (ignore points that fall outside image size)
valid_indices = (pixel_x >= 0) & (pixel_x < WIDTH) & \
                (pixel_y >= 0) & (pixel_y < HEIGHT)

pixel_x = pixel_x[valid_indices]
pixel_y = pixel_y[valid_indices]
final_depths = filtered_depths[valid_indices]

# 6. Fill the map
# Note: standard image convention is map[y, x]
depth_map[pixel_y, pixel_x] = final_depths

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.imshow(depth_map, cmap='inferno', interpolation='nearest')
plt.colorbar(label='Depth')
plt.title(f"Reconstructed Sparse Depth Map ({len(final_depths)} points)")
plt.show()