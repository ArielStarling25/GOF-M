import torch

if torch.cuda.is_available():
    print("CUDA is enabled.")
    # Optionally, print the number of GPUs and their names
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT enabled or accessible by PyTorch.")