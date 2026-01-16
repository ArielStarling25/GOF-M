# training scripts for the nerf-synthetic datasets
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import itertools
from pathlib import Path

scenes = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
# scenes = ["lego"]
# scenes = ['lego', 'materials', 'drums', 'ficus']
# scenes = ["materials"]
# scenes = ['hotdog', 'lego', 'materials', 'drums', 'ficus', 'drums']
# scenes = ['hotdog']

factors = [1]

output_dir = "exp_nerf_synthetic/release"

dataset_dir = "nerf_synthetic"

dry_run = False

set_iterations = 30000

excluded_gpus = set([])

jobs = list(itertools.product(scenes, factors))

def train_scene(gpu, scene, factor):
    current_file_path = Path(__file__).resolve()
    scripts_dir = current_file_path.parent
    project_root = scripts_dir.parent
    #dataset_path = project_root / "datasets" / "360_v2" / scene
    dataset_path = os.path.join(project_root, "datasets", dataset_dir, scene)
    print("Dataset Path set to: ", dataset_path)

    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 train.py -s {dataset_path} -m {output_dir}/{scene} --eval --white_background --port {6209+int(gpu)}"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 render.py -m {output_dir}/{scene} --skip_train"
    print(cmd)
    if not dry_run:
        os.system(cmd)
        
    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 metrics.py -m {output_dir}/{scene}"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 extract_mesh.py -m {output_dir}/{scene} --iteration {set_iterations} --texture_mesh"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    return True

    
def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet
    print("Starting Job Dispatcher... NeRF Synthetic")

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(1)
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

