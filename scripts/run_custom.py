# training scripts for the TNT datasets
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

# training_list = ['Panther']
# training_list = ['Figurine']
training_list = ['Hilux']

# split = "TrainingSet"
scenes = training_list

factors = [2] * len(scenes)

excluded_gpus = set([])

output_dir = "exp_Custom/release"

dry_run = False
RESULTS_ONLY = False
SEGMENTED = True
MESH_EXTRACT_ONLY = True

set_iterations = 5000

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor=None):
    current_file_path = Path(__file__).resolve()
    scripts_dir = current_file_path.parent
    project_root = scripts_dir.parent
    dataset_path = os.path.join(project_root, "datasets", "custom", scene)
    print("Dataset Path set to: ", dataset_path)

    if not RESULTS_ONLY:
        if not MESH_EXTRACT_ONLY:
            cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 train.py -s {dataset_path} -m {output_dir}/{scene} --eval -i images_{factor} -r {factor} --use_decoupled_appearance"
            if factor == 0:
                cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 train.py -s {dataset_path} -m {output_dir}/{scene} --eval -i images -r 1 --use_decoupled_appearance"
            if SEGMENTED:
                cmd += " --lambda_distortion 1000"
            print(cmd)
            if not dry_run:
                os.system(cmd)

        cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 render.py -m {output_dir}/{scene} --skip_train"
        print(cmd)
        if not dry_run:
            os.system(cmd)

        #fusion
        cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 extract_mesh.py -m {output_dir}/{scene} --iteration {set_iterations} --texture_mesh"
        print(cmd)
        if not dry_run:
            os.system(cmd)

        #tsdf fusion
        cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 extract_mesh_tsdf.py -m {output_dir}/{scene} --iteration {set_iterations}"
        print(cmd)
        if not dry_run:
            os.system(cmd)

    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 metrics.py -m {output_dir}/{scene}"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    # evaluation
    # You need to install open3d==0.9 for evaluation
    # Main issue with TNT is that it takes up way too much memory to store
    # cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/{scene} --traj-path TNT_GOF/TrainingSet/{scene}/{scene}_COLMAP_SfM.log --ply-path {output_dir}/{scene}/test/ours_{set_iterations}/fusion/mesh_binary_search_7.ply"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)

    return True

def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1, maxLoad=0.1))
        # all_available_gpus = set([6,7])
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
        time.sleep(5)
        
    print("All jobs have been processed.")

# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

