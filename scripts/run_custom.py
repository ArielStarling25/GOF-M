# training scripts for the TNT datasets
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from datetime import datetime

# training_list = ['Panther']
# training_list = ['Figurine']
# training_list = ['Hilux']
# training_list = ['Panther_1000', 'Panther']
training_list = [
    os.path.join("refnerf", "ball"), 
    os.path.join("refnerf", "car"),
    os.path.join("refnerf", "coffee"),
    os.path.join("refnerf", "helmet"),
    os.path.join("refnerf", "teapot"),
    ]

# split = "TrainingSet"
scenes = training_list

factors = [2] * len(scenes)

excluded_gpus = set([])

output_dir = "exp_Custom/release"

log_dir = os.path.join(output_dir, "run_logs")

dry_run = False
RESULTS_ONLY = False
MESH_EXTRACT_ONLY = False
ENABLE_MASK = False

set_iterations = 30000

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
            if ENABLE_MASK:
                cmd += " --lambda_distortion 1000 --enable_mask"
            print(cmd)
            if not dry_run:
                os.system(cmd)

        cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 render.py -m {output_dir}/{scene} --skip_train"
        print(cmd)
        if not dry_run:
            os.system(cmd)


        cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 extract_ply.py -m {output_dir}/{scene}"
        print(cmd)
        if not dry_run:
            os.system(cmd)

        ## fusion
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

def dispatch_jobs(jobs, executor, excluded_gpus=None):
    if excluded_gpus is None:
        excluded_gpus = set()

    os.makedirs(log_dir, exist_ok=True)
    
    # Create file
    run_timestamp = datetime.now().strftime("%d%m%y_run_%H%M%S")
    log_file_path = os.path.join(log_dir, f"{run_timestamp}.txt")
    
    print(f"Logging run statistics to: {log_file_path}")

    future_to_job = {}
    reserved_gpus = set()
    completed_jobs_stats = [] # List to store dicts of finished job data
    
    total_start_perf = time.perf_counter()

    # Helper function to generate table and save/print
    def report_stats(current_stats, final=False):
        # Define column widths
        w_name, w_dur, w_time = 30, 15, 22
        
        # Build Table Header
        header = (f"{'Job Name':<{w_name}} | {'Duration (s)':<{w_dur}} | "
                  f"{'Start Time':<{w_time}} | {'End Time':<{w_time}}")
        separator = "-" * len(header)
        lines = ["\n" + separator, header, separator]
        for stat in current_stats:
            lines.append(f"{str(stat['name']):<{w_name}} | "
                         f"{stat['duration']:<{w_dur}.2f} | "
                         f"{stat['start']:<{w_time}} | "
                         f"{stat['end']:<{w_time}}")
        lines.append(separator + "\n")
        report_text = "\n".join(lines)
        print(report_text)
        with open(log_file_path, "a", encoding="utf-8") as f:
            status = "FINAL SUMMARY" if final else "INTERMEDIATE UPDATE"
            f.write(f"\n[{status} - {datetime.now().strftime('%H:%M:%S')}]\n")
            f.write(report_text)

    while jobs or future_to_job:
        # Get available GPUs
        try:
            # Added try/except for local testing without GPUs
            all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1, maxLoad=0.1))
        except Exception:
            # Fallback for testing/debugging if GPUtil fails or no GPU found
            all_available_gpus = set() 

        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            
            start_perf = time.perf_counter()
            start_dt = datetime.now()
            
            future = executor.submit(worker, gpu, *job)
            
            future_to_job[future] = {
                "gpu": gpu, 
                "job": job, 
                "start_perf": start_perf,
                "start_dt": start_dt
            }
            reserved_gpus.add(gpu)

        # Check for completed jobs
        done_futures = [future for future in future_to_job if future.done()]
        
        for future in done_futures:
            data = future_to_job.pop(future)
            gpu = data["gpu"]
            job = data["job"]
            start_perf = data["start_perf"]
            start_dt = data["start_dt"]
            
            # Calculate timings
            end_perf = time.perf_counter()
            end_dt = datetime.now()
            duration = end_perf - start_perf
            reserved_gpus.discard(gpu)
            try:
                future.result()
            except Exception as exc:
                print(f"Job {job} generated an exception: {exc}")
            
            time_fmt = "%d/%m/%y | %H:%M:%S"
            
            completed_jobs_stats.append({
                "name": job,
                "duration": duration,
                "start": start_dt.strftime(time_fmt),
                "end": end_dt.strftime(time_fmt)
            })
            
            print(f"Job {job} finished. Releasing GPU {gpu}")
            
            # REPORTING: Print table and save to log after every job
            report_stats(completed_jobs_stats)

        time.sleep(5)
    
    total_duration = time.perf_counter() - total_start_perf
    
    print("-" * 40)
    print("All jobs have been processed.")
    print(f"SUMMARY: Total time taken for the entire batch: {total_duration:.2f} seconds.")
    print("-" * 40)
    
    # Final write to log with total batch duration note
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"\nBATCH COMPLETE. Total Duration: {total_duration:.2f} seconds.\n")

# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

