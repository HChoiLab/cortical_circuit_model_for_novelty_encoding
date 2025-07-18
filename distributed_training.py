import torch
import torch.multiprocessing as mp
import os
import main
from utils.analysis import process_outputs
import numpy as np
import argparse

SCRATCH = "."# "/storage/ice1/7/2/asharafeldin3"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", action='store_true', default=False)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--adaptation_model", type=bool, default=False)
    
    return parser.parse_known_args()[0]

# ------ training function for the predictive coding model -------- #
def train_model(seed, gpu_id, action=False):

    args = main.parse_args()

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(gpu_id)

    # Set arguments
    args.train_path = os.path.join(SCRATCH, "datasets/train")
    args.test_path = os.path.join(SCRATCH, "datasets/test")
    args.seed = seed
    args.device = torch.device(f'cuda:{gpu_id}')
    args.progress_mode = 'epoch'
    args.num_epochs = 70
    args.value_start = 1
    args.temporal_start = 15
    args.energy_start = 25
    args.lr = 0.001
    args.calculate_dprime = True
    
    if not action:
        args.perception_only = True
        args.lambda_energy = 2.0
        args.lambda_temporal = 0.5
        args.lambda_spatial = 1.0
        args.lambda_reward = 0.0
    else:
        args.perception_only = False
        args.lambda_energy = 2.0
        args.lambda_temporal = 0.5
        args.lambda_spatial = 1.0
        args.lambda_reward = 0.1
    
    # train
    model, data, output, training_progress = main.main(args)
    change_responses, omission_responses = process_outputs(args, model, data, output, abridged=True)
    save_dir = "results/experimental" if not action else "./results/temp"
    #save_dir = os.path.join(SCRATCH, "novelty_encoding_model/" + save_dir)
    save_prefix = "perception_only" if not action else "perception_action"
    torch.save({
        "args": vars(args),
        "model": model.state_dict(),
        "change_responses": change_responses,
        "omission_responses": omission_responses,
        "training_progress": training_progress
    }, os.path.join(save_dir, f"{save_prefix}_{args.seed}"))

# --------- training function for the adaptation baseline model ---------- #
def train_adaptation_model(seed, gpu_id):

    args = main.parse_args()

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(gpu_id)

    # Task parameters for the adaptation model
    args.blank_ts = 10
    args.img_ts = 5

    # Set arguments
    args.train_path = os.path.join(SCRATCH, "datasets/train")
    args.test_path = os.path.join(SCRATCH, "datasets/test")
    args.seed = seed
    args.device = torch.device(f'cuda:{gpu_id}')
    args.progress_mode = 'epoch'
    args.num_epochs = 20

    args.perception_only = True           # adaptation baseline doesn't do action
    args.adaptation_model = True

    # train
    model, data, output, training_progress = main.main(args)
    change_responses, omission_responses = process_outputs(args, model, data, output, abridged=True)

    save_dir = "results/adaptation_baseline"
    save_prefix = "no_hebbian"
    torch.save({
        "args": vars(args),
        "model": model.state_dict(),
        "change_responses": change_responses,
        "omission_responses": omission_responses,
        "training_progress": training_progress
    }, os.path.join(save_dir, f"{save_prefix}_{args.seed}"))

def run_training(gpu_seeds, action=False, adaptation=False):
    processes = []

    # Launch separate processes for each GPU
    for gpu_id, seed in gpu_seeds.items():
        if not adaptation:
            p = mp.Process(target=train_model, args=(seed, gpu_id, action))
        else:
            p = mp.Process(target=train_adaptation_model, args=(seed, gpu_id))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    
    local_args = parse_args()
    for _ in range(local_args.num_runs):
        # Make sure to initialize the multiprocessing environment
        mp.set_start_method('spawn', force=True)
        
        num_gpus = torch.cuda.device_count()

        # Define which seeds to use for each GPU
        gpu_seeds = {k: np.random.randint(1001, 9999) for k in range(num_gpus)}

        run_training(gpu_seeds, action=local_args.action, adaptation=local_args.adaptation_model)