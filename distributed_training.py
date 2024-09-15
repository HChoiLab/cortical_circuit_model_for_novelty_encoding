import torch
import torch.multiprocessing as mp
import os
from main import parse_args, main
from utils.analysis import process_outputs
import numpy as np

def train_model(seed, gpu_id):

    args = parse_args()

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(gpu_id)

    # Set arguments
    args.perception_only = True
    args.lambda_reward = 0.0
    args.seed = seed
    args.device = torch.device(f"cuda:{gpu_id}")
    args.progress_mode = 'epoch'
    
    # train
    model, data, output = main(args)
    change_responses, omission_responses = process_outputs(args, model, data, output)
    save_dir = "./results/perception_only"
    torch.save({
        "args": vars(args),
        "model": model.state_dict(),
        "change_responses": change_responses,
        "omission_responses": omission_responses
    }, os.path.join(save_dir, f"perception_only_{args.seed}"))

def run_training(gpu_seeds):
    processes = []

    # Launch separate processes for each GPU
    for gpu_id, seed in gpu_seeds.items():
        p = mp.Process(target=train_model, args=(seed, gpu_id))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    # Make sure to initialize the multiprocessing environment
    mp.set_start_method('spawn', force=True)

    # Define which seeds to use for each GPU
    gpu_seeds = {k: np.random.randint(1001, 9999) for k in range(4)}

    run_training(gpu_seeds)
