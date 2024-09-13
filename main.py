import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

from task import fetch_sequences, get_reward_sequence
from model import EnergyConstrainedPredictiveCodingModel
from training import SequenceDataset, train
from utils.schedules import ramp_schedule, decreasing_ramp_schedule

# argument parser 
def parse_args():
    parser = argparse.ArgumentParser()

    """ data paths """
    parser.add_argument("--train_path", type=str, default="./datasets/train", help="Path to training (familiar) images")
    parser.add_argument("--test_path", type=str, default="./datasets/test", help="Path to test (novel) images")
    parser.add_argument("--num_train", type=int, default=16, help="Number of training images to use")
    parser.add_argument("--num_test", type=int, default=24, help="Number of test images to use")

    """ sequence setup """
    parser.add_argument("--image_dim", type=int, default=32)
    parser.add_argument("--blank_ts", type=int, default=5, help="Number of blank time steps")
    parser.add_argument("--img_ts", type=int, default=3, help="Number of image time steps")
    parser.add_argument("--num_pres", type=int, default=6, help="Length of sequence, i.e. number of image presentations")
    parser.add_argument("--train_omission_prob", type=float, default=0.0, help="Omission probability for familiar sequences")
    parser.add_argument("--test_omission_prob", type=float, default=0.3, help="Omission probability for novel sequences")

    """ model parameters """
    parser.add_argument("--latent_dim", type=float, default=64)
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--lambda_spatial", type=float, default=1.0)
    parser.add_argument("--lambda_temporal", type=float, default=1.0)
    parser.add_argument("--lambda_energy", type=float, default=1.0)
    parser.add_argument("--lambda_reward", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--perception_only", type=bool, default=False)

    """ training parameters """
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress_mode", type=str, default='batch')

    return parser.parse_known_args()[0]

def main(args):

    if args.seed is None:
        seed = np.random.randint(1001, 9999)
    else:
        seed = args.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # train and test sequences
    args.train_omission_prob = 0.      # no omissions during training
    train_seqs, train_ts, _, test_seqs, test_ts, test_oms = fetch_sequences(args)
    y_dim = train_seqs.shape[-1]

    # train sequences with omissions (for omission analysis)
    args.train_omission_prob = args.test_omission_prob
    train_om_seqs, train_om_ts, train_oms, _, _, _ = fetch_sequences(args)

    # create reward tensor for train and test sequences
    R_train = get_reward_sequence(*train_seqs.shape[:2], train_ts, reward_window=args.img_ts + 2, reward_amount=6.0, action_cost=1.0)
    R_test = get_reward_sequence(*test_seqs.shape[:2], test_ts, reward_window=args.img_ts + 2, reward_amount=6.0, action_cost=1.0)
    R_train_om = get_reward_sequence(*train_seqs.shape[:2], train_om_ts, reward_window=args.img_ts + 2, reward_amount=6.0, action_cost=1.0)

    # image sequences
    Y_train = torch.Tensor(train_seqs).float().to(args.device)
    Y_test = torch.Tensor(test_seqs).float()
    Y_train_om = torch.Tensor(train_om_seqs).float().to(args.device)

    # create data loaders
    train_dataset = SequenceDataset(Y_train, R_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # create model
    model = EnergyConstrainedPredictiveCodingModel(
        input_dim=y_dim,
        latent_dim=args.latent_dim,
        higher_state_dim=args.h_dim,
        lambda_spatial=args.lambda_spatial,
        lambda_temporal=args.lambda_temporal,
        lambda_energy=args.lambda_energy,
        lambda_reward=args.lambda_reward,
        perception_only=args.perception_only
    ).to(args.device)

    # create lambda schedules
    lambda_temporal_sched = ramp_schedule(args.num_epochs, 50, stop=args.lambda_temporal)
    lambda_energy_sched = ramp_schedule(args.num_epochs, 100, stop=args.lambda_energy)
    lambda_rew_sched = ramp_schedule(args.num_epochs, 50, stop=args.lambda_reward)

    # epsilon schedule for greedy exploration
    epsilon_sched = decreasing_ramp_schedule(args.num_epochs, 50, 0.9, 0.001, decay_episodes=args.num_epochs)

    # create optimizer and learning rate schedule
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

    # train
    train(model, opt, train_dataloader,
          lambda_temporal_sched,
          lambda_energy_sched,
          lambda_rew_sched,
          epsilon_sched,
          lr_sched,
          progress_mode=args.progress_mode,
          num_epochs=args.num_epochs,
          device=args.device)
    
    # evaluation
    train_responses, _ = model.forward_sequence(Y_train.to(args.device), R_train.to(args.device), epsilon=0.)
    train_responses_om, _ = model.forward_sequence(Y_train_om.to(args.device), R_train_om.to(args.device), epsilon=0.)
    test_responses, _ = model.forward_sequence(Y_test.to(args.device), R_test.to(args.device), epsilon=0.)

    # indices of the omission trials
    test_om_indcs = np.where(np.array(test_oms) > 0)[0]
    train_om_indcs = np.where(np.array(train_oms) > 0)[0]

    data = {
        "Y_train": Y_train,
        "Y_test": Y_test,
        "Y_train_om": Y_train_om,

        "train_ts": train_ts,
        "test_ts": test_ts,
        "train_om_ts": train_om_ts,

        "train_oms": train_oms,
        "test_oms": test_oms
    }

    output = {
        "train_responses": train_responses,
        "train_responses_om": train_responses_om,
        "test_responses": test_responses,
        "test_om_indcs": test_om_indcs,
        "train_om_indcs": train_om_indcs
    }

    return model, data, output
