import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from task import fetch_sequences, get_reward_sequence
from model import EnergyConstrainedPredictiveCodingModel
from baselines import AdaptationModel
from training import SequenceDataset, train, SequenceCollate, train_adaptation_model
from utils.schedules import ramp_schedule, decreasing_ramp_schedule

# argument parser 
def parse_args():
    parser = argparse.ArgumentParser()

    """ data paths """
    parser.add_argument("--train_path", type=str, default="./datasets/train", help="Path to training (familiar) images")
    parser.add_argument("--test_path", type=str, default="./datasets/test", help="Path to test (novel) images")
    parser.add_argument("--num_train", type=int, default=8, help="Number of training images to use")
    parser.add_argument("--num_test", type=int, default=12, help="Number of test images to use")

    """ sequence setup """
    parser.add_argument("--image_dim", type=int, default=32)
    parser.add_argument("--blank_ts", type=int, default=4, help="Number of blank time steps")
    parser.add_argument("--img_ts", type=int, default=2, help="Number of image time steps")
    parser.add_argument("--num_pres", type=int, default=8, help="Length of sequence, i.e. number of image presentations")
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
    parser.add_argument("--calculate_dprime", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress_mode", type=str, default='batch')
    
    # when to start optimizing different objective function
    parser.add_argument("--temporal_start", type=int, default=50)
    parser.add_argument("--energy_start", type=int, default=75)
    parser.add_argument("--value_start", type=int, default=10)

    # whether we running the predictive coding model or the adaptation model baseline
    parser.add_argument("--adaptation_model", type=bool, default=False)

    return parser.parse_known_args()[0]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _adaptation_main(args, Y_train, Y_train_om, Y_test, train_dataloader):

    # create adaptation model
    model = AdaptationModel(num_neurons_E=32,
                            num_neurons_VIP=32,
                            num_neurons_SST=32,
                            num_inputs=Y_train.shape[-1],
                            tau_E=.3,
                            tau_VIP=.3,
                            tau_SST=.3,
                            tau_A=1.,
                            g_E=20.0,
                            g_VIP=20.0,
                            g_SST=10.0,
                            eta=0.00001,
                            alpha=0.001,
                            dt=0.1,
                            hebbian=False,
                            device=args.device).to(args.device)

    training_progress = {"loss": -1. * np.ones((args.num_epochs,))}
    if model.hebbian:
        training_progress = train_adaptation_model(model, train_dataloader,
                                                   num_epochs=args.num_epochs,
                                                   device=args.device,
                                                   progress_mode=args.progress_mode)
        training_progress = {"loss": np.array(training_progress)}

    train_responses = model.forward_sequence(Y_train.to(args.device))
    test_responses = model.forward_sequence(Y_test.to(args.device))
    train_responses_om = model.forward_sequence(Y_train_om.to(args.device))

    return model, train_responses, train_responses_om, test_responses, training_progress

def main(args):

    if args.seed is None:
        seed = np.random.randint(1001, 9999)
    else:
        seed = args.seed
    
    set_seed(seed)

    # train and test sequences
    args.train_omission_prob = 0.      # no omissions during training
    train_seqs, train_ts, _, test_seqs, test_ts, test_oms = fetch_sequences(args, seed)
    y_dim = train_seqs.shape[-1]

    # train sequences with omissions (for omission analysis)
    args.train_omission_prob = args.test_omission_prob
    train_om_seqs, train_om_ts, train_oms, _, _, _ = fetch_sequences(args, seed)

    # create reward tensor for train and test sequences
    rew_window = (0, args.img_ts + args.blank_ts)
    R_train = get_reward_sequence(*train_seqs.shape[:2], train_ts, reward_window=rew_window, reward_amount=10.0, action_cost=2.0)
    R_test = get_reward_sequence(*test_seqs.shape[:2], test_ts, reward_window=rew_window, reward_amount=10.0, action_cost=2.0)
    R_train_om = get_reward_sequence(*train_seqs.shape[:2], train_om_ts, reward_window=rew_window, reward_amount=10.0, action_cost=2.0)

    # indices of the omission trials
    test_om_indcs = np.where(np.array(test_oms) > 0)[0]
    train_om_indcs = np.where(np.array(train_oms) > 0)[0]

    # image sequences
    Y_train = torch.Tensor(train_seqs).float().to(args.device)
    Y_test = torch.Tensor(test_seqs).float()
    Y_train_om = torch.Tensor(train_om_seqs).float().to(args.device)

    # create data loaders
    train_dataset = SequenceDataset(Y_train, R_train, train_ts)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=SequenceCollate)

    if not args.adaptation_model:
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
        lambda_temporal_sched = ramp_schedule(args.num_epochs, args.temporal_start, stop=args.lambda_temporal, stop_epoch=args.num_epochs)
        lambda_energy_sched = ramp_schedule(args.num_epochs, args.energy_start, stop=args.lambda_energy, stop_epoch=args.num_epochs)
        lambda_rew_sched = ramp_schedule(args.num_epochs, args.value_start, start=args.lambda_reward/10., stop=args.lambda_reward, stop_epoch=args.num_epochs)

        # epsilon schedule for greedy exploration
        epsilon_sched = decreasing_ramp_schedule(args.num_epochs, args.value_start, 0.5, 0.01, stop_epoch=args.num_epochs-10)
        
        # create optimizer and learning rate schedule
        opt = model.get_optimizer(lr=args.lr)
        lr_sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

        # train
        calc_dprime = args.calculate_dprime
        training_progress = train(model, opt, train_dataloader,
                                lambda_temporal_sched,
                                lambda_energy_sched,
                                lambda_rew_sched,
                                epsilon_sched,
                                lr_sched,
                                progress_mode=args.progress_mode,
                                num_epochs=args.num_epochs,
                                device=args.device,
                                d_prime=calc_dprime,
                                response_window=rew_window,
                                test_sequences=(Y_test.to(args.device), R_test.to(args.device), test_ts))
        
        # evaluation
        train_responses, _ = model.forward_sequence(Y_train.to(args.device), R_train.to(args.device), epsilon=0.)
        train_responses_om, _ = model.forward_sequence(Y_train_om.to(args.device), R_train_om.to(args.device), epsilon=0.)
        test_responses, _ = model.forward_sequence(Y_test.to(args.device), R_test.to(args.device), epsilon=0.)
    
    else:
        model, train_responses, train_responses_om, test_responses, training_progress = _adaptation_main(args,
                                                                                                         Y_train,
                                                                                                         Y_train_om,
                                                                                                         Y_test,
                                                                                                         train_dataloader)

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

    return model, data, output, training_progress

if __name__ == '__main__':
    main()
