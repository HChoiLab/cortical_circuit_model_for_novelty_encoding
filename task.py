from utils import data as dutils
from utils.data import get_sequences
from utils import stimuli as sutils
import numpy as np
import torch
import os

# functions to retrieve training data
def fetch_sequences(args):
    
    # load the images
    all_train = dutils.load_images_to_numpy_array(args.train_path, grayscale=True, num_images=len(os.listdir(args.train_path))) / 255
    test_images = dutils.load_images_to_numpy_array(args.test_path, grayscale=True, num_images=args.num_test) / 255
    
    train_images = all_train[:args.num_train]
    
    # normalize images
    img_mean = np.mean(all_train, axis=0, keepdims=True)
    img_std = np.std(all_train, axis=0, keepdims=True)
    
    train_images = (train_images - img_mean) / img_std
    test_images = (test_images - img_mean) / img_std

    # processing params
    process_params = {
        "method": 'downsample',
        "num_pcs": 3,
        "vectorize": True,
        "new_size": (args.image_dim, args.image_dim),
        "downsample_mode": 'avg_pooling'  
    }

    # process images
    train_inds = np.random.choice(len(train_images), size=(args.num_train,), replace=False)
    test_inds = np.random.choice(len(test_images), size=(args.num_test,), replace=False)
    train_images = list(sutils.extract_features(train_images[train_inds], **process_params))
    test_images = list(sutils.extract_features(test_images[test_inds], **process_params))

    train_seqs, train_ts, train_oms = get_sequences(train_images,
                                                    blank_ts=args.blank_ts,
                                                    pres_ts=args.img_ts,
                                                    num_presentations=args.num_pres,
                                                    omission_prob=args.train_omission_prob)
    test_seqs, test_ts, test_oms = get_sequences(test_images, 
                                                 blank_ts=args.blank_ts,
                                                 pres_ts=args.img_ts,
                                                 num_presentations=args.num_pres,
                                                 omission_prob=args.test_omission_prob)

    return train_seqs, train_ts, train_oms, test_seqs, test_ts, test_oms 

def get_reward_sequence(batch_size, seq_len, seq_ts,
                        reward_window=6,
                        reward_amount=1.0,
                        action_cost=0.5):
    # for each sequence, creates a reward sequence with the reward an agent gets if it licks at time t
    
    reward_seq = torch.ones((batch_size, seq_len), requires_grad=False) * (-action_cost)
    for s in range(len(reward_seq)):
        
        change_time = seq_ts[s]['after'][0][0]
        reward_seq[s, change_time:change_time+reward_window] = reward_amount - action_cost
    
    return reward_seq