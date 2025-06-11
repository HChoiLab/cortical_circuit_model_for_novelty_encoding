from torch.utils.data import Dataset
import torch
from torch import nn
import tqdm
from utils.analysis import compute_dprime
import numpy as np

def SequenceCollate(batch):
    
    x = torch.stack([b[0] for b in batch])

    r = None
    if len(batch[0]) > 1:
        r = torch.stack([b[1] for b in batch])
    
    ts = None
    if len(batch[0]) > 2:
        ts = [b[-1] for b in batch]
    
    return x, r, ts

# dataset object to hold data
class SequenceDataset(Dataset):
    
    def __init__(self, data_tensor, rewards_tensor=None, ts_tensor=None):
        self.x = data_tensor
        self.R = rewards_tensor
        self.ts_tensor = ts_tensor
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):

        output = (self.x[idx],)

        if self.R is not None:
            output += (self.R[idx],)
        else:
            output += (None,)
        
        if self.ts_tensor is not None:
            output += (self.ts_tensor[idx],)
        else:
            output += (None,)

        return output


# ------------------------------- Training functions for the predictive coding model -------------------------------------- #
def training_epoch(model, optimizer, dataloader, epoch,
                   lambda_temporal_sched=None,
                   lambda_energy_sched=None,
                   lambda_reward_sched=None,
                   epsilon_sched=None,
                   progress_bar=True,
                   device='cuda',
                   d_prime=False,
                   response_window=None):
    
    lambda_temporal = None if lambda_temporal_sched is None else lambda_temporal_sched[epoch]
    lambda_energy = None if lambda_energy_sched is None else lambda_energy_sched[epoch]
    lambda_reward = None if lambda_reward_sched is None else lambda_reward_sched[epoch]
    epsilon = 0.1 if epsilon_sched is None else epsilon_sched[epoch]

    # Inter-layer feedforward weights become less plastic over time
    if epoch > 25:
        for param in model.posterior_mu.parameters():
            param.requires_grad = False
        for param in model.posterior_sigma.parameters():
            param.requires_grad = False
            
        if epoch > 35:
            for param in model.z_to_h.parameters():
                param.requires_grad = False
            for param in model.z_to_h2.parameters():
                param.requires_grad = False

    
    if progress_bar:
        pbar = tqdm.tqdm(dataloader, unit='batch')
        pbar.set_description(f"Epoch {epoch}")
    else:
        pbar = dataloader

    avg_losses = None
    avg_dprime = 0. if d_prime & (not model.perception_only) else torch.nan
    
    for _, (Y, R, ts) in enumerate(pbar):
        Y = Y.to(device)
        if R is not None:
            R = R.to(device)
        
        # run through the sequence
        responses, losses = model.forward_sequence(Y, R,
                                                   lambda_temporal=lambda_temporal,
                                                   lambda_energy=lambda_energy,
                                                   lambda_reward=lambda_reward,
                                                   epsilon=epsilon)
        total_loss = losses['total']
        
        # step the optimizer 
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        dp = torch.nan
        if d_prime and (not model.perception_only):
            dp = compute_dprime(ts, responses['action'], response_window=response_window)
            avg_dprime += dp
        
        if progress_bar:
            # update the progress bar
            pbar.set_postfix(total=total_loss.item(),
                            spatial=losses['spatial_error'].item(),
                            temporal=losses['temporal_error'].item(),
                            energy=losses['energy'].item(),
                            rewards=losses['episode_rewards'].item(),
                            value=losses['value_loss'].item(),
                            dprime=dp)

        if avg_losses is None:
            avg_losses = {k: v.item() for k, v in losses.items()}
        else:
            avg_losses = {k: avg_losses[k] + v.item() for k, v in losses.items()}
        
    avg_losses.update({'dprime': avg_dprime})
    
    avg_losses = {k: v / len(dataloader) for k, v in avg_losses.items()}

    return avg_losses

def train(model, optimizer, dataloader,
          lambda_temporal_sched=None,
          lambda_energy_sched=None,
          lambda_reward_sched=None,
          epsilon_sched=None,
          lr_sched=None,
          num_epochs=50,
          progress_mode='batch',
          device='cuda',
          d_prime=False,
          response_window=None,
          test_sequences=None):
    
    if progress_mode == 'none':
        pbar = range(num_epochs)
        epoch_bar = False
    elif progress_mode == 'batch':
        pbar = range(num_epochs)
        epoch_bar = True
    elif progress_mode == 'epoch':
        pbar = tqdm.tqdm(range(num_epochs), unit='epoch')
        epoch_bar = False
    else:
        raise
    
    training_progress = []
    dprime_familiar, dprime_novel = [], []
    learning_rates = []
    last_dprime = torch.nan
    
    if lr_sched is not None:
        learning_rates += [lr_sched.get_last_lr()[0]]
    
    for epoch in pbar:
        
        # if tracking dprime, we compute it every 5 epochs for efficiency
        if epoch % 5 != 0:
            dprime_epoch = False
        else:
            dprime_epoch = d_prime and (not model.perception_only)
        
        avg_losses = training_epoch(model, optimizer, dataloader, epoch, lambda_temporal_sched, lambda_energy_sched, lambda_reward_sched, epsilon_sched, device=device,
                                    progress_bar=epoch_bar, d_prime=dprime_epoch, response_window=response_window)
        
        if dprime_epoch:
            last_dprime = avg_losses['dprime']
            dprime_familiar.append(last_dprime)
            
            # compute d_prime for the novel sequences every 5 epochs
            with torch.no_grad():
                test_x, test_r, test_ts = test_sequences
                test_responses, _ = model.forward_sequence(test_x, test_r, epsilon=0.1 if epsilon_sched is None else epsilon_sched[epoch])
                test_dprime = compute_dprime(test_ts, test_responses['action'], response_window=response_window)
                dprime_novel.append(test_dprime)
        
        training_progress.append(avg_losses)
                
        if lr_sched is not None:
            lr_sched.step()
            learning_rates += [lr_sched.get_last_lr()[0]]
        
        if progress_mode == 'epoch':
            pbar.set_postfix(total=avg_losses['total'],
                            spatial=avg_losses['spatial_error'],
                            temporal=avg_losses['temporal_error'],
                            energy=avg_losses['energy'],
                            rewards=avg_losses['episode_rewards'],
                            value=avg_losses['value_loss'],
                            dprime=last_dprime)
            
    training_progress = {
            k: np.stack([d[k] for d in training_progress]) for k in training_progress[0].keys()
        }
    
    training_progress['dprime'] = np.array(dprime_familiar)
    training_progress['dprime_novel'] = np.array(dprime_novel)
    training_progress['lr'] = np.array(learning_rates)

    return training_progress

# ------------------------------- Training functions for the adaptaion model -------------------------------------- #
def adaptation_model_training_epoch(model, dataloader, epoch, optimizer=None, device='cuda', progress_bar=True):

    if progress_bar:
        pbar = tqdm.tqdm(dataloader, unit='batch')
        pbar.set_description(f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    avg_loss = 0.
    for _, (Y, _, _) in enumerate(pbar):
        Y = Y.to(device)
        
        output = model.forward_sequence(Y)
        
        if optimizer is not None:
            optimizer.zero_grad()
            output['MSE'].backward()
            optimizer.step()
            
            if progress_bar:
                pbar.set_postfix(mse=output['MSE'].item())
            
            avg_loss += (output['MSE'] / len(pbar))
    
    return avg_loss

def train_adaptation_model(model, dataloader, optimizer=None, num_epochs=10, device='cuda', progress_mode='batch'):

    if progress_mode == 'none':
        pbar = range(num_epochs)
        epoch_bar = False
    elif progress_mode == 'batch':
        pbar = range(num_epochs)
        epoch_bar = True
    elif progress_mode == 'epoch':
        pbar = tqdm.tqdm(range(num_epochs), unit='epoch')
        epoch_bar = False
    else:
        raise

    training_progress = []
    for epoch in pbar:
        
        avg_loss = adaptation_model_training_epoch(model, dataloader, epoch, optimizer, device, progress_bar=epoch_bar)

        training_progress.append(avg_loss)

        if progress_mode == 'epoch':
            pbar.set_postfix(mse=avg_loss)
        
    return training_progress