from torch.utils.data import Dataset
import tqdm

# dataset object to hold data
class SequenceDataset(Dataset):
    
    def __init__(self, data_tensor, rewards_tensor=None):
        self.x = data_tensor
        self.R = rewards_tensor
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):

        if self.R is not None:
            return self.x[idx], self.R[idx]

        return self.x[idx], None
    

# training functions
def training_epoch(model, optimizer, dataloader, epoch,
                   lambda_temporal_sched=None,
                   lambda_energy_sched=None,
                   lambda_reward_sched=None,
                   epsilon_sched=None,
                   progress_bar=True,
                   device='cuda'):
    
    lambda_temporal = None if lambda_temporal_sched is None else lambda_temporal_sched[epoch]
    lambda_energy = None if lambda_energy_sched is None else lambda_energy_sched[epoch]
    lambda_reward = None if lambda_reward_sched is None else lambda_reward_sched[epoch]
    epsilon = 0.1 if epsilon_sched is None else epsilon_sched[epoch]
    
    if progress_bar:
        pbar = tqdm.tqdm(dataloader, unit='batch')
        pbar.set_description(f"Epoch {epoch}")
    else:
        pbar = dataloader

    avg_losses = None
    
    for _, (Y, R) in enumerate(pbar):
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
        
        if progress_bar:
            # update the progress bar
            pbar.set_postfix(total=total_loss.item(),
                            spatial=losses['spatial_error'].item(),
                            temporal=losses['temporal_error'].item(),
                            energy=losses['energy'].item(),
                            rewards=losses['episode_rewards'].item(),
                            value=losses['value_loss'].item())

        if avg_losses is None:
            avg_losses = {k: v.item() for k, v in losses.items()}
        else:
            avg_losses = {k: avg_losses[k] + v.item() for k, v in losses.items()}
    
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
          device='cuda'):
    
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

    for epoch in pbar:
        
        avg_losses = training_epoch(model, optimizer, dataloader, epoch, lambda_temporal_sched, lambda_energy_sched, lambda_reward_sched, epsilon_sched, device=device,
                                    progress_bar=epoch_bar)
        
        if lr_sched is not None:
            lr_sched.step()
        
        if progress_mode == 'epoch':
            pbar.set_postfix(total=avg_losses['total'],
                            spatial=avg_losses['spatial_error'],
                            temporal=avg_losses['temporal_error'],
                            energy=avg_losses['energy'],
                            rewards=avg_losses['episode_rewards'],
                            value=avg_losses['value_loss'])