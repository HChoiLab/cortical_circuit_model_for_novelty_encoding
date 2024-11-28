import torch
from torch import nn
import math

EPS = 1e-10

# Parametrization classes to constrain weights
class Positive(nn.Module):
    def forward(self, X):
        return torch.relu(X)

class Negative(nn.Module):
    def forward(self, X):
        return -torch.relu(-X)
    
class StableRecurrent(nn.Module):
    MAX_NORM = 0.5
    def forward(self, X):
        norm = X.norm()
        if norm > self.MAX_NORM:
            return X * (self.MAX_NORM / norm)
        return X
    
        
class EnergyConstrainedPredictiveCodingModel(nn.Module):
    
    def __init__(self, input_dim, latent_dim, higher_state_dim,
                lambda_spatial=1.0,
                lambda_temporal=1.0,
                lambda_energy=1.0,
                lambda_reward=1.0,
                perception_only=True, 
                bbtt_chunk=None):           # TODO
        
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.higher_state_dim = higher_state_dim
        self.perception_only = perception_only
        self.bbtt_chunk = bbtt_chunk
        
        self.lambda_spatial = lambda_spatial            # regularizing scalar for the spatial error loss
        self.lambda_temporal = lambda_temporal          # regularizing scalar for the temporal error loss
        self.lambda_energy = lambda_energy              # regularizing scalar for the energy efficiency cost
        self.lambda_reward = lambda_reward              # regularizing scalar for the reward prediction loss
        
        # connections from input to latents
        self.posterior_mu = nn.Linear(input_dim, latent_dim, bias=False)                # encodes input to mean of posterior
        self.posterior_sigma = nn.Linear(input_dim, latent_dim, bias=False)            # encodes input to the variance of the posterior
        
        # connections to higher area
        self.z_to_h = nn.Linear(latent_dim, higher_state_dim, bias=False)                         # connections from latents to higher area
        self.h_to_h = nn.Linear(higher_state_dim, higher_state_dim, bias=False)                   # connections from higher area to itself
        self.h2_to_h2 = nn.Linear(higher_state_dim, higher_state_dim, bias=False)
        self.z_to_h2 = nn.Linear(latent_dim, higher_state_dim, bias=False)
        
        # connections from higher area
        self.prior_mu = nn.Linear(higher_state_dim, latent_dim, bias=False)                # encodes prior distribution mean (represented by an SST subpop)
        self.prior_sigma = nn.Linear(higher_state_dim, latent_dim, bias=True)              # encodes prior variance (represented by VIP neurons)
        
        # connections involved in the computation of theta
        self.theta_dim = latent_dim
        self.vip_to_theta = nn.Linear(latent_dim, self.theta_dim, bias=False)
        self.theta_to_z = nn.Linear(self.theta_dim, latent_dim, bias=False)
        self.I_to_theta = nn.Linear(input_dim, self.theta_dim, bias=False)                 # this is the main source of input drive to the sst theta population
    
        # connections to reconstruct input from z activity
        self.reconstruction = nn.Sequential(nn.Linear(latent_dim, 256, bias=False),
                                            nn.Linear(256, input_dim, bias=False))
        
        # decision making and value learning
        self.value_dim = higher_state_dim
        self.h_to_value = nn.Linear(higher_state_dim, self.value_dim)
        self.value_to_value = nn.Linear(self.value_dim, self.value_dim)
        
        # feedforward connections to SST neurons are not learnable
        for param in self.I_to_theta.parameters():
            param.requires_grad = False
            
        self.initialize_params()
        
        # weight constraints 
        # these weights should be inhibitory; we constrain them to be positive here but then 
        # multiply them by -1 when they're used for computation 
        nn.utils.parametrize.register_parametrization(self.vip_to_theta, 'weight', Positive())
        nn.utils.parametrize.register_parametrization(self.theta_to_z, 'weight', Positive())
        nn.utils.parametrize.register_parametrization(self.prior_sigma, 'bias', Positive())
        
        # this makes sure the recurrent connections between the first higher area are stable
        nn.utils.parametrize.register_parametrization(self.h_to_h, 'weight', StableRecurrent())
    
    def initialize_params(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        
        self.apply(init_weights)
        
        # specific initialization constraints
        nn.init.normal_(self.I_to_theta.weight, mean=1e-2, std=1e-3)
        nn.init.normal_(self.vip_to_theta.weight, mean=1e-1, std=1e-2)
        nn.init.normal_(self.theta_to_z.weight, mean=.2, std=1e-2)         
        nn.init.normal_(self.prior_sigma.bias, mean=5., std=0.1)
    
    def compute_reward_loss(self, R, actions, values):
        
        rewards = R * actions.squeeze().detach()
        
        value_loss = ((rewards - values.squeeze()) ** 2).mean()
        
        return value_loss, rewards.sum(-1)

    def act_with_exploration(self, lick_prob, epsilon):

        best_action = (lick_prob > 0.5).float().detach()
        random_action = torch.randint_like(best_action, 0, 2).float()
        u_sample = torch.rand_like(best_action)
        action = (u_sample < epsilon).float() * random_action + (u_sample >= epsilon).float() * best_action

        return action      
    
    def forward(self, I_t, responses_m_1, epsilon=0.1):      
        
        # get prior from previous higher area state
        mu_p = nn.functional.relu(self.prior_mu(responses_m_1['h2']))
        
        sigmap_h = self.prior_sigma(responses_m_1['h']) 
        sigma_p = 0.8 * torch.relu(sigmap_h) + 0.2 * responses_m_1['sigma_p']
            
        # compute thetas
        vip_inh = self.vip_to_theta(sigma_p)
        theta_ff = 0.4 * responses_m_1['theta_ff'] + torch.exp(-50 * responses_m_1['theta_ff'].abs()) * self.I_to_theta(I_t)
        theta_ff = torch.tanh(theta_ff)**2
        theta_h = theta_ff / (1 + vip_inh) #torch.sigmoid(-vip_inh)
        theta = 0.1 * responses_m_1['theta'] + theta_h
        
        # encode input to the posterior parameters
        mu_q = torch.relu(self.posterior_mu(I_t))
        sigma_q = torch.relu(self.posterior_sigma(I_t))
        
        # compute pyramidal activities (inferred z's)
        raw_z = mu_q + torch.randn_like(sigma_q) * (torch.sigmoid(0.01 * sigma_q) - 0.5)
        raw_z = torch.relu(torch.tanh(raw_z))

        # inhibitory input from SST to excitatory activity
        sst_inhibition = 0.8 * responses_m_1['sst_inh'] + self.theta_to_z(theta)
        z = nn.functional.relu(raw_z - sst_inhibition)
        z_energy = nn.functional.relu(raw_z.detach() - sst_inhibition)
        
        # compute reconstruction (top down prediction from the representation layer)
        I_hat = torch.sigmoid(self.reconstruction(z) - 2.0)
        
        # compute temporal prediction (top down prediction from the higher layer)
        z_hat = mu_p + torch.randn_like(mu_p) * sigma_p
        
         # evolve higher area activities
        h = torch.relu(self.z_to_h(z) + self.h_to_h(responses_m_1['h']))
        h2 = torch.relu(self.z_to_h2(z) + self.h2_to_h2(responses_m_1['h2']))

        # error population activities
        layer_1_error = (I_t - I_hat) ** 2
        layer_2_error = (z - z_hat) ** 2

        # decision making if we are doing action
        if not self.perception_only:
            value = self.h_to_value(h) + self.value_to_value(responses_m_1['value'])
            lick_value = torch.tanh(value[:, :self.value_dim//2].mean(-1, keepdim=True))
            nolick_value = torch.tanh(value[:, self.value_dim//2:].mean(-1, keepdim=True))

            # probability of licking 
            lick_prob = torch.exp(lick_value) / (torch.exp(lick_value) + torch.exp(nolick_value))
            
            # select action with exploration 
            action = self.act_with_exploration(lick_prob, epsilon)
            
            # predicted value of action
            action_value = action * lick_value + (1 - action) * nolick_value
            
            rl_gain =  0.2 * responses_m_1['rl_gain'] + 3. * torch.exp(-1e3 * responses_m_1['rl_gain']) * action_value.detach()
            rl_gain = torch.relu(rl_gain)
            
            # Update VIP activities based on RL gain modulation
            sigmap_h += rl_gain
            sigma_p = 0.8 * torch.relu(sigmap_h) + 0.2 * responses_m_1['sigma_p']

        # compute losses
        spatial_error_loss = layer_1_error.mean()
        temporal_error_loss = layer_2_error.mean()
        energy_loss = torch.abs(z_energy).mean()

        # put together all responses
        responses_t = {
            "mu_p": mu_p,
            "sigmap_h": sigmap_h,
            "sigma_p": sigma_p,
            "mu_q": mu_q,
            "sigma_q": sigma_q,
            "theta_ff": theta_ff,
            "theta_h": theta_h,
            "theta": theta,
            "vip_inh": vip_inh,
            "sst_inh": sst_inhibition,
            "z_h": raw_z,
            "z": z,
            "I_hat": I_hat,
            "temp_error": layer_2_error,
            "h": h,
            "h2": h2
        }

        if not self.perception_only:
            responses_t.update({
                "action": action,
                "value": value,
                "action_value": action_value,
                "lick_value": lick_value,
                "rl_gain": rl_gain
            })

        losses_t = {
            "spatial_error_loss": spatial_error_loss,
            "temporal_error_loss": temporal_error_loss,
            "energy_loss": energy_loss
        }
        
        return responses_t, losses_t
    
    def forward_sequence(self, Y, R=None, lambda_spatial=None, lambda_temporal=None, lambda_energy=None, lambda_reward=None, epsilon=None):
        
        T = Y.shape[1]
        
        sequence_responses = []
        
        mean_total, mean_spatial, mean_temporal, mean_energy = 0., 0., 0., 0.
        
        # get initial states
        responses_0 = self.init_state(Y.shape[0], Y.device)
        I_0 = torch.zeros_like(Y[:, 0])

        if lambda_spatial is None:
            lambda_spatial = self.lambda_spatial
        if lambda_temporal is None:
            lambda_temporal = self.lambda_temporal
        if lambda_energy is None:
            lambda_energy = self.lambda_energy
        if lambda_reward is None:
            lambda_reward = self.lambda_reward
        
        # go through the loop
        for t in range(T):                    
            
            I_t = I_0 if t == 0 else Y[:, t-1]       # 1 timestep delay between input layer and representation layer
            
            # responses from previous time step
            responses_m_1 = responses_0 if t == 0 else sequence_responses[t-1]
            
            # if we are doing chunked backpropagation through time
            if self.bbtt_chunk is not None:
                if (t + 1) % self.bbtt_chunk == 0:
                    h_m_1 = h_m_1.detach()
                    z_m_1 = z_m_1.detach()
                    sst_inh_prev = sst_inh_prev.detach()
            
            # forward the model
            responses_t, losses_t = self(I_t, responses_m_1, epsilon=epsilon)
            
            # compute the losses
            spatial_loss, temporal_loss, energy_loss = losses_t.values()
            total_loss = lambda_spatial * spatial_loss + lambda_temporal * temporal_loss + lambda_energy * energy_loss
            
            # update avg losses
            mean_spatial += spatial_loss / T
            mean_temporal += temporal_loss / T
            mean_energy += energy_loss / T
            mean_total += total_loss / T

            sequence_responses.append(responses_t)
        
        responses = {
            k: torch.stack([d[k] for d in sequence_responses], 1) for k in sequence_responses[0].keys()
        }
        
        if (R is not None) and (not self.perception_only):
            mean_value_loss, episode_rewards = self.compute_reward_loss(R, responses['action'], responses['action_value'])
            mean_total = mean_total + lambda_reward * mean_value_loss
            ep_rew_mean = episode_rewards.mean()
        else:
            mean_value_loss, ep_rew_mean = torch.Tensor([float('nan')]).to(mean_total), torch.Tensor([float('nan')]).to(mean_total)

        losses = {"total": mean_total,
                  "spatial_error": mean_spatial,
                  "temporal_error": mean_temporal,
                  "energy": mean_energy,
                  "value_loss": mean_value_loss,
                  "episode_rewards": ep_rew_mean}
        
        return responses, losses
    
    def stabilize_recurrent_weights(self):
        self.h_to_h.weight.data = StableRecurrent()(self.h_to_h.weight.data)
    
    def init_state(self, batch_size, device='cpu'):
        
        responses_0 = {
            "mu_p": torch.zeros((batch_size, self.latent_dim)).to(device),
            "sigmap_h": torch.zeros((batch_size, self.latent_dim)).to(device),
            "sigma_p": torch.zeros((batch_size, self.latent_dim)).to(device),
            "mu_q": torch.zeros((batch_size, self.latent_dim)).to(device),
            "sigma_q": torch.zeros((batch_size, self.latent_dim)).to(device),
            "theta_ff": torch.zeros((batch_size, self.theta_dim)).to(device),
            "theta_h": torch.zeros((batch_size, self.theta_dim)).to(device),
            "theta": torch.zeros((batch_size, self.theta_dim)).to(device),
            "vip_inh": torch.zeros((batch_size, self.latent_dim)).to(device),
            "sst_inh": torch.zeros((batch_size, self.latent_dim)).to(device),
            "z_h": torch.zeros((batch_size, self.latent_dim)).to(device),
            "z": torch.zeros((batch_size, self.latent_dim)).to(device),
            "h": torch.zeros((batch_size, self.higher_state_dim)).to(device),
            "h2": torch.zeros((batch_size, self.higher_state_dim)).to(device),
            "value": torch.zeros((batch_size, self.value_dim)).to(device),
            "rl_gain": torch.zeros((batch_size, 1)).to(device)
        }

        return responses_0
        