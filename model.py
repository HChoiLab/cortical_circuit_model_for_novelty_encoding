import torch
from torch import nn


# Parametrization classes to constrain weights
class Positive(nn.Module):
    def forward(self, X):
        return torch.relu(X)

class EnergyConstrainedPredictiveCodingModel(nn.Module):
    
    def __init__(self, input_dim, latent_dim, higher_state_dim,
                lambda_spatial=10.,
                lambda_temporal=1.0,
                lambda_energy=1.0,
                lambda_reward=1.0,
                perception_only=True):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.higher_state_dim = higher_state_dim
        self.perception_only = perception_only
        
        self.lambda_spatial = lambda_spatial            # regularizing scalar for the spatial error loss
        self.lambda_temporal = lambda_temporal          # regularizing scalar for the temporal error loss
        self.lambda_energy = lambda_energy              # regularizing scalar for the energy efficiency cost
        self.lambda_reward = lambda_reward              # regularizing scalar for the reward prediction loss
        
        # connections from input to latents
        self.posterior_mu = nn.Linear(input_dim, latent_dim, bias=False)                # encodes input to mean of posterior
        self.posterior_log_var = nn.Linear(input_dim, latent_dim, bias=False)           # encodes input to the log variance of the posterior
        
        # connections to higher area
        self.z_to_h = nn.Linear(latent_dim, higher_state_dim, bias=False)                         # connections from latents to higher area
        self.h_to_h = nn.Linear(higher_state_dim, higher_state_dim, bias=False)                   # connections from higher area to itself
        
        # connections from higher area
        self.prior_mu = nn.Linear(higher_state_dim, latent_dim, bias=False)                 # encodes the mean of the prior distribution (represented by an SST subpopulation)
        self.prior_log_var = nn.Linear(higher_state_dim, latent_dim, bias=False)            # encodes the log variance of the prior (represented by VIP neurons)
        
        # connections involved in the computation of theta
        self.theta_dim = 16
        self.h_to_theta = nn.Linear(higher_state_dim, self.theta_dim, bias=False)
        self.I_to_theta = nn.Linear(input_dim, self.theta_dim, bias=False)
        self.vip_to_theta = nn.Linear(latent_dim, self.theta_dim, bias=False)
        self.theta_to_z = nn.Linear(self.theta_dim, latent_dim, bias=False)

        # weight constraints 
        # these weights should be inhibitory; we constrain them to be positive here but then 
        # multiply them by -1 when they're used for computation 
        nn.utils.parametrize.register_parametrization(self.vip_to_theta, 'weight', Positive())
        nn.utils.parametrize.register_parametrization(self.theta_to_z, 'weight', Positive())
    
        # connections to reconstruct input from z activity
        self.reconstruction = nn.Sequential(nn.Linear(latent_dim, 256, bias=False),
                                            nn.Linear(256, input_dim, bias=False),
                                            nn.Sigmoid())
        
        # decision making and value learning
        self.h_to_value = nn.Sequential(nn.Linear(higher_state_dim, 256),
                                        nn.Linear(256, 2))

    
    def compute_reward_loss(self, R, actions, values):
        
        rewards = R * actions.detach()
        
        value_loss = ((rewards - values.squeeze()) ** 2).mean()
        
        return value_loss, rewards.sum(-1)

    def act_with_exploration(self, lick_prob, epsilon):

        best_action = (lick_prob > 0.5).float().detach()
        random_action = torch.randint_like(best_action, 0, 2).float()
        u_sample = torch.rand_like(best_action)
        action = (u_sample < epsilon).float() * random_action + (u_sample >= epsilon).float() * best_action

        return action      
    
    def forward(self, I_t, h_m_1, z_m_1, theta_m_1, epsilon=0.1):      
        
        # get prior from previous higher area state
        mu_p = nn.functional.relu(self.prior_mu(h_m_1))

        # value prediction and value-based modulation of VIP activity
        # value computation
        if not self.perception_only:
            values = torch.tanh(self.h_to_value(h_m_1))
            lick_value = values[:, 1]
            nolick_value = values[:, 0]

            # probability of licking 
            lick_prob = torch.softmax(values, -1)[:, 1]

            sigma_p = nn.functional.softplus(self.prior_log_var(h_m_1) + 0.5 * lick_prob.detach().reshape(-1, 1), beta=1.2)

            #rl_gain = torch.exp(lick_value.detach()).reshape(-1, 1)
            #sigma_p = rl_gain * sigma_p
        
        else:
            sigma_p = nn.functional.softplus(self.prior_log_var(h_m_1), beta=1.2)

        # compute thetas
        theta_h = 0.5 * theta_m_1 + 0.1 * self.I_to_theta(I_t) - self.vip_to_theta(sigma_p.detach())     # 0.2 -> 0.05 , 0.5 -> 0.3
        theta = 0.001 * nn.functional.softplus(theta_h, beta=0.5)
        
        # encode input to the posterior parameters
        mu_q = nn.functional.relu(self.posterior_mu(I_t))
        sigma_q = torch.relu(self.posterior_log_var(I_t)) 
        
        # compute pyramidal activities (inferred z's)
        raw_z = mu_q + torch.randn_like(sigma_q) * sigma_q
        raw_z = torch.clamp(raw_z, min=0, max=1)

        # inhibitory input from SST to excitatory activity
        thresholds = 10. * self.theta_to_z(theta)
        z = nn.functional.relu(raw_z - thresholds)
        
        # compute reconstruction (top down prediction from the representation layer)
        I_hat = self.reconstruction(z)
        
        # evolve higher area activities
        h = nn.functional.relu(self.z_to_h(z_m_1) + self.h_to_h(h_m_1))
        
        # compute temporal prediction (top down prediction from the higher layer)
        z_hat = mu_p + torch.randn_like(mu_p) * sigma_p                      

        # error population activities
        layer_1_error = (I_t - I_hat) ** 2
        layer_2_error = (z.detach() - z_hat) ** 2

        # decision making if we are doing action
        if not self.perception_only:
            # select action with exploration 
            action = self.act_with_exploration(lick_prob, epsilon)
            
            # predicted value of action
            action_value = action * lick_value + (1 - action) * nolick_value

        # compute losses
        spatial_error_loss = layer_1_error.mean()
        temporal_error_loss = layer_2_error.mean()
        energy_loss = torch.abs(z).mean()

        # put together all responses
        responses_t = {
            "mu_p": mu_p,
            "sigma_p": sigma_p,
            "mu_q": mu_q,
            "sigma_q": sigma_q,
            "theta_h": theta_h,
            "theta": theta,
            "z": z,
            "I_hat": I_hat,
            "h": h
        }

        if not self.perception_only:
            responses_t.update({
                "action": action,
                "action_value": action_value,
                "lick_value": lick_value
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
        h_0, z_0, theta_0 = self.init_state(Y.shape[0])
        h_0, z_0, theta_0 = h_0.to(Y.device), z_0.to(Y.device), theta_0.to(Y.device)
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
            
            # 1 timestep delay between higher layer and representation layer
            z_m_1 = z_0 if t == 0 else sequence_responses[t-1]['z']
            h_m_1 = h_0 if t == 0 else sequence_responses[t-1]['h']
            theta_m_1 = theta_0 if t == 0 else sequence_responses[t-1]['theta_h']
            
            # forward the model
            responses_t, losses_t = self(I_t, h_m_1, z_m_1, theta_m_1, epsilon=epsilon)
            
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
    
    def init_state(self, batch_size):

        return torch.zeros((batch_size, self.higher_state_dim)), torch.zeros((batch_size, self.latent_dim)), torch.zeros((batch_size, self.theta_dim))
        