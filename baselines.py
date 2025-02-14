import torch
import torch.nn as nn

# Parametrization classes to constrain weights
class Positive(nn.Module):
    def forward(self, X):
        return torch.relu(X)
    
class Negative(nn.Module):
    def forward(self, X):
        return -torch.relu(-X)

class ZeroDiagonal(nn.Module):
    def forward(self, X):
        mask = 1. - torch.eye(X.shape[0])
        return mask.to(X.device) * X

class AdaptationModel(nn.Module):
    def __init__(self, num_neurons_E=100, num_neurons_VIP=50, num_neurons_SST=50, num_inputs=100,
                 tau_E=20.0, tau_VIP=20.0, tau_SST=20.0, tau_A=200.0,
                 g_E=1.0, g_VIP=1.0, g_SST=0.5,
                 eta=0.001, alpha=0.001, dt=1.0, hebbian=True, device='cpu'):
        super(AdaptationModel, self).__init__()
        
        self.device = device
        
        # Model parameters
        self.num_neurons_E = num_neurons_E
        self.num_neurons_VIP = num_neurons_VIP
        self.num_neurons_SST = num_neurons_SST
        self.num_inputs = num_inputs
        self.hebbian = hebbian
        
        # Time constants
        self.tau_E = tau_E
        self.tau_VIP = tau_VIP
        self.tau_SST = tau_SST
        self.tau_A = tau_A
        self.dt = dt
        
        # Adaptation strengths
        self.g_E = g_E
        self.g_VIP = g_VIP
        self.g_SST = g_SST
        
        # Learning parameters
        self.eta = eta
        self.alpha = alpha
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Initialize synaptic weights
        # Inter-population connection
        self.I_to_E = nn.Linear(num_inputs, num_neurons_E, bias=False)
        self.I_to_SST = nn.Linear(num_inputs, num_neurons_SST, bias=False)
        self.E_to_VIP = nn.Linear(num_neurons_E, num_neurons_VIP, bias=False)
        self.VIP_to_E = nn.Linear(num_neurons_VIP, num_neurons_E, bias=False)
        self.E_to_SST = nn.Linear(num_neurons_E, num_neurons_SST, bias=False)
        self.SST_to_E = nn.Linear(num_neurons_SST, num_neurons_E, bias=False)
        self.VIP_to_SST = nn.Linear(num_neurons_VIP, num_neurons_SST, bias=False)
        
        # Recurrent connections
        self.E_rec = nn.Linear(num_neurons_E, num_neurons_E, bias=False)
        self.VIP_rec = nn.Linear(num_neurons_VIP, num_neurons_VIP, bias=False)
        self.SST_rec = nn.Linear(num_neurons_SST, num_neurons_SST, bias=False)
        
        # Initialize firing rates and adaptation variables
        self.reset_state()
        
        # Enforcing weight constraints
        self.enforce_weight_constraints()
    
    def reset_state(self, batch_size=1):
        # Firing rates
        self.r_E = torch.zeros(batch_size, self.num_neurons_E, device=self.device)
        self.r_VIP = torch.zeros(batch_size, self.num_neurons_VIP, device=self.device)
        self.r_SST = torch.zeros(batch_size, self.num_neurons_SST, device=self.device)
        
        # Adaptation variables
        self.A_E = torch.zeros(batch_size, self.num_neurons_E, device=self.device)
        self.A_VIP = torch.zeros(batch_size, self.num_neurons_VIP, device=self.device)
        self.A_SST = torch.zeros(batch_size, self.num_neurons_SST, device=self.device)       
    
    def forward(self, I_input):
        """
        Forward pass for one time step.
        
        Args:
            I_input (torch.Tensor): Input stimulus at current time step (shape: [num_inputs])
        
        Returns:
            dict: Dictionary containing current firing rates and adaptation variables
        """
        # External input to Excitatory neurons
        I_E = self.I_to_E(I_input)
        
        # Update adaptation variables
        dA_E = (-self.A_E + self.g_E * self.r_E) * (self.dt / self.tau_A)
        self.A_E = self.A_E + dA_E
        dA_VIP = (-self.A_VIP + self.g_VIP * self.r_VIP) * (self.dt / self.tau_A)
        self.A_VIP = self.A_VIP + dA_VIP
        dA_SST = (-self.A_SST + self.g_SST * self.r_SST) * (self.dt / self.tau_A)
        self.A_SST = self.A_SST + dA_SST
        
        # Compute net input currents
        # From SST and VIP to E neurons (inhibition)
        I_SST_E = self.SST_to_E(self.r_SST)
        I_VIP_E = self.VIP_to_E(self.r_VIP)
        I_E_E = self.E_rec(self.r_E)  # recurrent input
        # Total input to E neurons
        net_input_E = (I_E + I_SST_E + I_VIP_E + I_E_E) - self.A_E
        
        # Update firing rates of Excitatory neurons
        dr_E = (-self.r_E + self.activation(net_input_E)) * (self.dt / self.tau_E)
        self.r_E = self.r_E + dr_E
        
        # Compute input to VIP neurons
        I_VIP = self.E_to_VIP(self.r_E) + self.VIP_rec(self.r_VIP)
        # Update firing rates of VIP neurons
        net_input_VIP = I_VIP - self.A_VIP
        dr_VIP = (-self.r_VIP + self.activation(net_input_VIP)) * (self.dt / self.tau_VIP)
        self.r_VIP = self.r_VIP + dr_VIP
        
        # Compute input to SST
        I_SST = self.VIP_to_SST(self.r_VIP) + self.SST_rec(self.r_SST) + self.I_to_SST(I_input) #self.E_to_SST(self.r_E) 
        # Update firing rates of SST neurons
        net_input_SST = I_SST - self.A_SST
        dr_SST = (-self.r_SST + self.activation(net_input_SST)) * (self.dt / self.tau_SST)
        self.r_SST = self.r_SST + dr_SST
        
        # Hebbian learning updates
        if self.hebbian:
            self.hebbian_update_step(I_input)
        
        # Enforce constraints on the weights
        self.enforce_weight_constraints()
        
        # Normalize weights
        self.normalize_weights()
        
        # Return current state
        return {
            'r_E': self.r_E.clone(),
            'r_VIP': self.r_VIP.clone(),
            'r_SST': self.r_SST.clone(),
            'A_E': self.A_E.clone(),
            'A_VIP': self.A_VIP.clone(),
            'A_SST': self.A_SST.clone()
        }
    
    def forward_sequence(self, input_sequence):
        """
        Forward pass for a sequence of inputs.
        
        Args:
            input_sequence (torch.Tensor): Sequence of inputs (shape: [batch_size, sequence_length, num_inputs])
        
        Returns:
            dict: Dictionary containing histories of firing rates and adaptation variables
        """
        
        self.reset_state(batch_size=input_sequence.shape[0])
        
        # Lists to store history
        r_E_history = []
        r_VIP_history = []
        r_SST_history = []
        
        I0 = torch.zeros_like(input_sequence[:, 0])
        
        # Process each input in the sequence
        for t in range(input_sequence.shape[1]):
            
            # 1 timestep delay between input and E layer
            It = I0 if t == 0 else input_sequence[:, t-1]
            
            state = self.forward(It)
            r_E_history.append(state['r_E'])
            r_VIP_history.append(state['r_VIP'])
            r_SST_history.append(state['r_SST'])
        
        # Convert histories to tensors
        r_E_history = torch.stack(r_E_history, dim=1)  # Shape: [sequence_length, num_neurons_E]
        r_VIP_history = torch.stack(r_VIP_history, dim=1)  # Shape: [sequence_length, num_neurons_VIP]
        r_SST_history = torch.stack(r_SST_history, dim=1)  # Shape: [sequence_length, num_neurons_SST]
        
        return {
            'E': r_E_history,
            'VIP': r_VIP_history,
            'SST': r_SST_history
        }       
        
    
    def hebbian_update(self, W, pre_rates, post_rates):
        """
        Hebbian learning rule to update synaptic weights.
        
        Args:
            W (nn.Parameter): Synaptic weight matrix
            pre_rates (torch.Tensor): Presynaptic firing rates
            post_rates (torch.Tensor): Postsynaptic firing rates
        """
        post_pre_corr = torch.bmm(post_rates.unsqueeze(2), pre_rates.unsqueeze(1)).mean(0)
        delta_W = self.eta * (post_pre_corr - self.alpha * W.weight.data)
        W.weight.data += delta_W
    
    def hebbian_update_step(self, I_input):
        
        # Inter-population weights
        self.hebbian_update(self.I_to_E, I_input, self.r_E)
        self.hebbian_update(self.E_to_VIP, self.r_E, self.r_VIP)
        self.hebbian_update(self.VIP_to_E, self.r_VIP, self.r_E)
        self.hebbian_update(self.E_to_SST, self.r_E, self.r_SST)
        self.hebbian_update(self.SST_to_E, self.r_SST, self.r_E)
        self.hebbian_update(self.VIP_to_SST, self.r_VIP, self.r_SST)
        
        # Recurrent weights
        self.hebbian_update(self.E_rec, self.r_E, self.r_E)
        self.hebbian_update(self.VIP_rec, self.r_VIP, self.r_VIP)
        self.hebbian_update(self.SST_rec, self.r_SST, self.r_SST)
    
    def enforce_weight_constraints(self):
        
        # Inter-population weights
        self.E_to_VIP.weight.data = Positive()(self.E_to_VIP.weight.data)
        self.E_to_SST.weight.data = Positive()(self.E_to_SST.weight.data)
        
        self.VIP_to_E.weight.data = Negative()(self.VIP_to_E.weight.data)
        self.SST_to_E.weight.data = Negative()(self.SST_to_E.weight.data)
        self.VIP_to_SST.weight.data = Negative()(self.VIP_to_SST.weight.data)
        
        # Recurrent weights
        self.E_rec.weight.data = ZeroDiagonal()(Positive()(self.E_rec.weight.data))
        self.VIP_rec.weight.data = ZeroDiagonal()(Negative()(self.VIP_rec.weight.data))
        self.SST_rec.weight.data = ZeroDiagonal()(Negative()(self.SST_rec.weight.data))
        
    def normalize_weights(self):
        """
        Normalize synaptic weights to prevent unlimited growth.
        """
        # interpopulation weights
        self.E_to_VIP.weight.data = self.E_to_VIP.weight.data / self.E_to_VIP.weight.data.norm(dim=1, keepdim=True)
        self.VIP_to_E.weight.data = self.VIP_to_E.weight.data / self.VIP_to_E.weight.data.norm(dim=1, keepdim=True)
        self.E_to_SST.weight.data = self.E_to_SST.weight.data / self.E_to_SST.weight.data.norm(dim=1, keepdim=True)
        self.SST_to_E.weight.data = self.SST_to_E.weight.data / self.SST_to_E.weight.data.norm(dim=1, keepdim=True)
        self.VIP_to_SST.weight.data = self.VIP_to_SST.weight.data / self.VIP_to_SST.weight.data.norm(dim=1, keepdim=True)
        
        # Recurrent connections
        self.E_rec.weight.data = self.E_rec.weight.data / self.E_rec.weight.data.norm(dim=1, keepdim=True)
        self.VIP_rec.weight.data = self.VIP_rec.weight.data / self.VIP_rec.weight.data.norm(dim=1, keepdim=True)
        self.SST_rec.weight.data = self.SST_rec.weight.data /  self.SST_rec.weight.data.norm(dim=1, keepdim=True)
