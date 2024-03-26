import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_action, use_std=False) -> None:
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_action

        self.fc1 = nn.Linear(num_obs, 128)
        self.fc_mu = nn.Linear(128, num_action)
        self.fc_v = nn.Linear(128, 1)

        self.use_std = use_std
        if self.use_std:
            self.fc_std = nn.Linear(128, num_action)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))
        
        if self.use_std:
            std = F.softplus(self.fc_std(x))
        else:
            std = torch.tensor([0.5]) 

        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v    