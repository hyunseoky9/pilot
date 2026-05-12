import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, CosineAnnealingLR

class Critic(nn.Module):
    def __init__(self, input_dims,
                 hidden_size, hidden_num, 
                 lrdecayrate, lr,
                 min_lr, lrdecaytype,
                 scheduler_info, device):
        super(Critic, self).__init__()

        # build the model
        layers = [nn.Linear(input_dims, hidden_size[0]), nn.ReLU()]
        for i in range(hidden_num - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size[-1], 1))  # Output layer for value function
        
        # Creating the Sequential module
        self.critic = nn.Sequential(*layers)
        self._init_weights()
        # set up optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-8)
        self.device = T.device(device)
        self.to(self.device)
        # set up learning rate scheduler
        if lrdecaytype == 'exp':
            self.scheduler = ExponentialLR(self.optimizer, gamma=lrdecayrate)  # Exponential decay
        elif lrdecaytype == 'multistep':
            self.scheduler = MultiStepLR(self.optimizer, milestones=scheduler_info['lr_drop_ep'],
                                          gamma=scheduler_info['lr_drop_gamma'])

    def _init_weights(self):
        linears = [m for m in self.critic if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
        # Policy head: small gain so initial logits are near-uniform
        nn.init.orthogonal_(linears[-1].weight, gain=1.0)
        nn.init.zeros_(linears[-1].bias)

    def forward(self, state):
        value = self.critic(state)
        return value
