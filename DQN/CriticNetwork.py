import torch.nn as nn


class CriticNetwork(nn.Module):
    def __init__(self, n_act, n_obs):
        super(CriticNetwork, self).__init__()
        h_dim = 64
        self.layers = nn.Sequential(
            nn.Linear(n_obs, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, n_act)
        )

    def forward(self, obs):
        # Observation shape: (n_batches, n_features)
        return self.layers(obs)
