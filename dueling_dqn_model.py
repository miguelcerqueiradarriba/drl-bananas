import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        internal_layers = state_size * 50

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, internal_layers),
            nn.ReLU(),
            nn.Linear(internal_layers, internal_layers),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(internal_layers, internal_layers),
            nn.ReLU(),
            nn.Linear(internal_layers, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(internal_layers, internal_layers),
            nn.ReLU(),
            nn.Linear(internal_layers, action_size)
        )

    def forward(self, state):
        x = self.feature_layer(state)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        output = values + (advantages - advantages.mean())

        return output
