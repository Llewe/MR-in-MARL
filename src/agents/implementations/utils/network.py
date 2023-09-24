import torch
from gymnasium import Space
from torch.nn import Module, Linear, Sequential, ELU
from torch.nn.functional import softmax


def preprocessing_module(nr_input_features, nr_hidden_units):
    return Sequential(
        Linear(nr_input_features, nr_hidden_units),
        ELU(),
        Linear(nr_hidden_units, nr_hidden_units),
        ELU(),
    )


class ActorNetwork(Module):
    # Takes in observations and outputs actions
    def __init__(
        self,
        observation_space,
        action_space: Space,
        hidden_units: int,
        learning_rate: float,
    ):
        super(ActorNetwork, self).__init__()
        self.action_space = action_space
        self.fc_net = preprocessing_module(observation_space, hidden_units)
        self.action_head = Linear(hidden_units, action_space.n)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    # forward pass
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return softmax(self.action_head(x), dim=-1)


class CriticNetwork(Module):
    # Takes in state
    def __init__(self, observation_space, hidden_units: int, learning_rate: float):
        super(CriticNetwork, self).__init__()

        self.fc_net = preprocessing_module(observation_space, hidden_units)
        self.value_head = Linear(hidden_units, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return self.value_head(x)
