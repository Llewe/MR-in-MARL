import torch
from gymnasium import Space
from torch.nn import Module, Linear, Sequential, ELU
from torch.nn.functional import softmax


def preprocessing_module(nr_input_features, nr_hidden_units, last_layer_units):
    return Sequential(
        Linear(nr_input_features, nr_hidden_units),
        ELU(),
        Linear(nr_hidden_units, nr_hidden_units),
        ELU(),
        Linear(nr_hidden_units, last_layer_units),
    )


class ActorNetwork(Module):
    # Takes in observations and outputs actions
    def __init__(
        self,
        observation_space,
        num_actions: int,
        hidden_units: int,
        learning_rate: float,
    ):
        super(ActorNetwork, self).__init__()
        self.num_actions = num_actions
        self.fc_net = preprocessing_module(observation_space, hidden_units, num_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    # forward pass
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return softmax(self.fc_net(x), dim=-1)


class CriticNetwork(Module):
    # Takes in state
    def __init__(self, observation_space, hidden_units: int, learning_rate: float):
        super(CriticNetwork, self).__init__()

        self.fc_net = preprocessing_module(observation_space, hidden_units, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc_net(x)
