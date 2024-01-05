import numpy
import torch
from torch.nn import ELU, Linear, Module, Sequential
from torch.nn.functional import softmax
import torch.nn.init as init

def initialize_weights(layer):
    """
    Initialize weights of a layer with Kaiming normal initialization and biases with zeros.
    I currently dont use it as it had poor performance on the mate in CoinGame-2 etc.
    Parameters
    ----------
    layer

    Returns
    -------

    """
    if isinstance(layer, Linear):
        init.kaiming_normal_(layer.weight, nonlinearity='relu')
        init.constant_(layer.bias, 0)

def preprocessing_module(nr_input_features, nr_hidden_units, last_layer_units):
    model=  Sequential(
        Linear(nr_input_features, nr_hidden_units),
        ELU(),
        Linear(nr_hidden_units, nr_hidden_units),
        ELU(),
        Linear(nr_hidden_units, last_layer_units),
    )
    # if needed here initialize weights can be applied
    # model.apply(initialize_weights)

    return model


class ActorNetwork(Module):
    # Takes in observations and outputs actions
    num_actions: numpy.int32
    fc_net: Sequential
    optimizer: torch.optim.Adam

    def __init__(
            self,
            observation_space,
            num_actions: numpy.int32,
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
    fc_net: Sequential
    optimizer: torch.optim.Adam

    # Takes in state
    def __init__(self, observation_space, hidden_units: int, learning_rate: float):
        super(CriticNetwork, self).__init__()

        self.fc_net = preprocessing_module(observation_space, hidden_units, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc_net(x)
