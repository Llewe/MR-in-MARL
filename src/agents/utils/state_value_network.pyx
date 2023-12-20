from torch.nn import ELU, Linear, Module, Sequential


# from torch.nn.functional import relu


def preprocessing_module(nr_input_features, nr_hidden_units):
    return Sequential(
        Linear(nr_input_features, nr_hidden_units),
        ELU(),
        Linear(nr_hidden_units, nr_hidden_units),
        ELU(),
    )


class StateValueNetwork(Module):
    # Takes in state
    def __init__(self, observation_space, hidden_units):
        super(StateValueNetwork, self).__init__()

        self.fc_net = preprocessing_module(observation_space, hidden_units)
        self.value_head = Linear(hidden_units, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return self.value_head(x)
