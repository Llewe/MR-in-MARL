from torch.nn import Module, Linear, Sequential, ELU, Softmax
from torch.nn.functional import relu, softmax, elu

NR_HIDDEN_UNITS = 128


def preprocessing_module(nr_input_features, nr_hidden_units):
    return Sequential(
        Linear(nr_input_features, nr_hidden_units),
        ELU(),
        Linear(nr_hidden_units, nr_hidden_units),
        ELU(),
    )


class PolicyNetwork(Module):
    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space, hidden_units):
        super(PolicyNetwork, self).__init__()

        self.fc_net = preprocessing_module(observation_space, hidden_units)
        self.action_head = Linear(hidden_units, action_space)

    # forward pass
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return softmax(self.action_head(x), dim=-1)
