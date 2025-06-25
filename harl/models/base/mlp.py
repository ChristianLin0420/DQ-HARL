import torch.nn as nn
from harl.utils.models_tools import init, get_active_func, get_init_method

"""MLP modules."""


def get_activation_gain(activation_func):
    """Get the gain for activation function, with custom handling for newer activations."""
    # Handle activations that PyTorch's calculate_gain doesn't support
    if activation_func in ["silu", "swish"]:
        # SiLU/Swish typically uses a gain similar to ReLU
        return nn.init.calculate_gain("relu")
    elif activation_func == "gelu":
        # GELU typically uses a gain similar to ReLU
        return nn.init.calculate_gain("relu")
    elif activation_func == "mish":
        # Mish typically uses a gain similar to ReLU
        return nn.init.calculate_gain("relu")
    else:
        # Use PyTorch's built-in calculation for standard activations
        try:
            return nn.init.calculate_gain(activation_func)
        except ValueError:
            # Fallback to 1.0 if the activation is not recognized
            print(f"Warning: Unknown activation function '{activation_func}', using gain=1.0")
            return 1.0


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = get_activation_gain(activation_func)  # Use our custom function

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    """A MLP base module."""

    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]

        obs_dim = obs_shape[0]

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func
        )

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x
