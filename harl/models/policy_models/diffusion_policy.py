import torch
import torch.nn as nn
import numpy as np

from harl.utils.envs_tools import check
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.utils.models_tools import init, get_active_func, get_init_method


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal position embeddings for time steps in diffusion process."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device = device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


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


class DiffusionMLP(nn.Module):
    """MLP network for diffusion model that takes observation, time, and current action."""
    
    def __init__(self, args, obs_dim, action_dim, time_embed_dim=32):
        super().__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.activation_func = args["activation_func"]
        self.initialization_method = args["initialization_method"]
        self.use_feature_normalization = args.get("use_feature_normalization", True)

        # Time embedding
        self.time_embed_dim = time_embed_dim
        self.time_embed = SinusoidalPositionalEncoding(time_embed_dim)

        # Input dimension: obs + action + time_embedding
        input_dim = obs_dim + action_dim + time_embed_dim

        if self.use_feature_normalization:
            self.input_norm = nn.LayerNorm(input_dim)

        # Build MLP Layers
        active_func = get_active_func(self.activation_func)
        init_method = get_init_method(self.initialization_method)
        gain = get_activation_gain(self.activation_func)  # Use our custom function

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        layers = []
        prev_dim = input_dim

        for hidden_size in self.hidden_sizes:
            # Linear layer
            layers.append(init_(nn.Linear(prev_dim, hidden_size)))
            # Activation function - add the module directly, not as a call
            layers.append(active_func)
            # Layer normalization
            layers.append(nn.LayerNorm(hidden_size))
            prev_dim = hidden_size

        # Output layer predicts noise/velocity
        layers.append(init_(nn.Linear(prev_dim, action_dim)))

        self.net = nn.Sequential(*layers)

    def forward(self, obs, action, time):
        """Forward pass for diffusion model.
        Args:
            obs: (torch.Tensor) observation tensor.
            action: (torch.Tensor) current action tensor.
            time: (torch.Tensor) time step tensor.
        Returns:
            noise: (torch.Tensor) predicted noise/velocity tensor.
        """

        # Embed time step
        time_embed = self.time_embed(time)

        # Concatenate inputs
        x = torch.cat([obs, action, time_embed], dim=-1)

        if self.use_feature_normalization:
            x = self.input_norm(x)

        return self.net(x)
    
class DiffusionPolicy(nn.Module):
    """
    Diffusion-based policy that generates actions through iterative denoising.
    Implements the GenPO diffusion process with coupled noise vectors.
    """
    
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Diffusion parameters
        self.num_diffusion_steps = args.get("num_diffusion_steps", 10)
        self.mixing_p = args.get("mixing_p", 0.9)
        self.beta_start = args.get("beta_start", 1e-4)
        self.beta_end = args.get("beta_end", 0.02)
        self.time_embed_dim = args.get("time_embed_dim", 32)

        # RNN parameters
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        
        # Get dimensions
        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs_dim = obs_shape[0]
        self.action_dim = action_space.shape[0]
        
        # Observation encoder
        self.obs_encoder = MLPBase(args, obs_shape)
        
        # RNN layer if needed
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )
        
        # Diffusion network that predicts noise
        self.diffusion_net = DiffusionMLP(
            args, 
            self.hidden_sizes[-1],  # Use encoded observation dimension
            self.action_dim, 
            self.time_embed_dim
        )
        
        # Set up diffusion schedule
        self.register_diffusion_schedule()
        
        self.to(device)

    def register_diffusion_schedule(self):
        """Register diffusion schedule parameters."""
        # Linear beta schedule
        betas = torch.linspace(
            self.beta_start, self.beta_end, self.num_diffusion_steps
        ).to(self.device)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", 
                           torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Generate actions using the diffusion process.
        
        Args:
            obs: (batch_size, obs_dim)
            rnn_states: RNN states if using recurrent policy
            masks: Mask for RNN states
            available_actions: Available actions (not used in continuous case)
            deterministic: Whether to use deterministic sampling
        
        Returns:
            actions: (batch_size, action_dim)
            action_log_probs: (batch_size, action_dim) - approximate log probabilities
            rnn_states: Updated RNN states
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        batch_size = obs.shape[0]
        
        # Encode observations
        obs_features = self.obs_encoder(obs)
        
        # Apply RNN if needed
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            obs_features, rnn_states = self.rnn(obs_features, rnn_states, masks)
        
        # Generate actions using diffusion process and compute log probabilities
        actions, action_log_probs = self.sample_actions_with_log_prob(obs_features, deterministic)
        
        return actions, action_log_probs, rnn_states

    def sample_actions_with_log_prob(self, obs_features, deterministic=False):
        """
        Sample actions using the reverse diffusion process and compute log probabilities.
        This implements a more sophisticated log probability computation.
        """
        batch_size = obs_features.shape[0]
        
        # Initialize coupled noise vectors (x, y)
        if deterministic:
            x = torch.zeros(batch_size, self.action_dim, device=self.device)
            y = torch.zeros(batch_size, self.action_dim, device=self.device)
        else:
            x = torch.randn(batch_size, self.action_dim, device=self.device)
            y = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Track log probabilities through the reverse process
        log_prob_x = torch.zeros(batch_size, 1, device=self.device)
        log_prob_y = torch.zeros(batch_size, 1, device=self.device)
        
        # Store initial noise for log probability computation
        x_init, y_init = x.clone(), y.clone()
        
        # Reverse diffusion process
        for t in reversed(range(self.num_diffusion_steps)):
            # Create time tensor
            time_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.float32)
            
            # Predict noise for both components
            predicted_noise_x = self.diffusion_net(obs_features, x, time_tensor)
            predicted_noise_y = self.diffusion_net(obs_features, y, time_tensor)
            
            # Compute denoising step parameters
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            
            # Compute mean of reverse process
            mean_x = (x - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise_x) / sqrt_alpha_t
            mean_y = (y - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise_y) / sqrt_alpha_t
            
            if t > 0:
                # Add noise for non-final steps
                if not deterministic:
                    noise_x = torch.randn_like(x)
                    noise_y = torch.randn_like(y)
                    
                    # Compute log probability contribution from this step
                    # This is based on the Gaussian transition probability
                    sigma_t = torch.sqrt(beta_t)
                    log_prob_x += -0.5 * torch.sum((noise_x)**2, dim=-1, keepdim=True) - \
                                 0.5 * self.action_dim * torch.log(2 * np.pi * sigma_t**2)
                    log_prob_y += -0.5 * torch.sum((noise_y)**2, dim=-1, keepdim=True) - \
                                 0.5 * self.action_dim * torch.log(2 * np.pi * sigma_t**2)
                else:
                    noise_x = torch.zeros_like(x)
                    noise_y = torch.zeros_like(y)
                
                x_tilde = mean_x + torch.sqrt(beta_t) * noise_x
                y_tilde = mean_y + torch.sqrt(beta_t) * noise_y
            else:
                # Final step - no noise
                x_tilde = mean_x
                y_tilde = mean_y
            
            # Mixing step (Eq. 7 from paper)
            x_new = self.mixing_p * x_tilde + (1 - self.mixing_p) * y_tilde
            y_new = self.mixing_p * y_tilde + (1 - self.mixing_p) * x_tilde
            
            x, y = x_new, y_new
        
        # Final action is average of coupled components
        actions = (x + y) / 2.0
        
        # Compute final log probability
        # Include the initial noise distribution and the Jacobian of the averaging operation
        initial_log_prob = -0.5 * torch.sum(x_init**2 + y_init**2, dim=-1, keepdim=True) - \
                          self.action_dim * np.log(2 * np.pi)
        
        # Add log probability from the reverse process
        total_log_prob = initial_log_prob + log_prob_x + log_prob_y
        
        # Jacobian correction for the averaging operation (x + y) / 2
        # The Jacobian determinant is 1/2^action_dim, so log|J| = -action_dim * log(2)
        jacobian_correction = -self.action_dim * np.log(2)
        total_log_prob += jacobian_correction
        
        return actions, total_log_prob

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Align evaluation with the actual diffusion generation process.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # Encode observations
        obs_features = self.obs_encoder(obs)
        
        # Apply RNN if needed
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            obs_features, _ = self.rnn(obs_features, rnn_states, masks)
        
        # Use the SAME process as action generation for evaluation
        batch_size = obs_features.shape[0]
        
        # Sample multiple actions and find the one closest to the target
        num_samples = 5  # Sample multiple actions for better evaluation
        log_probs = []
        
        for _ in range(num_samples):
            # Generate action using the same process as forward()
            sample_action, sample_log_prob = self.sample_actions_with_log_prob(obs_features, deterministic=False)
            
            # Compute likelihood of the target action given this sample
            mse = torch.sum((action - sample_action)**2, dim=-1, keepdim=True)
            log_prob = -0.5 * mse / (0.1**2) - 0.5 * self.action_dim * np.log(2 * np.pi * 0.1**2)
            log_probs.append(log_prob)
        
        # Use the maximum likelihood among samples
        action_log_probs = torch.stack(log_probs, dim=0).max(dim=0)[0]
        
        # Compute entropy based on action diversity
        if hasattr(self, '_recent_actions'):
            self._recent_actions = torch.cat([self._recent_actions[-100:], action], dim=0)
        else:
            self._recent_actions = action
        
        action_var = torch.var(self._recent_actions, dim=0).mean()
        dist_entropy = 0.5 * torch.log(2 * np.pi * torch.e * (action_var + 1e-8))
        dist_entropy = dist_entropy.expand(batch_size, 1)
        
        return action_log_probs, dist_entropy, None