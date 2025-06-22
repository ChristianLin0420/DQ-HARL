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
        gain = nn.init.calculate_gain(self.activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain = gain)
        
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
        
        # Generate actions using diffusion process
        actions = self.sample_actions(obs_features, deterministic)
        
        # Compute approximate log probabilities
        # For simplicity, we use a Gaussian approximation
        # In practice, this would require more sophisticated computation
        action_log_probs = -0.5 * torch.sum(actions**2, dim=-1, keepdim=True) - \
                          0.5 * self.action_dim * np.log(2 * np.pi)
        
        return actions, action_log_probs, rnn_states

    def sample_actions(self, obs_features, deterministic=False):
        """
        Sample actions using the reverse diffusion process.
        Implements the GenPO algorithm with coupled noise vectors.
        """
        batch_size = obs_features.shape[0]
        
        # Initialize coupled noise vectors (x, y)
        if deterministic:
            x = torch.zeros(batch_size, self.action_dim, device=self.device)
            y = torch.zeros(batch_size, self.action_dim, device=self.device)
        else:
            x = torch.randn(batch_size, self.action_dim, device=self.device)
            y = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_diffusion_steps)):
            # Create time tensor
            time_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise for both components
            predicted_noise_x = self.diffusion_net(obs_features, x, time_tensor.float())
            predicted_noise_y = self.diffusion_net(obs_features, y, time_tensor.float())
            
            # Compute denoising step size
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            
            # Reverse step (Eq. 7 from paper)
            if t > 0:
                noise_x = torch.randn_like(x) if not deterministic else torch.zeros_like(x)
                noise_y = torch.randn_like(y) if not deterministic else torch.zeros_like(y)
            else:
                noise_x = torch.zeros_like(x)
                noise_y = torch.zeros_like(y)
            
            # Update using predicted noise
            x_tilde = (x - beta_t / torch.sqrt(1 - self.alphas_cumprod[t]) * predicted_noise_x) / torch.sqrt(alpha_t) + torch.sqrt(beta_t) * noise_x
            y_tilde = (y - beta_t / torch.sqrt(1 - self.alphas_cumprod[t]) * predicted_noise_y) / torch.sqrt(alpha_t) + torch.sqrt(beta_t) * noise_y
            
            # Mixing step (Eq. 7 from paper)
            x_new = self.mixing_p * x_tilde + (1 - self.mixing_p) * y_tilde
            y_new = self.mixing_p * y_tilde + (1 - self.mixing_p) * x_tilde
            
            x, y = x_new, y_new
        
        # Final action is average of coupled components
        actions = (x + y) / 2.0
        
        return actions
    
    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate log probabilities and entropy for given actions.
        This is complex for diffusion models - we use approximations.
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
        
        # For diffusion models, computing exact log probabilities is complex
        # We need to compute them through the diffusion network to maintain gradients
        
        # Use the diffusion network to compute a score/log probability
        # This is a simplified approach - in practice you might want the exact computation
        batch_size = obs_features.shape[0]
        
        # Create a dummy time step (we'll use the middle of the diffusion process)
        time_step = torch.full((batch_size,), self.num_diffusion_steps // 2, 
                              device=self.device, dtype=torch.float32)
        
        # Compute noise prediction for the given action
        # This creates a computational graph connected to the diffusion network
        predicted_noise = self.diffusion_net(obs_features, action, time_step)
        
        # Compute log probability based on the noise prediction
        # This is an approximation but maintains gradients
        noise_mse = torch.sum((predicted_noise - action) ** 2, dim=-1, keepdim=True)
        action_log_probs = -0.5 * noise_mse  # Negative MSE as log prob approximation
        
        # Compute entropy based on the noise prediction variance
        # This also maintains gradients through the network
        noise_var = torch.var(predicted_noise, dim=-1, keepdim=True)
        dist_entropy = 0.5 * torch.log(2 * np.pi * torch.e * (noise_var + 1e-8))
        
        # Return None for action distribution as it's complex for diffusion models
        return action_log_probs, dist_entropy, None