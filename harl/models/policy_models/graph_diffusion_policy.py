import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from harl.models.policy_models.diffusion_policy import DiffusionPolicy, SinusoidalPositionalEncoding
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_active_func, get_init_method


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


class GraphConvLayer(nn.Module):
    """Graph convolution layer for processing agent interactions."""
    
    def __init__(self, input_dim, output_dim, activation_func="relu"):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = get_active_func(activation_func)
        
    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: [batch_size, num_agents, input_dim]
            adjacency_matrix: [batch_size, num_agents, num_agents] or [num_agents, num_agents]
        """
        # Aggregate neighbor features
        if adjacency_matrix.dim() == 2:
            adjacency_matrix = adjacency_matrix.unsqueeze(0).expand(node_features.size(0), -1, -1)
        
        # Normalize adjacency matrix (add self-connections and normalize)
        adjacency_matrix = adjacency_matrix + torch.eye(adjacency_matrix.size(-1), device=adjacency_matrix.device)
        degree = adjacency_matrix.sum(dim=-1, keepdim=True).clamp(min=1)
        adjacency_matrix = adjacency_matrix / degree
        
        # Apply graph convolution: A * X * W
        aggregated = torch.bmm(adjacency_matrix, node_features)
        output = self.linear(aggregated)
        return self.activation(output)


class GraphAttentionLayer(nn.Module):
    """Numerically stable graph attention layer."""
    
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Initialize with smaller variance to prevent exploding gradients
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights with smaller variance
        for module in [self.query, self.key, self.value, self.output_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
        
        # Temperature parameter for more stable attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, node_features, adjacency_matrix=None):
        """Numerically stable attention computation."""
        batch_size, num_agents, _ = node_features.shape
        
        # Add small epsilon to prevent NaN in gradients
        node_features = node_features + 1e-8 * torch.randn_like(node_features)
        
        # Linear projections with gradient clipping
        Q = torch.clamp(self.query(node_features), -10, 10)
        K = torch.clamp(self.key(node_features), -10, 10)
        V = torch.clamp(self.value(node_features), -10, 10)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with temperature
        scale = self.temperature / np.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Clamp scores to prevent overflow
        scores = torch.clamp(scores, -10, 10)
        
        # Handle adjacency matrix masking more carefully
        if adjacency_matrix is not None:
            if adjacency_matrix.dim() == 2:
                adjacency_matrix = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Expand for multi-head
            mask = adjacency_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # Use large negative value instead of -inf to prevent NaN
            masked_scores = scores.masked_fill(mask == 0, -1e9)
            
            # Check if any row is completely masked
            row_sums = mask.sum(dim=-1)  # [batch, heads, agents]
            all_masked = (row_sums == 0).any()
            
            if all_masked:
                # If some agents have no connections, use uniform attention
                uniform_attention = torch.ones_like(scores) / num_agents
                attention_weights = torch.where(
                    row_sums.unsqueeze(-1) > 0,
                    F.softmax(masked_scores, dim=-1),
                    uniform_attention
                )
            else:
                attention_weights = F.softmax(masked_scores, dim=-1)
        else:
            attention_weights = F.softmax(scores, dim=-1)
        
        # Check for NaN in attention weights
        if torch.isnan(attention_weights).any():
            print("Warning: NaN detected in attention weights, using uniform weights")
            attention_weights = torch.ones_like(attention_weights) / num_agents
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_agents, self.output_dim)
        output = torch.clamp(self.output_proj(attended), -10, 10)
        
        return output


class GraphDiffusionMLP(nn.Module):
    """Graph-enhanced MLP for diffusion model."""
    
    def __init__(self, args, obs_dim, action_dim, num_agents, time_embed_dim=32):
        super().__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.activation_func = args["activation_func"]
        self.initialization_method = args["initialization_method"]
        self.use_feature_normalization = args.get("use_feature_normalization", True)
        self.num_agents = num_agents
        
        # Graph convolution parameters
        self.num_graph_layers = args.get("num_graph_layers", 2)
        self.graph_hidden_dim = args.get("graph_hidden_dim", 64)
        self.use_graph_attention = args.get("use_graph_attention", False)
        self.num_attention_heads = args.get("num_attention_heads", 4)
        
        # Time embedding
        self.time_embed_dim = time_embed_dim
        self.time_embed = SinusoidalPositionalEncoding(time_embed_dim)
        
        # Graph processing layers
        self.graph_layers = nn.ModuleList()
        graph_input_dim = obs_dim
        
        for i in range(self.num_graph_layers):
            if self.use_graph_attention:
                self.graph_layers.append(
                    GraphAttentionLayer(
                        graph_input_dim, 
                        self.graph_hidden_dim, 
                        self.num_attention_heads
                    )
                )
            else:
                self.graph_layers.append(
                    GraphConvLayer(
                        graph_input_dim, 
                        self.graph_hidden_dim, 
                        self.activation_func
                    )
                )
            graph_input_dim = self.graph_hidden_dim
        
        # Input dimension: graph_features + individual_action + time_embedding
        input_dim = self.graph_hidden_dim + action_dim + time_embed_dim
        
        if self.use_feature_normalization:
            self.input_norm = nn.LayerNorm(input_dim)
        
        # Build MLP layers
        active_func = get_active_func(self.activation_func)
        init_method = get_init_method(self.initialization_method)
        gain = get_activation_gain(self.activation_func)
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_size in self.hidden_sizes:
            layers.append(init_(nn.Linear(prev_dim, hidden_size)))
            layers.append(active_func)
            layers.append(nn.LayerNorm(hidden_size))
            prev_dim = hidden_size
        
        layers.append(init_(nn.Linear(prev_dim, action_dim)))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs_features, action, time, adjacency_matrix=None, agent_id=None):
        """
        Args:
            obs_features: [batch_size, num_agents, obs_dim] or [batch_size, obs_dim]
            action: [batch_size, action_dim]
            time: [batch_size]
            adjacency_matrix: [batch_size, num_agents, num_agents] or None
            agent_id: int, specific agent ID for parameter sharing scenario
        """
        batch_size = obs_features.size(0)
        
        # Handle single agent case
        if obs_features.dim() == 2:
            obs_features = obs_features.unsqueeze(1)  # [batch_size, 1, obs_dim]
        
        # Default adjacency matrix (fully connected)
        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(
                batch_size, self.num_agents, self.num_agents, 
                device=obs_features.device
            )
        
        # Apply graph layers
        graph_features = obs_features
        for graph_layer in self.graph_layers:
            if isinstance(graph_layer, GraphAttentionLayer):
                graph_features = graph_layer(graph_features, adjacency_matrix)
            else:
                graph_features = graph_layer(graph_features, adjacency_matrix)
        
        # Extract agent-specific features
        if agent_id is not None:
            agent_graph_features = graph_features[:, agent_id, :]  # [batch_size, graph_hidden_dim]
        else:
            # Use mean pooling or first agent for single-agent case
            agent_graph_features = graph_features.mean(dim=1)  # [batch_size, graph_hidden_dim]
        
        # Time embedding
        time_embed = self.time_embed(time)
        
        # Concatenate inputs
        x = torch.cat([agent_graph_features, action, time_embed], dim=-1)
        
        if self.use_feature_normalization:
            x = self.input_norm(x)
        
        return self.net(x)


class GraphDiffusionPolicy(DiffusionPolicy):
    """Graph-enhanced diffusion policy for multi-agent settings."""
    
    def __init__(self, args, obs_space, action_space, num_agents=1, device=torch.device("cpu")):
        # Initialize parent class first
        super().__init__(args, obs_space, action_space, device)
        
        self.num_agents = num_agents
        
        # Replace the diffusion network with graph-enhanced version
        self.diffusion_net = GraphDiffusionMLP(
            args,
            self.hidden_sizes[-1],  # Use encoded observation dimension
            self.action_dim,
            num_agents,
            self.time_embed_dim
        )
        
        self.to(device)
    
    def forward(self, obs, rnn_states, masks, available_actions=None, 
                deterministic=False, adjacency_matrix=None, agent_id=None):
        """Enhanced forward pass with graph structure."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        batch_size = obs.shape[0]
        
        # Encode observations
        obs_features = self.obs_encoder(obs)
        
        # Apply RNN if needed
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            obs_features, rnn_states = self.rnn(obs_features, rnn_states, masks)
        
        # Generate actions using graph-enhanced diffusion process
        actions, action_log_probs = self.sample_actions_with_graph(
            obs_features, adjacency_matrix, agent_id, deterministic
        )
        
        return actions, action_log_probs, rnn_states
    
    def sample_actions_with_graph(self, obs_features, adjacency_matrix=None, 
                                 agent_id=None, deterministic=False):
        """Sample actions using graph-enhanced reverse diffusion process."""
        batch_size = obs_features.shape[0]
        
        # Initialize coupled noise vectors
        if deterministic:
            x = torch.zeros(batch_size, self.action_dim, device=self.device)
            y = torch.zeros(batch_size, self.action_dim, device=self.device)
        else:
            x = torch.randn(batch_size, self.action_dim, device=self.device)
            y = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Store initial noise for log probability computation
        x_init, y_init = x.clone(), y.clone()
        
        # Prepare observations for graph processing
        if obs_features.dim() == 2 and self.num_agents > 1:
            # Reshape for multi-agent case
            obs_for_graph = obs_features.unsqueeze(1).expand(-1, self.num_agents, -1)
        else:
            obs_for_graph = obs_features
        
        # Reverse diffusion process with graph enhancement
        for t in reversed(range(self.num_diffusion_steps)):
            time_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.float32)
            
            # Predict noise using graph-enhanced network
            predicted_noise_x = self.diffusion_net(
                obs_for_graph, x, time_tensor, adjacency_matrix, agent_id
            )
            predicted_noise_y = self.diffusion_net(
                obs_for_graph, y, time_tensor, adjacency_matrix, agent_id
            )
            
            # Compute denoising step parameters
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            
            # Compute mean of reverse process
            mean_x = (x - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise_x) / sqrt_alpha_t
            mean_y = (y - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise_y) / sqrt_alpha_t
            
            if t > 0:
                if not deterministic:
                    noise_x = torch.randn_like(x)
                    noise_y = torch.randn_like(y)
                else:
                    noise_x = torch.zeros_like(x)
                    noise_y = torch.zeros_like(y)
                
                x_tilde = mean_x + torch.sqrt(beta_t) * noise_x
                y_tilde = mean_y + torch.sqrt(beta_t) * noise_y
            else:
                x_tilde = mean_x
                y_tilde = mean_y
            
            # Mixing step (Eq. 7 from GenPO paper)
            x_new = self.mixing_p * x_tilde + (1 - self.mixing_p) * y_tilde
            y_new = self.mixing_p * y_tilde + (1 - self.mixing_p) * x_tilde
            
            x, y = x_new, y_new
        
        # Final action
        actions = (x + y) / 2.0
        
        # Compute log probabilities (simplified for now)
        initial_log_prob = -0.5 * torch.sum(x_init**2 + y_init**2, dim=-1, keepdim=True) - \
                          self.action_dim * np.log(2 * np.pi)
        jacobian_correction = -self.action_dim * np.log(2)
        total_log_prob = initial_log_prob + jacobian_correction
        
        return actions, total_log_prob
    
    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, 
                        active_masks=None, adjacency_matrix=None, agent_id=None):
        """Evaluate actions with graph structure."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # Encode observations
        obs_features = self.obs_encoder(obs)
        
        # Apply RNN if needed
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            obs_features, _ = self.rnn(obs_features, rnn_states, masks)
        
        # Use graph-enhanced evaluation
        batch_size = obs_features.shape[0]
        
        # Sample multiple actions for better evaluation
        num_samples = 5
        log_probs = []
        
        for _ in range(num_samples):
            sample_action, sample_log_prob = self.sample_actions_with_graph(
                obs_features, adjacency_matrix, agent_id, deterministic=False
            )
            
            # Compute likelihood of target action
            mse = torch.sum((action - sample_action)**2, dim=-1, keepdim=True)
            log_prob = -0.5 * mse / (0.1**2) - 0.5 * self.action_dim * np.log(2 * np.pi * 0.1**2)
            log_probs.append(log_prob)
        
        action_log_probs = torch.stack(log_probs, dim=0).max(dim=0)[0]
        
        # Compute entropy
        if hasattr(self, '_recent_actions'):
            self._recent_actions = torch.cat([self._recent_actions[-100:], action], dim=0)
        else:
            self._recent_actions = action
        
        action_var = torch.var(self._recent_actions, dim=0).mean()
        dist_entropy = 0.5 * torch.log(2 * np.pi * torch.e * (action_var + 1e-8))
        dist_entropy = dist_entropy.expand(batch_size, 1)
        
        return action_log_probs, dist_entropy, None 