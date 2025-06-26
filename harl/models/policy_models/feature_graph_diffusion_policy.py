import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from harl.models.policy_models.diffusion_policy import DiffusionPolicy, SinusoidalPositionalEncoding
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_active_func, get_init_method


def get_activation_gain(activation_func):
    """Get the gain for activation function, with custom handling for newer activations."""
    if activation_func in ["silu", "swish"]:
        return nn.init.calculate_gain("relu")
    elif activation_func == "gelu":
        return nn.init.calculate_gain("relu")
    elif activation_func == "mish":
        return nn.init.calculate_gain("relu")
    else:
        try:
            return nn.init.calculate_gain(activation_func)
        except ValueError:
            print(f"Warning: Unknown activation function '{activation_func}', using gain=1.0")
            return 1.0


class FeatureGraphBuilder(nn.Module):
    """
    Builds graph structure from individual agent observations.
    Each scalar feature becomes a node in the graph.
    CTDE compliant - operates only on single agent's observation.
    """
    
    def __init__(self, obs_dim, adjacency_type="spatial", args=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.adjacency_type = adjacency_type
        self.args = args or {}
        
        if adjacency_type == "learned":
            # Learn pairwise feature relationships
            self.feature_relation_net = nn.Sequential(
                nn.Linear(2, 64),  # Takes pair of feature values
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
    
    def get_feature_adjacency(self, obs):
        """
        Generate adjacency matrix for observation features.
        
        Args:
            obs: [batch_size, obs_dim] - single agent observation
        
        Returns:
            adjacency: [batch_size, obs_dim, obs_dim] - feature adjacency matrix
        """
        batch_size, obs_dim = obs.shape
        
        if self.adjacency_type == "spatial":
            return self._spatial_adjacency(batch_size, obs_dim, obs.device)
            
        elif self.adjacency_type == "correlation":
            return self._correlation_adjacency(obs)
            
        elif self.adjacency_type == "learned":
            return self._learned_adjacency(obs)
            
        elif self.adjacency_type == "semantic":
            return self._semantic_adjacency(batch_size, obs_dim, obs.device)
            
        else:  # fully_connected
            return self._fully_connected_adjacency(batch_size, obs_dim, obs.device)
    
    def _spatial_adjacency(self, batch_size, obs_dim, device):
        """Connect spatially nearby features (good for structured observations)."""
        adj = torch.zeros(batch_size, obs_dim, obs_dim, device=device)
        
        for i in range(obs_dim):
            # Connect to nearby features in observation vector
            for j in range(max(0, i-2), min(obs_dim, i+3)):
                if i != j:
                    # Distance-based weight (closer features have stronger connections)
                    weight = 1.0 / (1.0 + abs(i - j))
                    adj[:, i, j] = weight
        
        return adj
    
    def _correlation_adjacency(self, obs):
        """Connect features based on correlation across batch."""
        batch_size, obs_dim = obs.shape
        
        if batch_size < 2:
            # Fallback to spatial if batch too small
            return self._spatial_adjacency(batch_size, obs_dim, obs.device)
        
        # Compute pairwise correlations
        obs_centered = obs - obs.mean(dim=0, keepdim=True)
        obs_std = obs_centered.std(dim=0, keepdim=True) + 1e-8
        obs_norm = obs_centered / obs_std
        
        # Correlation matrix
        corr_matrix = torch.mm(obs_norm.T, obs_norm) / batch_size
        
        # Convert to adjacency (absolute correlation > threshold)
        threshold = 0.3
        adj = (torch.abs(corr_matrix) > threshold).float()
        
        # Remove self-connections
        adj = adj - torch.eye(obs_dim, device=obs.device)
        
        # Expand to batch dimension
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        return adj
    
    def _learned_adjacency(self, obs):
        """Learn feature relationships through neural network."""
        batch_size, obs_dim = obs.shape
        
        # Create all pairwise feature combinations
        obs_i = obs.unsqueeze(2).expand(-1, -1, obs_dim)  # [B, D, D]
        obs_j = obs.unsqueeze(1).expand(-1, obs_dim, -1)  # [B, D, D]
        
        # Pairwise features [value_i, value_j]
        pairwise_features = torch.stack([obs_i, obs_j], dim=-1)  # [B, D, D, 2]
        
        # Flatten for processing
        pairwise_flat = pairwise_features.view(-1, 2)  # [B*D*D, 2]
        
        # Compute adjacency weights
        adj_weights = self.feature_relation_net(pairwise_flat)  # [B*D*D, 1]
        
        # Reshape back
        adj = adj_weights.view(batch_size, obs_dim, obs_dim)  # [B, D, D]
        
        # Remove self-connections
        eye = torch.eye(obs_dim, device=obs.device).unsqueeze(0)
        adj = adj * (1 - eye)
        
        return adj
    
    def _semantic_adjacency(self, batch_size, obs_dim, device):
        """Connect features based on semantic meaning (domain-specific)."""
        adj = torch.zeros(batch_size, obs_dim, obs_dim, device=device)
        
        # Example semantic groupings for common observation structures
        if obs_dim >= 6:
            # Position features (assuming first 2-3 features are position-related)
            adj[:, 0, 1] = adj[:, 1, 0] = 1.0  # pos_x - pos_y
            if obs_dim > 2:
                adj[:, 0, 2] = adj[:, 2, 0] = 0.8  # pos_x - pos_z
                adj[:, 1, 2] = adj[:, 2, 1] = 0.8  # pos_y - pos_z
            
            # Velocity features (assuming next 2-3 features are velocity-related)
            if obs_dim > 5:
                adj[:, 3, 4] = adj[:, 4, 3] = 1.0  # vel_x - vel_y
                if obs_dim > 5:
                    adj[:, 3, 5] = adj[:, 5, 3] = 0.8  # vel_x - vel_z
                    adj[:, 4, 5] = adj[:, 5, 4] = 0.8  # vel_y - vel_z
                
                # Cross-category connections (pos-vel relationships)
                adj[:, 0, 3] = adj[:, 3, 0] = 0.6  # pos_x - vel_x
                adj[:, 1, 4] = adj[:, 4, 1] = 0.6  # pos_y - vel_y
                if obs_dim > 5:
                    adj[:, 2, 5] = adj[:, 5, 2] = 0.6  # pos_z - vel_z
        
        # Add some random connectivity for remaining features
        for i in range(6, obs_dim):
            for j in range(i+1, min(i+4, obs_dim)):
                adj[:, i, j] = adj[:, j, i] = 0.3
        
        return adj
    
    def _fully_connected_adjacency(self, batch_size, obs_dim, device):
        """Fully connected feature graph."""
        adj = torch.ones(batch_size, obs_dim, obs_dim, device=device)
        
        # Remove self-connections
        eye = torch.eye(obs_dim, device=device).unsqueeze(0)
        adj = adj - eye
        
        return adj


class FeatureGraphConvolution(nn.Module):
    """Graph convolution operating on observation features."""
    
    def __init__(self, feature_dim, output_dim, activation_func="relu"):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # Transform scalar features to vectors
        self.feature_embed = nn.Linear(1, feature_dim)
        
        # Graph convolution components
        self.self_transform = nn.Linear(feature_dim, output_dim)
        self.neighbor_transform = nn.Linear(feature_dim, output_dim)
        
        self.activation = get_active_func(activation_func)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Initialize with smaller weights for stability
        nn.init.xavier_uniform_(self.feature_embed.weight, gain=0.1)
        nn.init.xavier_uniform_(self.self_transform.weight, gain=0.1)
        nn.init.xavier_uniform_(self.neighbor_transform.weight, gain=0.1)
    
    def forward(self, features, adjacency_matrix):
        """
        Args:
            features: [batch_size, obs_dim] - scalar observation features
            adjacency_matrix: [batch_size, obs_dim, obs_dim]
        
        Returns:
            processed_features: [batch_size, obs_dim, output_dim]
        """
        batch_size, obs_dim = features.shape
        
        # Transform scalar features to vectors
        feature_vectors = self.feature_embed(features.unsqueeze(-1))  # [B, D, feature_dim]
        
        # Self transformation
        self_features = self.self_transform(feature_vectors)  # [B, D, output_dim]
        
        # Neighbor aggregation with normalized adjacency
        degree = adjacency_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized_adj = adjacency_matrix / degree
        
        # Aggregate neighbor features
        neighbor_features = torch.bmm(normalized_adj, feature_vectors)  # [B, D, feature_dim]
        neighbor_features = self.neighbor_transform(neighbor_features)  # [B, D, output_dim]
        
        # Combine self and neighbor information
        output = self.activation(self_features + neighbor_features)
        output = self.layer_norm(output)
        
        return output


class FeatureGraphAttention(nn.Module):
    """Graph attention mechanism for observation features."""
    
    def __init__(self, feature_dim, output_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Feature embedding
        self.feature_embed = nn.Linear(1, feature_dim)
        
        # Attention components
        self.query = nn.Linear(feature_dim, output_dim)
        self.key = nn.Linear(feature_dim, output_dim)
        self.value = nn.Linear(feature_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Temperature for stable attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Initialize with smaller weights
        for module in [self.feature_embed, self.query, self.key, self.value, self.output_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
    
    def forward(self, features, adjacency_matrix=None):
        """
        Args:
            features: [batch_size, obs_dim] - scalar observation features
            adjacency_matrix: [batch_size, obs_dim, obs_dim] - optional mask
        
        Returns:
            attended_features: [batch_size, obs_dim, output_dim]
        """
        batch_size, obs_dim = features.shape
        
        # Transform scalar features to vectors
        feature_vectors = self.feature_embed(features.unsqueeze(-1))  # [B, D, feature_dim]
        
        # Attention computation
        Q = self.query(feature_vectors)  # [B, D, output_dim]
        K = self.key(feature_vectors)
        V = self.value(feature_vectors)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, obs_dim, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, obs_dim, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, obs_dim, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with temperature
        scale = self.temperature / np.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Clamp scores for numerical stability
        scores = torch.clamp(scores, -10, 10)
        
        # Apply adjacency mask if provided
        if adjacency_matrix is not None:
            mask = adjacency_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Check for NaN and use uniform weights as fallback
        if torch.isnan(attention_weights).any():
            print("Warning: NaN in attention weights, using uniform weights")
            attention_weights = torch.ones_like(attention_weights) / obs_dim
        
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, obs_dim, self.output_dim)
        output = self.output_proj(attended)
        output = self.layer_norm(output)
        
        return output


class FeatureGraphDiffusionMLP(nn.Module):
    """
    Feature Graph Diffusion MLP for FGAPPO.
    CTDE compliant - processes only single agent's observation features.
    """
    
    def __init__(self, args, obs_dim, action_dim, time_embed_dim=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = args["hidden_sizes"]
        self.activation_func = args["activation_func"]
        self.use_feature_normalization = args.get("use_feature_normalization", True)
        
        # Feature graph parameters
        self.num_feature_graph_layers = args.get("num_feature_graph_layers", 2)
        self.feature_graph_hidden_dim = args.get("feature_graph_hidden_dim", 32)
        self.use_feature_attention = args.get("use_feature_attention", False)
        self.feature_adjacency_type = args.get("feature_adjacency_type", "spatial")
        self.feature_embed_dim = args.get("feature_embed_dim", 16)
        
        # Time embedding
        self.time_embed_dim = time_embed_dim
        self.time_embed = SinusoidalPositionalEncoding(time_embed_dim)
        
        # Feature graph builder
        self.feature_graph_builder = FeatureGraphBuilder(
            obs_dim, self.feature_adjacency_type, args
        )
        
        # Feature graph processing layers
        self.feature_graph_layers = nn.ModuleList()
        
        for i in range(self.num_feature_graph_layers):
            if self.use_feature_attention:
                layer = FeatureGraphAttention(
                    self.feature_embed_dim if i == 0 else self.feature_graph_hidden_dim,
                    self.feature_graph_hidden_dim,
                    num_heads=2,
                    dropout=0.1
                )
            else:
                layer = FeatureGraphConvolution(
                    self.feature_embed_dim if i == 0 else self.feature_graph_hidden_dim,
                    self.feature_graph_hidden_dim,
                    self.activation_func
                )
            self.feature_graph_layers.append(layer)
        
        # Feature aggregation: convert from [B, D, H] to [B, agg_dim]
        self.feature_aggregator = nn.Sequential(
            nn.Linear(self.feature_graph_hidden_dim * obs_dim, self.hidden_sizes[0]),
            get_active_func(self.activation_func),
            nn.LayerNorm(self.hidden_sizes[0])
        )
        
        # Input dimension: aggregated_features + action + time_embedding
        input_dim = self.hidden_sizes[0] + action_dim + time_embed_dim
        
        if self.use_feature_normalization:
            self.input_norm = nn.LayerNorm(input_dim)
        
        # Build final MLP layers
        init_method = get_init_method(args.get("initialization_method", "orthogonal_"))
        gain = get_activation_gain(self.activation_func)
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                init_(nn.Linear(prev_dim, hidden_size)),
                get_active_func(self.activation_func),
                nn.LayerNorm(hidden_size)
            ])
            prev_dim = hidden_size
        
        layers.append(init_(nn.Linear(prev_dim, action_dim)))
        self.final_mlp = nn.Sequential(*layers)
    
    def forward(self, obs, action, time):
        """
        Process single agent observation through feature graph.
        
        Args:
            obs: [batch_size, obs_dim] - single agent observation  
            action: [batch_size, action_dim] - current action in diffusion
            time: [batch_size] - diffusion time step
        
        Returns:
            noise_prediction: [batch_size, action_dim]
        """
        batch_size = obs.size(0)
        
        # Build feature adjacency matrix (CTDE compliant - only uses this agent's obs)
        feature_adjacency = self.feature_graph_builder.get_feature_adjacency(obs)
        
        # Process features through graph layers
        current_features = obs  # Start with raw scalar features [B, D]
        
        for i, layer in enumerate(self.feature_graph_layers):
            if isinstance(layer, FeatureGraphAttention):
                graph_output = layer(current_features, feature_adjacency)  # [B, D, H]
            else:
                graph_output = layer(current_features, feature_adjacency)  # [B, D, H]
            
            # For next iteration, we need scalar features
            # Use mean pooling across feature dimension to get [B, D]
            if i < len(self.feature_graph_layers) - 1:
                current_features = graph_output.mean(dim=-1)  # [B, D]
        
        # Final output is [B, D, H], flatten for aggregation
        flattened_features = graph_output.view(batch_size, -1)  # [B, D*H]
        
        # Aggregate features
        processed_features = self.feature_aggregator(flattened_features)  # [B, hidden_size[0]]
        
        # Time embedding
        time_embed = self.time_embed(time)  # [B, time_embed_dim]
        
        # Combine all inputs
        combined_input = torch.cat([processed_features, action, time_embed], dim=-1)
        
        if self.use_feature_normalization:
            combined_input = self.input_norm(combined_input)
        
        # Generate noise prediction
        noise_prediction = self.final_mlp(combined_input)
        
        return noise_prediction


class FeatureGraphDiffusionPolicy(DiffusionPolicy):
    """
    CTDE-compliant diffusion policy using feature-level graphs.
    Each agent processes only their own observation features as graph nodes.
    """
    
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        # Initialize parent class first
        super().__init__(args, obs_space, action_space, device)
        
        # Determine the encoded observation dimension by doing a test forward pass
        with torch.no_grad():
            test_obs = torch.zeros(1, self.obs_dim, device=device)
            encoded_test_obs = self.obs_encoder(test_obs)
            encoded_obs_dim = encoded_test_obs.shape[-1]
        
        # Replace diffusion network with feature graph version using encoded obs dim
        self.diffusion_net = FeatureGraphDiffusionMLP(
            args,
            encoded_obs_dim,  # Use actual encoded dimension
            self.action_dim,
            self.time_embed_dim
        )
        
        self.to(device)
    
    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        CTDE-compliant forward pass.
        Each agent processes only their own observation features.
        
        Args:
            obs: [batch_size, obs_dim] - single agent observation
            rnn_states: RNN states for this agent
            masks: Masks for this agent
            available_actions: Available actions for this agent
            deterministic: Whether to use deterministic sampling
        
        Returns:
            actions: [batch_size, action_dim]
            action_log_probs: [batch_size, 1]
            rnn_states: Updated RNN states
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # Encode observations to get processed features
        obs_features = self.obs_encoder(obs)
        
        # Apply RNN if needed
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            obs_features, rnn_states = self.rnn(obs_features, rnn_states, masks)
        
        # Generate actions using feature graph enhanced diffusion
        # Now obs_features has the correct encoded dimension
        actions, action_log_probs = self.sample_actions_with_feature_graph(
            obs_features, deterministic
        )
        
        return actions, action_log_probs, rnn_states
    
    def sample_actions_with_feature_graph(self, obs_features, deterministic=False):
        """
        Sample actions using feature graph enhanced reverse diffusion process.
        """
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
        
        # Reverse diffusion process with feature graph enhancement
        for t in reversed(range(self.num_diffusion_steps)):
            time_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.float32)
            
            # Predict noise using feature graph network
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
            
            # Mixing step (GenPO)
            x_new = self.mixing_p * x_tilde + (1 - self.mixing_p) * y_tilde
            y_new = self.mixing_p * y_tilde + (1 - self.mixing_p) * x_tilde
            
            x, y = x_new, y_new
        
        # Final action is average of coupled components
        actions = (x + y) / 2.0
        
        # Compute log probabilities (simplified)
        initial_log_prob = -0.5 * torch.sum(x_init**2 + y_init**2, dim=-1, keepdim=True) - \
                          self.action_dim * np.log(2 * np.pi)
        jacobian_correction = -self.action_dim * np.log(2)
        total_log_prob = initial_log_prob + jacobian_correction
        
        return actions, total_log_prob
    
    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate actions using feature graph structure.
        CTDE compliant - uses only agent's own observation.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # Encode observations to get processed features
        obs_features = self.obs_encoder(obs)
        
        # Apply RNN if needed
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            obs_features, _ = self.rnn(obs_features, rnn_states, masks)
        
        # Use feature graph enhanced evaluation
        batch_size = obs_features.shape[0]
        
        # Sample multiple actions for better evaluation
        num_samples = 5
        log_probs = []
        
        for _ in range(num_samples):
            sample_action, sample_log_prob = self.sample_actions_with_feature_graph(
                obs_features, deterministic=False
            )
            
            # Compute likelihood of target action
            mse = torch.sum((action - sample_action)**2, dim=-1, keepdim=True)
            log_prob = -0.5 * mse / (0.1**2) - 0.5 * self.action_dim * np.log(2 * np.pi * 0.1**2)
            log_probs.append(log_prob)
        
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