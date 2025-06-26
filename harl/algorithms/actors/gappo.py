import torch
import torch.nn as nn
import numpy as np
from harl.algorithms.actors.dappo import DAPPO
from harl.models.policy_models.graph_diffusion_policy import GraphDiffusionPolicy
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm


class GAPPO(DAPPO):
    """
    Graph-Augmented PPO (GAPPO) Actor.
    Extends DAPPO with graph neural network capabilities for multi-agent coordination.
    """
    
    def __init__(self, args, obs_space, act_space, num_agents=1, device=torch.device("cpu")):
        # Store graph-specific parameters
        self.num_agents = num_agents
        self.use_graph_attention = args.get("use_graph_attention", False)
        self.adjacency_type = args.get("adjacency_type", "fully_connected")  # fully_connected, distance_based, learned
        
        # Initialize parent class first
        super().__init__(args, obs_space, act_space, device)
        
        # Replace the actor with graph-enhanced version
        self.actor = GraphDiffusionPolicy(args, obs_space, act_space, num_agents, device)
        
        # Recreate optimizer with new parameters
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        
        # Graph-specific components
        if self.adjacency_type == "learned":
            self.adjacency_net = self._create_adjacency_network(args)
            self.adjacency_net.to(self.device)
            self.adjacency_optimizer = torch.optim.Adam(
                self.adjacency_net.parameters(),
                lr=self.lr * 0.1,  # Use smaller learning rate for adjacency learning
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            )
    
    def _create_adjacency_network(self, args):
        """Create a neural network to learn adjacency matrices."""
        from harl.models.base.mlp import MLPLayer
        
        # Network that takes agent observations and outputs adjacency weights
        obs_dim = self.actor.obs_dim
        input_dim = obs_dim * self.num_agents  # Concatenated observations
        hidden_sizes = args.get("adjacency_hidden_sizes", [128, 64])
        output_dim = self.num_agents * self.num_agents  # Flattened adjacency matrix
        
        adjacency_net = MLPLayer(
            input_dim, 
            hidden_sizes + [output_dim], 
            args["initialization_method"], 
            args["activation_func"]
        )
        
        return adjacency_net
    
    def get_adjacency_matrix(self, obs_batch, agent_positions=None):
        """
        Generate adjacency matrix based on observations or environment.
        
        Args:
            obs_batch: [batch_size, obs_dim] or [batch_size, num_agents, obs_dim]
            agent_positions: [batch_size, num_agents, position_dim] for distance-based adjacency
        
        Returns:
            adjacency_matrix: [batch_size, num_agents, num_agents]
        """
        batch_size = obs_batch.shape[0]
        
        if self.adjacency_type == "fully_connected":
            # Default: fully connected graph without self-connections
            adj_matrix = torch.ones(
                batch_size, self.num_agents, self.num_agents,
                device=self.device
            )
            # Remove self-connections
            adj_matrix = adj_matrix - torch.eye(
                self.num_agents, device=self.device
            ).unsqueeze(0)
            
        elif self.adjacency_type == "distance_based" and agent_positions is not None:
            # Create adjacency based on distance between agents
            adj_matrix = self._compute_distance_adjacency(agent_positions)
            
        elif self.adjacency_type == "learned":
            # Learn adjacency matrix using neural network
            adj_matrix = self._compute_learned_adjacency(obs_batch)
            
        else:
            # Fallback to fully connected
            adj_matrix = torch.ones(
                batch_size, self.num_agents, self.num_agents,
                device=self.device
            )
            adj_matrix = adj_matrix - torch.eye(
                self.num_agents, device=self.device
            ).unsqueeze(0)
        
        return adj_matrix
    
    def _compute_distance_adjacency(self, agent_positions, threshold=2.0):
        """Compute adjacency matrix based on agent distances."""
        batch_size, num_agents, _ = agent_positions.shape
        
        # Compute pairwise distances
        pos_expanded_i = agent_positions.unsqueeze(2)  # [batch, agents, 1, pos_dim]
        pos_expanded_j = agent_positions.unsqueeze(1)  # [batch, 1, agents, pos_dim]
        distances = torch.norm(pos_expanded_i - pos_expanded_j, dim=-1)  # [batch, agents, agents]
        
        # Create adjacency matrix based on distance threshold
        adj_matrix = (distances < threshold).float()
        
        # Remove self-connections
        adj_matrix = adj_matrix - torch.eye(num_agents, device=self.device).unsqueeze(0)
        
        return adj_matrix
    
    def _compute_learned_adjacency(self, obs_batch):
        """Compute adjacency matrix using learned neural network."""
        batch_size = obs_batch.shape[0]
        
        # Handle different observation formats
        if obs_batch.dim() == 3:  # [batch_size, num_agents, obs_dim]
            obs_flat = obs_batch.view(batch_size, -1)
        else:  # [batch_size, obs_dim] - replicate for all agents
            obs_flat = obs_batch.unsqueeze(1).expand(-1, self.num_agents, -1)
            obs_flat = obs_flat.view(batch_size, -1)
        
        # Generate adjacency weights
        adj_logits = self.adjacency_net(obs_flat)  # [batch_size, num_agents * num_agents]
        adj_logits = adj_logits.view(batch_size, self.num_agents, self.num_agents)
        
        # Apply sigmoid to get weights in [0, 1]
        adj_matrix = torch.sigmoid(adj_logits)
        
        # Remove self-connections
        adj_matrix = adj_matrix * (1 - torch.eye(self.num_agents, device=self.device).unsqueeze(0))
        
        return adj_matrix
    
    def get_agent_id(self, agent_idx=None):
        """Get agent ID for parameter sharing scenarios."""
        if agent_idx is not None:
            return agent_idx
        return None  # Use mean pooling in graph layers
    
    def update(self, sample, agent_id=None, agent_positions=None):
        """Enhanced update with graph structure."""
        # Handle different sample tuple lengths
        if len(sample) == 9:
            (
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                factor_batch,
            ) = sample
        else:
            (
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            ) = sample
            factor_batch = None
        
        # Convert to tensors
        obs_batch = check(obs_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        
        if factor_batch is None:
            factor_batch = torch.ones_like(adv_targ)
        else:
            factor_batch = check(factor_batch).to(**self.tpdv)
        
        # Get adjacency matrix
        adjacency_matrix = self.get_adjacency_matrix(obs_batch, agent_positions)
        
        # Get new action log probabilities with graph structure
        action_log_probs, dist_entropy, _ = self.evaluate_actions_with_graph(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            adjacency_matrix,
            agent_id,
        )
        
        # Compute importance weights
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        
        # Standard PPO clipped loss (L^PPO)
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )
        
        if self.use_policy_active_masks:
            policy_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()
        
        # Entropy loss with importance sampling (λL^ENT from GenPO)
        entropy_loss = -torch.mean(imp_weights.detach() * action_log_probs)
        
        # Compression loss (νE[(x1 - y1)²] from GenPO)
        compression_loss = self.compute_compression_loss_with_graph(
            obs_batch, rnn_states_batch, masks_batch, adjacency_matrix, agent_id
        )
        
        # Graph regularization loss (optional)
        graph_reg_loss = self.compute_graph_regularization(adjacency_matrix)
        
        # Ensure all losses are scalars
        if dist_entropy.dim() > 0:
            dist_entropy = dist_entropy.mean()
        
        # Complete GAPPO loss
        total_loss = (
            policy_loss + 
            self.lambda_ent * entropy_loss + 
            self.nu_compression * compression_loss +
            0.001 * graph_reg_loss -  # Small weight for graph regularization
            dist_entropy * self.entropy_coef
        )
        
        # Update policy
        self.actor_optimizer.zero_grad()
        if self.adjacency_type == "learned":
            self.adjacency_optimizer.zero_grad()
        
        total_loss.backward()
        
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
            if self.adjacency_type == "learned":
                nn.utils.clip_grad_norm_(
                    self.adjacency_net.parameters(), self.max_grad_norm
                )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        
        self.actor_optimizer.step()
        if self.adjacency_type == "learned":
            self.adjacency_optimizer.step()
        
        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
    
    def compute_compression_loss_with_graph(self, obs_batch, rnn_states_batch, masks_batch, 
                                          adjacency_matrix, agent_id):
        """Compute compression loss with graph structure."""
        obs_batch = check(obs_batch).to(**self.tpdv)
        rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        
        with torch.no_grad():
            # Get observation features
            obs_features = self.actor.obs_encoder(obs_batch)
            if self.actor.use_naive_recurrent_policy or self.actor.use_recurrent_policy:
                obs_features, _ = self.actor.rnn(obs_features, rnn_states_batch, masks_batch)
        
        # Sample dummy actions with graph structure
        batch_size = obs_features.shape[0]
        
        # Initialize coupled noise vectors
        x = torch.randn(batch_size, self.actor.action_dim, device=self.device, requires_grad=True)
        y = torch.randn(batch_size, self.actor.action_dim, device=self.device, requires_grad=True)
        
        # Run a few steps of the diffusion process with graph enhancement
        for t in reversed(range(min(3, self.actor.num_diffusion_steps))):
            time_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.float32)
            
            # Prepare observations for graph processing
            if obs_features.dim() == 2 and self.num_agents > 1:
                obs_for_graph = obs_features.unsqueeze(1).expand(-1, self.num_agents, -1)
            else:
                obs_for_graph = obs_features
            
            # Predict noise for both components using graph network
            predicted_noise_x = self.actor.diffusion_net(
                obs_for_graph, x, time_tensor, adjacency_matrix, agent_id
            )
            predicted_noise_y = self.actor.diffusion_net(
                obs_for_graph, y, time_tensor, adjacency_matrix, agent_id
            )
            
            # Simple denoising step
            alpha_t = self.actor.alphas[t]
            beta_t = self.actor.betas[t]
            
            x = (x - beta_t * predicted_noise_x) / torch.sqrt(alpha_t)
            y = (y - beta_t * predicted_noise_y) / torch.sqrt(alpha_t)
            
            # Mixing step
            x_new = self.actor.mixing_p * x + (1 - self.actor.mixing_p) * y
            y_new = self.actor.mixing_p * y + (1 - self.actor.mixing_p) * x
            x, y = x_new, y_new
        
        # Compute compression loss: E[(x - y)²]
        compression_loss = torch.mean((x - y) ** 2)
        
        return compression_loss
    
    def compute_graph_regularization(self, adjacency_matrix):
        """Compute graph regularization loss to encourage meaningful connections."""
        if adjacency_matrix is None:
            return torch.tensor(0.0, device=self.device)
        
        # Encourage sparsity in learned adjacency matrices
        if self.adjacency_type == "learned":
            # L1 regularization to promote sparsity
            sparsity_loss = torch.mean(torch.abs(adjacency_matrix))
            
            # Symmetry regularization (optional)
            symmetry_loss = torch.mean((adjacency_matrix - adjacency_matrix.transpose(-2, -1)) ** 2)
            
            return sparsity_loss + 0.1 * symmetry_loss
        
        return torch.tensor(0.0, device=self.device)
    
    def evaluate_actions_with_graph(self, obs, rnn_states, action, masks, 
                                   available_actions=None, active_masks=None, 
                                   adjacency_matrix=None, agent_id=None):
        """Evaluate actions with graph structure."""
        return self.actor.evaluate_actions(
            obs, rnn_states, action, masks, available_actions, 
            active_masks, adjacency_matrix, agent_id
        )
    
    def get_actions(self, obs, rnn_states, masks, available_actions=None, 
                   deterministic=False, adjacency_matrix=None, agent_id=None):
        """Get actions with graph structure."""
        return self.actor.forward(
            obs, rnn_states, masks, available_actions, deterministic, 
            adjacency_matrix, agent_id
        )
    
    def train(self, actor_buffer, advantages, state_type, agent_positions=None):
        """
        Train the GAPPO actor using the graph diffusion policy (non-parameter-sharing version).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample, agent_positions=agent_positions
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type, agent_positions=None):
        """
        Train the GAPPO actor using parameter sharing across multiple agents.
        This method is called when share_param is True in the configuration.
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (
                std_advantages + 1e-5
            )
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(8)]
                for agent_id, generator in enumerate(data_generators):
                    sample = next(generator)
                    for i in range(8):
                        batches[i].append(sample[i])
                
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[7][0] is None:
                    batches[7] = None
                else:
                    batches[7] = np.concatenate(batches[7], axis=0)
                
                # For parameter sharing, we update with all agents' data
                # but use a specific agent_id for graph structure
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    tuple(batches), agent_id=0, agent_positions=agent_positions  # Use agent 0 as reference
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info 