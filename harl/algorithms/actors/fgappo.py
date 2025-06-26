import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from harl.algorithms.actors.dappo import DAPPO
from harl.models.policy_models.feature_graph_diffusion_policy import FeatureGraphDiffusionPolicy
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm


class FGAPPO(DAPPO):
    """
    Feature Graph-Augmented PPO (FGAPPO) Actor.
    CTDE-compliant extension of DAPPO that treats observation features as graph nodes.
    Each agent processes only their own observation features through graph neural networks.
    """
    
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        # Store feature graph specific parameters
        self.feature_adjacency_type = args.get("feature_adjacency_type", "spatial")
        self.use_feature_attention = args.get("use_feature_attention", False)
        self.num_feature_graph_layers = args.get("num_feature_graph_layers", 2)
        self.feature_graph_hidden_dim = args.get("feature_graph_hidden_dim", 32)
        
        # Initialize parent class first
        super().__init__(args, obs_space, act_space, device)
        
        # Replace the actor with feature graph enhanced version
        self.actor = FeatureGraphDiffusionPolicy(args, obs_space, act_space, device)
        
        # Recreate optimizer with new parameters
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        
        # Handle learned feature adjacency
        if self.feature_adjacency_type == "learned":
            # The feature graph builder's learned network is part of actor parameters
            # No separate optimizer needed as it's included in actor_optimizer
            pass
    
    def get_actions(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Get actions for a single agent using only their observation features.
        CTDE compliant - no inter-agent communication required.
        
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
        return self.actor.forward(obs, rnn_states, masks, available_actions, deterministic)
    
    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate actions using only agent's own observation features.
        CTDE compliant - no inter-agent information required.
        
        Args:
            obs: [batch_size, obs_dim] - single agent observation
            rnn_states: RNN states for this agent
            action: [batch_size, action_dim] - actions to evaluate
            masks: Masks for this agent
            available_actions: Available actions for this agent
            active_masks: Active masks for this agent
        
        Returns:
            action_log_probs: [batch_size, 1] - log probabilities
            dist_entropy: [batch_size, 1] - entropy
            rnn_states: None (not returned by evaluate_actions)
        """
        return self.actor.evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks)
    
    def update(self, sample):
        """
        Enhanced update with feature graph structure.
        Maintains CTDE compliance during training.
        """
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
        
        # Get new action log probabilities using feature graph structure
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
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
        compression_loss = self.compute_compression_loss_with_feature_graph(
            obs_batch, rnn_states_batch, masks_batch
        )
        
        # Feature graph regularization loss
        feature_graph_reg_loss = self.compute_feature_graph_regularization(obs_batch)
        
        # Ensure all losses are scalars
        if dist_entropy.dim() > 0:
            dist_entropy = dist_entropy.mean()
        
        # Complete FGAPPO loss
        total_loss = (
            policy_loss + 
            self.lambda_ent * entropy_loss + 
            self.nu_compression * compression_loss +
            0.001 * feature_graph_reg_loss -  # Small weight for feature graph regularization
            dist_entropy * self.entropy_coef
        )
        
        # Update policy
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        
        self.actor_optimizer.step()
        
        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
    
    def compute_compression_loss_with_feature_graph(self, obs_batch, rnn_states_batch, masks_batch):
        """
        Compute compression loss with feature graph structure.
        Uses only single agent's observation features (CTDE compliant).
        """
        obs_batch = check(obs_batch).to(**self.tpdv)
        rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        
        # Get encoded observation features (matching the policy forward pass)
        obs_features = self.actor.obs_encoder(obs_batch)
        if self.actor.use_naive_recurrent_policy or self.actor.use_recurrent_policy:
            obs_features, _ = self.actor.rnn(obs_features, rnn_states_batch, masks_batch)
        
        # Sample dummy actions with feature graph structure
        batch_size = obs_features.shape[0]
        
        # Initialize coupled noise vectors
        x = torch.randn(batch_size, self.actor.action_dim, device=self.device, requires_grad=True)
        y = torch.randn(batch_size, self.actor.action_dim, device=self.device, requires_grad=True)
        
        # Run a few steps of the diffusion process with feature graph enhancement
        for t in reversed(range(min(3, self.actor.num_diffusion_steps))):
            time_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.float32)
            
            # Predict noise for both components using feature graph network
            predicted_noise_x = self.actor.diffusion_net(obs_features, x, time_tensor)
            predicted_noise_y = self.actor.diffusion_net(obs_features, y, time_tensor)
            
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
    
    def compute_feature_graph_regularization(self, obs_batch):
        """
        Compute feature graph regularization loss.
        Encourages meaningful feature relationships.
        """
        if self.feature_adjacency_type != "learned":
            return torch.tensor(0.0, device=self.device)
        
        # Get feature adjacency matrix
        feature_adjacency = self.actor.diffusion_net.feature_graph_builder.get_feature_adjacency(obs_batch)
        
        # Sparsity regularization - encourage sparse connections
        sparsity_loss = torch.mean(torch.abs(feature_adjacency))
        
        # Symmetry regularization - encourage symmetric connections
        symmetry_loss = torch.mean((feature_adjacency - feature_adjacency.transpose(-2, -1)) ** 2)
        
        # Connectivity regularization - ensure each feature has some connections
        degree = feature_adjacency.sum(dim=-1)  # [batch_size, obs_dim]
        connectivity_loss = torch.mean(F.relu(1.0 - degree))  # Penalize isolated features
        
        total_reg_loss = sparsity_loss + 0.1 * symmetry_loss + 0.5 * connectivity_loss
        
        return total_reg_loss
    
    def train(self, actor_buffer, advantages, state_type):
        """
        Train the FGAPPO actor using feature graph diffusion policy.
        CTDE compliant - each agent trains using only their own data.
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
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """
        Train the FGAPPO actor using parameter sharing across multiple agents.
        CTDE compliant - each agent's data is processed independently.
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
                
                # Update with all agents' data (each agent's features processed independently)
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    tuple(batches)
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
    
    def debug_feature_graph(self, obs_batch, verbose=False):
        """
        Debug function to visualize feature graph structure.
        Useful for understanding how features are connected.
        """
        if verbose:
            print("=" * 50)
            print("FGAPPO Feature Graph Debug Info")
            print("=" * 50)
            print(f"Observation dimension: {obs_batch.shape[-1]}")
            print(f"Feature adjacency type: {self.feature_adjacency_type}")
            print(f"Use feature attention: {self.use_feature_attention}")
            print(f"Number of feature graph layers: {self.num_feature_graph_layers}")
            print(f"Feature graph hidden dimension: {self.feature_graph_hidden_dim}")
        
        # Get feature adjacency matrix
        with torch.no_grad():
            obs_batch = check(obs_batch).to(**self.tpdv)
            feature_adjacency = self.actor.diffusion_net.feature_graph_builder.get_feature_adjacency(obs_batch)
            
            # Compute statistics
            avg_adjacency = feature_adjacency.mean(dim=0)  # Average across batch
            sparsity = (avg_adjacency > 0.1).float().mean()
            max_degree = avg_adjacency.sum(dim=-1).max()
            min_degree = avg_adjacency.sum(dim=-1).min()
            
            if verbose:
                print(f"Average adjacency matrix shape: {avg_adjacency.shape}")
                print(f"Sparsity (connections > 0.1): {sparsity:.3f}")
                print(f"Max degree: {max_degree:.3f}")
                print(f"Min degree: {min_degree:.3f}")
                print("=" * 50)
            
            return {
                "feature_adjacency": feature_adjacency,
                "avg_adjacency": avg_adjacency,
                "sparsity": sparsity.item(),
                "max_degree": max_degree.item(),
                "min_degree": min_degree.item()
            } 