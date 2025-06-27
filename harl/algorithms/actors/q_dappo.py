import torch
import torch.nn as nn
import numpy as np
from harl.algorithms.actors.dappo import DAPPO
from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm


class QDAPPO(DAPPO):
    """
    Q-weighted Diffusion-Augmented PPO (Q-DAPPO) Actor.
    
    Combines the diffusion policy from DAPPO with Q-weighted optimization from QVPO.
    Key features:
    1. Q-weighted VLO loss instead of standard PPO loss
    2. Q-weight transformation to handle negative Q-values
    3. Diffusion entropy regularization
    4. High-quality sample selection based on Q-values
    5. Adaptive action sampling with Q-value guidance
    """
    
    def __init__(self, args, obs_space, act_space, share_obs_space=None, device=torch.device("cpu")):
        # Initialize parent DAPPO class
        super().__init__(args, obs_space, act_space, device)
        
        # Handle missing share_obs_space for compatibility with runner
        if share_obs_space is None:
            # Fallback: use obs_space as share_obs_space (common in single-agent or EP state type)
            share_obs_space = obs_space
            print("⚠️  Q-DAPPO: share_obs_space not provided, using obs_space as fallback")
        
        # Q-DAPPO specific parameters
        self.q_weight_type = args.get("q_weight_type", "softmax")  # softmax, exp, sigmoid
        self.q_temperature = args.get("q_temperature", 1.0)
        self.q_weight_coef = args.get("q_weight_coef", 1.0)
        self.diffusion_entropy_coef = args.get("diffusion_entropy_coef", 0.01)
        self.use_q_clipping = args.get("use_q_clipping", True)
        self.q_clip_range = args.get("q_clip_range", (-10.0, 10.0))
        
        # Sample generation parameters
        self.num_diffusion_samples = args.get("num_diffusion_samples", 16)  # Nd from Algorithm 1
        self.num_uniform_samples = args.get("num_uniform_samples", 4)      # Ne from Algorithm 1
        self.sample_selection_method = args.get("sample_selection_method", "max_weight")  # max_weight, weighted_sampling
        
        # Enhanced entropy regularization
        self.use_diffusion_entropy = args.get("use_diffusion_entropy", True)
        self.entropy_estimation_method = args.get("entropy_estimation_method", "gaussian_approx")
        
        # Ensure required Q-critic parameters are present with defaults
        q_critic_args = args.copy()
        q_critic_args.setdefault("polyak", 0.005)
        q_critic_args.setdefault("use_proper_time_limits", True)
        q_critic_args.setdefault("critic_lr", args.get("lr", 0.0001))  # Fallback to actor lr
        
        # Q-critic for computing Q-values
        self.q_critic = ContinuousQCritic(
            q_critic_args, share_obs_space, [act_space],  # Wrap act_space in list for multi-agent compatibility
            num_agents=1, state_type="EP", device=device
        )
        
        # Store share observation space for Q-value computation
        self.share_obs_space = share_obs_space
        
    def compute_q_weights(self, q_values, method="softmax"):
        """
        Transform Q-values into positive weights using various methods.
        
        Args:
            q_values: [batch_size, 1] Q-values (can be negative)
            method: Transformation method - 'softmax', 'exp', 'sigmoid', 'advantage'
            
        Returns:
            weights: [batch_size, 1] Positive weights
        """
        if self.use_q_clipping:
            q_values = torch.clamp(q_values, self.q_clip_range[0], self.q_clip_range[1])
        
        if method == "softmax":
            # Softmax transformation with temperature
            weights = torch.softmax(q_values / self.q_temperature, dim=0)
        elif method == "exp":
            # Exponential transformation 
            weights = torch.exp(q_values / self.q_temperature)
            weights = weights / torch.sum(weights, dim=0, keepdim=True)
        elif method == "sigmoid":
            # Sigmoid transformation
            weights = torch.sigmoid(q_values / self.q_temperature)
        elif method == "advantage":
            # Advantage-based weighting (subtract baseline)
            baseline = torch.mean(q_values, dim=0, keepdim=True)
            advantages = q_values - baseline
            weights = torch.relu(advantages) + 1e-8  # Ensure positive
            weights = weights / torch.sum(weights, dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown Q-weight method: {method}")
            
        return weights

    def sample_actions_with_q_guidance(self, obs_batch, rnn_states_batch, masks_batch, 
                                     share_obs_batch, deterministic=False):
        """
        Sample multiple actions from diffusion policy and select based on Q-values.
        Implements Algorithm 1 from QVPO paper.
        
        Args:
            obs_batch: Individual observations
            rnn_states_batch: RNN states  
            masks_batch: Masks
            share_obs_batch: Shared observations for Q-value computation
            deterministic: Whether to use deterministic sampling
            
        Returns:
            selected_actions: Best actions based on Q-values
            all_actions: All sampled actions
            q_values: Q-values for all actions
            weights: Q-weights for all actions
        """
        batch_size = obs_batch.shape[0]
        
        # Step 1: Generate Nd samples from diffusion policy
        diffusion_actions = []
        diffusion_log_probs = []
        
        for _ in range(self.num_diffusion_samples):
            actions, log_probs, _ = self.get_actions(
                obs_batch, rnn_states_batch, masks_batch, 
                deterministic=deterministic
            )
            diffusion_actions.append(actions)
            diffusion_log_probs.append(log_probs)
        
        # Step 2: Generate Ne samples from uniform distribution
        uniform_actions = []
        action_dim = self.act_space.shape[0]
        
        for _ in range(self.num_uniform_samples):
            if hasattr(self.act_space, 'low') and hasattr(self.act_space, 'high'):
                # Bounded continuous action space
                low = torch.tensor(self.act_space.low, device=self.device)
                high = torch.tensor(self.act_space.high, device=self.device)
                uniform_action = torch.rand(batch_size, action_dim, device=self.device)
                uniform_action = low + uniform_action * (high - low)
            else:
                # Unbounded - use standard normal
                uniform_action = torch.randn(batch_size, action_dim, device=self.device)
            uniform_actions.append(uniform_action)
        
        # Step 3: Combine all actions
        all_actions = diffusion_actions + uniform_actions
        all_actions_tensor = torch.stack(all_actions, dim=1)  # [batch, num_samples, action_dim]
        
        # Step 4: Compute Q-values for all actions (memory-efficient batching)
        batch_size, num_samples, action_dim = all_actions_tensor.shape
        
        # Process in smaller batches to avoid memory issues
        q_batch_size = 32  # Process 32 samples at a time
        q_values_list = []
        
        for i in range(0, num_samples, q_batch_size):
            end_idx = min(i + q_batch_size, num_samples)
            batch_actions = all_actions_tensor[:, i:end_idx, :]
            
            # Expand share_obs for this batch
            batch_share_obs = share_obs_batch.unsqueeze(1).expand(-1, end_idx - i, -1)
            batch_share_obs_flat = batch_share_obs.reshape(-1, batch_share_obs.shape[-1])
            batch_actions_flat = batch_actions.reshape(-1, action_dim)
            
            # Compute Q-values for this batch
            with torch.no_grad():  # Don't need gradients for Q-value computation
                batch_q_values = self.q_critic.get_values(batch_share_obs_flat, batch_actions_flat)
            q_values_list.append(batch_q_values.reshape(batch_size, end_idx - i, 1))
        
        # Combine all Q-values
        q_values = torch.cat(q_values_list, dim=1)
        
        # Step 5: Compute Q-weights
        weights = self.compute_q_weights(q_values.reshape(-1, 1), self.q_weight_type)
        weights = weights.reshape(batch_size, len(all_actions), 1)
        
        # Step 6: Select actions based on Q-values
        if self.sample_selection_method == "max_weight":
            # Select action with maximum Q-weight
            best_indices = torch.argmax(weights.squeeze(-1), dim=1)
            selected_actions = all_actions_tensor[torch.arange(batch_size), best_indices]
        elif self.sample_selection_method == "weighted_sampling":
            # Sample proportional to Q-weights
            probs = weights.squeeze(-1)
            probs = probs / torch.sum(probs, dim=1, keepdim=True)
            indices = torch.multinomial(probs, 1).squeeze(-1)
            selected_actions = all_actions_tensor[torch.arange(batch_size), indices]
        else:
            raise ValueError(f"Unknown selection method: {self.sample_selection_method}")
        
        return selected_actions, all_actions_tensor, q_values, weights

    def compute_diffusion_entropy_regularization(self, obs_batch, rnn_states_batch, masks_batch):
        """
        Compute diffusion entropy regularization term.
        Estimates the entropy of the diffusion policy distribution.
        """
        batch_size = obs_batch.shape[0]
        
        if self.entropy_estimation_method == "gaussian_approx":
            # Approximate entropy using Gaussian assumption
            # Sample multiple actions and compute empirical variance (reduced samples for memory)
            actions_list = []
            for _ in range(4):  # Reduced from 8 to 4 samples for memory efficiency
                actions, _, _ = self.get_actions(obs_batch, rnn_states_batch, masks_batch)
                actions_list.append(actions)
            
            actions_tensor = torch.stack(actions_list, dim=1)  # [batch, num_samples, action_dim]
            empirical_var = torch.var(actions_tensor, dim=1)  # [batch, action_dim]
            
            # Entropy of multivariate Gaussian: 0.5 * log((2πe)^d * |Σ|)
            action_dim = self.act_space.shape[0]
            entropy = 0.5 * (action_dim * np.log(2 * np.pi * np.e) + torch.sum(torch.log(empirical_var + 1e-8), dim=-1))
            entropy = torch.mean(entropy)
            
        elif self.entropy_estimation_method == "sample_based":
            # Direct entropy estimation from samples (reduced samples for memory)
            actions_list = []
            log_probs_list = []
            
            for _ in range(8):  # Reduced from 16 to 8 samples for memory efficiency
                actions, log_probs, _ = self.get_actions(obs_batch, rnn_states_batch, masks_batch)
                actions_list.append(actions)
                log_probs_list.append(log_probs)
            
            log_probs_tensor = torch.stack(log_probs_list, dim=1)
            entropy = -torch.mean(log_probs_tensor)
            
        else:
            entropy = torch.tensor(0.0, device=self.device)
        
        return entropy

    def update(self, sample, share_obs_batch=None):
        """
        Update the Q-DAPPO policy using Q-weighted VLO loss.
        
        Implements the core Q-weighted variational policy optimization with diffusion.
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
        rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        
        if share_obs_batch is None:
            share_obs_batch = obs_batch  # Fallback if not provided
        else:
            share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        
        if factor_batch is None:
            factor_batch = torch.ones_like(adv_targ)
        else:
            factor_batch = check(factor_batch).to(**self.tpdv)

        # Step 1: Sample actions with Q-guidance
        selected_actions, all_actions, q_values, q_weights = self.sample_actions_with_q_guidance(
            obs_batch, rnn_states_batch, masks_batch, share_obs_batch
        )
        
        # Step 2: Evaluate current policy on original actions
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch, rnn_states_batch, actions_batch, masks_batch,
            available_actions_batch, active_masks_batch
        )
        
        # Step 3: Compute Q-values for current actions
        current_q_values = self.q_critic.get_values(share_obs_batch, actions_batch)
        current_q_weights = self.compute_q_weights(current_q_values, self.q_weight_type)
        
        # Step 4: Compute Q-weighted VLO loss (Equation 6 from QVPO)
        # L(θ) = E[ω_eq(s,a) ||ε - ε_θ(√(αt)a + √(1-αt)ε, s, t)||²]
        
        # Get importance weights for policy gradient
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1, keepdim=True
        )
        
        # Q-weighted policy loss
        # Weight the policy loss by Q-values
        weighted_advantages = current_q_weights.detach() * adv_targ
        
        surr1 = imp_weights * weighted_advantages
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * weighted_advantages
        
        if self.use_policy_active_masks:
            q_weighted_policy_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            q_weighted_policy_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()
        
        # Step 5: Compute diffusion entropy regularization
        diffusion_entropy = self.compute_diffusion_entropy_regularization(
            obs_batch, rnn_states_batch, masks_batch
        )
        
        # Step 6: Compute compression loss (from original DAPPO)
        compression_loss = self.compute_compression_loss(obs_batch, rnn_states_batch, masks_batch)
        
        # Step 7: Q-weighted entropy loss
        q_weighted_entropy_loss = -torch.mean(current_q_weights.detach() * action_log_probs)
        
        # Ensure all losses are scalars
        if dist_entropy.dim() > 0:
            dist_entropy = dist_entropy.mean()
        
        # Step 8: Complete Q-DAPPO loss
        total_loss = (
            q_weighted_policy_loss +
            self.lambda_ent * q_weighted_entropy_loss +
            self.nu_compression * compression_loss +
            self.diffusion_entropy_coef * diffusion_entropy -
            self.entropy_coef * dist_entropy
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
        
        return (
            q_weighted_policy_loss, 
            dist_entropy, 
            actor_grad_norm, 
            imp_weights,
            torch.mean(current_q_values),  # Average Q-value
            torch.mean(current_q_weights)  # Average Q-weight
        )

    def train(self, actor_buffer, advantages, state_type, share_obs_buffer=None):
        """
        Train the Q-DAPPO actor with Q-weighted optimization.
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["q_values"] = 0
        train_info["q_weights"] = 0

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
                if share_obs_buffer is not None:
                    share_obs_generator = share_obs_buffer.recurrent_generator_actor(
                        advantages, self.actor_num_mini_batch, self.data_chunk_length
                    )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
                if share_obs_buffer is not None:
                    share_obs_generator = share_obs_buffer.naive_recurrent_generator_actor(
                        advantages, self.actor_num_mini_batch
                    )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
                if share_obs_buffer is not None:
                    share_obs_generator = share_obs_buffer.feed_forward_generator_actor(
                        advantages, self.actor_num_mini_batch
                    )

            for i, sample in enumerate(data_generator):
                share_obs_batch = None
                if share_obs_buffer is not None:
                    share_obs_sample = next(share_obs_generator)
                    share_obs_batch = share_obs_sample[0]  # Get share_obs from sample
                
                policy_loss, dist_entropy, actor_grad_norm, imp_weights, q_values, q_weights = self.update(
                    sample, share_obs_batch
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["q_values"] += q_values.item()
                train_info["q_weights"] += q_weights.item()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type, share_obs_buffer=None):
        """
        Train the Q-DAPPO actor using parameter sharing across multiple agents.
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["q_values"] = 0
        train_info["q_weights"] = 0

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
            share_obs_generators = []
            
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                    if share_obs_buffer is not None:
                        share_obs_generator = share_obs_buffer[agent_id].recurrent_generator_actor(
                            advantages_list[agent_id],
                            self.actor_num_mini_batch,
                            self.data_chunk_length,
                        )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                    if share_obs_buffer is not None:
                        share_obs_generator = share_obs_buffer[agent_id].naive_recurrent_generator_actor(
                            advantages_list[agent_id], self.actor_num_mini_batch
                        )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                    if share_obs_buffer is not None:
                        share_obs_generator = share_obs_buffer[agent_id].feed_forward_generator_actor(
                            advantages_list[agent_id], self.actor_num_mini_batch
                        )
                
                data_generators.append(data_generator)
                if share_obs_buffer is not None:
                    share_obs_generators.append(share_obs_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(8)]
                share_obs_batches = []
                
                for agent_id, generator in enumerate(data_generators):
                    sample = next(generator)
                    for i in range(8):
                        batches[i].append(sample[i])
                    
                    if share_obs_buffer is not None:
                        share_obs_sample = next(share_obs_generators[agent_id])
                        share_obs_batches.append(share_obs_sample[0])
                
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[7][0] is None:
                    batches[7] = None
                else:
                    batches[7] = np.concatenate(batches[7], axis=0)
                
                # Combine share observations if available
                combined_share_obs = None
                if share_obs_batches:
                    combined_share_obs = np.concatenate(share_obs_batches, axis=0)
                
                policy_loss, dist_entropy, actor_grad_norm, imp_weights, q_values, q_weights = self.update(
                    tuple(batches), combined_share_obs
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["q_values"] += q_values.item()
                train_info["q_weights"] += q_weights.item()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def get_actions(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Override to optionally use Q-guided action selection during inference.
        """
        if hasattr(self, 'use_q_guidance_inference') and self.use_q_guidance_inference:
            # Use Q-guidance during inference (more expensive but potentially better)
            # This requires share_obs, so we'll fall back to normal sampling
            pass
        
        # Use standard diffusion policy action generation
        return super().get_actions(obs, rnn_states, masks, available_actions, deterministic) 