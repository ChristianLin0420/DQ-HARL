"""QAPPO algorithm - Q-weighted Heterogeneous-Agent Proximal Policy Optimization."""
import torch
import torch.nn as nn
import numpy as np
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm


class QAPPO(HAPPO):
    """
    Q-weighted Heterogeneous-Agent Proximal Policy Optimization (QAPPO).
    
    Combines HAPPO's heterogeneous multi-agent capabilities with Q-weighted optimization.
    Key features:
    1. Q-weighted PPO loss instead of standard PPO loss
    2. Q-weight transformation to handle negative Q-values
    3. Multi-sample action generation with Q-value guidance
    4. Heterogeneous agent support (different obs/action spaces)
    5. Factor-based sequential agent updates
    """
    
    def __init__(self, args, obs_space, act_space, share_obs_space=None, device=torch.device("cpu")):
        # Initialize parent HAPPO class
        super().__init__(args, obs_space, act_space, device)
        
        # Handle missing share_obs_space for compatibility
        if share_obs_space is None:
            share_obs_space = obs_space
            print("⚠️  QAPPO: share_obs_space not provided, using obs_space as fallback")
        
        # Q-APPO specific parameters
        self.q_weight_type = args.get("q_weight_type", "softmax")  # softmax, exp, sigmoid, advantage
        self.q_temperature = args.get("q_temperature", 1.0)
        self.q_weight_coef = args.get("q_weight_coef", 1.0)
        self.use_q_clipping = args.get("use_q_clipping", True)
        self.q_clip_range = args.get("q_clip_range", (-10.0, 10.0))
        
        # Q-guided sampling parameters  
        self.num_q_samples = args.get("num_q_samples", 8)  # Number of action samples for Q-guidance
        self.q_sample_method = args.get("q_sample_method", "policy_sampling")  # policy_sampling, mixed_sampling
        self.use_q_guidance = args.get("use_q_guidance", True)
        
        # Q-value entropy regularization
        self.q_entropy_coef = args.get("q_entropy_coef", 0.01)
        self.use_q_entropy = args.get("use_q_entropy", True)
        
        # Ensure required Q-critic parameters are present
        q_critic_args = args.copy()
        q_critic_args.setdefault("polyak", 0.005)
        q_critic_args.setdefault("use_proper_time_limits", True)
        q_critic_args.setdefault("critic_lr", args.get("lr", 0.0005))
        
        # Q-critic for computing Q-values
        self.q_critic = ContinuousQCritic(
            q_critic_args, share_obs_space, [act_space],
            num_agents=1, state_type="EP", device=device
        )
        
        # Store share observation space
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
            # Advantage-based weighting
            baseline = torch.mean(q_values, dim=0, keepdim=True)
            advantages = q_values - baseline
            weights = torch.relu(advantages) + 1e-8
            weights = weights / torch.sum(weights, dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown Q-weight method: {method}")
            
        return weights

    def sample_actions_with_q_guidance(self, obs_batch, rnn_states_batch, masks_batch, 
                                     share_obs_batch, available_actions_batch=None):
        """
        Sample multiple actions and select based on Q-values for Q-guided policy improvement.
        
        Args:
            obs_batch: Individual observations
            rnn_states_batch: RNN states
            masks_batch: Masks  
            share_obs_batch: Shared observations for Q-value computation
            available_actions_batch: Available actions
            
        Returns:
            selected_actions: Best actions based on Q-values
            all_actions: All sampled actions
            q_values: Q-values for all actions
            weights: Q-weights for all actions
        """
        batch_size = obs_batch.shape[0]
        
        # Sample multiple actions from current policy
        all_actions = []
        all_log_probs = []
        
        for _ in range(self.num_q_samples):
            if self.q_sample_method == "policy_sampling":
                # Sample from current policy
                actions, log_probs, _ = self.get_actions(
                    obs_batch, rnn_states_batch, masks_batch, available_actions_batch
                )
            elif self.q_sample_method == "mixed_sampling":
                # Mix of policy sampling and noise
                if np.random.rand() < 0.7:  # 70% policy sampling
                    actions, log_probs, _ = self.get_actions(
                        obs_batch, rnn_states_batch, masks_batch, available_actions_batch
                    )
                else:  # 30% noise sampling
                    actions, log_probs, _ = self.get_actions(
                        obs_batch, rnn_states_batch, masks_batch, available_actions_batch
                    )
                    # Add noise to actions
                    action_dim = actions.shape[-1]
                    noise = torch.randn_like(actions) * 0.1
                    actions = actions + noise
                    
            all_actions.append(actions)
            all_log_probs.append(log_probs)
        
        # Stack all actions
        all_actions_tensor = torch.stack(all_actions, dim=1)  # [batch, num_samples, action_dim]
        
        # Compute Q-values for all actions (memory-efficient)
        batch_size, num_samples, action_dim = all_actions_tensor.shape
        q_values_list = []
        
        # Process in batches to avoid memory issues
        batch_size_limit = 16
        for i in range(0, num_samples, batch_size_limit):
            end_idx = min(i + batch_size_limit, num_samples)
            batch_actions = all_actions_tensor[:, i:end_idx, :]
            
            # Expand share_obs for this batch
            batch_share_obs = share_obs_batch.unsqueeze(1).expand(-1, end_idx - i, -1)
            batch_share_obs_flat = batch_share_obs.reshape(-1, batch_share_obs.shape[-1])
            batch_actions_flat = batch_actions.reshape(-1, action_dim)
            
            # Compute Q-values
            with torch.no_grad():
                batch_q_values = self.q_critic.get_values(batch_share_obs_flat, batch_actions_flat)
            q_values_list.append(batch_q_values.reshape(batch_size, end_idx - i, 1))
        
        # Combine all Q-values
        q_values = torch.cat(q_values_list, dim=1)
        
        # Compute Q-weights
        weights = self.compute_q_weights(q_values.reshape(-1, 1), self.q_weight_type)
        weights = weights.reshape(batch_size, num_samples, 1)
        
        # Select best actions based on Q-values
        best_indices = torch.argmax(weights.squeeze(-1), dim=1)
        selected_actions = all_actions_tensor[torch.arange(batch_size), best_indices]
        
        return selected_actions, all_actions_tensor, q_values, weights

    def compute_q_entropy_regularization(self, q_values, weights):
        """
        Compute Q-value entropy regularization to encourage exploration.
        
        Args:
            q_values: Q-values for sampled actions
            weights: Q-weights for sampled actions
            
        Returns:
            entropy_reg: Entropy regularization term
        """
        if not self.use_q_entropy:
            return torch.tensor(0.0, device=self.device)
        
        # Compute entropy of Q-weight distribution
        # H = -sum(w * log(w))
        weights_flat = weights.reshape(-1, weights.shape[1])  # [batch*1, num_samples]
        log_weights = torch.log(weights_flat + 1e-8)
        entropy = -torch.sum(weights_flat * log_weights, dim=1)
        
        return torch.mean(entropy)

    def update(self, sample, share_obs_batch=None):
        """
        Update QAPPO policy using Q-weighted PPO loss.
        
        Args:
            sample: Training sample tuple
            share_obs_batch: Shared observations for Q-value computation
            
        Returns:
            Tuple of loss values and metrics
        """
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

        # Convert to tensors
        obs_batch = check(obs_batch).to(**self.tpdv)
        rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        
        if share_obs_batch is None:
            share_obs_batch = obs_batch
        else:
            share_obs_batch = check(share_obs_batch).to(**self.tpdv)

        # Evaluate current policy on actions
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch, rnn_states_batch, actions_batch, masks_batch,
            available_actions_batch, active_masks_batch
        )

        # Compute Q-values for current actions
        current_q_values = self.q_critic.get_values(share_obs_batch, actions_batch)
        
        # Q-guided action sampling and selection
        if self.use_q_guidance:
            selected_actions, all_actions, sampled_q_values, q_weights = self.sample_actions_with_q_guidance(
                obs_batch, rnn_states_batch, masks_batch, share_obs_batch, available_actions_batch
            )
            
            # Compute Q-entropy regularization
            q_entropy_reg = self.compute_q_entropy_regularization(sampled_q_values, q_weights)
        else:
            q_entropy_reg = torch.tensor(0.0, device=self.device)

        # Compute Q-weights for current actions
        current_q_weights = self.compute_q_weights(current_q_values, self.q_weight_type)
        
        # Q-weighted PPO loss
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1, keepdim=True
        )
        
        # Weight advantages by Q-values
        q_weighted_advantages = current_q_weights.detach() * adv_targ
        
        surr1 = imp_weights * q_weighted_advantages
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * q_weighted_advantages
        
        if self.use_policy_active_masks:
            q_weighted_policy_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            q_weighted_policy_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        # Q-weighted entropy loss
        q_weighted_entropy_loss = -torch.mean(current_q_weights.detach() * action_log_probs)
        
        # Total loss
        total_loss = (
            q_weighted_policy_loss +
            self.entropy_coef * q_weighted_entropy_loss +
            self.q_entropy_coef * q_entropy_reg -
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
        Train QAPPO actor with Q-weighted optimization.
        
        Args:
            actor_buffer: Actor buffer containing training data
            advantages: Computed advantages
            state_type: State type (EP/FP)
            share_obs_buffer: Shared observation buffer
            
        Returns:
            train_info: Training information dictionary
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

        # Advantage normalization
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

            for sample in data_generator:
                share_obs_batch = None
                if share_obs_buffer is not None:
                    share_obs_sample = next(share_obs_generator)
                    share_obs_batch = share_obs_sample[0]
                
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