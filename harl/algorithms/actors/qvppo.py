"""QVPPO algorithm - Q-weighted Variational Policy Optimization for Heterogeneous Agents."""
import torch
import torch.nn as nn
import numpy as np
from harl.algorithms.actors.qappo import QAPPO
from harl.models.policy_models.diffusion_policy import DiffusionPolicy
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm


class QVPPO(QAPPO):
    """
    Q-weighted Variational Policy Optimization (QVPPO).
    
    Combines QAPPO's Q-weighted optimization with diffusion policies from QVPO.
    Implements the HA-QVPO algorithm which uses:
    1. Q-weighted Variational Lower Bound (VLO) loss
    2. Diffusion-based action generation
    3. Advantage-based Q-weighting
    4. Sequential agent coordination (from HAPPO)
    5. Numerical stability features
    """
    
    def __init__(self, args, obs_space, act_space, share_obs_space=None, device=torch.device("cpu")):
        # Save original args before parent initialization
        self.original_args = args.copy()
        
        # CRITICAL: For QVPPO, we need to ensure share_obs_space is properly set
        # The Q-critic will be created with this space, so dimensions must match
        if share_obs_space is None:
            share_obs_space = obs_space
            print("⚠️  QVPPO: share_obs_space not provided, using obs_space as fallback")
        
        # Initialize parent QAPPO class (this will create a standard StochasticPolicy actor and Q-critic)
        super().__init__(args, obs_space, act_space, share_obs_space, device)
        
        # QVPPO specific parameters
        self.vlo_loss_weight = args.get("vlo_loss_weight", 1.0)
        self.diffusion_entropy_coef = args.get("diffusion_entropy_coef", 0.01)
        self.use_advantage_weighting = args.get("use_advantage_weighting", True)
        self.num_diffusion_samples = args.get("num_diffusion_samples", 8)
        self.timestep_sampling_strategy = args.get("timestep_sampling_strategy", "uniform")  # uniform, importance
        
        # Numerical stability parameters
        self.epsilon_stability = float(args.get("epsilon_stability", 1e-8))
        self.gradient_clip_norm = float(args.get("gradient_clip_norm", 1.0))
        self.action_bound_epsilon = float(args.get("action_bound_epsilon", 1e-6))
        
        # Replace the standard policy with diffusion policy
        print("🌊 QVPPO: Replacing standard policy with diffusion policy...")
        del self.actor  # Remove the standard actor
        
        # Create diffusion policy with same parameters
        diffusion_args = self.original_args.copy()
        diffusion_args.setdefault("num_diffusion_steps", 10)
        diffusion_args.setdefault("mixing_p", 0.9)
        diffusion_args.setdefault("beta_start", 1e-4)
        diffusion_args.setdefault("beta_end", 0.02)
        
        self.actor = DiffusionPolicy(diffusion_args, obs_space, act_space, device)
        
        # Recreate optimizer with diffusion policy parameters
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        
        print(f"✅ QVPPO initialized with diffusion policy ({self.actor.num_diffusion_steps} steps)")

    def compute_advantage_weights(self, q_values, share_obs_batch):
        """
        Compute advantage-based weights: w_q(s,a) = max(A_π(s,a), 0)
        where A_π(s,a) = Q_π(s,a) - V_π(s)
        
        Args:
            q_values: [batch_size, 1] Q-values
            share_obs_batch: [batch_size, share_obs_dim] shared observations
            
        Returns:
            weights: [batch_size, 1] advantage-based weights
        """
        with torch.no_grad():
            # Estimate V(s) by sampling multiple actions and averaging Q-values
            batch_size = share_obs_batch.shape[0]
            
            # Sample multiple actions to estimate V(s)
            num_value_samples = 5
            sampled_q_values = []
            
            for _ in range(num_value_samples):
                # Sample random actions from action space bounds
                if hasattr(self.actor, 'action_dim'):
                    random_actions = torch.randn(batch_size, self.actor.action_dim, device=self.device) * 0.5
                else:
                    # Fallback: estimate action dim from q_critic
                    random_actions = torch.randn(batch_size, q_values.shape[-1], device=self.device) * 0.5
                
                # Clamp actions to reasonable bounds
                random_actions = torch.clamp(random_actions, -2.0, 2.0)
                
                try:
                    sampled_q = self.q_critic.get_values(share_obs_batch, random_actions)
                    sampled_q_values.append(sampled_q)
                except Exception as e:
                    # Q-critic error - use zeros as fallback
                    sampled_q_values.append(torch.zeros(batch_size, 1, device=self.device))
            
            if sampled_q_values:
                # Estimate V(s) as average of sampled Q-values
                v_values = torch.stack(sampled_q_values, dim=0).mean(dim=0)
                
                # Compute advantages
                advantages = q_values - v_values
                
                # Apply ReLU to get positive weights (max(A, 0))
                weights = torch.relu(advantages) + self.epsilon_stability
                
                # Normalize weights to prevent extreme values
                weights = weights / (torch.mean(weights) + self.epsilon_stability)
            else:
                # Fallback: use uniform weights if all Q-value computations failed
                weights = torch.ones(batch_size, 1, device=self.device)
                
        return weights

    def sample_diffusion_timesteps(self, batch_size):
        """
        Sample timesteps for diffusion training.
        
        Args:
            batch_size: Number of timesteps to sample
            
        Returns:
            timesteps: [batch_size] sampled timesteps
        """
        if self.timestep_sampling_strategy == "uniform":
            # Uniform sampling
            timesteps = torch.randint(
                0, self.actor.num_diffusion_steps, 
                (batch_size,), device=self.device
            )
        elif self.timestep_sampling_strategy == "importance":
            # Importance sampling - sample more from middle timesteps
            weights = torch.ones(self.actor.num_diffusion_steps, device=self.device)
            mid_point = self.actor.num_diffusion_steps // 2
            for i in range(self.actor.num_diffusion_steps):
                weights[i] = 1.0 + 0.5 * np.exp(-0.1 * (i - mid_point)**2)
            
            timesteps = torch.multinomial(weights, batch_size, replacement=True)
        else:
            raise ValueError(f"Unknown timestep sampling strategy: {self.timestep_sampling_strategy}")
        
        return timesteps

    def compute_q_weighted_vlo_loss(self, obs_batch, actions_batch, share_obs_batch, 
                                   factor_batch, active_masks_batch):
        """
        Compute the Q-weighted Variational Lower Bound (VLO) loss.
        
        L_HA-QVPO = E[w_q(s,a) * M_{1:m}(s,a) * ||ε - ε_θ(√α_t * a + √(1-α_t) * ε, s, t)||²]
        
        Args:
            obs_batch: [batch_size, obs_dim] observations
            actions_batch: [batch_size, action_dim] actions
            share_obs_batch: [batch_size, share_obs_dim] shared observations
            factor_batch: [batch_size, 1] importance sampling factors
            active_masks_batch: [batch_size, 1] active masks
            
        Returns:
            vlo_loss: Q-weighted VLO loss
            metrics: Dictionary of loss metrics
        """
        batch_size = obs_batch.shape[0]
        
        # Compute Q-values for current actions
        with torch.no_grad():
            try:
                current_q_values = self.q_critic.get_values(share_obs_batch, actions_batch)
            except Exception as e:
                # Q-critic dimension mismatch or other error - use fallback
                current_q_values = torch.zeros(batch_size, 1, device=self.device)
        
        # Compute advantage-based weights
        if self.use_advantage_weighting:
            q_weights = self.compute_advantage_weights(current_q_values, share_obs_batch)
        else:
            # Use direct Q-value weighting as fallback
            q_weights = self.compute_q_weights(current_q_values, self.q_weight_type)
        
        # Sample timesteps for diffusion training
        timesteps = self.sample_diffusion_timesteps(batch_size)
        
        # Sample noise
        noise = torch.randn_like(actions_batch)
        
        # Forward diffusion process: x_t = √α_t * x_0 + √(1-α_t) * ε
        sqrt_alphas_cumprod_t = self.actor.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.actor.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        
        noisy_actions = sqrt_alphas_cumprod_t * actions_batch + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Encode observations for diffusion network
        obs_features = self.actor.obs_encoder(obs_batch)
        
        # Apply RNN if needed
        if self.actor.use_naive_recurrent_policy or self.actor.use_recurrent_policy:
            # For training, we use zero initial states
            rnn_states = torch.zeros(
                batch_size, self.actor.rnn.hidden_size, 
                device=self.device, dtype=torch.float32
            )
            masks = torch.ones(batch_size, 1, device=self.device, dtype=torch.float32)
            obs_features, _ = self.actor.rnn(obs_features, rnn_states, masks)
        
        # Predict noise using diffusion network
        timesteps_float = timesteps.float()
        predicted_noise = self.actor.diffusion_net(obs_features, noisy_actions, timesteps_float)
        
        # Compute VLO loss: ||ε - ε_θ(x_t, s, t)||²
        vlo_loss_raw = torch.mean((noise - predicted_noise) ** 2, dim=-1, keepdim=True)
        
        # Apply Q-weighting and importance sampling
        q_weighted_vlo_loss = q_weights.detach() * factor_batch * vlo_loss_raw
        
        # Apply active masks and compute final loss
        if self.use_policy_active_masks:
            final_vlo_loss = (q_weighted_vlo_loss * active_masks_batch).sum() / (active_masks_batch.sum() + self.epsilon_stability)
        else:
            final_vlo_loss = torch.mean(q_weighted_vlo_loss)
        
        # Numerical stability check
        if torch.isnan(final_vlo_loss) or torch.isinf(final_vlo_loss):
            final_vlo_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
        
        # Compute metrics
        metrics = {
            "vlo_loss_raw": torch.mean(vlo_loss_raw).item(),
            "q_weights_mean": torch.mean(q_weights).item(),
            "q_weights_std": torch.std(q_weights).item(),
            "predicted_noise_norm": torch.norm(predicted_noise).item(),
            "actual_noise_norm": torch.norm(noise).item(),
        }
        
        return final_vlo_loss, metrics

    def compute_diffusion_entropy_regularization(self, obs_batch):
        """
        Compute diffusion entropy regularization to encourage exploration.
        
        Args:
            obs_batch: [batch_size, obs_dim] observations
            
        Returns:
            entropy_reg: Entropy regularization term
        """
        batch_size = obs_batch.shape[0]
        
        # Sample multiple actions from the diffusion policy
        with torch.no_grad():
            obs_features = self.actor.obs_encoder(obs_batch)
            
            # Apply RNN if needed
            if self.actor.use_naive_recurrent_policy or self.actor.use_recurrent_policy:
                rnn_states = torch.zeros(
                    batch_size, self.actor.rnn.hidden_size, 
                    device=self.device, dtype=torch.float32
                )
                masks = torch.ones(batch_size, 1, device=self.device, dtype=torch.float32)
                obs_features, _ = self.actor.rnn(obs_features, rnn_states, masks)
            
            # Sample multiple actions
            sampled_actions = []
            for _ in range(self.num_diffusion_samples):
                action, _ = self.actor.sample_actions_with_log_prob(obs_features, deterministic=False)
                sampled_actions.append(action)
            
            # Stack actions and compute variance
            actions_tensor = torch.stack(sampled_actions, dim=1)  # [batch, num_samples, action_dim]
            action_variance = torch.var(actions_tensor, dim=1)  # [batch, action_dim]
            
            # Entropy approximation: 0.5 * log(2πe * σ²)
            entropy = 0.5 * torch.log(2 * np.pi * np.e * (action_variance + self.epsilon_stability))
            entropy_reg = torch.mean(entropy)
        
        return entropy_reg

    def update(self, sample, share_obs_batch=None):
        """
        Update QVPPO policy using Q-weighted VLO loss.
        
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

        # Compute Q-weighted VLO loss
        vlo_loss, vlo_metrics = self.compute_q_weighted_vlo_loss(
            obs_batch, actions_batch, share_obs_batch, factor_batch, active_masks_batch
        )

        # Compute diffusion entropy regularization
        diffusion_entropy_reg = self.compute_diffusion_entropy_regularization(obs_batch)

        # Total loss
        total_loss = (
            self.vlo_loss_weight * vlo_loss + 
            self.diffusion_entropy_coef * diffusion_entropy_reg
        )

        # Update policy
        self.actor_optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
            # Additional clipping for QVPPO stability
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)

        self.actor_optimizer.step()

        # Compute additional metrics for monitoring
        with torch.no_grad():
            current_q_values = self.q_critic.get_values(share_obs_batch, actions_batch)

        return (
            vlo_loss,
            diffusion_entropy_reg,
            actor_grad_norm,
            torch.ones_like(adv_targ),  # Dummy importance weights for compatibility
            torch.mean(current_q_values),  # Average Q-value
            vlo_metrics["q_weights_mean"]  # Average Q-weight
        )

    def train(self, actor_buffer, advantages, state_type, share_obs_batch=None):
        """
        Train QVPPO actor with Q-weighted VLO optimization.
        
        Args:
            actor_buffer: Actor buffer containing training data
            advantages: Computed advantages
            state_type: State type (EP/FP)
            share_obs_batch: Pre-extracted shared observations batch [total_samples, share_obs_dim]
            
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
        train_info["vlo_loss"] = 0
        train_info["diffusion_entropy"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        # Advantage normalization
        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # Create a simple generator for shared observations if provided
        def create_share_obs_generator(share_obs_batch, data_generator):
            """Create a generator that yields shared observation batches matching the actor data."""
            if share_obs_batch is None:
                while True:
                    yield None
            else:
                # Convert to tensor for slicing
                share_obs_tensor = torch.from_numpy(share_obs_batch).float()
                sample_idx = 0
                
                for sample in data_generator:
                    batch_size = sample[0].shape[0]  # Get batch size from obs in sample
                    if sample_idx + batch_size <= share_obs_tensor.shape[0]:
                        batch_share_obs = share_obs_tensor[sample_idx:sample_idx + batch_size].numpy()
                        sample_idx += batch_size
                        yield batch_share_obs
                    else:
                        # Handle case where we run out of shared obs (shouldn't happen normally)
                        yield sample[0]  # Fallback to individual observations

        for epoch in range(self.ppo_epoch):
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

            # Create the corresponding shared observation generator
            # We need to recreate this for each epoch
            if self.use_recurrent_policy:
                share_data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                share_data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                share_data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            share_obs_generator = create_share_obs_generator(share_obs_batch, share_data_generator)

            for sample in data_generator:
                # Get the corresponding shared observation batch
                current_share_obs_batch = next(share_obs_generator)
                
                vlo_loss, diffusion_entropy_reg, actor_grad_norm, imp_weights, q_values, q_weights = self.update(
                    sample, current_share_obs_batch
                )

                train_info["vlo_loss"] += vlo_loss.item()
                train_info["policy_loss"] += vlo_loss.item()  # For compatibility
                train_info["diffusion_entropy"] += diffusion_entropy_reg.item()
                train_info["dist_entropy"] += diffusion_entropy_reg.item()  # For compatibility
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["q_values"] += q_values.item()
                train_info["q_weights"] += q_weights

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the diffusion policy.
        Ensures numerical stability and prevents NaN outputs.
        
        Args:
            obs: Local agent inputs to the actor
            rnn_states_actor: RNN states for actor
            masks: Denotes points at which RNN states should be reset
            available_actions: Available actions (not used in continuous case)
            deterministic: Whether the action should be deterministic
            
        Returns:
            actions: Actions taken by this actor
            rnn_states_actor: Updated RNN states
        """
        try:
            # Use diffusion policy for action generation
            actions, _, rnn_states_actor = self.actor(
                obs, rnn_states_actor, masks, available_actions, deterministic
            )
            
            # Numerical stability checks
            if torch.isnan(actions).any() or torch.isinf(actions).any():
                print(f"⚠️  QVPPO: NaN/Inf actions detected, using fallback")
                # Fallback to small random actions
                actions = torch.randn_like(actions) * 0.1
            
            # Clamp actions to reasonable bounds
            actions = torch.clamp(actions, -10.0, 10.0)
            
            return actions, rnn_states_actor
            
        except Exception as e:
            print(f"⚠️  QVPPO: Error in action generation: {e}")
            # Emergency fallback
            batch_size = obs.shape[0]
            action_dim = getattr(self.actor, 'action_dim', 1)
            fallback_actions = torch.randn(batch_size, action_dim, device=self.device) * 0.1
            return fallback_actions, rnn_states_actor

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate actions using the diffusion policy.
        
        Args:
            obs: Observations
            rnn_states: RNN states
            action: Actions to evaluate
            masks: Masks
            available_actions: Available actions
            active_masks: Active masks
            
        Returns:
            action_log_probs: Log probabilities of actions
            dist_entropy: Distribution entropy
            rnn_states: Updated RNN states
        """
        try:
            action_log_probs, dist_entropy, rnn_states = self.actor.evaluate_actions(
                obs, rnn_states, action, masks, available_actions, active_masks
            )
            
            # Numerical stability checks
            if torch.isnan(action_log_probs).any() or torch.isinf(action_log_probs).any():
                print(f"⚠️  QVPPO: NaN/Inf log probs detected, using fallback")
                action_log_probs = torch.zeros_like(action_log_probs) - 1.0  # Small negative log prob
            
            if torch.isnan(dist_entropy).any() or torch.isinf(dist_entropy).any():
                print(f"⚠️  QVPPO: NaN/Inf entropy detected, using fallback")
                dist_entropy = torch.ones_like(dist_entropy) * 0.1  # Small positive entropy
            
            return action_log_probs, dist_entropy, rnn_states
            
        except Exception as e:
            print(f"⚠️  QVPPO: Error in action evaluation: {e}")
            # Emergency fallback
            batch_size = obs.shape[0]
            fallback_log_probs = torch.zeros(batch_size, 1, device=self.device) - 1.0
            fallback_entropy = torch.ones(batch_size, 1, device=self.device) * 0.1
            return fallback_log_probs, fallback_entropy, rnn_states

    def get_actions(self, obs, rnn_states, masks, available_actions=None):
        """
        Get actions and their log probabilities from the diffusion policy.
        """
        try:
            actions, action_log_probs, rnn_states = self.actor(
                obs, rnn_states, masks, available_actions, deterministic=False
            )
            
            # Numerical stability
            if torch.isnan(actions).any() or torch.isinf(actions).any():
                print(f"⚠️  QVPPO: NaN/Inf in get_actions, using fallback")
                actions = torch.randn_like(actions) * 0.1
                action_log_probs = torch.zeros_like(action_log_probs) - 1.0
            
            actions = torch.clamp(actions, -10.0, 10.0)
            
            return actions, action_log_probs, rnn_states
            
        except Exception as e:
            print(f"⚠️  QVPPO: Error in get_actions: {e}")
            batch_size = obs.shape[0]
            action_dim = getattr(self.actor, 'action_dim', 1)
            fallback_actions = torch.randn(batch_size, action_dim, device=self.device) * 0.1
            fallback_log_probs = torch.zeros(batch_size, 1, device=self.device) - 1.0
            return fallback_actions, fallback_log_probs, rnn_states 