"""Runner for on-policy HARL algorithms with QVPPO support and Q-value based agent ordering."""
import os
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_ha_runner_q_order import OnPolicyHARunnerQOrder


class OnPolicyHARunnerQVPPO(OnPolicyHARunnerQOrder):
    """Runner for on-policy HA algorithms with QVPPO support and Q-value based agent ordering."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the runner with QVPPO-specific features."""
        super().__init__(args, algo_args, env_args)
        
        # QVPPO specific configuration
        self.use_share_obs_for_qvppo = algo_args["algo"].get("use_share_obs_for_qvppo", True)
        self.qvppo_critic_update_freq = algo_args["algo"].get("qvppo_critic_update_freq", 1)
        
        print(f"🌊 QVPPO Runner initialized with Q-value ordering and diffusion policy support")
        print(f"   - Share obs for QVPPO: {self.use_share_obs_for_qvppo}")
        print(f"   - Q-ordering mode: {self.q_ordering_mode}")

    def train(self):
        """Train the model with QVPPO-specific optimizations and Q-value based agent ordering."""
        actor_train_infos = []

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # Determine agent order based on Q-values
        agent_order = self.get_agent_order_by_q_values(step=0)
        
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # Prepare shared observations for QVPPO
            share_obs_batch = None
            if self.use_share_obs_for_qvppo and hasattr(self.actor[agent_id], 'q_critic'):
                # QVPPO actor detected - extract shared observations from critic buffer
                if hasattr(self, 'critic_buffer'):
                    # Extract the shared observations that correspond to the actor buffer data
                    if self.state_type == "EP":
                        # For EP, shared obs are the same for all agents
                        raw_share_obs = self.critic_buffer.share_obs[:-1].reshape(-1, *self.critic_buffer.share_obs.shape[2:])
                        
                        # Check if we need to extract agent-specific portion
                        expected_share_obs_dim = self.actor[agent_id].share_obs_space.shape[0]
                        actual_share_obs_dim = raw_share_obs.shape[-1]
                        
                        if actual_share_obs_dim == expected_share_obs_dim:
                            # Dimensions match - use as is
                            share_obs_batch = raw_share_obs
                        elif actual_share_obs_dim == expected_share_obs_dim * self.num_agents:
                            # Critic buffer has concatenated obs from all agents
                            # Extract the portion for the current agent
                            agent_start_idx = agent_id * expected_share_obs_dim
                            agent_end_idx = (agent_id + 1) * expected_share_obs_dim
                            share_obs_batch = raw_share_obs[:, agent_start_idx:agent_end_idx]
                        else:
                            # Dimension mismatch - use individual observations as fallback
                            individual_obs = self.actor_buffer[agent_id].obs[:-1].reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:])
                            share_obs_batch = individual_obs
                            
                    elif self.state_type == "FP":
                        # For FP, shared obs are agent-specific
                        share_obs_batch = self.critic_buffer.share_obs[:-1, :, agent_id].reshape(-1, *self.critic_buffer.share_obs.shape[3:])
                    else:
                        # Fallback
                        share_obs_batch = self.critic_buffer.share_obs[:-1].reshape(-1, *self.critic_buffer.share_obs.shape[2:])

            # update actor
            if self.state_type == "EP":
                if hasattr(self.actor[agent_id], 'q_critic'):
                    # QVPPO actor - pass shared observations
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP", share_obs_batch
                    )
                else:
                    # Regular actor
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
            elif self.state_type == "FP":
                if hasattr(self.actor[agent_id], 'q_critic'):
                    # QVPPO actor - pass shared observations
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP", share_obs_batch
                    )
                else:
                    # Regular actor
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                    )

            # Log QVPPO specific metrics
            # if 'vlo_loss' in actor_train_info:
            #     print(f"🌊 Agent {agent_id} QVPPO metrics:")
            #     print(f"   - VLO Loss: {actor_train_info['vlo_loss']:.6f}")
            #     print(f"   - Q-values: {actor_train_info['q_values']:.6f}")
            #     print(f"   - Q-weights: {actor_train_info['q_weights']:.6f}")
            #     if 'diffusion_entropy' in actor_train_info:
            #         print(f"   - Diffusion Entropy: {actor_train_info['diffusion_entropy']:.6f}")

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update factor for next agent with numerical stability
            logprob_diff = new_actions_logprob - old_actions_logprob
            
            # Clip log probability differences to prevent overflow in exp()
            # Clamping to [-10, 10] means ratios stay in [exp(-10), exp(10)] = [0.000045, 22026]
            logprob_diff_clamped = torch.clamp(logprob_diff, -10.0, 10.0)
            
            # Compute importance ratios with clamped differences
            importance_ratios = torch.exp(logprob_diff_clamped)
            
            # Additional clipping on the ratios themselves for safety
            importance_ratios = torch.clamp(importance_ratios, 0.01, 100.0)
            
            # Apply action aggregation
            aggregated_ratios = getattr(torch, self.action_aggregation)(importance_ratios, dim=-1)
            
            # Reshape and convert to numpy
            ratio_factor = _t2n(aggregated_ratios.reshape(
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ))
            
            # Check for NaN/Inf before multiplication
            if np.isnan(ratio_factor).any() or np.isinf(ratio_factor).any():
                print(f"⚠️  NaN/Inf in ratio_factor for agent {agent_id}, using ones")
                ratio_factor = np.ones_like(ratio_factor)
            
            # Update factor with additional overflow protection
            new_factor = factor * ratio_factor
            
            # Final safety check on the accumulated factor
            if np.isnan(new_factor).any() or np.isinf(new_factor).any():
                print(f"⚠️  NaN/Inf in accumulated factor for agent {agent_id}, resetting")
                factor = np.ones_like(factor)  # Reset to neutral factor
            else:
                # Clamp factor to reasonable range to prevent explosive growth
                factor = np.clip(new_factor, 0.001, 1000.0)
            actor_train_infos.append(actor_train_info)

        # update critic (update frequency can be adjusted for QVPPO)
        if self.training_episode % self.qvppo_critic_update_freq == 0:
            critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        else:
            # Skip critic update but provide dummy info for compatibility
            critic_train_info = {
                "value_loss": 0.0,
                "policy_loss": 0.0,
                "dist_entropy": 0.0,
                "actor_grad_norm": 0.0,
                "ratio": 0.0
            }

        return actor_train_infos, critic_train_info

    def save(self):
        """Save the model including Q-critic and any QVPPO-specific components."""
        super().save()
        
        # Save QVPPO-specific components if they exist
        for agent_id in range(self.num_agents):
            if hasattr(self.actor[agent_id], 'q_critic'):
                try:
                    qvppo_save_dir = f"{self.save_dir}/qvppo_agent_{agent_id}"
                    os.makedirs(qvppo_save_dir, exist_ok=True)
                    self.actor[agent_id].q_critic.save(qvppo_save_dir)
                    print(f"💾 QVPPO agent {agent_id} Q-critic saved to {qvppo_save_dir}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not save QVPPO agent {agent_id} Q-critic: {e}")

    def restore(self):
        """Restore the model including Q-critic and any QVPPO-specific components."""
        super().restore()
        
        # Restore QVPPO-specific components if they exist
        if self.algo_args["train"]["model_dir"] is not None:
            for agent_id in range(self.num_agents):
                if hasattr(self.actor[agent_id], 'q_critic'):
                    try:
                        qvppo_restore_dir = self.algo_args["train"]["model_dir"] / f"qvppo_agent_{agent_id}"
                        if qvppo_restore_dir.exists():
                            self.actor[agent_id].q_critic.restore(qvppo_restore_dir)
                            print(f"🔄 QVPPO agent {agent_id} Q-critic restored from {qvppo_restore_dir}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not restore QVPPO agent {agent_id} Q-critic: {e}")

    def log_train_infos_to_wandb(self, train_infos):
        """Enhanced logging for QVPPO metrics."""
        # Call parent logging first
        if hasattr(super(), 'log_train_infos_to_wandb'):
            super().log_train_infos_to_wandb(train_infos)
        
        # Add QVPPO-specific logging
        for agent_id, train_info in enumerate(train_infos):
            if isinstance(train_info, dict):
                # Log QVPPO-specific metrics
                qvppo_metrics = {}
                if 'vlo_loss' in train_info:
                    qvppo_metrics[f'agent_{agent_id}/vlo_loss'] = train_info['vlo_loss']
                if 'diffusion_entropy' in train_info:
                    qvppo_metrics[f'agent_{agent_id}/diffusion_entropy'] = train_info['diffusion_entropy']
                if 'q_values' in train_info:
                    qvppo_metrics[f'agent_{agent_id}/q_values'] = train_info['q_values']
                if 'q_weights' in train_info:
                    qvppo_metrics[f'agent_{agent_id}/q_weights'] = train_info['q_weights']
                
                # Log to wandb if available
                if qvppo_metrics and hasattr(self, 'wandb') and self.wandb is not None:
                    self.wandb.log(qvppo_metrics)

    def run(self):
        """Enhanced run method with QVPPO monitoring."""
        # Store training episode counter for critic update frequency
        self.training_episode = 0
        
        print("🚀 Starting QVPPO training with Q-value based agent ordering...")
        print(f"   - Q-ordering: {self.use_q_ordering}")
        print(f"   - Share obs for QVPPO: {self.use_share_obs_for_qvppo}")
        
        # Call parent run method with enhanced monitoring
        try:
            return super().run()
        except Exception as e:
            print(f"❌ Error during QVPPO training: {e}")
            # Try to save emergency checkpoint
            try:
                emergency_save_dir = self.save_dir / "emergency_checkpoint"
                emergency_save_dir.mkdir(exist_ok=True)
                self.save()
                print(f"🆘 Emergency checkpoint saved to {emergency_save_dir}")
            except:
                pass
            raise

    def eval(self):
        """Enhanced evaluation with QVPPO monitoring."""
        print("🧪 Starting QVPPO evaluation...")
        
        # For QVPPO, we might want to use deterministic actions during evaluation
        for agent_id in range(self.num_agents):
            if hasattr(self.actor[agent_id], 'actor') and hasattr(self.actor[agent_id].actor, 'num_diffusion_steps'):
                print(f"🎯 Agent {agent_id}: Using deterministic diffusion sampling for evaluation")
        
        try:
            return super().eval()
        except Exception as e:
            print(f"❌ Error during QVPPO evaluation: {e}")
            raise 