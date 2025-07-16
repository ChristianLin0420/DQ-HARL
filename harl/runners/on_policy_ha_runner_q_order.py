"""Runner for on-policy HARL algorithms with Q-value based agent ordering."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic


class OnPolicyHARunnerQOrder(OnPolicyHARunner):
    """Runner for on-policy HA algorithms with Q-value based agent ordering."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the runner with Q-critic for agent ordering."""
        super().__init__(args, algo_args, env_args)
        
        # Initialize Q-critic for computing Q-values for agent ordering
        # Use the same configuration as the V-critic but create a Q-critic
        q_critic_args = {**algo_args["model"], **algo_args["algo"]}
        q_critic_args.setdefault("critic_lr", algo_args.get("lr", 0.0005))
        q_critic_args.setdefault("polyak", 0.005)
        q_critic_args.setdefault("use_proper_time_limits", True)
        q_critic_args.setdefault("gamma", 0.99)
        
        self.q_critic = ContinuousQCritic(
            q_critic_args,
            self.envs.share_observation_space[0], 
            self.envs.action_space,
            self.num_agents,
            self.state_type,
            device=self.device
        )
        
        # Q-value ordering configuration
        self.q_ordering_mode = algo_args["algo"].get("q_ordering_mode", "descending")  # "ascending" or "descending"
        self.use_q_ordering = algo_args["algo"].get("use_q_ordering", True)
        
        print(f"Q-value based agent ordering initialized with mode: {self.q_ordering_mode}")

    def compute_agent_q_values(self, step):
        """Compute Q-values for all agents at the current step.
        
        Args:
            step: Current step in the episode
            
        Returns:
            agent_q_values: List of Q-values for each agent
        """
        agent_q_values = []
        
        # Get shared observations for Q-value computation
        if self.state_type == "EP":
            share_obs = self.critic_buffer.share_obs[step]  # (n_threads, share_obs_dim)
        elif self.state_type == "FP":
            share_obs = self.critic_buffer.share_obs[step]  # (n_threads, n_agents, share_obs_dim)
        
        # Collect all agent actions at this step
        all_actions = []
        for agent_id in range(self.num_agents):
            actions = self.actor_buffer[agent_id].actions[step]  # (n_threads, action_dim)
            all_actions.append(actions)
        
        # Stack actions: (n_agents, n_threads, action_dim) -> (n_threads, n_agents, action_dim)
        actions_array = np.array(all_actions).transpose(1, 0, 2)
        
        # Compute agent-specific Q-values using different approaches
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                if self.state_type == "EP":
                    # For EP, use the same shared observation for all agents
                    agent_share_obs = share_obs  # (n_threads, share_obs_dim)
                elif self.state_type == "FP":
                    # For FP, use agent-specific shared observation
                    agent_share_obs = share_obs[:, agent_id]  # (n_threads, share_obs_dim)
                
                # Approach 1: Use agent's individual action repeated for all positions
                # This gives each agent a Q-value based on their own action contribution
                agent_action = actions_array[:, agent_id]  # (n_threads, action_dim)
                
                # Create a joint action where this agent's action is used for all positions
                # This evaluates "what if this agent's action was taken by everyone"
                repeated_agent_action = np.tile(agent_action[:, np.newaxis, :], (1, self.num_agents, 1))
                joint_actions_repeated = repeated_agent_action.reshape(repeated_agent_action.shape[0], -1)
                
                # Convert to tensors
                agent_share_obs_tensor = torch.FloatTensor(agent_share_obs).to(self.device)
                joint_actions_tensor = torch.FloatTensor(joint_actions_repeated).to(self.device)
                
                # Compute Q-values for this specific agent configuration
                q_values_repeated = self.q_critic.get_values(agent_share_obs_tensor, joint_actions_tensor)
                
                # Approach 2: Also compute Q-value with actual joint actions but weight by agent contribution
                actual_joint_actions = actions_array.reshape(actions_array.shape[0], -1)
                actual_joint_actions_tensor = torch.FloatTensor(actual_joint_actions).to(self.device)
                q_values_actual = self.q_critic.get_values(agent_share_obs_tensor, actual_joint_actions_tensor)
                
                # Approach 3: Compute Q-value difference (counterfactual reasoning)
                # What happens if we replace this agent's action with zero action?
                zero_action = np.zeros_like(agent_action)
                counterfactual_actions = actions_array.copy()
                counterfactual_actions[:, agent_id] = zero_action
                counterfactual_joint = counterfactual_actions.reshape(counterfactual_actions.shape[0], -1)
                counterfactual_joint_tensor = torch.FloatTensor(counterfactual_joint).to(self.device)
                q_values_counterfactual = self.q_critic.get_values(agent_share_obs_tensor, counterfactual_joint_tensor)
                
                # Combine different approaches for agent-specific Q-value
                # Use the difference between actual and counterfactual as the agent's contribution
                q_contribution = torch.mean(q_values_actual - q_values_counterfactual).item()
                
                # Add some randomness based on agent-specific action magnitude to break ties
                action_magnitude = np.mean(np.abs(agent_action))
                agent_specific_value = q_contribution + 0.01 * action_magnitude + 0.001 * agent_id
                
                agent_q_values.append(agent_specific_value)
        
        return agent_q_values

    def get_agent_order_by_q_values(self, step):
        """Determine agent update order based on Q-values.
        
        Args:
            step: Current step in the episode
            
        Returns:
            agent_order: List of agent IDs ordered by Q-values
        """
        if not self.use_q_ordering or self.fixed_order:
            # Fall back to original behavior
            if self.fixed_order:
                return list(range(self.num_agents))
            else:
                return list(torch.randperm(self.num_agents).numpy())
        
        try:
            # Compute Q-values for all agents
            agent_q_values = self.compute_agent_q_values(step)
            
            # Sort agents by Q-values
            agent_q_pairs = list(enumerate(agent_q_values))
            
            if self.q_ordering_mode == "descending":
                # Highest Q-value first
                agent_q_pairs.sort(key=lambda x: x[1], reverse=True)
            elif self.q_ordering_mode == "ascending":
                # Lowest Q-value first
                agent_q_pairs.sort(key=lambda x: x[1], reverse=False)
            else:
                raise ValueError(f"Unknown q_ordering_mode: {self.q_ordering_mode}")
            
            agent_order = [agent_id for agent_id, _ in agent_q_pairs]
            
            # Log Q-values for debugging (optional)
            if hasattr(self, 'logger') and step % 100 == 0:  # Log every 100 steps
                q_values_str = ", ".join([f"Agent {i}: {q:.3f}" for i, q in agent_q_pairs])
                print(f"Step {step} Q-values: {q_values_str}")
                print(f"Agent order: {agent_order}")
            
            return agent_order
            
        except Exception as e:
            print(f"Error computing Q-value based ordering: {e}")
            print("Falling back to random ordering")
            return list(torch.randperm(self.num_agents).numpy())

    def train(self):
        """Train the model with Q-value based agent ordering."""
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

        # Determine agent order based on Q-values (using step 0 as representative)
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

            # update actor
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), "EP"
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                )

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

            # update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info

    def save(self):
        """Save the model including Q-critic."""
        super().save()
        # Save Q-critic state
        if hasattr(self, 'q_critic'):
            self.q_critic.save(self.save_dir)

    def restore(self):
        """Restore the model including Q-critic."""
        super().restore()
        # Restore Q-critic state
        if hasattr(self, 'q_critic') and self.algo_args["train"]["model_dir"] is not None:
            try:
                self.q_critic.restore(self.algo_args["train"]["model_dir"])
                print("Q-critic model restored successfully")
            except Exception as e:
                print(f"Warning: Could not restore Q-critic model: {e}") 