import torch
import torch.nn as nn
import numpy as np
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.models.policy_models.diffusion_policy import DiffusionPolicy
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm, update_linear_schedule


class DAPPO(OnPolicyBase):
    """
    Diffusion-Augmented PPO (DAPPO) Actor.
    Implements the GenPO algorithm with diffusion-based action generation.
    """
    
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        # Override the actor creation to use DiffusionPolicy
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.data_chunk_length = args["data_chunk_length"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.action_aggregation = args["action_aggregation"]

        self.lr = args["lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        
        # Save observation and action spaces
        self.obs_space = obs_space
        self.act_space = act_space
        
        # DAPPO specific parameters
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        
        # GenPO specific parameters
        self.lambda_ent = args.get("lambda_ent", 0.01)
        self.nu_compression = args.get("nu_compression", 0.01)
        self.adaptive_lr = args.get("adaptive_lr", True)
        self.kl_threshold_high = args.get("kl_threshold_high", 0.02)
        self.kl_threshold_low = args.get("kl_threshold_low", 0.005)
        self.lr_adjustment_factor = args.get("lr_adjustment_factor", 2.0)
        
        # Create diffusion policy
        self.actor = DiffusionPolicy(args, self.obs_space, self.act_space, self.device)
        
        # Create optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
    
    def update(self, sample):
        """
        Update the diffusion policy using the GenPO loss function.
        Handle both HAPPO-style (9 values) and MAPPO-style (8 values) sample tuples.
        """
        # Handle different sample tuple lengths
        if len(sample) == 9:
            # HAPPO-style with factor_batch
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
            # MAPPO-style without factor_batch
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
            factor_batch = None  # Set to None, will handle after tensor conversion

        # Convert to tensors
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        
        # Handle factor_batch after tensor conversion
        if factor_batch is None:
            # Create default factor_batch as ones with same shape as adv_targ
            factor_batch = torch.ones_like(adv_targ)
        else:
            factor_batch = check(factor_batch).to(**self.tpdv)

        # Get new action log probabilities and entropy
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

        # Standard PPO clipped loss
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        # Ensure policy_action_loss is scalar
        policy_loss = policy_action_loss

        # Ensure dist_entropy is scalar
        if dist_entropy.dim() > 0:
            dist_entropy = dist_entropy.mean()

        # Create scalar total loss
        total_loss = policy_loss - dist_entropy * self.entropy_coef

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

        # Compute KL divergence for adaptive learning rate
        with torch.no_grad():
            kl_div = torch.mean(old_action_log_probs_batch - action_log_probs).item()
            
            # Adaptive learning rate adjustment (Algorithm 1, line 11)
            if self.adaptive_lr:
                if kl_div > self.kl_threshold_high:
                    # Decrease learning rate
                    new_lr = self.lr / self.lr_adjustment_factor
                elif kl_div < self.kl_threshold_low:
                    # Increase learning rate
                    new_lr = self.lr * self.lr_adjustment_factor
                else:
                    new_lr = self.lr
                
                # Update learning rate in optimizer
                if new_lr != self.lr:
                    self.lr = new_lr
                    for param_group in self.actor_optimizer.param_groups:
                        param_group['lr'] = self.lr

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """
        Train the DAPPO actor using the diffusion policy (non-parameter-sharing version).
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
        Train the DAPPO actor using parameter sharing across multiple agents.
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
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(8):
                        batches[i].append(sample[i])
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[7][0] is None:
                    batches[7] = None
                else:
                    batches[7] = np.concatenate(batches[7], axis=0)
                
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