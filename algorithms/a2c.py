"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""


import torch
import torch.nn as nn
import torch.optim as optim

from utils import helpers as utl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A2C_RL2:
    def __init__(
        self,
        args,
        actor_critic,
        critic_loss_coeff,
        entropy_loss_coeff,
        policy_optimizer,
        policy_eps,
        policy_lr,
        policy_anneal_lr,
        train_steps
    ):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        # loss function
        # coefficients for mixing the value and entropy loss
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff

        # optimizer
        if policy_optimizer == 'adam':
            self.optimizer = optim.Adam(
                actor_critic.parameters(), 
                lr=policy_lr,
                eps=policy_eps
            )
        elif policy_optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), 
                policy_lr, 
                eps=policy_eps, 
                alpha=0.99
            )

        # learning rate annealing
        self.lr_scheduler_policy = None
        if policy_anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler_policy = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lam)
        

    def get_losses(self, policy_storage):

        # re-build computational graph
        if not self.args.shared_rnn:
            action_logits, state_values, actor_hidden_states, critic_hidden_states = \
                self.actor_critic(
                    curr_states=policy_storage.states_for_policy[:-1,:,:], 
                    prev_actions=policy_storage.actions[:-1,:,:], 
                    prev_rewards=policy_storage.rewards[:-1,:,:],
                    actor_prev_hidden_states=None,
                    critic_prev_hidden_states=None)
        elif self.args.shared_rnn:
            action_logits, state_values, rnn_hidden_states = \
                self.actor_critic(
                    curr_states=policy_storage.states_for_policy[:-1,:,:], 
                    prev_actions=policy_storage.actions[:-1,:,:], 
                    prev_rewards=policy_storage.rewards[:-1,:,:],
                    rnn_prev_hidden_states=None)
        
        # Handle both discrete and continuous action spaces
        if hasattr(self.actor_critic, 'action_space_type') and self.actor_critic.action_space_type == 'Box':
            # Continuous action space - Gaussian policy
            action_means, action_stds = action_logits
            action_pd = torch.distributions.Normal(action_means, action_stds)
            policy_entropy = action_pd.entropy()
            
            # Get selected actions (stored actions for continuous case)
            actions = policy_storage.actions[1:,:,:]
            action_log_probs = action_pd.log_prob(actions)
            
            # Sum log probs across action dimensions for multi-dimensional actions
            if len(actions.shape) > 2 and actions.shape[-1] > 1:
                action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
                policy_entropy = policy_entropy.sum(dim=-1, keepdim=True)
        else:
            # Discrete action space - Categorical policy (backward compatibility)
            action_pd = torch.distributions.Categorical(logits=action_logits)
            policy_entropy = action_pd.entropy()
            # get selected actions
            actions = torch.argmax(policy_storage.actions[1:,:,:], dim=2)
            action_log_probs = action_pd.log_prob(actions)

        # compute advantages
        advantages = (policy_storage.returns - policy_storage.state_values).squeeze()
        # multiply by the mask
        advantages = advantages * policy_storage.masks_ongoing[:-1,:,0]
        # Normalize advantages (per batch)
        if self.args.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.optimizer.zero_grad()

        # compute actor loss
        # squeeze if continuous action space
        if hasattr(self.actor_critic, 'action_space_type') and self.actor_critic.action_space_type == 'Box':
            actor_loss = -(advantages.detach() * action_log_probs.squeeze()).mean()
        else:
            actor_loss = -(advantages.detach() * action_log_probs).mean()

        # compute critic loss
        # multiply by the mask
        critic_loss = (policy_storage.returns - state_values) * policy_storage.masks_ongoing[:-1,:,:]
        critic_loss = critic_loss.pow(2).mean()
        
        # (loss = value loss + action loss + entropy loss, weighted)
        # give bonus for higher policy entropy
        loss = actor_loss - policy_entropy.mean() * self.entropy_loss_coeff\
                        + critic_loss * self.critic_loss_coeff

        return loss, actor_loss, critic_loss, policy_entropy.mean()


    def update_parameters(self, loss):
        
        # zero out the gradients
        self.optimizer.zero_grad()
        
        # compute gradients 
        # (will attach to all networks involved in this computation)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)

        # update
        self.optimizer.step()

        if self.lr_scheduler_policy is not None:
            self.lr_scheduler_policy.step()


class A2C_MPC:
    def __init__(
        self,
        args,
        actor_critic,
        critic_loss_coeff,
        entropy_loss_coeff,
        policy_optimizer,
        policy_eps,
        policy_lr,
        policy_anneal_lr,
        train_steps
    ):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        # loss function
        # coefficients for mixing the value and entropy loss
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff

        # optimizer
        if policy_optimizer == 'adam':
            self.optimizer = optim.Adam(
                actor_critic.parameters(), 
                lr=policy_lr,
                eps=policy_eps
            )
        elif policy_optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), 
                policy_lr, 
                eps=policy_eps, 
                alpha=0.99
            )

        # learning rate annealing
        self.lr_scheduler_policy = None
        if policy_anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler_policy = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lam)
        

    def get_losses(
        self, 
        policy_storage,
        encoder=None  # encoder is only used if args.policy_gradient_through_encoder
    ):

        # Re-compute latents through encoder to maintain computational graph
        if self.args.policy_gradient_through_encoder:
            # Re-encode the trajectories with encoder to maintain gradients
            _, latent_means, latent_logvars, _ = encoder(
                curr_states=policy_storage.states_for_policy[:-1,:,:].to(device),
                prev_actions=policy_storage.actions[:-1,:,:], 
                prev_rewards=policy_storage.rewards[:-1,:,:],
                rnn_prev_hidden_states=None, 
                return_prior=False, 
                sample=True,  # Use mean instead of sampling for deterministic gradients
                detach_every=None
            )
        else:
            # Use the latents from the policy storage
            latent_means = policy_storage.latent_means[:,:,:].to(device)
            latent_logvars = policy_storage.latent_logvars[:,:,:].to(device)
        
        # re-build computational graph
        action_logits, state_values = \
            self.actor_critic(
                curr_states=policy_storage.states_for_policy[:-1,:,:].to(device), 
                curr_latent_means=latent_means,
                curr_latent_logvars=latent_logvars,
            )
        
        # Handle both discrete and continuous action spaces
        if hasattr(self.actor_critic, 'action_space_type') and self.actor_critic.action_space_type == 'Box':
            # Continuous action space - Gaussian policy
            action_means, action_stds = action_logits
            action_pd = torch.distributions.Normal(action_means, action_stds)
            policy_entropy = action_pd.entropy()
            
            # Get selected actions (stored actions for continuous case)
            actions = policy_storage.actions[1:,:,:]
            action_log_probs = action_pd.log_prob(actions)
            
            # Sum log probs across action dimensions for multi-dimensional actions
            if len(actions.shape) > 2 and actions.shape[-1] > 1:
                action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
                policy_entropy = policy_entropy.sum(dim=-1, keepdim=True)
        else:
            # Discrete action space - Categorical policy (backward compatibility)
            action_pd = torch.distributions.Categorical(logits=action_logits)
            policy_entropy = action_pd.entropy()
            # get selected actions
            actions = torch.argmax(policy_storage.actions[1:,:,:], dim=2)
            action_log_probs = action_pd.log_prob(actions)

        # compute advantages
        advantages = (policy_storage.returns - policy_storage.state_values).squeeze()
        # Normalize advantages (per batch)
        if self.args.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.optimizer.zero_grad()
        # compute actor loss
        # squeeze if continuous action space
        if hasattr(self.actor_critic, 'action_space_type') and self.actor_critic.action_space_type == 'Box':
            actor_loss = -(advantages.detach() * action_log_probs.squeeze()).mean()
        else:
            actor_loss = -(advantages.detach() * action_log_probs).mean()

        # compute critic loss
        critic_loss = (policy_storage.returns - state_values).pow(2).mean()
        
        # (loss = value loss + action loss + entropy loss, weighted)
        # give bonus for higher policy entropy
        loss = actor_loss - policy_entropy.mean() * self.entropy_loss_coeff\
                        + critic_loss * self.critic_loss_coeff

        return loss, actor_loss, critic_loss, policy_entropy.mean()

    def _clip_and_step(self):
        """Helper method for gradient clipping and optimizer step"""
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)
        
        # update
        self.optimizer.step()

        if self.lr_scheduler_policy is not None:
            self.lr_scheduler_policy.step()

    def update_parameters(self, loss):
        """
        if not args.policy_gradient_through_encoder, 
        performs complete update cycle
        """
        # zero out the gradients
        self.optimizer.zero_grad()
        # compute gradients 
        # (will attach to all networks involved in this computation)
        loss.backward()
        self._clip_and_step()

    def step_optimizer(self):
        """
        if args.policy_gradient_through_encoder, 
        only performs gradient clipping and optimizer step
        """
        self._clip_and_step()
