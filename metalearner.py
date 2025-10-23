import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from algorithms.a2c import A2C_RL2, A2C_MPC
from algorithms.online_storage import OnlineStorageRL2, OnlineStorageMPC
from environments.parallel_envs import make_vec_envs
from models.policy_net import ActorCriticRNN, SharedActorCriticRNN, ActorCriticMLP
from utils import helpers as utl
from utils.tb_logger import TBLogger
from vae import VAE_MPC

from utils.evaluation import get_empirical_returns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaLearner:
    """
    Meta-Learner class with the main training loop for RL2 and MPC.
    """

    def __init__(self, args):

        self.args = args
        self.iter_idx = -1
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # -- initialize environments --
        self.envs = make_vec_envs(
            env_name=self.args.env_name,
            max_episode_steps=self.args.max_episode_steps,
            num_processes=self.args.num_processes
        )
        print(f'initialize envs done: {self.envs}')

        # -- get dimensions: state, action, reward --
        # state
        assert isinstance(self.envs.single_observation_space, gym.spaces.Dict)
        # for trial-based bandit tasks
        if self.args.env_name.split('-')[0] in [
            'StatBernoulliBandit2ArmIndependent',
            'OracleBanditDeterministic', 
            'DisSymmetricStickyMarkovBandit2State2Arm', 'DisAsymmetricRewardStickyMarkovBandit2State2Arm',
            'DisAsymmetricTransitionStickyMarkovBandit2State2Arm',
        ]:
            self.args.state_dim = self.envs.single_observation_space['timestep'].shape[0]
        # for tiger task
        elif self.args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
            self.args.state_dim = self.envs.single_observation_space['hear'].n
        # for latent cart task
        elif self.args.env_name.split('-')[0] in ['LatentGoalCart']:
            self.args.state_dim = self.envs.single_observation_space['position'].shape[0]
        else:
            raise ValueError(f'Env is not supported yet: {self.args.env_name}')
            
        # whether to use time as state to policy
        if 'Bandit' in self.args.env_name:
            if self.args.time_as_state:  # if time as input to the policy
                self.args.input_state_dim_for_policy = self.args.state_dim
            else:  # not use time as input to the policy
                self.args.input_state_dim_for_policy = self.args.state_dim - 1
        else:
            self.args.input_state_dim_for_policy = self.args.state_dim
        # action
        if isinstance(self.envs.single_action_space, gym.spaces.discrete.Discrete):
            self.args.action_space_type = 'Discrete'
            self.args.action_dim = self.envs.single_action_space.n
        elif isinstance(self.envs.single_action_space, gym.spaces.box.Box):
            self.args.action_space_type = 'Box'
            self.args.action_dim = self.envs.single_action_space.shape[0]
        else:
            raise NotImplementedError(f'unsupported action space type: {self.envs.single_action_space}')
        # reward
        self.args.reward_dim = 1

        # -- initialize tensorboard logger and save args --
        self.logger = TBLogger(self.args, self.args.exp_label)

        # -- model initialization --
        if self.args.exp_label == 'rl2':
            # initialize policy
            self.policy = self.initialize_policy_rl2()
            self.policy_storage = OnlineStorageRL2(self.args)
        elif self.args.exp_label == 'MPC':
            # initialize VAE and policy
            self.vae = VAE_MPC(self.args, self.logger)
            self.policy = self.initialize_policy_MPC()
            self.policy_storage = OnlineStorageMPC(self.args)
        else:
            raise NotImplementedError
        
        # -- initialize return normalizer --
        if self.args.normalize_rew_for_policy:
            self.return_normalizer = utl.ReturnNormalizer(size=1)


    def initialize_policy_rl2(self):
        # initialize policy network
        if not self.args.shared_rnn:
            actor_critic = ActorCriticRNN(
                args=self.args,
                layers_before_rnn=self.args.layers_before_rnn,
                rnn_hidden_dim=self.args.rnn_hidden_dim,
                layers_after_rnn=self.args.layers_after_rnn,
                activation_function=self.args.policy_net_activation_function,
                rnn_cell_type=self.args.rnn_cell_type,
                initialization_method=self.args.policy_net_initialization_method,
                state_dim=self.args.input_state_dim_for_policy,
                state_embed_dim=self.args.state_embed_dim,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embed_dim,
                action_space_type=self.args.action_space_type,
                reward_dim=self.args.reward_dim,
                reward_embed_dim=self.args.reward_embed_dim
            ).to(device)
        elif self.args.shared_rnn:
            actor_critic = SharedActorCriticRNN(
                args=self.args,
                layers_before_rnn=self.args.layers_before_rnn,
                rnn_hidden_dim=self.args.rnn_hidden_dim,
                layers_after_rnn=self.args.layers_after_rnn,
                activation_function=self.args.policy_net_activation_function,
                rnn_cell_type=self.args.rnn_cell_type,
                initialization_method=self.args.policy_net_initialization_method,
                state_dim=self.args.input_state_dim_for_policy,
                state_embed_dim=self.args.state_embed_dim,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embed_dim,
                action_space_type=self.args.action_space_type,
                reward_dim=self.args.reward_dim,
                reward_embed_dim=self.args.reward_embed_dim
            ).to(device)
        else:
            raise ValueError(f'invalid args.shared_rnn: {self.args.shared_rnn}')
        
        # initialize policy trainer
        if self.args.policy_algorithm == 'a2c':
            policy = A2C_RL2(
                args=self.args,
                actor_critic=actor_critic,
                critic_loss_coeff=self.args.policy_critic_loss_coeff,
                entropy_loss_coeff=self.args.policy_entropy_loss_coeff,
                policy_optimizer=self.args.policy_optimizer,
                policy_eps=self.args.policy_eps,
                policy_lr=self.args.policy_lr,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.args.num_updates,
            )
        else:
            raise NotImplementedError

        return policy
            
            
    def initialize_policy_MPC(self):
        # initialize policy network
        actor_critic = ActorCriticMLP(
            args=self.args,
            # inputs
            state_dim=self.args.input_state_dim_for_policy,
            latent_dim=self.args.latent_dim,
            # network
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_net_activation_function,  # tanh, relu, leaky-relu
            initialization_method=self.args.policy_net_initialization_method, # orthogonal, normc
            # outputs
            action_dim=self.args.action_dim,
            action_space_type=self.args.action_space_type
        ).to(device)

        # initialize policy trainer
        if self.args.policy_algorithm == 'a2c':
            policy = A2C_MPC(
                args=self.args,
                actor_critic=actor_critic,
                # loss function
                critic_loss_coeff=self.args.policy_critic_loss_coeff,
                entropy_loss_coeff=self.args.policy_entropy_loss_coeff,
                # optimization
                policy_optimizer=self.args.policy_optimizer,
                policy_eps=self.args.policy_eps,
                policy_lr=self.args.policy_lr,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.args.num_updates
            )
        else:
            raise NotImplementedError

        return policy


    def train(self):
        """ Main Meta-Training loop """

        # loss
        train_stats = {
            'episode_returns': [],
            'actor_losses': [],
            'critic_losses': [],
            'policy_entropies': [],
            'elbo_losses': [],
            'reward_reconstruction_losses': [],
            'state_reconstruction_losses': [],
            'kl_losses': []
        }
        # evaluation: pass to log function
        evaluation_stats = {
            'eval_epoch_ids': [],
            'empirical_return_avgs': [],
            'empirical_return_stds': []
        }

        # log once before training
        start_time = time.time()
        with torch.no_grad():
            self.log(evaluation_stats, start_time)

        # training starts
        for self.iter_idx in range(int(self.args.num_updates)):
            print(f'training epoch: {self.iter_idx}')

            # -- COLLECT DATA -- #
            # reset all envs
            curr_states_dict, infos = self.envs.reset(seed=self.args.seed, options={})
            curr_states = utl.get_states_from_state_dicts(
                curr_states_dict, self.args, True)
            curr_states_for_policy = utl.get_states_from_state_dicts(
                curr_states_dict, self.args, self.args.time_as_state)
            curr_states = torch.from_numpy(curr_states).float().\
                reshape((1, self.args.num_processes, self.args.state_dim)).to(device)
            curr_states_for_policy = torch.from_numpy(curr_states_for_policy).float().\
                reshape((1, self.args.num_processes, self.args.input_state_dim_for_policy)).to(device)

            prev_actions = torch.zeros(1, self.args.num_processes, self.args.action_dim).to(device)
            prev_rewards = torch.zeros(1, self.args.num_processes, self.args.reward_dim).to(device)
            # Update return statistics (with raw rewards)
            if self.args.normalize_rew_for_policy:
                self.return_normalizer.update(prev_rewards.squeeze())
                # Normalize the reward
                normalized_prev_rewards = self.return_normalizer.normalize(prev_rewards)
            else:
                normalized_prev_rewards = prev_rewards
            
            # initialize rnn hidden states
            if self.args.exp_label in ['rl2', 'noisy_rl2']:
                if not self.args.shared_rnn:
                    # initialize ActorCriticRNN hidden states
                    actor_prev_hidden_states = torch.zeros(1, self.args.num_processes, self.args.rnn_hidden_dim).to(device)
                    critic_prev_hidden_states = torch.zeros(1, self.args.num_processes, self.args.rnn_hidden_dim).to(device)
                elif self.args.shared_rnn:
                    # initialize SharedActorCriticRNN hidden states
                    rnn_prev_hidden_states = torch.zeros(1, self.args.num_processes, self.args.rnn_hidden_dim).to(device)
            elif self.args.exp_label == 'MPC':
                # initialize RNNEncoder hidden states
                encoder_rnn_prev_hidden_states = torch.zeros(1, self.args.num_processes, self.args.encoder_rnn_hidden_dim).to(device)
            else:
                raise ValueError

            # insert initial data to policy_storage
            if self.args.exp_label in ['rl2', 'noisy_rl2']:
                if not self.args.shared_rnn:
                    self.policy_storage.insert_initial(
                        states=curr_states.squeeze(0),
                        states_for_policy=curr_states_for_policy.squeeze(0),
                        actions=prev_actions.squeeze(0),
                        rewards=prev_rewards.squeeze(0),
                        normalized_rewards=normalized_prev_rewards.squeeze(0),
                        actor_hidden_states=actor_prev_hidden_states.squeeze(0),
                        critic_hidden_states=critic_prev_hidden_states.squeeze(0)
                    )
                elif self.args.shared_rnn:
                    # if shared_rnn, then both actor_hidden_states and
                    # critic_hidden_states are rnn_hidden_states
                    self.policy_storage.insert_initial(
                        states=curr_states.squeeze(0),
                        states_for_policy=curr_states_for_policy.squeeze(0),
                        actions=prev_actions.squeeze(0),
                        rewards=prev_rewards.squeeze(0),
                        normalized_rewards=normalized_prev_rewards.squeeze(0),
                        actor_hidden_states=rnn_prev_hidden_states.squeeze(0),
                        critic_hidden_states=rnn_prev_hidden_states.squeeze(0)
                    )
            elif self.args.exp_label == 'MPC':
                self.vae.rollout_storage.insert_running_initial(
                    states=curr_states.squeeze(0),
                    states_for_policy=curr_states_for_policy.squeeze(0),
                    actions=prev_actions.squeeze(0),
                    rewards=prev_rewards.squeeze(0),
                    encoder_hidden_states=encoder_rnn_prev_hidden_states.squeeze(0)
                )
                self.policy_storage.insert_initial(
                    states=curr_states.squeeze(0),
                    states_for_policy=curr_states_for_policy.squeeze(0),
                    actions=prev_actions.squeeze(0),
                    rewards=prev_rewards.squeeze(0),
                    normalized_rewards=normalized_prev_rewards.squeeze(0),
                )
            else:
                raise ValueError(f'incompatible model type: {self.args.exp_label}')

            # rollout current policy for n steps in parallel environments
            for step in range(self.args.policy_num_steps_per_update):
                
                # sample actions from policy: act based on current policy
                with torch.no_grad():
                    if self.args.exp_label in ['rl2', 'noisy_rl2']:
                        if not self.args.shared_rnn:
                            actions_categorical, action_log_probs, entropy, state_values, \
                                actor_hidden_states, critic_hidden_states = \
                                    self.policy.actor_critic.act(
                                        curr_states=curr_states_for_policy,
                                        prev_actions=prev_actions,
                                        prev_rewards=prev_rewards,
                                        actor_prev_hidden_states=actor_prev_hidden_states, 
                                        critic_prev_hidden_states=critic_prev_hidden_states,
                                        return_prior=False, 
                                        deterministic=self.args.deterministic_policy)
                        elif self.args.shared_rnn:
                            actions_categorical, action_log_probs, entropy, state_values, \
                                rnn_hidden_states = \
                                    self.policy.actor_critic.act(
                                        curr_states=curr_states_for_policy,
                                        prev_actions=prev_actions,
                                        prev_rewards=prev_rewards,
                                        rnn_prev_hidden_states=rnn_prev_hidden_states,
                                        return_prior=False, 
                                        deterministic=self.args.deterministic_policy)
                    elif self.args.exp_label == 'MPC':
                        curr_latent_samples, curr_latent_means, curr_latent_logvars, encoder_rnn_hidden_states = \
                            self.vae.encoder(
                                curr_states=curr_states_for_policy, 
                                prev_actions=prev_actions, 
                                prev_rewards=prev_rewards,
                                rnn_prev_hidden_states=encoder_rnn_prev_hidden_states, 
                                return_prior=False, 
                                sample=True, 
                                detach_every=None
                            )
                        # cast latents to tensor (step, batch, feature)
                        curr_latent_means = curr_latent_means.reshape(1, self.args.num_processes, self.args.latent_dim).to(device)
                        curr_latent_logvars = curr_latent_logvars.reshape(1, self.args.num_processes, self.args.latent_dim).to(device)
                        actions_categorical, action_log_probs, entropy, state_values = \
                            self.policy.actor_critic.act(
                                curr_states=curr_states_for_policy, 
                                curr_latent_means=curr_latent_means, 
                                curr_latent_logvars=curr_latent_logvars, 
                                deterministic=self.args.deterministic_policy
                            )
                    else:
                        raise ValueError(f'incompatible model type: {self.args.exp_label}')
                
                # perform the action a_{t} in the environment to get s_{t+1} and r_{t+1}
                next_states_dict, rewards, terminated, truncated, infos = self.envs.step(
                    actions_categorical.squeeze(0).cpu().numpy())
                # cast states and rewards to tensor (step, batch, feature)
                next_states = utl.get_states_from_state_dicts(
                    next_states_dict, self.args, True)
                next_states_for_policy = utl.get_states_from_state_dicts(
                    next_states_dict, self.args, self.args.time_as_state)
                next_states = torch.from_numpy(next_states).float().\
                    reshape((1, self.args.num_processes, self.args.state_dim)).to(device)
                next_states_for_policy = torch.from_numpy(next_states_for_policy).float().\
                    reshape((1, self.args.num_processes, self.args.input_state_dim_for_policy)).to(device)

                rewards = torch.from_numpy(rewards).float()\
                    .reshape(1, self.args.num_processes, self.args.reward_dim).to(device)
                # Update return statistics (with raw rewards)
                if self.args.normalize_rew_for_policy:
                    self.return_normalizer.update(rewards.squeeze())
                    # Normalize the reward
                    normalized_rewards = self.return_normalizer.normalize(rewards)
                else:
                    normalized_rewards = rewards

                
                # create masks for return calculation:
                # for each env 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                
                ## if episodic task: use the episode_terminated flag from infos
                ## as AsyncVecEnv automatically resets the envs upon the termination flag
                ## which we do no want
                if self.args.env_name.split('-')[0] in[
                    'StatTiger', 'MarkovTiger'
                ]:
                    if step == self.args.policy_num_steps_per_update-1:
                        masks_ongoing = torch.tensor([0 for epi_term in infos['episode_terminated']]).float()\
                            .reshape(1, self.args.num_processes, 1).to(device)
                        # TODO: fix this last step mask issue
                    else:
                        masks_ongoing = torch.tensor([not epi_term for epi_term in infos['episode_terminated']]).float()\
                            .reshape(1, self.args.num_processes, 1).to(device)
                elif self.args.env_name.split('-')[0] in [
                    'OracleBanditDeterministic'
                ]:
                    masks_ongoing = torch.tensor([1 for _ in terminated]).float()\
                        .reshape(1, self.args.num_processes, 1).to(device)
                ## non episodic tasks: use the terminated flag from env.step
                else:
                    masks_ongoing = torch.tensor([not term for term in terminated]).float()\
                        .reshape(1, self.args.num_processes, 1).to(device)

                # update inputs for next step
                if self.args.action_space_type == 'Discrete':    
                    actions = F.one_hot(actions_categorical, num_classes=self.args.action_dim)\
                        .float().reshape((1, self.args.num_processes, self.args.action_dim))
                elif self.args.action_space_type == 'Box':
                    actions = actions_categorical.reshape((1, self.args.num_processes, self.args.action_dim))
                
                curr_states = next_states.to(device)
                curr_states_for_policy = next_states_for_policy.to(device)
                prev_actions = actions.to(device)
                prev_rewards = rewards.to(device)
                normalized_prev_rewards = normalized_rewards.to(device)
                if self.args.exp_label in ['rl2', 'noisy_rl2']:
                    if not self.args.shared_rnn:
                        actor_prev_hidden_states = actor_hidden_states.to(device)
                        critic_prev_hidden_states = critic_hidden_states.to(device)
                    elif self.args.shared_rnn:
                        rnn_prev_hidden_states = rnn_hidden_states.to(device)
                elif self.args.exp_label == 'MPC':
                    encoder_rnn_prev_hidden_states = encoder_rnn_hidden_states.to(device)
                else:
                    raise ValueError
                
                # insert experience to policy/ vae storage
                if self.args.exp_label in ['rl2', 'noisy_rl2']:
                    if not self.args.shared_rnn:
                        self.policy_storage.insert(
                            states=curr_states.squeeze(0),
                            states_for_policy=curr_states_for_policy.squeeze(0),
                            actions=prev_actions.squeeze(0),
                            action_log_probs=action_log_probs.reshape(self.args.num_processes, 1),
                            rewards=prev_rewards.squeeze(0),
                            normalized_rewards=normalized_prev_rewards.squeeze(0),
                            actor_hidden_states=actor_prev_hidden_states.squeeze(0),
                            critic_hidden_states=critic_prev_hidden_states.squeeze(0),
                            state_values=state_values.squeeze(0),
                            masks_ongoing=masks_ongoing.squeeze(0)
                        )
                    elif self.args.shared_rnn:
                        # again, if shared_rnn, then actor_hidden_states are just rnn_hidden_states,
                        # and critic_hidden_states are empty
                        self.policy_storage.insert(
                            states=curr_states.squeeze(0),
                            states_for_policy=curr_states_for_policy.squeeze(0),
                            actions=prev_actions.squeeze(0),
                            action_log_probs=action_log_probs.reshape(self.args.num_processes, 1),
                            rewards=prev_rewards.squeeze(0),
                            normalized_rewards=normalized_prev_rewards.squeeze(0),
                            actor_hidden_states=rnn_prev_hidden_states.squeeze(0),
                            critic_hidden_states=rnn_prev_hidden_states.squeeze(0),
                            state_values=state_values.squeeze(0),
                            masks_ongoing=masks_ongoing.squeeze(0)
                        )
                elif self.args.exp_label == 'MPC':
                    self.vae.rollout_storage.insert_running(
                        states=curr_states.squeeze(0),
                        states_for_policy=curr_states_for_policy.squeeze(0),
                        actions=prev_actions.squeeze(0),
                        rewards=prev_rewards.squeeze(0),
                        masks_ongoing=masks_ongoing.squeeze(0),
                        latent_means=curr_latent_means.squeeze(0),
                        latent_logvars=curr_latent_logvars.squeeze(0),
                        encoder_hidden_states=encoder_rnn_prev_hidden_states.squeeze(0)
                    )
                    self.policy_storage.insert(
                        states=curr_states.squeeze(0),
                        states_for_policy=curr_states_for_policy.squeeze(0),
                        latent_means=curr_latent_means.squeeze(0),
                        latent_logvars=curr_latent_logvars.squeeze(0),
                        actions=prev_actions.squeeze(0),
                        action_log_probs=action_log_probs.reshape(self.args.num_processes, 1),
                        rewards=prev_rewards.squeeze(0),
                        normalized_rewards=normalized_prev_rewards.squeeze(0),
                        state_values=state_values.squeeze(0),
                        masks_ongoing=masks_ongoing.squeeze(0)
                    )
                else:
                    raise ValueError(f'incompatible model type: {self.args.exp_label}')
        
        
            # -- UPDATE POLICY --
            # compute return
            self.policy_storage.compute_returns(
                self.args.policy_gamma, 
                self.args.policy_use_gae,
                self.args.policy_lambda
            )
            
            # for MPC: dump running to buffer
            if self.args.exp_label == 'MPC':
                self.vae.rollout_storage.dump_running_to_buffer()
                print(f'vae_storage ready for update: {self.vae.rollout_storage.ready_for_update()}')

            # compute loss
            # policy loss
            if self.args.exp_label in ['rl2', 'noisy_rl2']:
                loss, actor_loss, critic_loss, policy_entropy = \
                    self.policy.get_losses(self.policy_storage)
                
            elif self.args.exp_label == 'MPC':
                if self.args.policy_gradient_through_encoder:
                    loss, actor_loss, critic_loss, policy_entropy = \
                    self.policy.get_losses(self.policy_storage, self.vae.encoder)
                else:
                    loss, actor_loss, critic_loss, policy_entropy = \
                        self.policy.get_losses(self.policy_storage)
            else:
                raise ValueError(f'incompatible model type: {self.args.exp_label}')
            
            print(f'policy_loss: {loss}')
            print(f'actor_loss: {actor_loss}')
            print(f'critic_loss: {critic_loss}')
            print(f'policy_entropy: {policy_entropy}')

            # vae loss: by default do not backprop policy loss through the encoder
            if self.args.exp_label == 'MPC':
                elbo_loss, reward_reconstruction_loss, state_reconstruction_loss, kl_loss = \
                    self.vae.compute_vae_loss()
                
                print(f'elbo_loss: {elbo_loss}')
                print(f'reward_reconstruction_loss: {reward_reconstruction_loss}')
                print(f'state_reconstruction_loss: {state_reconstruction_loss}')
                print(f'kl_loss: {kl_loss}')

            # update parameters
            if self.args.exp_label in ['rl2', 'noisy_rl2']:
                self.policy.update_parameters(loss)

            elif self.args.exp_label == 'MPC':
                if self.args.policy_gradient_through_encoder:
                    # When policy backprops through encoder, we need to:
                    # 1. Zero gradients for both optimizers
                    self.policy.optimizer.zero_grad()
                    self.vae.optimizer_vae.zero_grad()
                    
                    # 2. Compute combined loss (VAE loss + Policy loss)
                    total_loss = elbo_loss + self.args.policy_gradient_encoder_coeff*loss
                    
                    # 3. Backward the combined loss
                    total_loss.backward()
                    
                    # 4. Step both optimizers
                    self.policy.step_optimizer()
                    self.vae.step_optimizer()
            
                else:
                    # ordinary: separate updates
                    self.policy.update_parameters(loss)
                    self.vae.update_parameters(elbo_loss)
            
            else:
                raise ValueError(f'incompatible model type: {self.args.exp_label}')

            # -- LOG --
            with torch.no_grad():
                # log the losses and entropy
                episode_return = (self.policy_storage.rewards[1:, :, :] * self.policy_storage.masks_ongoing[:-1, :, :]).squeeze().sum(dim=0).mean()
                print(f'episode_return: {episode_return}')
                train_stats['episode_returns'].append(episode_return.detach().cpu().numpy())
                train_stats['actor_losses'].append(actor_loss.detach().cpu().numpy())
                train_stats['critic_losses'].append(critic_loss.detach().cpu().numpy())
                train_stats['policy_entropies'].append(policy_entropy.detach().cpu().numpy())
                if self.args.exp_label == 'MPC':
                    train_stats['elbo_losses'].append(elbo_loss.detach().cpu().numpy())
                    train_stats['reward_reconstruction_losses'].append(reward_reconstruction_loss.detach().cpu().numpy())
                    if torch.is_tensor(state_reconstruction_loss):
                        train_stats['state_reconstruction_losses'].append(state_reconstruction_loss.detach().cpu().numpy())
                    else:
                        train_stats['state_reconstruction_losses'].append(state_reconstruction_loss)
                    if torch.is_tensor(kl_loss):
                        train_stats['kl_losses'].append(kl_loss.detach().cpu().numpy())
                    else:
                        train_stats['kl_losses'].append(kl_loss)
            # evaluation
            with torch.no_grad():
                self.log(evaluation_stats, start_time)

            # clear up running storage after update
            self.policy_storage.after_update()
            if self.args.exp_label == 'MPC':
                self.vae.rollout_storage.after_update()
        
        # return
        return train_stats, evaluation_stats
        

    def log(self, evaluation_stats, start_time):
            
        # --- evaluate policy ----
        if (self.iter_idx+1) in self.args.eval_ids:
            print(f'EVALUATION: epoch {self.iter_idx}')
            evaluation_stats['eval_epoch_ids'].append(self.iter_idx)

            if self.args.exp_label == 'MPC':
                encoder = self.vae.encoder
                policy_network = self.policy.actor_critic
            elif self.args.exp_label in ['rl2', 'noisy_rl2']:
                encoder = None
                policy_network = self.policy.actor_critic
                
            # for all environments: get empirical return
            num_eval_envs = self.args.num_eval_envs
            if self.args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
                empirical_return_avg, empirical_return_std, epi_length_avg, epi_length_std = get_empirical_returns(
                    env_name=self.args.env_name,
                    args=self.args,
                    encoder=encoder,  # None if rl2
                    policy_network=policy_network,
                    num_envs=num_eval_envs
                )
            else:
                empirical_return_avg, empirical_return_std = get_empirical_returns(
                    env_name=self.args.env_name,
                    args=self.args,
                    encoder=encoder,  # None if rl2
                    policy_network=policy_network,
                    num_envs=num_eval_envs
                )
            evaluation_stats['empirical_return_avgs'].append(empirical_return_avg)
            evaluation_stats['empirical_return_stds'].append(empirical_return_std)


        # --- save models ---
        if (self.iter_idx + 1) % self.args.save_interval == 0:
            print(f'SAVING MODEL: epoch {self.iter_idx}')
            save_path = self.logger.full_output_folder
            print(f'save_path: {save_path}')

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:
                # save model
                actor_critic_path = os.path.join(save_path, f'actor_critic_weights{idx_label}.h5')
                torch.save(self.policy.actor_critic.state_dict(), actor_critic_path)
                
                if self.args.exp_label == 'MPC':
                    encoder_path = os.path.join(save_path, f'encoder_weights{idx_label}.h5')
                    torch.save(self.vae.encoder.state_dict(), encoder_path)
                    if self.args.decode_reward:
                        reward_decoder_path = os.path.join(save_path, f'reward_decoder_weights{idx_label}.h5')
                        torch.save(self.vae.reward_decoder.state_dict(), reward_decoder_path)
                    if self.args.decode_state:
                        state_decoder_path = os.path.join(save_path, f'state_decoder_weights{idx_label}.h5')
                        torch.save(self.vae.state_decoder.state_dict(), state_decoder_path)