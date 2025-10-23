"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Used for on-policy rollout storages.
"""


import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OnlineStorageRL2(object):
    def __init__(self, args):
        '''
        storage tensor shape: (num_steps+1, num_processes, feature_dim)
        --
        num_steps: how many timesteps per update (= length of online buffer)
        num_processes: number of parallel processes
        '''
        self.args = args
        self.state_dim = self.args.state_dim
        self.input_state_dim_for_policy = self.args.input_state_dim_for_policy
        self.action_dim = self.args.action_dim
        self.num_steps = self.args.policy_num_steps_per_update
        self.num_processes = self.args.num_processes  
        self.rnn_hidden_dim = self.args.rnn_hidden_dim

        self.step = 0  # keep track of current environment step

        # for re-building computational graph
        # inputs to the policy
        # states will include s_0 when reset (hence num_steps+1): (s_0, s_1, ..., s_T)
        self.states = torch.zeros(self.num_steps+1, self.num_processes, self.state_dim)
        self.states_for_policy = torch.zeros(self.num_steps+1, self.num_processes, self.input_state_dim_for_policy)
        # hidden states of RNN (necessary if we want to re-compute embeddings): 
        # (h_-1, h_0, ..., h_T-1)
        self.actor_hidden_states = torch.zeros(self.num_steps+1, 
                                               self.num_processes,
                                               self.rnn_hidden_dim)
        self.critic_hidden_states = torch.zeros(self.num_steps+1, 
                                                self.num_processes,
                                                self.rnn_hidden_dim)
        
        # actions: (a_-1, a_0, ..., a_T-1)
        self.actions = torch.zeros(self.num_steps+1, self.num_processes, self.action_dim)
        # action_log_prob: (a_0, a_1, ..., a_T-1)
        self.action_log_probs = torch.zeros(self.num_steps, self.num_processes, 1)
        # rewards: (r_-1, r_0, ..., r_T-1)
        self.rewards = torch.zeros(self.num_steps+1, self.num_processes, 1)
        # normalized_rewards: (r_-1, r_0, ..., r_T-1)
        self.normalized_rewards = torch.zeros(self.num_steps+1, self.num_processes, 1)
        # state_values: (v_0, v_1, ..., v_T-1)
        self.state_values = torch.zeros(self.num_steps, self.num_processes, 1)
        # returns: (G_0, G_1, ..., G_T-1)
        self.returns = torch.zeros(self.num_steps, self.num_processes, 1)
        # masks_ongoing: (m_-1, m_0, m_1, ..., m_T-1)
        self.masks_ongoing = torch.ones(self.num_steps+1, self.num_processes, 1)

        self.to_device()

    def to_device(self):
        self.states = self.states.to(device)
        self.states_for_policy = self.states_for_policy.to(device)
        self.actor_hidden_states = self.actor_hidden_states.to(device)
        self.critic_hidden_states = self.critic_hidden_states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.normalized_rewards = self.normalized_rewards.to(device)
        self.state_values = self.state_values.to(device)
        self.returns = self.returns.to(device)
        self.masks_ongoing = self.masks_ongoing.to(device)

    def insert_initial(
        self,
        states,
        states_for_policy,
        actions,
        rewards,
        normalized_rewards,
        actor_hidden_states,
        critic_hidden_states
    ):
        self.states[self.step].copy_(states)
        self.states_for_policy[self.step].copy_(states_for_policy)
        self.actor_hidden_states[self.step].copy_(actor_hidden_states)
        self.critic_hidden_states[self.step].copy_(critic_hidden_states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.normalized_rewards[self.step].copy_(normalized_rewards)
        self.step = self.step + 1

    def insert(
        self,
        states,
        states_for_policy,
        actions,
        action_log_probs,
        rewards,
        normalized_rewards,
        actor_hidden_states,
        critic_hidden_states,
        state_values,
        masks_ongoing
    ):
        self.states[self.step].copy_(states)
        self.states_for_policy[self.step].copy_(states_for_policy)
        self.actor_hidden_states[self.step].copy_(actor_hidden_states)
        self.critic_hidden_states[self.step].copy_(critic_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step-1].copy_(action_log_probs)
        self.rewards[self.step].copy_(rewards)
        self.normalized_rewards[self.step].copy_(normalized_rewards)
        self.state_values[self.step-1].copy_(state_values)
        self.masks_ongoing[self.step].copy_(masks_ongoing)

        self.step = self.step + 1

    def compute_returns(
        self, gamma, gae_lambda,
        use_gae=False
    ):
        T = self.normalized_rewards.size(0) - 1  # remove r_-1

        if use_gae:
            # generalized advantage estimator
            self.returns[-1] = self.normalized_rewards[-1] * self.masks_ongoing[-1]
            gae = 0
            for step in reversed(range(T-1)):
                td_error = self.normalized_rewards[step]\
                            + gamma * self.state_values[step+1] * self.masks_ongoing[step]\
                            - self.state_values[step]
                gae = td_error + gamma * gae_lambda * self.masks_ongoing[step] * gae
                self.returns[step] = gae + self.state_values[step]
        else:
            # Monte Carlo estimator
            self.returns[-1] = self.normalized_rewards[-1] * self.masks_ongoing[-1]
            for step in reversed(range(T-1)):
                self.returns[step] = self.returns[step+1] * gamma * self.masks_ongoing[step]\
                                        + self.normalized_rewards[step]

    def after_update(self):
        self.step = 0


class OnlineStorageMPC(object):
    def __init__(self, args):
        '''
        storage tensor shape: (num_steps+1, num_processes, feature_dim)
        --
        num_steps: how many timesteps per update (= length of online buffer)
        num_processes: number of parallel processes
        '''
        self.args = args
        self.state_dim = self.args.state_dim
        self.latent_dim = self.args.latent_dim
        self.input_state_dim_for_policy = self.args.input_state_dim_for_policy
        self.action_dim = self.args.action_dim
        self.rnn_hidden_dim = self.args.encoder_rnn_hidden_dim
        self.num_steps = self.args.policy_num_steps_per_update
        self.num_processes = self.args.num_processes  

        self.step = 0  # keep track of current environment step

        # for re-building computational graph
        # inputs to the policy
        # states will include s_0 when reset (hence num_steps+1): 
        # (s_0, s_1, ..., s_T)
        self.states = torch.zeros(self.num_steps+1, self.num_processes, self.state_dim)
        self.states_for_policy = torch.zeros(self.num_steps+1, self.num_processes, self.input_state_dim_for_policy)
        # latents: (z_0, z_1, ..., z_T-1)
        self.latent_means = torch.zeros(self.num_steps, self.num_processes, self.latent_dim)
        self.latent_logvars = torch.zeros(self.num_steps, self.num_processes, self.latent_dim)
        # actions: (a_-1, a_0, ..., a_T-1)
        self.actions = torch.zeros(self.num_steps+1, self.num_processes, self.action_dim)
        # action_log_prob: (a_0, a_1, ..., a_T-1)
        self.action_log_probs = torch.zeros(self.num_steps, self.num_processes, 1)

        # for computing returns
        self.rewards = torch.zeros(self.num_steps+1, self.num_processes, 1)
        # normalized_rewards: (r_-1, r_0, ..., r_T-1)
        self.normalized_rewards = torch.zeros(self.num_steps+1, self.num_processes, 1)
        # state_values: (v_0, v_1, ..., v_T-1)
        self.state_values = torch.zeros(self.num_steps, self.num_processes, 1)
        # returns: (G_0, G_1, ..., G_T-1)
        self.returns = torch.zeros(self.num_steps, self.num_processes, 1)
        # masks_ongoing: (m_-1, m_0, m_1, ..., m_T-1)
        self.masks_ongoing = torch.ones(self.num_steps+1, self.num_processes, 1)

        self.to_device()

    def to_device(self):
        self.states = self.states.to(device)
        self.states_for_policy = self.states_for_policy.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.normalized_rewards = self.normalized_rewards.to(device)
        self.state_values = self.state_values.to(device)
        self.returns = self.returns.to(device)
        self.masks_ongoing = self.masks_ongoing.to(device)

    def insert_initial(
        self,
        states,
        states_for_policy,
        actions,
        rewards,
        normalized_rewards
    ):
        self.states[self.step].copy_(states)
        self.states_for_policy[self.step].copy_(states_for_policy)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.normalized_rewards[self.step].copy_(normalized_rewards)

        self.step = self.step + 1

    def insert(
        self,
        states,
        states_for_policy,
        latent_means,
        latent_logvars,
        actions,
        action_log_probs,
        rewards,
        normalized_rewards,
        state_values,
        masks_ongoing
    ):
        self.states[self.step].copy_(states)
        self.states_for_policy[self.step].copy_(states_for_policy)
        self.latent_means[self.step-1].copy_(latent_means)
        self.latent_logvars[self.step-1].copy_(latent_logvars)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step-1].copy_(action_log_probs)
        self.rewards[self.step].copy_(rewards)
        self.normalized_rewards[self.step].copy_(normalized_rewards)
        self.state_values[self.step-1].copy_(state_values)
        self.masks_ongoing[self.step].copy_(masks_ongoing)

        self.step = self.step + 1

    def compute_returns(
        self, gamma, gae_lambda,
        use_gae=False
    ):
        T = self.normalized_rewards.size(0) - 1  # remove r_-1

        if use_gae:
            # generalized advantage estimator
            self.returns[-1] = self.normalized_rewards[-1] * self.masks_ongoing[-1]
            gae = 0
            for step in reversed(range(T-1)):
                td_error = self.normalized_rewards[step]\
                            + gamma * self.state_values[step+1] * self.masks_ongoing[step]\
                            - self.state_values[step]
                gae = td_error + gamma * gae_lambda * self.masks_ongoing[step] * gae
                self.returns[step] = gae + self.state_values[step]
        else:
            # Monte Carlo estimator
            self.returns[-1] = self.normalized_rewards[-1] * self.masks_ongoing[-1]
            for step in reversed(range(T-1)):
                self.returns[step] = self.returns[step+1] * gamma * self.masks_ongoing[step]\
                                        + self.normalized_rewards[step]

    def after_update(self):
        self.step = 0