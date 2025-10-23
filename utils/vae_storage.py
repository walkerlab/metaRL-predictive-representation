import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutStorageVAE(object):
    def __init__(self, args):
        """
        Store everything needed for VAE update
        storage tensor shape: (num_steps+1, num_processes, feature_dim)
        --
        num_steps: how many timesteps per update (= length of online buffer)
        num_processes: number of parallel processes
        """
        self.args = args
        self.state_dim = self.args.state_dim
        self.input_state_dim_for_policy = self.args.input_state_dim_for_policy
        self.latent_dim = self.args.latent_dim
        self.action_dim = self.args.action_dim
        self.rnn_hidden_dim = self.args.encoder_rnn_hidden_dim

        self.max_num_steps = self.args.vae_storage_max_num_steps  # how long a trajectory can be at max (horizon)
        self.num_processes = self.args.num_processes
        self.max_buffer_size = self.args.vae_storage_max_buffer_size  # maximum buffer size (number of trajectories)
        
        # running storage: for each process (stored on GPU/ device)
        self.curr_timestep = 0  # count environment steps so we know where to insert
        # states will include s_0 when reset (hence num_steps+1): 
        # (s_0, s_1, ..., s_T)
        self.running_states = torch.zeros(
            self.max_num_steps+1, self.num_processes, self.state_dim)
        self.running_states_for_policy = torch.zeros(
            self.max_num_steps+1, self.num_processes, self.input_state_dim_for_policy)
        # actions: (a_-1, a_0, ..., a_T-1)
        self.running_actions = torch.zeros(
            self.max_num_steps+1, self.num_processes, self.action_dim)
        # rewards: (r_-1, r_0, ..., r_T-1)
        self.running_rewards = torch.zeros(
            self.max_num_steps+1, self.num_processes, 1)
        # masks_ongoing: (m_-1, m_0, m_1, ..., m_T-1)
        self.running_masks_ongoing = torch.ones(
            self.max_num_steps+1, self.num_processes, 1)     
        self.running_encoder_hidden_states = torch.zeros(
            self.max_num_steps+1, self.num_processes, self.rnn_hidden_dim)
        # latents: (z_0, z_1, ..., z_T-1)
        self.running_latent_means = torch.zeros(
            self.max_num_steps, self.num_processes, self.latent_dim)
        self.running_latent_logvars = torch.zeros(
            self.max_num_steps, self.num_processes, self.latent_dim)

        # buffer: for completed rollouts (stored on CPU)
        self.filled_buffer_size = 0  # how much of the buffer has been filled
        self.insert_idx = 0  # at which index we're currently inserting new data
        if self.max_buffer_size > 0:
            # states will include s_0 when reset (hence num_steps+1): (s_0, s_1, ..., s_T)
            self.buffer_states = torch.zeros(
                self.max_num_steps+1, self.max_buffer_size, self.state_dim)
            self.buffer_states_for_policy = torch.zeros(
                self.max_num_steps+1, self.max_buffer_size, self.input_state_dim_for_policy)
            # actions: (a_-1, a_0, ..., a_T-1)
            self.buffer_actions = torch.zeros(
                self.max_num_steps+1, self.max_buffer_size, self.action_dim)
            # rewards: (r_-1, r_0, ..., r_T-1)
            self.buffer_rewards = torch.zeros(
                self.max_num_steps+1, self.max_buffer_size, 1)
            # masks_ongoing: (m_-1, m_0, m_1, ..., m_T-1)
            self.buffer_masks_ongoing = torch.ones(
                self.max_num_steps+1, self.max_buffer_size, 1)  
            self.buffer_encoder_hidden_states = torch.zeros(
                self.max_num_steps+1, self.max_buffer_size, self.rnn_hidden_dim)
            # latents: (z_0, z_1, ..., z_T-1)
            self.buffer_latent_means = torch.zeros(
                self.max_num_steps, self.max_buffer_size, self.latent_dim)
            self.buffer_latent_logvars = torch.zeros(
                self.max_num_steps, self.max_buffer_size, self.latent_dim)


    def to_device(self):
        self.running_states = self.running_states.to(device)
        self.running_states_for_policy = self.running_states_for_policy.to(device)
        self.running_actions = self.running_actions.to(device)
        self.running_rewards = self.running_rewards.to(device)
        self.running_masks_ongoing = self.running_masks_ongoing.to(device)
        self.running_encoder_hidden_states = self.running_encoder_hidden_states.to(device)

    def insert_running_initial(
        self,
        states,
        states_for_policy,
        actions,
        rewards,
        encoder_hidden_states
    ):
        self.running_states[self.curr_timestep].copy_(states)
        self.running_states_for_policy[self.curr_timestep].copy_(states_for_policy)
        self.running_actions[self.curr_timestep].copy_(actions)
        self.running_rewards[self.curr_timestep].copy_(rewards)
        self.running_encoder_hidden_states[self.curr_timestep].copy_(encoder_hidden_states)

        self.curr_timestep = self.curr_timestep + 1

    def insert_running(
        self,
        states,
        states_for_policy,
        actions,
        rewards,
        masks_ongoing,
        latent_means,
        latent_logvars,
        encoder_hidden_states
    ):
        self.running_states[self.curr_timestep].copy_(states)
        self.running_states_for_policy[self.curr_timestep].copy_(states_for_policy)
        self.running_actions[self.curr_timestep].copy_(actions)
        self.running_rewards[self.curr_timestep].copy_(rewards)
        self.running_masks_ongoing[self.curr_timestep].copy_(masks_ongoing)
        self.running_encoder_hidden_states[self.curr_timestep].copy_(encoder_hidden_states)
        self.running_latent_means[self.curr_timestep-1].copy_(latent_means)
        self.running_latent_logvars[self.curr_timestep-1].copy_(latent_logvars)

        self.curr_timestep = self.curr_timestep + 1


    def dump_running_to_buffer(self):        
        # at the end of a rollout, 
        # dump the data into the (permanent) buffer
        if self.max_buffer_size > 0:
            if self.args.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                # check where to insert data
                if self.insert_idx + self.num_processes > self.max_buffer_size:
                    # keep track of how much we filled the buffer 
                    # (for sampling from it)
                    self.filled_buffer_size = self.insert_idx
                    # this will keep some entries at the end of the buffer without overwriting them,
                    # but the buffer is large enough to make this negligible
                    self.insert_idx = 0

                # add running to buffer
                # note: num_trajectories are along dim=1,
                # trajectory_length along dim=0, to match pytorch RNN interface
                self.buffer_states[:, self.insert_idx:self.insert_idx+self.num_processes, :].copy_(self.running_states)
                self.buffer_states_for_policy[:, self.insert_idx:self.insert_idx+self.num_processes, :].copy_(self.running_states_for_policy)
                self.buffer_actions[:, self.insert_idx:self.insert_idx+self.num_processes, :].copy_(self.running_actions)
                self.buffer_rewards[:, self.insert_idx:self.insert_idx+self.num_processes, :].copy_(self.running_rewards)
                self.buffer_masks_ongoing[:, self.insert_idx:self.insert_idx+self.num_processes, :].copy_(self.running_masks_ongoing)
                self.buffer_encoder_hidden_states[:, self.insert_idx:self.insert_idx+self.num_processes, :].copy_(self.running_encoder_hidden_states)
                self.buffer_latent_means[:, self.insert_idx:self.insert_idx+self.num_processes, :].copy_(self.running_latent_means)
                self.buffer_latent_logvars[:, self.insert_idx:self.insert_idx+self.num_processes, :].copy_(self.running_latent_logvars)

                self.insert_idx += self.num_processes
        
        self.filled_buffer_size = max(self.filled_buffer_size, self.insert_idx)

    
    def after_update(self):
        self.curr_timestep = 0

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.filled_buffer_size

    def get_batch(self, batch_size=5, replace=False):
        batch_size = min(self.filled_buffer_size, batch_size)

        # select the indices for the processes from which we pick
        batch_traj_indices = np.random.choice(
            range(self.filled_buffer_size), batch_size, replace=replace
        )

        # select the rollouts we want
        batch_states_for_policy = self.buffer_states_for_policy[:, batch_traj_indices, :]
        batch_actions = self.buffer_actions[:, batch_traj_indices, :]
        batch_rewards = self.buffer_rewards[:, batch_traj_indices, :]
        batch_masks_ongoing = self.buffer_masks_ongoing[:, batch_traj_indices, :]

        return batch_states_for_policy.to(device), batch_actions.to(device), \
               batch_rewards.to(device), batch_masks_ongoing.to(device)
