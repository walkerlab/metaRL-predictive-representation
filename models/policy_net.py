"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ActorCriticRNN(nn.Module):
    def __init__(self,
                 args,
                 layers_before_rnn, # list
                 rnn_hidden_dim, # int
                 layers_after_rnn, # list
                 rnn_cell_type, # vanilla, gru
                 activation_function,  # tanh, relu, leaky-relu
                 initialization_method, # orthogonal, normc
                 state_dim,
                 state_embed_dim,
                 action_dim,
                 action_embed_dim,
                 action_space_type, # Discrete, Box
                 reward_dim,
                 reward_embed_dim
                 ):
        '''
        Separate single-layered RNNs for actor and critic
        '''
        super(ActorCriticRNN, self).__init__()
        
        self.args = args

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.state_embed_dim = state_embed_dim
        self.action_embed_dim = action_embed_dim
        self.reward_embed_dim = reward_embed_dim
        self.action_space_type = action_space_type

        # set activation function
        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        # set initialization method
        if initialization_method == 'normc':
            self.init_ = lambda m: init(m, init_normc_, 
                                        lambda x: nn.init.constant_(x, 0), 
                                        nn.init.calculate_gain(activation_function))
        elif initialization_method == 'orthogonal':
            self.init_ = lambda m: init(m, nn.init.orthogonal_, 
                                        lambda x: nn.init.constant_(x, 0), 
                                        nn.init.calculate_gain(activation_function))

        # embedder for state, action, reward
        if (self.state_embed_dim!=0) & (self.action_embed_dim!=0) & (self.reward_embed_dim!=0):
            self.state_encoder = utl.FeatureExtractor(self.state_dim, self.state_embed_dim, F.relu)
            self.action_encoder = utl.FeatureExtractor(self.action_dim, self.action_embed_dim, F.relu)
            self.reward_encoder = utl.FeatureExtractor(self.reward_dim, self.reward_embed_dim, F.relu)
            curr_input_dim =  self.state_embed_dim + self.action_embed_dim + self.reward_embed_dim
        else:
            curr_input_dim = self.state_dim + self.action_dim + self.reward_dim

        # initialize actor and critic
        # fully connected layers before the recurrent cell
        self.actor_fc_before_rnn, actor_fc_before_rnn_final_dim = self.gen_fc_layers(layers_before_rnn, 
                                                                                     curr_input_dim)
        self.critic_fc_before_rnn, critic_fc_before_rnn_final_dim = self.gen_fc_layers(layers_before_rnn, 
                                                                                       curr_input_dim)

        # recurrent layer
        self.rnn_hidden_dim = rnn_hidden_dim
        if rnn_cell_type == 'vanilla':
            self.actor_rnn = nn.RNN(
                input_size=actor_fc_before_rnn_final_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=1)
            self.critic_rnn = nn.RNN(
                input_size=critic_fc_before_rnn_final_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=1)
        elif rnn_cell_type == 'gru':
            self.actor_rnn = nn.GRU(
                input_size=actor_fc_before_rnn_final_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=1)
            self.critic_rnn = nn.GRU(
                input_size=critic_fc_before_rnn_final_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=1)
        else:
            raise ValueError(f'invalid rnn_cell_type: {rnn_cell_type}')
        
        for name, param in self.actor_rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        for name, param in self.critic_rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = self.rnn_hidden_dim
        self.actor_fc_after_rnn, actor_fc_after_rnn_final_dim = self.gen_fc_layers(layers_after_rnn, 
                                                                                   curr_input_dim)
        self.critic_fc_after_rnn, critic_fc_after_rnn_final_dim = self.gen_fc_layers(layers_after_rnn, 
                                                                                     curr_input_dim)
        
        # output layer
        if action_space_type == 'Discrete':
            self.actor_output = self.init_(nn.Linear(actor_fc_after_rnn_final_dim, action_dim))
            self.policy_dist = torch.distributions.Categorical
        elif action_space_type == 'Box':
            # For Gaussian policy, we need mean and log_std
            self.actor_mean = self.init_(nn.Linear(actor_fc_after_rnn_final_dim, action_dim))
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            self.policy_dist = torch.distributions.Normal
        else:
            raise NotImplementedError(f"Action space type {action_space_type} not supported")
        
        self.critic_output = self.init_(nn.Linear(critic_fc_after_rnn_final_dim, 1))

    def gen_fc_layers(self, layers, curr_input_dim):
        fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            fc = self.init_(nn.Linear(curr_input_dim, layers[i]))
            fc_layers.append(fc)
            curr_input_dim = layers[i]
        return fc_layers, curr_input_dim

    def forward_actor(self, inputs, prev_hidden_states):
        h = inputs
        # fc before RNN
        for i in range(len(self.actor_fc_before_rnn)):
            h = self.actor_fc_before_rnn[i](h)
            h = self.activation_function(h)
        # RNN
        hidden_states, _ = self.actor_rnn(h, prev_hidden_states) # rnn output: output, h_n
        h = hidden_states.clone()
        # fc after RNN
        for i in range(len(self.actor_fc_after_rnn)):
            h = self.actor_fc_after_rnn[i](h)
            h = self.activation_function(h)
        return h, hidden_states

    def forward_critic(self, inputs, prev_hidden_states):
        h = inputs
        # fc before RNN
        for i in range(len(self.critic_fc_before_rnn)):
            h = self.critic_fc_before_rnn[i](h)
            h = self.activation_function(h)
        # RNN
        hidden_states, _ = self.critic_rnn(h, prev_hidden_states)
        h = hidden_states.clone()
        # fc after RNN
        for i in range(len(self.critic_fc_after_rnn)):
            h = self.critic_fc_after_rnn[i](h)
            h = self.activation_function(h)
        return h, hidden_states

    def init_hidden(self, batch_size, model):
        '''
        start out with a hidden state of all zeros
        ---
        model: str, "actor" or "critic"
        '''
        prior_hidden_states = torch.zeros((1, batch_size, self.rnn_hidden_dim), 
                                         requires_grad=True).to(device)
        h = prior_hidden_states
        # forward through fully connected layers after RNN
        if model == 'actor':
            for i in range(len(self.actor_fc_after_rnn)):
                h = F.relu(self.actor_fc_after_rnn[i](h))
            if self.action_space_type == 'Discrete':
                prior_output = self.actor_output(h)
            else:  # Box/Continuous
                prior_means = self.actor_output_mean(h)
                prior_stds = torch.exp(self.actor_log_std).expand_as(prior_means)
                prior_output = (prior_means, prior_stds)
        elif model == 'critic':
            for i in range(len(self.critic_fc_after_rnn)):
                h = F.relu(self.critic_fc_after_rnn[i](h))
            prior_output = self.critic_output(h)
        else:
            raise ValueError(f'model can only be actor or critic')

        return prior_output, prior_hidden_states

    def forward(self, 
                curr_states, prev_actions, prev_rewards,
                actor_prev_hidden_states, critic_prev_hidden_states,
                return_prior=False):
        """
        Actions, states, rewards should be given in shape [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_states!=None.
        (hidden_states = [hidden_actor, hidden_critic])
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In either case, we may return embeddings of length sequence_len+1 
        if they include the prior.
        """
        # input shape: sequence_len x batch_size x feature_dim
        # extract features for states, actions, rewards
        if (self.state_embed_dim!=0) & (self.action_embed_dim!=0) & (self.reward_embed_dim!=0):
            hs = self.state_encoder(curr_states)
            ha = self.action_encoder(prev_actions)
            hr = self.reward_encoder(prev_rewards)
            h = torch.cat((hs, ha, hr), dim=-1)
        else:
            h = torch.cat((curr_states, prev_actions, prev_rewards), dim=-1)

        # initialize hidden state
        # if hidden_states is none, start with the prior
        if (actor_prev_hidden_states is None) and (critic_prev_hidden_states is None):
            batch_size = curr_states.shape[1]
            prior_action_logits, actor_prior_hidden_states = self.init_hidden(batch_size, 'actor')
            prior_state_values, critic_prior_hidden_states = self.init_hidden(batch_size, 'critic')
            actor_prev_hidden_states = actor_prior_hidden_states.clone()
            critic_prev_hidden_states = critic_prior_hidden_states.clone()

        # forward through actor_critic
        actor_h, actor_hidden_states = self.forward_actor(h, actor_prev_hidden_states)
        critic_h, critic_hidden_states = self.forward_critic(h, critic_prev_hidden_states)
        # print(f'actor_hidden_states: {actor_hidden_states.shape}')
        # print(f'critic_hidden_states: {critic_hidden_states.shape}')

        # outputs
        if self.action_space_type == 'Discrete':
            action_logits = self.actor_output(actor_h)
        else:  # Box/Continuous
            action_means = self.actor_mean(actor_h)
            action_stds = torch.exp(self.actor_log_std).expand_as(action_means)
            action_logits = (action_means, action_stds)
            
        state_values = self.critic_output(critic_h)

        if return_prior:
            if self.action_space_type == 'Discrete':
                action_logits = torch.cat((prior_action_logits, action_logits))
            else:  # Box/Continuous
                prior_means, prior_stds = prior_action_logits
                action_logits = (torch.cat((prior_means, action_means)), 
                               torch.cat((prior_stds, action_stds)))
            state_values = torch.cat((prior_state_values, state_values))
            actor_hidden_states = torch.cat((actor_prior_hidden_states, actor_hidden_states))
            critic_hidden_states = torch.cat((critic_prior_hidden_states, critic_hidden_states))
        
        return action_logits, state_values, actor_hidden_states, critic_hidden_states

    def act(self, 
            curr_states, prev_actions, prev_rewards, 
            actor_prev_hidden_states, critic_prev_hidden_states,
            return_prior=False, deterministic=False):
        """
        Returns the (raw) actions and their value.
        """
        # forward once
        action_logits, state_values, actor_hidden_states, critic_hidden_states = self.forward(
            curr_states, prev_actions, prev_rewards,
            actor_prev_hidden_states, critic_prev_hidden_states,
            return_prior=return_prior
        )
        
        # sample action
        if self.action_space_type == 'Discrete':
            action_pd = self.policy_dist(logits=action_logits)
        else:  # Box/Continuous
            action_means, action_stds = action_logits
            action_pd = self.policy_dist(action_means, action_stds)
            
        if deterministic:
            if isinstance(action_pd, torch.distributions.Categorical):
                actions = action_pd.mode
            elif isinstance(action_pd, torch.distributions.Normal):
                actions = action_pd.mean
            else:
                actions = action_pd.mean
        else:
            actions = action_pd.sample()
            
        action_log_probs = action_pd.log_prob(actions)
        
        # For continuous actions, sum log probs across action dimensions
        if isinstance(action_pd, torch.distributions.Normal) and len(actions.shape) > 1:
            action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
            
        entropy = action_pd.entropy()
        if isinstance(action_pd, torch.distributions.Normal) and len(actions.shape) > 1:
            entropy = entropy.sum(dim=-1, keepdim=True)

        return actions, action_log_probs, entropy, state_values, actor_hidden_states, critic_hidden_states


class SharedActorCriticRNN(nn.Module):
    def __init__(self,
                 args,
                 layers_before_rnn, # list
                 rnn_hidden_dim, # int
                 layers_after_rnn, # list
                 rnn_cell_type, # vanilla, gru
                 activation_function,  # tanh, relu, leaky-relu
                 initialization_method, # orthogonal, normc
                 state_dim,
                 state_embed_dim,
                 action_dim,
                 action_embed_dim,
                 action_space_type, # Discrete, Box
                 reward_dim,
                 reward_embed_dim
                 ):
        '''
        use shared single-layered RNNs for both actor and critic
        '''
        super(SharedActorCriticRNN, self).__init__()
        
        self.args = args

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.state_embed_dim = state_embed_dim
        self.action_embed_dim = action_embed_dim
        self.reward_embed_dim = reward_embed_dim

        # set activation function
        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        # set initialization method
        if initialization_method == 'normc':
            self.init_ = lambda m: init(m, init_normc_, 
                                        lambda x: nn.init.constant_(x, 0), 
                                        nn.init.calculate_gain(activation_function))
        elif initialization_method == 'orthogonal':
            self.init_ = lambda m: init(m, nn.init.orthogonal_, 
                                        lambda x: nn.init.constant_(x, 0), 
                                        nn.init.calculate_gain(activation_function))

        # embedder for state, action, reward
        if (self.state_embed_dim!=0) & (self.action_embed_dim!=0) & (self.reward_embed_dim!=0):
            self.state_encoder = utl.FeatureExtractor(self.state_dim, self.state_embed_dim, F.relu)
            self.action_encoder = utl.FeatureExtractor(self.action_dim, self.action_embed_dim, F.relu)
            self.reward_encoder = utl.FeatureExtractor(self.reward_dim, self.reward_embed_dim, F.relu)
            curr_input_dim =  self.state_embed_dim + self.action_embed_dim + self.reward_embed_dim
        else:
            curr_input_dim = self.state_dim + self.action_dim + self.reward_dim

        # initialize actor and critic
        # fully connected layers before the recurrent cell
        self.fc_before_rnn, fc_before_rnn_final_dim = self.gen_fc_layers(layers_before_rnn, 
                                                                         curr_input_dim)

        # recurrent layer
        self.rnn_hidden_dim = rnn_hidden_dim
        if rnn_cell_type == 'vanilla':
            self.shared_rnn = nn.RNN(
                input_size=fc_before_rnn_final_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=1)
        elif rnn_cell_type == 'gru':
            self.shared_rnn = nn.GRU(
                input_size=fc_before_rnn_final_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=1)
        else:
            raise ValueError(f'invalid rnn_cell_type: {rnn_cell_type}')
        
        for name, param in self.shared_rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = self.rnn_hidden_dim
        self.fc_after_rnn, fc_after_rnn_final_dim = self.gen_fc_layers(layers_after_rnn, 
                                                                                   curr_input_dim)
        
        # Store action space type for later use
        self.action_space_type = action_space_type
        
        # output layer
        if action_space_type == 'Discrete':
            self.actor_output = self.init_(nn.Linear(fc_after_rnn_final_dim, action_dim))
            self.policy_dist = torch.distributions.Categorical
        elif action_space_type == 'Box':
            self.actor_mean = self.init_(nn.Linear(fc_after_rnn_final_dim, action_dim))
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            self.policy_dist = torch.distributions.Normal
        else:
            raise NotImplementedError(f"Action space type {action_space_type} not supported")
            
        self.critic_output = self.init_(nn.Linear(fc_after_rnn_final_dim, 1))

    def gen_fc_layers(self, layers, curr_input_dim):
        fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            fc = self.init_(nn.Linear(curr_input_dim, layers[i]))
            fc_layers.append(fc)
            curr_input_dim = layers[i]
        return fc_layers, curr_input_dim

    def forward_shared_actor_critic(self, inputs, prev_hidden_states):
        # fc before RNN
        h = inputs
        for i in range(len(self.fc_before_rnn)):
            h = self.fc_before_rnn[i](h)
            h = self.activation_function(h)
        # RNN
        hidden_states, _ = self.shared_rnn(h, prev_hidden_states) # rnn output: output, h_n
        h = hidden_states.clone()
        # fc after RNN
        for i in range(len(self.fc_after_rnn)):
            h = self.fc_after_rnn[i](h)
            h = self.activation_function(h)
        
        return h, hidden_states

    def init_hidden(self, batch_size):
        '''
        start out with a hidden state of all zeros
        '''
        prior_hidden_states = torch.zeros((1, batch_size, self.rnn_hidden_dim), 
                                          requires_grad=True).to(device)
        
        # forward through fully connected layers after RNN
        h = prior_hidden_states
        for i in range(len(self.fc_after_rnn)):
            h = F.relu(self.fc_after_rnn[i](h))
    
        if self.action_space_type == 'Discrete':
            prior_action_logits = self.actor_output(h)
        else:  # Box/Continuous
            prior_means = self.actor_mean(h)
            prior_stds = torch.exp(self.actor_log_std).expand_as(prior_means)
            prior_action_logits = (prior_means, prior_stds)
            
        prior_state_values = self.critic_output(h)

        return prior_action_logits, prior_state_values, prior_hidden_states

    def forward(self, 
                curr_states, prev_actions, prev_rewards,
                rnn_prev_hidden_states,
                return_prior=False):
        """
        Actions, states, rewards should be given in shape [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_states!=None.
        (hidden_states = [hidden_actor, hidden_critic])
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In either case, we may return embeddings of length sequence_len+1 
        if they include the prior.
        """
        # input shape: sequence_len x batch_size x feature_dim
        # extract features for states, actions, rewards
        if (self.state_embed_dim!=0) & (self.action_embed_dim!=0) & (self.reward_embed_dim!=0):
            hs = self.state_encoder(curr_states)
            ha = self.action_encoder(prev_actions)
            hr = self.reward_encoder(prev_rewards)
            h = torch.cat((hs, ha, hr), dim=-1)
        else:
            h = torch.cat((curr_states, prev_actions, prev_rewards), dim=-1)

        # initialize hidden state
        # if hidden_states is none, start with the prior
        if rnn_prev_hidden_states is None:
            batch_size = curr_states.shape[1]
            prior_action_logits, prior_state_values, rnn_prior_hidden_states = self.init_hidden(batch_size)
            rnn_prev_hidden_states = rnn_prior_hidden_states.clone()

        # forward through shared actor_critic network
        shared_actor_critic_h, rnn_hidden_states = self.forward_shared_actor_critic(h, rnn_prev_hidden_states)

        # outputs
        if self.action_space_type == 'Discrete':
            action_logits = self.actor_output(shared_actor_critic_h)
        else:  # Box/Continuous
            action_means = self.actor_mean(shared_actor_critic_h)
            action_stds = torch.exp(self.actor_log_std).expand_as(action_means)
            action_logits = (action_means, action_stds)
            
        state_values = self.critic_output(shared_actor_critic_h)

        if return_prior:
            if self.action_space_type == 'Discrete':
                action_logits = torch.cat((prior_action_logits, action_logits))
            else:  # Box/Continuous
                prior_means, prior_stds = prior_action_logits
                action_logits = (torch.cat((prior_means, action_means)), 
                               torch.cat((prior_stds, action_stds)))
            state_values = torch.cat((prior_state_values, state_values))
            rnn_hidden_states = torch.cat((rnn_prior_hidden_states, rnn_hidden_states))
        
        return action_logits, state_values, rnn_hidden_states

    def act(self, 
            curr_states, prev_actions, prev_rewards, 
            rnn_prev_hidden_states,
            return_prior=False, deterministic=False):
        """
        Returns the (raw) actions and their value.
        """
        # forward once
        action_logits, state_values, rnn_hidden_states = self.forward(
            curr_states, prev_actions, prev_rewards,
            rnn_prev_hidden_states,
            return_prior=return_prior
        )
        
        # sample action
        if self.action_space_type == 'Discrete':
            action_pd = self.policy_dist(logits=action_logits)
        else:  # Box/Continuous
            action_means, action_stds = action_logits
            action_pd = self.policy_dist(action_means, action_stds)
            
        if deterministic:
            if isinstance(action_pd, torch.distributions.Categorical):
                actions = action_pd.mode
            else:
                actions = action_pd.mean
        else:
            actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()

        return actions, action_log_probs, entropy, state_values, rnn_hidden_states


# -- for MPC --
class ActorCriticMLP(nn.Module):
    def __init__(
        self,
        args,
        # inputs
        state_dim,
        latent_dim,
        # network
        hidden_layers,
        activation_function,  # tanh, relu, leaky-relu
        initialization_method, # orthogonal, normc
        # outputs
        action_dim,
        action_space_type # Discrete, Box
    ):
        '''
        Separate MLPs for actor and critic
        The policy can get any of these as input:
        - state (given by environment)
        - task (in the (belief) oracle setting)
        - latent variable (from VAE)
        '''
        super(ActorCriticMLP, self).__init__()
        
        self.args = args
        self.policy_mlp_use_observations = getattr(args, 'policy_mlp_use_observations', True) # Default to True for backward compatibility

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # set activation function
        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        # set initialization method
        if initialization_method == 'normc':
            self.init_ = lambda m: init(
                m, init_normc_, 
                lambda x: nn.init.constant_(x, 0), 
                nn.init.calculate_gain(activation_function)
            )
        elif initialization_method == 'orthogonal':
            self.init_ = lambda m: init(
                m, nn.init.orthogonal_, 
                lambda x: nn.init.constant_(x, 0), 
                nn.init.calculate_gain(activation_function)
            )

        # Calculate input dimension based on whether we're using observations
        if self.policy_mlp_use_observations:
            curr_input_dim = self.state_dim + self.latent_dim * 2  # to accomodate latent_mean and latent_logvar
        else:
            curr_input_dim = self.latent_dim * 2

        # initialize actor and critic
        # fully connected hidden layers
        self.actor_layers, actor_fc_final_dim = self.gen_fc_layers(
            hidden_layers, curr_input_dim)
        self.critic_layers, critic_fc_final_dim = self.gen_fc_layers(
            hidden_layers, curr_input_dim)
        
        # output layer
        self.action_space_type = action_space_type
        
        if action_space_type == 'Discrete':
            self.actor_output = self.init_(nn.Linear(actor_fc_final_dim, action_dim))
            self.policy_dist = torch.distributions.Categorical
        elif action_space_type == 'Box':
            self.actor_mean = self.init_(nn.Linear(actor_fc_final_dim, action_dim))
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            self.policy_dist = torch.distributions.Normal
        else:
            raise NotImplementedError(f"Action space type {action_space_type} not supported")
            
        self.critic_output = self.init_(nn.Linear(critic_fc_final_dim, 1))

    def gen_fc_layers(self, layers, curr_input_dim):
        fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            fc = self.init_(nn.Linear(curr_input_dim, layers[i]))
            fc_layers.append(fc)
            curr_input_dim = layers[i]
        return fc_layers, curr_input_dim

    def forward_actor(self, inputs):
        h = inputs
        for i in range(len(self.actor_layers)):
            h = self.actor_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward_critic(self, inputs):
        h = inputs
        for i in range(len(self.critic_layers)):
            h = self.critic_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward(self, curr_states, curr_latent_means, curr_latent_logvars):
        """
        states, latents should be given in shape [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_states!=None.
        (hidden_states = [hidden_actor, hidden_critic])
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In either case, we may return embeddings of length sequence_len+1 
        if they include the prior.
        
        The forward method with original parameter signature for backward compatibility.
        Now decides internally whether to use curr_states based on self.policy_mlp_use_observations.
        
        """
        # input shape: sequence_len x batch_size x feature_dim
        # Input preparation - use observations only if configured to do so
        if self.policy_mlp_use_observations:
            h = torch.cat((curr_states, curr_latent_means, curr_latent_logvars), dim=-1)
        else:
            h = torch.cat((curr_latent_means, curr_latent_logvars), dim=-1)
        # print(f'input h: {h.shape}')

        # forward through actor & critic
        actor_h = self.forward_actor(h)
        critic_h = self.forward_critic(h)
        # print(f'actor_hidden_states: {actor_hidden_states.shape}')
        # print(f'critic_hidden_states: {critic_hidden_states.shape}')

        # outputs
        if self.action_space_type == 'Discrete':
            action_logits = self.actor_output(actor_h)
        else:  # Box/Continuous
            action_means = self.actor_mean(actor_h)
            action_stds = torch.exp(self.actor_log_std).expand_as(action_means)
            action_logits = (action_means, action_stds)
            
        state_values = self.critic_output(critic_h)
        
        return action_logits, state_values

    def act(
        self, 
        curr_states, curr_latent_means, curr_latent_logvars, 
        deterministic=False
    ):
        """
        Returns the (raw) actions and their value.
        """
        # forward once
        action_logits, state_values = self.forward(curr_states, curr_latent_means, curr_latent_logvars)
        
        # sample action
        if self.action_space_type == 'Discrete':
            action_pd = self.policy_dist(logits=action_logits)
        else:  # Box/Continuous
            action_means, action_stds = action_logits
            action_pd = self.policy_dist(action_means, action_stds)
            
        if deterministic:
            if isinstance(action_pd, torch.distributions.Categorical):
                actions = action_pd.mode
            elif isinstance(action_pd, torch.distributions.Normal):
                actions = action_pd.mean
            else:
                actions = action_pd.mean
        else:
            actions = action_pd.sample()
            
        action_log_probs = action_pd.log_prob(actions)
        
        # For continuous actions, sum log probs across action dimensions
        if isinstance(action_pd, torch.distributions.Normal) and len(actions.shape) > 1:
            action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
            
        entropy = action_pd.entropy()
        if isinstance(action_pd, torch.distributions.Normal) and len(actions.shape) > 1:
            entropy = entropy.sum(dim=-1, keepdim=True)

        return actions, action_log_probs, entropy, state_values
