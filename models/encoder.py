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
    

class RNNEncoder(nn.Module):
    def __init__(
        self,
        args,
        latent_dim,  # int
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
        reward_dim,
        reward_embed_dim
    ):
        super(RNNEncoder, self).__init__()

        self.args = args

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.state_embed_dim = state_embed_dim
        self.action_embed_dim = action_embed_dim
        self.reward_embed_dim = reward_embed_dim

        self.latent_dim = latent_dim

        self.reparameterise = self._sample_gaussian

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
        self.fc_before_rnn, fc_before_rnn_final_dim = self.gen_fc_layers(
            layers_before_rnn, curr_input_dim
        )

        # recurrent layer
        self.rnn_hidden_dim = rnn_hidden_dim
        if rnn_cell_type == 'vanilla':
            self.rnn = nn.RNN(
                input_size=fc_before_rnn_final_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=1)
        elif rnn_cell_type == 'gru':
            self.rnn = nn.GRU(
                input_size=fc_before_rnn_final_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=1)
        else:
            raise ValueError(f'invalid rnn_cell_type: {rnn_cell_type}')
        
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = self.rnn_hidden_dim
        self.fc_after_rnn, fc_after_rnn_final_dim = self.gen_fc_layers(
            layers_after_rnn, curr_input_dim
        )

        # output layer
        self.encoder_mu = self.init_(nn.Linear(fc_after_rnn_final_dim, latent_dim))
        self.encoder_logvar = self.init_(nn.Linear(fc_after_rnn_final_dim, latent_dim))

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        
        # TODO: double check this code, maybe we should use 
        # .unsqueeze(0).expand((num, *logvar.shape))
        else:
            raise NotImplementedError
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def gen_fc_layers(self, layers, curr_input_dim):
        fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            fc = self.init_(nn.Linear(curr_input_dim, layers[i]))
            fc_layers.append(fc)
            curr_input_dim = layers[i]
        return fc_layers, curr_input_dim

    def init_hidden(self, batch_size, sample=True):
        '''
        start out with a hidden state of all zeros
        ---
        model: str, "actor" or "critic"
        '''
        # TODO: add option to incorporate the initial state
        prior_hidden_states = torch.zeros((1, batch_size, self.rnn_hidden_dim), 
                                          requires_grad=True).to(device)
        
        # forward through fully connected layers after RNN
        h = prior_hidden_states
        for i in range(len(self.fc_after_rnn)):
            h = F.relu(self.actor_fc_after_rnn[i](h))

        # outputs
        prior_latent_means = self.encoder_mu(h)
        prior_latent_logvars = self.encoder_logvar(h)
        if sample:
            prior_latent_samples = self.reparameterise(
                prior_latent_means, prior_latent_logvars
            )
        else:
            prior_latent_samples = prior_latent_means

        return prior_latent_samples, prior_latent_means, prior_latent_logvars, \
            prior_hidden_states

    def forward(self,
                curr_states, prev_actions, prev_rewards,
                rnn_prev_hidden_states, 
                return_prior, 
                sample=True, 
                detach_every=None
        ):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
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
            prior_latent_samples, prior_latent_means, prior_latent_logvars, \
                rnn_prior_hidden_states = self.init_hidden(batch_size, sample=sample)
            rnn_prev_hidden_states = rnn_prior_hidden_states.clone()

        # fc before RNN
        for i in range(len(self.fc_before_rnn)):
            h = self.fc_before_rnn[i](h)
            h = self.activation_function(h)
        # RNN
        rnn_hidden_states, _ = self.rnn(h, rnn_prev_hidden_states) # rnn output: output, h_n
        h = rnn_hidden_states.clone()
        # fc after RNN
        for i in range(len(self.fc_after_rnn)):
            h = self.fc_after_rnn[i](h)
            h = self.activation_function(h)

        # outputs
        latent_means = self.encoder_mu(h)
        latent_logvars = self.encoder_logvar(h)
        if sample:
            latent_samples = self.reparameterise(
                latent_means, latent_logvars
            )
        else:
            latent_samples = latent_means

        if return_prior:
            latent_samples = torch.cat((prior_latent_samples, latent_samples))
            latent_means = torch.cat((prior_latent_means, latent_means))
            latent_logvars = torch.cat((prior_latent_logvars, latent_logvars))
            rnn_hidden_states = torch.cat((rnn_prior_hidden_states, rnn_hidden_states))

        if latent_means.shape[0] == 1:
            latent_samples, latent_means, latent_logvars = latent_samples[0], latent_means[0], latent_logvars[0]

        return latent_samples, latent_means, latent_logvars, rnn_hidden_states
