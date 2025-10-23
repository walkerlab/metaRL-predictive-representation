import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StateTransitionDecoder(nn.Module):
    def __init__(
        self,
        args,
        layers,
        pred_type,  # deterministic, Gaussian
        latent_dim,
        state_dim,
        action_dim,
    ):
        super(StateTransitionDecoder, self).__init__()

        self.args = args

        curr_input_dim = latent_dim + state_dim + action_dim
        self.fc_layers, curr_input_dim = self.gen_fc_layers(
            layers, curr_input_dim
        )
        # output layer
        if pred_type == 'gaussian':
            self.decoder_output = nn.Linear(curr_input_dim, 2 * state_dim)
        else:
            self.decoder_output = nn.Linear(curr_input_dim, state_dim)

    def gen_fc_layers(self, layers, curr_input_dim):
        fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            fc = nn.Linear(curr_input_dim, layers[i])
            fc_layers.append(fc)
            curr_input_dim = layers[i]
        return fc_layers, curr_input_dim

    def forward(self, latent_states, prev_states, actions):

        h = torch.cat((latent_states, prev_states, actions), 
                      dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.decoder_output(h)


class RewardDecoder(nn.Module):
    def __init__(
        self,
        args,
        layers,
        pred_type,  # bernoulli, categorical, deterministic
        latent_dim,
        state_dim,
        action_dim,
        input_action,  # bool
        input_prev_state  # bool
    ):
        super(RewardDecoder, self).__init__()

        self.args = args

        self.pred_type = pred_type
        self.input_action = input_action
        self.input_prev_state = input_prev_state
        
        # single head to predict reward conditioned on the state: i.e.
        # predict p(reward | latent, state=s) for the given s
        curr_input_dim = latent_dim + state_dim
        
        if self.input_action:
            # predict p(reward | latent, state=s, action=a) 
            # for the given s, a
            curr_input_dim += action_dim
            
        if input_prev_state:
            # predict p(reward | latent, states=s, prev_state=s') 
            # for the given s, s'
            curr_input_dim += state_dim

        self.fc_layers, curr_input_dim = self.gen_fc_layers(
            layers, curr_input_dim
        )
        # output layer
        if pred_type == 'gaussian':
            self.decoder_output = nn.Linear(curr_input_dim, 2)
        else:
            self.decoder_output = nn.Linear(curr_input_dim, 1)

    def gen_fc_layers(self, layers, curr_input_dim):
        fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            fc = nn.Linear(curr_input_dim, layers[i])
            fc_layers.append(fc)
            curr_input_dim = layers[i]
        return fc_layers, curr_input_dim

    def forward(
        self, 
        latent_states, 
        next_states, 
        actions=None,
        prev_states=None, 
    ):

        h = torch.cat((latent_states, next_states), dim=-1)
        if self.input_action:
            h = torch.cat((h, actions), dim=-1)
        if self.input_prev_state:
            h = torch.cat((h, prev_states), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.decoder_output(h)