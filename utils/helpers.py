from distutils.util import strtobool
import os
import random

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def seed(seed, deterministic_execution=False):
    print('Seeding: random, torch, numpy')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing \
               results will be similar but not identical.'
              'Use only one process and set --deterministic_execution to True \
               if you want identical results (only recommended for debugging).')


class FeatureExtractor(nn.Module):
    """ 
    Single layered feed-forward network 
    for embedding of states/actions/rewards
    """
    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(device)


def get_states_from_state_dicts(state_dicts, args, keep_time):
        '''
        get curr_states based on environment types
        ---
        state_dicts: Dict or OrderedDict
        args: args
        keep_time: boolean, whether to keep time
        '''
        
        # for trial-based bandit tasks
        if args.env_name.split('-')[0] in [
            'StatBernoulliBandit2ArmIndependent', 
            'OracleBanditDeterministic', 
            'DisSymmetricStickyMarkovBandit2State2Arm', 'DisAsymmetricRewardStickyMarkovBandit2State2Arm',
            'DisAsymmetricTransitionStickyMarkovBandit2State2Arm',
        ]:
            if not keep_time:
                states = np.array([])
            else:
                states = state_dicts['timestep']
        
        # for tiger task
        elif args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
            states = state_dicts['hear']
        
        # for latent cart task
        elif args.env_name.split('-')[0] in ['LatentGoalCart']:
            states = state_dicts['position']

        # raise error if task not identified
        else:
            raise ValueError(f'Env is not supported yet: {args.env_name}')
        
        return states


# for reward normalization
class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        self.var = m_2 / tot_count
        self.count = tot_count


class ReturnNormalizer:
    def __init__(self, size, epsilon=1e-8, clip=10.0):
        self.epsilon = epsilon
        self.clip = clip
        self.return_rms = RunningMeanStd(shape=(size,))
        
    def normalize(self, x):
        normalized_x = (x - self.return_rms.mean) / (self.return_rms.var.sqrt() + self.epsilon)
        if self.clip:
            normalized_x = torch.clamp(normalized_x, -self.clip, self.clip)
        return normalized_x
        
    def update(self, x):
        self.return_rms.update(x)


def plot_training_curves(
    args, out_dir,
    episode_returns, actor_losses, critic_losses, policy_entropies,
    elbo_losses, reward_reconstruction_losses, state_reconstruction_losses, kl_losses,
    rolling_length=10
):

    fig, axs = plt.subplots(
        nrows=4, ncols=2, 
        figsize=(12, 10),
        dpi=300
    )
    fig.suptitle(
        f"Training plots for model {out_dir.split('/')[-2]}/{out_dir.split('/')[-1]}\n"
        f"(n_envs={args.num_processes}, n_steps_per_update={args.policy_num_steps_per_update})"
    )

    # -- POLICY --
    # episode return
    ax = axs[0, 0]
    ax.set_title("Episode Returns")
    episode_returns_moving_average = (
        np.convolve(np.array(episode_returns), np.ones(rolling_length),
            mode="valid") 
        / rolling_length 
    )
    ax.plot(
        np.arange(len(episode_returns_moving_average)),
        episode_returns_moving_average,
    )
    ax.set_xlabel("Number of episodes")

    # actor loss
    ax = axs[1, 0]
    ax.set_title("Actor Loss")
    actor_losses_moving_average = (
        np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    ax.plot(actor_losses_moving_average)
    ax.set_xlabel("Number of updates")

    # critic loss
    ax = axs[2, 0]
    ax.set_title("Critic Loss")
    critic_losses_moving_average = (
        np.convolve(
            np.array(critic_losses).flatten(), np.ones(rolling_length), 
                mode="valid")
        / rolling_length
    )
    ax.plot(critic_losses_moving_average)
    ax.set_xlabel("Number of updates")

    # entropy
    ax = axs[3, 0]
    ax.set_title("Entropy")
    entropy_moving_average = (
        np.convolve(np.array(policy_entropies), np.ones(rolling_length), 
            mode="valid")
        / rolling_length
    )
    ax.plot(entropy_moving_average)
    ax.set_xlabel("Number of updates")


    # -- VAE --
    if args.exp_label == 'mpc':
        # vae_total_loss
        ax = axs[0, 1]
        ax.set_title("VAE loss")
        elbo_loss_moving_average = (
            np.convolve(np.array(elbo_losses), np.ones(rolling_length), 
                mode="valid")
            / rolling_length
        )
        ax.plot(elbo_loss_moving_average)
        ax.set_xlabel("Number of updates")

        # reward_reconstruction_loss
        ax = axs[1, 1]
        ax.set_title("Reward reconstruction loss")
        reward_reconstruction_loss_moving_average = (
            np.convolve(np.array(reward_reconstruction_losses), np.ones(rolling_length), 
                mode="valid")
            / rolling_length
        )
        ax.plot(reward_reconstruction_loss_moving_average)
        ax.set_xlabel("Number of updates")

        # state_reconstruction_loss
        ax = axs[2, 1]
        ax.set_title("State reconstruction loss")
        state_reconstruction_loss_moving_average = (
            np.convolve(np.array(state_reconstruction_losses), np.ones(rolling_length), 
                mode="valid")
            / rolling_length
        )
        ax.plot(state_reconstruction_loss_moving_average)
        ax.set_xlabel("Number of updates")

        # kl_loss
        ax = axs[3, 1]
        ax.set_title("KL loss")
        kl_loss_moving_average = (
            np.convolve(np.array(kl_losses), np.ones(rolling_length), 
                mode="valid")
            / rolling_length
        )
        ax.plot(kl_loss_moving_average)
        ax.set_xlabel("Number of updates")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'training_curves.png'))


def plot_evaluation_curves(
    out_dir,
    eval_epoch_ids,
    empirical_return_avgs,
    empirical_return_stds,
    num_eval_runs
):
    fig, axs = plt.subplots(
        nrows=1, ncols=1, 
        figsize=(6, 4),
        dpi=300
    )
    fig.suptitle(
        f"Eval empirical returns, model {out_dir.split('/')[-2]}/{out_dir.split('/')[-1]}\n"
    )
    axs.errorbar(
        eval_epoch_ids,
        empirical_return_avgs,
        yerr=empirical_return_stds / np.sqrt(num_eval_runs)
    )
    axs.set_xlabel("Epoch")
    axs.set_ylabel("episodic return")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eval_empirical_returns.png'))



#################################################
# STATE MACHINE ANALYSIS  
#################################################
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class StateSpaceMapper(nn.Module):
    def __init__(
        self,
        # inputs
        source_dim,
        # network
        hidden_layers,
        activation_function,  # tanh, relu, leaky-relu
        initialization_method, # orthogonal, normc
        # outputs
        target_dim
    ):
        super(StateSpaceMapper, self).__init__()

        self.source_dim = source_dim
        self.target_dim = target_dim

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

        curr_input_dim = self.source_dim
        self.mapper_layers, mapper_fc_final_dim = self.gen_fc_layers(
            hidden_layers, curr_input_dim)

        # output layer
        self.mapper_output = self.init_(nn.Linear(
            mapper_fc_final_dim, self.target_dim))


    def gen_fc_layers(self, layers, curr_input_dim):
        fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            fc = self.init_(nn.Linear(curr_input_dim, layers[i]))
            fc_layers.append(fc)
            curr_input_dim = layers[i]
        return fc_layers, curr_input_dim
    
    def forward_mapper(self, inputs):
        h = inputs
        for i in range(len(self.mapper_layers)):
            h = self.mapper_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward(self, source_states):
        # input shape: sequence_len x batch_size x feature_dim
        # forward through the mapper
        mapper_h = self.forward_mapper(source_states)
        mapped_states = self.mapper_output(mapper_h)

        return mapped_states


class StateMapperTrainer:
    def __init__(
        self,
        state_mapper,
        # optimization
        lr,
        eps,
        anneal_lr,
        train_steps,
        patience=10,
        min_delta=0.001,
        min_training_epochs=100
    ):
        self.state_mapper = state_mapper.to(device)
        self.patience = patience
        self.min_delta = min_delta
        self.min_training_epochs = min_training_epochs
        
        # initialize optimizer
        self.criterion = nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(
            self.state_mapper.parameters(), 
            lr=lr,
            eps=eps
        )
        # learning rate annealing
        self.lr_scheduler = None
        if anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lam)
    
    def evaluate(self, dataset, batch_size):
        """Evaluate the model on a dataset"""
        self.state_mapper.eval()
        total_loss = 0.0
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for x_, t_ in data_loader:
                x = x_.type(torch.FloatTensor).to(device)
                t = t_.type(torch.FloatTensor).to(device)
                
                y = self.state_mapper(x)
                loss = self.criterion(y, t)
                total_loss += loss.item() * x.size(0)
        
        return total_loss / len(dataset)

    def train(
        self, 
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        data_sampler=None
    ):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_state_dict = None
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.state_mapper.train()
            train_loss_epoch = 0
            
            # load data
            if data_sampler is not None:
                train_loader = DataLoader(
                    train_dataset, 
                    sampler=data_sampler, 
                    batch_size=batch_size
                )
            else:
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True
                )
            
            for x_, t_ in train_loader:
                x = x_.type(torch.FloatTensor).to(device)  # (batch_size, source_dim)
                t = t_.type(torch.FloatTensor).to(device)  # (batch_size, target_dim)

                # -- training --
                self.optimizer.zero_grad()

                y = self.state_mapper(x)  

                loss_mse_batch = self.criterion(y, t)
                
                loss_mse_batch.backward()
                self.optimizer.step()
                
                train_loss_epoch += loss_mse_batch.item() * x.size(0)
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Calculate average loss for the epoch
            train_loss_epoch /= len(train_dataset)
            train_losses.append(train_loss_epoch)
            
            # Validation phase
            val_loss = self.evaluate(val_dataset, batch_size)
            val_losses.append(val_loss)
            
            # Early stopping check
            if epoch > self.min_training_epochs:
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    best_state_dict = self.state_mapper.state_dict().copy()
                    patience_counter = 0
                    best_epoch = epoch
                else:
                    patience_counter += 1
            
            # Verbose output
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}')
                print(f' Train loss: {train_loss_epoch:.6f}')
                print(f' Val loss: {val_loss:.6f}')
                print(f' Best val loss: {best_val_loss:.6f}')
                print(f' Patience counter: {patience_counter}/{self.patience}')
            
            # Check if we should stop training
            if patience_counter >= self.patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Load the best model
        if best_state_dict is not None:
            self.state_mapper.load_state_dict(best_state_dict)
        
        return train_losses, val_losses, best_epoch

