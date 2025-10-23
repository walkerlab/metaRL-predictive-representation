import torch
from torch.nn import functional as F
import torch.nn as nn

from models.decoder import StateTransitionDecoder, RewardDecoder
from models.encoder import RNNEncoder
from utils.vae_storage import RolloutStorageVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE_MPC:
    """
    VAE of MPC:
    - RNN encoder and MLP decoders
    - Compute the ELBO loss
    - Update the VAE (encoder + decoder)
    """

    def __init__(self, args, logger):

        self.args = args
        self.logger = logger

        # initialize rnn encoder
        self.encoder = self.initialize_encoder()

        # initialize mlp decoders 
        self.state_decoder, self.reward_decoder = self.initialize_decoder()

        # initialize rollout storage for VAE update
        self.rollout_storage = RolloutStorageVAE(self.args)

        # initalize optimizer for the encoder and decoders
        decoder_params = []
        if not self.args.disable_decoder:
            if self.args.decode_reward:
                decoder_params.extend(self.reward_decoder.parameters())
            if self.args.decode_state:
                decoder_params.extend(self.state_decoder.parameters())
        self.optimizer_vae = torch.optim.Adam(
            [*self.encoder.parameters(), *decoder_params], 
            lr=self.args.lr_vae
        )

    def initialize_encoder(self):
        """ Initialize and return an RNN encoder """
        encoder = RNNEncoder(
            args=self.args,
            latent_dim=self.args.latent_dim,  # int
            layers_before_rnn=self.args.encoder_layers_before_rnn, # list
            rnn_hidden_dim=self.args.encoder_rnn_hidden_dim, # int
            layers_after_rnn=self.args.encoder_layers_after_rnn, # list
            rnn_cell_type=self.args.encoder_rnn_cell_type, # vanilla, gru
            activation_function=self.args.encoder_activation_function,  # tanh, relu, leaky-relu
            initialization_method=self.args.encoder_initialization_method, # orthogonal, normc
            state_dim=self.args.input_state_dim_for_policy,
            state_embed_dim=self.args.state_embed_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embed_dim,
            reward_dim=self.args.reward_dim,
            reward_embed_dim=self.args.reward_embed_dim
        ).to(device)
        return encoder

    def initialize_decoder(self):
        """
        Initialize and return the (state/ reward) decoder 
        as specified in self.args
        """
        # TODO: task decoders

        if self.args.disable_decoder:
            return None, None

        latent_dim = self.args.latent_dim
        # if we don't sample embeddings for the decoder, 
        # we feed in mean & variance
        if self.args.disable_stochasticity_in_latent:
            latent_dim *= 2

        # initialize state decoder for VAE
        if self.args.decode_state:
            state_decoder = StateTransitionDecoder(
                args=self.args,
                layers=self.args.state_decoder_layers,
                pred_type=self.args.state_pred_type,  # deterministic, Gaussian
                latent_dim=latent_dim,
                state_dim=self.args.input_state_dim_for_policy,
                action_dim=self.args.action_dim
            ).to(device)
        else:
            state_decoder = None

        # initialize reward decoder for VAE
        if self.args.decode_reward:
            reward_decoder = RewardDecoder(
                args=self.args,
                layers=self.args.reward_decoder_layers,
                pred_type=self.args.reward_pred_type,  # bernoulli, categorical, deterministic
                latent_dim=latent_dim,
                state_dim=self.args.input_state_dim_for_policy,
                action_dim=self.args.action_dim,
                input_action=self.args.input_action,  # bool
                input_prev_state=self.args.input_prev_state  # bool
            ).to(device)
        else:
            reward_decoder = None

        return state_decoder, reward_decoder
    

    def compute_state_reconstruction_loss(
        self, 
        latent_samples, states, actions, masks_ongoing,
        return_predictions=False
    ):
        """ 
        Compute state reconstruction loss.
        (No reduction of loss along batch dimension is done here; 
        sum/avg has to be done outside) 
        """

        pred_states = self.state_decoder(
            latent_states=latent_samples[:-1,:,:],
            prev_states=states[:-1,:,:],
            actions=actions[1:,:,:]
        )
        next_states = states[1:,:,:]
        
        if self.args.state_pred_type == 'gaussian':
            raise NotImplementedError('state_pred_type=gaussian not implemented')

        elif self.args.state_pred_type == 'deterministic':
            loss_state_reconstruction = pred_states - next_states

            # apply the masks
            loss_state_reconstruction = loss_state_reconstruction * masks_ongoing[:-1,:,:]
            
            loss_state_reconstruction = loss_state_reconstruction.pow(2).mean(dim=-1)
        
        elif self.args.state_pred_type == 'bernoulli':
            pred_states = torch.sigmoid(pred_states)
            target_states = (next_states == 1).float()[:]  # TODO: necessary?
            loss_state_reconstruction = F.binary_cross_entropy(pred_states, target_states, reduction='none')
            # apply the masks
            loss_state_reconstruction = loss_state_reconstruction * masks_ongoing[:-1,:,:]
            
            loss_state_reconstruction = loss_state_reconstruction.mean(dim=-1)

        if return_predictions:
            return loss_state_reconstruction, pred_states
        else:
            return loss_state_reconstruction


    def compute_reward_reconstruction_loss(
        self, 
        latent_samples, states, actions, rewards, masks_ongoing,
        return_predictions=False
    ):
        """ 
        Compute reward reconstruction loss.
        (No reduction of loss along batch dimension is done here; 
        sum/avg has to be done outside) 
        """

        pred_rewards = self.reward_decoder(
            latent_states=latent_samples[:-1,:,:],
            next_states=states[1:,:,:], 
            actions=actions[1:,:,:],
            prev_states=states[:-1,:,:]
        )
        
        if self.args.reward_pred_type == 'bernoulli':
            pred_rewards = torch.sigmoid(pred_rewards)
            target_rewards = (rewards == 1).float()[1:]  # TODO: necessary?
            loss_reward_reconstruction = F.binary_cross_entropy(pred_rewards, target_rewards, reduction='none')
            # apply the masks
            loss_reward_reconstruction = loss_reward_reconstruction * masks_ongoing[:-1,:,:]

            loss_reward_reconstruction = loss_reward_reconstruction.mean(dim=-1)

        elif self.args.reward_pred_type == 'deterministic':
            loss_reward_reconstruction = pred_rewards - rewards[1:]
            # apply the masks
            loss_reward_reconstruction = loss_reward_reconstruction * masks_ongoing[:-1,:,:]

            loss_reward_reconstruction = loss_reward_reconstruction.pow(2).mean(dim=-1)

        if return_predictions:
            return loss_reward_reconstruction, pred_rewards
        else:
            return loss_reward_reconstruction


    def compute_kl_loss(self, latent_means, latent_logvars, masks_ongoing):
        # -- KL divergence
        # returns, for each ELBO_t term, one KL (so H+1 kl's)
        if self.args.kl_to_gauss_prior:
            kl_divergences = - 0.5 * (1 + latent_logvars - latent_means.pow(2) - latent_logvars.exp())
            # apply the masks
            kl_divergences = kl_divergences[:-1,:,:] * masks_ongoing[:-1,:,:]
            kl_divergences = kl_divergences.sum(dim=-1)

        else:
            gauss_dim = latent_means.shape[-1]
            # add the gaussian prior
            all_means = torch.cat((torch.zeros(1, *latent_means.shape[1:]).to(device), latent_means))
            all_logvars = torch.cat((torch.zeros(1, *latent_logvars.shape[1:]).to(device), latent_logvars))
            # https://arxiv.org/pdf/1811.09975.pdf
            # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
                1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))
            
            # apply the masks
            kl_divergences = kl_divergences[:-1,:] * masks_ongoing[:-1,:,0]
                    
        return kl_divergences


    def compute_loss(
        self, 
        latent_means, latent_logvars, 
        states, actions, rewards, masks_ongoing
    ):
        """
        Computes the VAE loss for the given data.
        Batches everything together and therefore needs all trajectories to be of the same length.
        (Important because we need to separate ELBOs and decoding terms so can't collapse those dimensions)
        """

        # take one sample for each ELBO term
        if not self.args.disable_stochasticity_in_latent:
            latent_samples = self.encoder._sample_gaussian(latent_means, latent_logvars)
        else:
            latent_samples = torch.cat((latent_means, latent_logvars), dim=-1)
        
        if self.args.decode_state:
            # compute reconstruction loss for this trajectory 
            # (for each timestep that was encoded, decode the next timestep)
            # shape of loss: time_steps x num_trajectories
            loss_state_reconstruction = self.compute_state_reconstruction_loss(
                latent_samples=latent_samples,
                states=states, 
                actions=actions,
                masks_ongoing=masks_ongoing,
                return_predictions=False
            )
            # sum across all terms
            loss_state_reconstruction = loss_state_reconstruction.sum()

        else:
            loss_state_reconstruction = 0

        if self.args.decode_reward:
            # compute reconstruction loss for this trajectory 
            # (for each timestep that was encoded, decode the next timestep)
            # shape of loss: time_steps x num_trajectories
            loss_reward_reconstruction = self.compute_reward_reconstruction_loss(
                latent_samples=latent_samples,
                states=states, 
                actions=actions,
                rewards=rewards,
                masks_ongoing=masks_ongoing,
                return_predictions=False
            )

            # sum across all terms
            loss_reward_reconstruction = loss_reward_reconstruction.sum()
            
        else:
            loss_reward_reconstruction = 0
        
        if not self.args.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            # shape: time_steps+1 x num_trajectories
            loss_kl = self.compute_kl_loss(latent_means, latent_logvars, masks_ongoing)

            # avg/sum the elbos
            loss_kl = loss_kl.sum()

        else:
            loss_kl = 0
        
        return loss_reward_reconstruction, loss_state_reconstruction, loss_kl

        
    def compute_vae_loss(self):
        """ Returns the VAE loss """
        if not self.rollout_storage.ready_for_update():
            return 0
        
        if self.args.disable_decoder and self.args.disable_kl_term:
            print(f'self.args.disable_decoder: {self.args.disable_decoder}')
            print(f'self.args.disable_kl_term: {self.args.disable_kl_term}')
            return 0
        
        # get a batch
        batch_states_for_policy, batch_actions, batch_rewards, batch_masks_ongoing = \
            self.rollout_storage.get_batch(
                batch_size=self.args.vae_batch_size)
        # batch_states shape: (max_num_steps+1) x batch_size x state_dim

        # re-build the computational graph
        # pass through encoder (outputs will be: 
        # (max_num_steps+1) x batch_size x latent_dim -- includes the prior!)
        _, latent_means, latent_logvars, _ = self.encoder(
            curr_states=batch_states_for_policy[:-1,:,:],
            prev_actions=batch_actions[:-1,:,:], 
            prev_rewards=batch_rewards[:-1,:,:],
            rnn_prev_hidden_states=None, 
            return_prior=True, 
            sample=True, 
            detach_every=None
        )

        # compute loss
        loss_reward_reconstruction, loss_state_reconstruction, loss_kl = \
            self.compute_loss(
                latent_means=latent_means, 
                latent_logvars=latent_logvars, 
                states=batch_states_for_policy, 
                actions=batch_actions,
                rewards=batch_rewards,
                masks_ongoing=batch_masks_ongoing
            )     

        # VAE loss = KL loss + reward reconstruction + state transition reconstruction
        # take average (this is the expectation over p(M))
        loss_total = (self.args.vae_reward_reconstruction_loss_coeff * loss_reward_reconstruction +
            self.args.vae_state_reconstruction_loss_coeff * loss_state_reconstruction +
            self.args.vae_kl_loss_coeff * loss_kl
        ).mean()

        # make sure we can compute gradients
        if not self.args.disable_kl_term:
            assert loss_kl.requires_grad
        if self.args.decode_reward:
            assert loss_reward_reconstruction.requires_grad
        if self.args.decode_state:
            assert loss_state_reconstruction.requires_grad

        # overall loss
        elbo_loss = loss_total.mean()

        return elbo_loss, loss_reward_reconstruction, loss_state_reconstruction, loss_kl

    def _clip_and_step(self):
        """Helper method for gradient clipping and optimizer step"""
        # clip gradients
        if self.args.encoder_max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
        if self.args.decoder_max_grad_norm is not None:
            if self.args.decode_reward:
                nn.utils.clip_grad_norm_(self.reward_decoder.parameters(), self.args.decoder_max_grad_norm)
            if self.args.decode_state:
                nn.utils.clip_grad_norm_(self.state_decoder.parameters(), self.args.decoder_max_grad_norm)
        # update
        self.optimizer_vae.step()

    def update_parameters(self, elbo_loss):
        """
        if not args.policy_gradient_through_encoder, 
        performs complete update cycle
        """
        self.optimizer_vae.zero_grad()
        elbo_loss.backward()
        self._clip_and_step()

    def step_optimizer(self):
        """
        if args.policy_gradient_through_encoder, 
        only performs gradient clipping and optimizer step
        """
        self._clip_and_step()