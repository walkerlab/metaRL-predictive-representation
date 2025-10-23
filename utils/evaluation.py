import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from models.policy_net import ActorCriticRNN, SharedActorCriticRNN, ActorCriticMLP
from models.encoder import RNNEncoder
from models.decoder import StateTransitionDecoder, RewardDecoder

from utils import helpers as utl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_network(
    network_type,  # 'a2crnn', 'a2cmlp', 'rnn_encoder', 'reward_decoder'
    path_to_trained_network_state_dict,
    args,
    device=device
):

    if network_type == 'a2crnn':
        if not args.shared_rnn:
            network = ActorCriticRNN(
                args=args,
                layers_before_rnn=args.layers_before_rnn,
                rnn_hidden_dim=args.rnn_hidden_dim,
                layers_after_rnn=args.layers_after_rnn,
                rnn_cell_type=args.rnn_cell_type,
                activation_function=args.policy_net_activation_function,
                initialization_method=args.policy_net_initialization_method,
                state_dim=args.input_state_dim_for_policy,
                state_embed_dim=args.state_embed_dim,
                action_dim=args.action_dim,
                action_embed_dim=args.action_embed_dim,
                action_space_type=args.action_space_type,
                reward_dim=args.reward_dim,
                reward_embed_dim=args.reward_embed_dim
            )
        else:
            network = SharedActorCriticRNN(
                args=args,
                layers_before_rnn=args.layers_before_rnn,
                rnn_hidden_dim=args.rnn_hidden_dim,
                layers_after_rnn=args.layers_after_rnn,
                rnn_cell_type=args.rnn_cell_type,
                activation_function=args.policy_net_activation_function,
                initialization_method=args.policy_net_initialization_method,
                state_dim=args.input_state_dim_for_policy,
                state_embed_dim=args.state_embed_dim,
                action_dim=args.action_dim,
                action_embed_dim=args.action_embed_dim,
                action_space_type=args.action_space_type,
                reward_dim=args.reward_dim,
                reward_embed_dim=args.reward_embed_dim
            )
    elif network_type == 'a2cmlp':
        network = ActorCriticMLP(
            args=args,
            state_dim=args.input_state_dim_for_policy,
            latent_dim=args.latent_dim,
            hidden_layers=args.policy_layers,
            activation_function=args.policy_net_activation_function,
            initialization_method=args.policy_net_initialization_method,
            action_dim=args.action_dim,
            action_space_type=args.action_space_type
        )
    elif network_type == 'rnn_encoder':
        network = RNNEncoder(
            args=args,
            latent_dim=args.latent_dim,
            layers_before_rnn=args.encoder_layers_before_rnn,
            rnn_hidden_dim=args.encoder_rnn_hidden_dim,
            layers_after_rnn=args.encoder_layers_after_rnn,
            rnn_cell_type=args.encoder_rnn_cell_type,
            activation_function=args.encoder_activation_function,
            initialization_method=args.encoder_initialization_method,
            state_dim=args.input_state_dim_for_policy,
            state_embed_dim=args.state_embed_dim,
            action_dim=args.action_dim,
            action_embed_dim=args.action_embed_dim,
            reward_dim=args.reward_dim,
            reward_embed_dim=args.reward_embed_dim
        )
    elif network_type == 'reward_decoder':
        network = RewardDecoder(
            args=args,
            layers=args.reward_decoder_layers,
            pred_type=args.reward_pred_type,
            latent_dim=args.latent_dim,
            state_dim=args.input_state_dim_for_policy,
            action_dim=args.action_dim,
            input_action=args.input_action,
            input_prev_state=args.input_prev_state
        )
    else:
        raise ValueError(f'invalid networ_type: {network_type}')
        
    network.load_state_dict(torch.load(
        path_to_trained_network_state_dict,
        map_location=device))
    network.to(device)

    return network


#########################################################
# TEST ROLLOUT
#########################################################
def rollout_one_episode_rl2(
    env,
    policy_network,
    args,
    deterministic=False
):
    '''
    rollout the trained policy_network to 
    play one episode in the env
    '''
    infos = []
    states = []
    actions = []
    rewards = []
    state_values = []
    action_log_probs = []
    all_action_logits = []
    entropies = []
    actor_hidden_states = []
    critic_hidden_states = []

    # reset the env to get an initial state
    curr_state_dict, info = env.reset()
    infos.append(info)
    # input_state_for_policy: already accounted for time
    curr_state = utl.get_states_from_state_dicts(
        curr_state_dict, args, args.time_as_state
    )
    curr_state = torch.from_numpy(curr_state).float().\
        reshape((1, 1, args.input_state_dim_for_policy)).to(device)
    prev_action = torch.zeros(1, 1, args.action_dim).to(device)
    prev_reward = torch.zeros(1, 1, 1).to(device)
    # initialize ActorCriticRNN hidden states
    if not args.shared_rnn:
        actor_prev_hidden_state = torch.zeros(1, 1, args.rnn_hidden_dim).to(device)
        critic_prev_hidden_state = torch.zeros(1, 1, args.rnn_hidden_dim).to(device)
    elif args.shared_rnn:
        rnn_prev_hidden_state = torch.zeros(1, 1, args.rnn_hidden_dim).to(device)

    states.append(curr_state.squeeze().detach().cpu().numpy())

    # rollout
    done = False
    while not done:
        # select an action A_{t} with inputs
        # S_{t}, A_{t-1}, R_{t-1} and hidden states
        with torch.no_grad():
            if not args.shared_rnn:
                # act
                action_categorical, action_log_prob, entropy, state_value, \
                    actor_hidden_state, critic_hidden_state = \
                        policy_network.act(
                            curr_states=curr_state,
                            prev_actions=prev_action,
                            prev_rewards=prev_reward,
                            actor_prev_hidden_states=actor_prev_hidden_state,
                            critic_prev_hidden_states=critic_prev_hidden_state,
                            return_prior=False, 
                            deterministic=deterministic
                        )
                # get action_prob
                all_action_logit, _, _, _ = policy_network(
                    curr_states=curr_state, 
                    prev_actions=prev_action, 
                    prev_rewards=prev_reward,
                    actor_prev_hidden_states=actor_prev_hidden_state, 
                    critic_prev_hidden_states=critic_prev_hidden_state,
                    return_prior=False
                )
            elif args.shared_rnn:
                # act
                action_categorical, action_log_prob, entropy, state_value, \
                    rnn_hidden_state = \
                        policy_network.act(
                            curr_states=curr_state,
                            prev_actions=prev_action,
                            prev_rewards=prev_reward,
                            rnn_prev_hidden_states=rnn_prev_hidden_state,
                            return_prior=False, 
                            deterministic=deterministic
                        )
                # get action_prob
                all_action_logit, _, _, = policy_network(
                    curr_states=curr_state, 
                    prev_actions=prev_action, 
                    prev_rewards=prev_reward,
                    rnn_prev_hidden_states=rnn_prev_hidden_state,
                    return_prior=False
                )

        # perform the action A_{t} in the environment 
        # to get S_{t+1} and R_{t+1}
        next_state_dict, reward, terminated, truncated, info = env.step(
            action_categorical.squeeze().cpu().numpy()
        )
        next_state = utl.get_states_from_state_dicts(
            next_state_dict, args, args.time_as_state
        )
        next_state = torch.from_numpy(next_state).float()\
            .reshape(1, 1, args.input_state_dim_for_policy)
        if policy_network.action_space_type == 'Discrete':    
            action = F.one_hot(action_categorical, num_classes=args.action_dim).\
                float().reshape((1, 1, args.action_dim))
        elif policy_network.action_space_type == 'Box':
            action = action_categorical.reshape((1, 1, args.action_dim))
        reward = torch.from_numpy(np.array(reward)).float().reshape(1, 1, 1)

        # update for next step
        curr_state = next_state.to(device)
        prev_action = action.to(device)
        prev_reward = reward.to(device)
        if not args.shared_rnn:
            actor_prev_hidden_state = actor_hidden_state.to(device)
            critic_prev_hidden_state = critic_hidden_state.to(device)
        elif args.shared_rnn:
            rnn_prev_hidden_state = rnn_hidden_state.to(device)

        # update if the environment is done
        ## if episodic task: use the episode_terminated flag from infos
        if args.env_name.split('-')[0] in[
                'StatTiger', 'MarkovTiger'
            ]:
            done = info['episode_terminated']
        else:
            done = terminated or truncated
        
        infos.append(info)
        states.append(curr_state.squeeze().detach().cpu().numpy())
        actions.append(action_categorical.squeeze().detach().cpu().numpy())
        action_log_probs.append(action_log_prob.squeeze().detach().cpu().numpy())
        if policy_network.action_space_type == 'Discrete':
            all_action_logits.append(all_action_logit.squeeze().detach().cpu().numpy())
        elif policy_network.action_space_type == 'Box':
            all_action_logits.append(torch.cat(all_action_logit, dim=0).squeeze().detach().cpu().numpy())
        entropies.append(entropy.squeeze().detach().cpu().numpy())
        rewards.append(reward.squeeze().detach().cpu().numpy())
        state_values.append(state_value.squeeze().detach().cpu().numpy())
        if not args.shared_rnn:
            actor_hidden_states.append(actor_hidden_state.squeeze().detach().cpu().numpy())
            critic_hidden_states.append(critic_hidden_state.squeeze().detach().cpu().numpy())
        elif args.shared_rnn:
            # if shared_rnn: use actor_hidden_states to store rnn_hidden_states
            # thus critic_hidden_states will just be all zeros
            actor_hidden_states.append(rnn_hidden_state.squeeze().detach().cpu().numpy())

    env.close()

    states = np.array(states)
    actions = np.array(actions)
    action_log_probs = np.array(action_log_probs)
    all_action_logits = np.array(all_action_logits)
    entropies = np.array(entropies)
    rewards = np.array(rewards)
    state_values = np.array(state_values)
    actor_hidden_states = np.array(actor_hidden_states)
    critic_hidden_states = np.array(critic_hidden_states)

    return infos, states, actions, rewards, \
        action_log_probs, all_action_logits, entropies, state_values, \
        actor_hidden_states, critic_hidden_states


def rollout_one_episode_mpc(
    env,
    encoder,
    policy_network,
    args,
    deterministic=False
):
    '''
    rollout the trained encoder & policy_network to 
    play one episode in the env
    '''
    infos = []
    states = []
    latent_means = []
    latent_logvars = []
    actions = []
    rewards = []
    state_values = []
    action_log_probs = []
    all_action_logits = []
    entropies = []
    encoder_hidden_states = []
    
    # reset the env to get an initial state
    curr_state_dict, info = env.reset()
    infos.append(info)
    # input_state_for_policy: already accounted for time
    curr_state = utl.get_states_from_state_dicts(
        curr_state_dict, args, args.time_as_state
    )
    curr_state = torch.from_numpy(curr_state).float().\
        reshape((1, 1, args.input_state_dim_for_policy)).to(device)
    prev_action = torch.zeros(1, 1, args.action_dim).to(device)
    prev_reward = torch.zeros(1, 1, 1).to(device)
    # initialize RNNEncoder hidden states
    encoder_rnn_prev_hidden_state = torch.zeros(1, 1, args.encoder_rnn_hidden_dim).to(device)

    states.append(curr_state.squeeze().detach().cpu().numpy())

    # rollout
    done = False
    while not done:
        # select an action A_{t} with inputs
        # S_{t}, A_{t-1}, R_{t-1} and hidden states
        with torch.no_grad():
            curr_latent_sample, curr_latent_mean, curr_latent_logvar, encoder_rnn_hidden_state = \
                encoder(
                    curr_states=curr_state, 
                    prev_actions=prev_action, 
                    prev_rewards=prev_reward,
                    rnn_prev_hidden_states=encoder_rnn_prev_hidden_state, 
                    return_prior=False, 
                    sample=True, 
                    detach_every=None
                )
            # cast latents to tensor (step, batch, feature)
            curr_latent_mean = curr_latent_mean.reshape(1, 1, args.latent_dim).to(device)
            curr_latent_logvar = curr_latent_logvar.reshape(1, 1, args.latent_dim).to(device)
            # act
            action_categorical, action_log_prob, entropy, state_value = \
                policy_network.act(
                    curr_states=curr_state, 
                    curr_latent_means=curr_latent_mean, 
                    curr_latent_logvars=curr_latent_logvar, 
                    deterministic=deterministic
                )
            # get action_prob
            all_action_logit, _ = policy_network(
                curr_states=curr_state, 
                curr_latent_means=curr_latent_mean, 
                curr_latent_logvars=curr_latent_logvar
            )

        # perform the action A_{t} in the environment 
        # to get S_{t+1} and R_{t+1}
        next_state_dict, reward, terminated, truncated, info = env.step(
            action_categorical.squeeze().cpu().numpy()
        )
        next_state = utl.get_states_from_state_dicts(
            next_state_dict, args, args.time_as_state
        )
        next_state = torch.from_numpy(next_state).float()\
            .reshape(1, 1, args.input_state_dim_for_policy)
        
        if policy_network.action_space_type == 'Discrete':
            action = F.one_hot(action_categorical, num_classes=args.action_dim).\
                float().reshape((1, 1, args.action_dim))
        elif policy_network.action_space_type == 'Box':
            action = action_categorical.reshape((1, 1, args.action_dim))
        reward = torch.from_numpy(np.array(reward)).float().reshape(1, 1, 1)

        # update for next step
        curr_state = next_state.to(device)
        prev_action = action.to(device)
        prev_reward = reward.to(device)
        encoder_rnn_prev_hidden_state = encoder_rnn_hidden_state.to(device)

        # update if the environment is done
        ## if episodic task: use the episode_terminated flag from infos
        if args.env_name.split('-')[0] in[
                'StatTiger', 'MarkovTiger'
            ]:
            # print(f'action: {action_categorical}, {info}')
            done = info['episode_terminated']
        else:
            done = terminated or truncated
        
        infos.append(info)
        states.append(curr_state.squeeze().detach().cpu().numpy())
        latent_means.append(curr_latent_mean.squeeze().detach().cpu().numpy())
        latent_logvars.append(curr_latent_logvar.squeeze().detach().cpu().numpy())
        actions.append(action_categorical.squeeze().detach().cpu().numpy())
        action_log_probs.append(action_log_prob.squeeze().detach().cpu().numpy())
        if policy_network.action_space_type == 'Discrete':
            all_action_logits.append(all_action_logit.squeeze().detach().cpu().numpy())
        elif policy_network.action_space_type == 'Box':
            all_action_logits.append(torch.cat(all_action_logit, dim=0).squeeze().detach().cpu().numpy())
        entropies.append(entropy.squeeze().detach().cpu().numpy())
        rewards.append(reward.squeeze().detach().cpu().numpy())
        state_values.append(state_value.squeeze().detach().cpu().numpy())
        encoder_hidden_states.append(encoder_rnn_hidden_state.squeeze().detach().cpu().numpy())

    env.close()

    states = np.array(states)
    latent_means = np.array(latent_means)
    latent_logvars = np.array(latent_logvars)
    actions = np.array(actions)
    action_log_probs = np.array(action_log_probs)
    all_action_logits = np.array(all_action_logits)
    entropies = np.array(entropies)
    rewards = np.array(rewards)
    state_values = np.array(state_values)
    encoder_hidden_states = np.array(encoder_hidden_states)

    return infos, states, actions, rewards, \
        action_log_probs, all_action_logits, entropies, state_values, \
        latent_means, latent_logvars, encoder_hidden_states


def get_pred_reward_probs_mpc(
    reward_decoder,
    latent_means,  # np array
    args
):

    latent_means_tensor = torch.from_numpy(latent_means).float().reshape(
        latent_means.shape[0], 1, -1).to(device)
    next_states_tensor = torch.zeros(latent_means.shape[0], 1, args.input_state_dim_for_policy).to(device)
    prev_states_tensor = torch.zeros(latent_means.shape[0], 1, args.input_state_dim_for_policy).to(device)

    # for action_0
    actions_tensor = F.one_hot(torch.zeros(latent_means.shape[0], dtype=torch.int64), num_classes=args.action_dim).\
        float().reshape(latent_means.shape[0], 1, args.action_dim).to(device)
    pred_reward_logits_0 = reward_decoder(
        latent_states=latent_means_tensor,
        next_states=next_states_tensor,
        actions=actions_tensor,
        prev_states=prev_states_tensor
    )

    # for action_1
    actions_tensor = F.one_hot(torch.ones(latent_means.shape[0], dtype=torch.int64), num_classes=args.action_dim).\
        float().reshape(latent_means.shape[0], 1, args.action_dim).to(device)
    pred_reward_logits_1 = reward_decoder(
        latent_states=latent_means_tensor,
        next_states=next_states_tensor,
        actions=actions_tensor,
        prev_states=prev_states_tensor
    )

    pred_reward_probs_0 = torch.sigmoid(pred_reward_logits_0).squeeze().detach().cpu().numpy()
    pred_reward_probs_1 = torch.sigmoid(pred_reward_logits_1).squeeze().detach().cpu().numpy()

    pred_reward_probs = np.vstack((pred_reward_probs_0, pred_reward_probs_1))

    return pred_reward_probs


#########################################################
# PERFORMANCE
#########################################################
def get_empirical_returns(
    env_name,
    args,
    encoder,  # None if rl2
    policy_network,
    num_envs=10
):
    empirical_returns = []
    episode_lengths = []
    for test_env_id in range(num_envs):
        test_env = gym.make(
            f'environments.bandit:{env_name}'
        )
        if args.exp_label == 'mpc':
            info, _, actions, rewards, \
            _, _, _, _, \
            _, _, _ = rollout_one_episode_mpc(
                test_env,
                encoder,
                policy_network,
                args,
                deterministic=False
            )
        elif args.exp_label in ['rl2', 'noisy_rl2']:
            info, _, actions, rewards, \
            _, _, _, _, \
            _, _ = rollout_one_episode_rl2(
                test_env,
                policy_network,
                args,
                deterministic=False
            )
        else:
            raise ValueError(f'incompatible model type: {args.exp_label}')

        empirical_returns.append(np.sum(rewards))
        episode_lengths.append(len(actions))
    
    empirical_returns = np.array(empirical_returns)
    episode_lengths = np.array(episode_lengths)

    if env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        return np.average(empirical_returns), np.std(empirical_returns), np.average(episode_lengths), np.std(episode_lengths)
    else:
        return np.average(empirical_returns), np.std(empirical_returns)
