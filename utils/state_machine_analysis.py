import random
import os

import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
import gymnasium as gym

from utils.evaluation import rollout_one_episode_mpc, rollout_one_episode_rl2
from utils.helpers import StateSpaceMapper, StateMapperTrainer
from utils.bayes_optimal_agents import BernoulliGittinsAgent, rollout_one_episode_bernoulli_gittins_agent, \
    rollout_one_episode_bernoulli_gittins_agent_given_p_bandits, get_expected_regret_one_run_bernoulli_bandit, \
    cumulative_expected_regret_bernoulli_bandit_Gittins_agent
from utils.bayes_optimal_agents import OracleBanditSolver, rollout_one_episode_oracle_bandit_solver_given_r_bandits
from utils.bayes_optimal_agents import rollout_one_episode_dis_markovian_bandit_solver_given_env
from utils.bayes_optimal_agents import rollout_one_episode_tiger_solver_given_env, rollout_one_episode_latent_goal_cart_solver_given_env


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#########################################################
# Bernoulli bandit
#########################################################
def cumulative_expected_regret_bernoulli_bandit_metaRL_agent(
    env_name,
    args,
    encoder,  # None if rl2
    policy_network,
    num_test_envs=10,
    num_runs_per_test_env=5,
    total_trials=20
):
    '''
    calculate cumulative expected regret 
    for a metaRL_agent (MPC or rl2)
    in stationary multi-armed bernoulli bandit env
    '''

    regret_per_timesteps = []
    for test_env_id in range(num_test_envs):
        test_env = gym.make(
            f'environments.bandit:{env_name}',
            total_trials=total_trials
        )

        regret_per_timestep_per_test_env = []
        for run_id in range(num_runs_per_test_env):
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
            elif args.exp_label == 'rl2':
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

            regret_per_timestep = get_expected_regret_one_run_bernoulli_bandit(
                test_env, actions)
            regret_per_timestep_per_test_env.append(regret_per_timestep)

        regret_per_timestep_per_test_env = np.array(regret_per_timestep_per_test_env)
        regret_per_timesteps.append(np.average(regret_per_timestep_per_test_env, axis=0))
        
    regret_per_timesteps = np.array(regret_per_timesteps)
    cumulative_expected_regret_mean = np.average(regret_per_timesteps, axis=0)
    cumulative_expected_regret_std = np.std(regret_per_timesteps, axis=0)
    
    return cumulative_expected_regret_mean, cumulative_expected_regret_std


def plot_regret_comparison_bernoulli_bandit(
    env_name,
    encoder, # None if rl2
    policy_network,
    args,
    biased_beta_prior,
    total_trials=40,
    num_test_envs=150,
    num_runs_per_test_env=5
):

    '''
    plot performance in stationary bernoulli bandit env
    by comparison of regret with gitiins index agent
    '''

    if not biased_beta_prior:
        # get expected regret for gittins agents
        cum_expected_regret_mean_gittins, cum_expected_regret_std_gittins = \
        cumulative_expected_regret_bernoulli_bandit_Gittins_agent(
            env_name=env_name,
            biased_beta_prior=biased_beta_prior,
            gittins_look_ahead_window=total_trials,
            gittins_discounting=0.95,
            num_test_envs=num_test_envs,
            num_runs_per_test_env=num_runs_per_test_env,
            total_trials=total_trials
        )
    else:
        # get expected regret for biased gittins agents
        cum_expected_regret_mean_gittins, cum_expected_regret_std_gittins = \
        cumulative_expected_regret_bernoulli_bandit_Gittins_agent(
            env_name=env_name,
            biased_beta_prior=biased_beta_prior,
            gittins_look_ahead_window=total_trials,
            gittins_discounting=0.95,
            num_test_envs=num_test_envs,
            num_runs_per_test_env=num_runs_per_test_env,
            total_trials=total_trials
        )


    # get expected regret for metaRL agent
    cum_expected_regret_mean_metaRL, cum_expected_regret_std_metaRL = \
    cumulative_expected_regret_bernoulli_bandit_metaRL_agent(
        env_name=env_name,
        args=args,
        encoder=encoder,
        policy_network=policy_network,
        num_test_envs=num_test_envs,
        num_runs_per_test_env=num_runs_per_test_env,
        total_trials=total_trials
    )


    # plot: regret comparison with gittins agent
    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(8, 6), dpi=300
    )
    ax.set_title(f'Regret comparison (sem): {args.exp_label} in {env_name}')
    ax.errorbar(
        np.arange(len(cum_expected_regret_mean_gittins)),
        cum_expected_regret_mean_gittins,
        yerr=cum_expected_regret_std_gittins// np.sqrt(num_test_envs*num_runs_per_test_env),
        label='gittins'
    )
    ax.errorbar(
        np.arange(len(cum_expected_regret_mean_metaRL)),
        cum_expected_regret_mean_metaRL,
        yerr=cum_expected_regret_std_metaRL/ np.sqrt(num_test_envs),
        label=f'{args.exp_label}'
    )

    plt.errorbar(
        np.arange(len(cum_expected_regret_mean_metaRL)),
        cum_expected_regret_mean_metaRL - cum_expected_regret_mean_gittins,
        yerr=cum_expected_regret_std_metaRL/ np.sqrt(num_test_envs),
    label='diff'
    )
    ax.set_xlabel('trial')
    ax.set_ylabel('expected regret')
    ax.set_ylim(0, 3)
    ax.legend()

    fig.tight_layout()

    return fig



###############################################################################################
# Bernoulli bandit
###############################################################################################
def get_bayes_optimal_states_bernoulli_bandit_given_ref_one_run(
    actions,
    rewards,
    biased_beta_prior,
    num_bandits=2,
    prior_alphas=[1, 1],
    prior_betas=[1, 1]
):
    '''
    for a given run (trajectory of actions and rewards)
    in Bernoulli bandits,
    convert to bayes optimal states:
    per timestep, (ct_unrewarded, ct_rewarded) * num_bandits,
    240415 update: always starting from t_0
    '''

    bayes_optimal_states = np.zeros(
        (len(actions)+1, num_bandits*2),
        dtype=np.int64
    )
    # per timestep, (ct_unrewarded, ct_rewarded) * num_bandits
    # 1 extra timestep for t_0
    
    for timestep in range(len(actions)):
        if timestep == 0:
            if not biased_beta_prior:
                pass
            else:
                bayes_optimal_states[timestep][1] += 1  # beta_prior_a0=[2,1]
                bayes_optimal_states[timestep][2] += 1  # beta_prior_a1=[1,2]
        else:
            curr_ct = np.zeros(num_bandits*2, dtype=np.int64)
            action = actions[timestep-1]
            reward = rewards[timestep-1]
            curr_ct[int(action*num_bandits+reward)] += 1

            bayes_optimal_states[timestep] = bayes_optimal_states[timestep-1] + curr_ct
        
    return bayes_optimal_states


def get_bayes_optimal_actions_given_states_bernoulli_bandit(
    bayes_optimal_states,
    total_trials
):

    bayes_gittins_indices = []
    bayes_actions = []

    gittins_agent = BernoulliGittinsAgent(
        lookahead_window=total_trials, 
        discount_factor=0.95,
        biased_beta_prior=False  # states already takes care of beta_prior
    )

    for step in range(len(bayes_optimal_states)):
        bayes_state = bayes_optimal_states[step]
        gittins_index = []
        for arm_idx in [0, 1]:
            unrewarded_ct = bayes_state[arm_idx*2]
            rewarded_ct = bayes_state[arm_idx*2+1]

            exploring = unrewarded_ct + rewarded_ct
            positive_feedback = rewarded_ct
            
            negative_feedback = exploring - positive_feedback + 1
            positive_feedback = positive_feedback + 1

            gittins_index.append(gittins_agent.calc_gittins_index(positive_feedback, negative_feedback))

        # bayes_action
        bayes_action = gittins_index.index(max(gittins_index))
        
        bayes_gittins_indices.append(gittins_index)
        bayes_actions.append(bayes_action)

    bayes_gittins_indices = np.array(bayes_gittins_indices)
    bayes_actions = np.array(bayes_actions)        
    
    return bayes_gittins_indices, bayes_actions


def get_empirical_returns_given_actions_bernoulli_bandit_one_run(
    actions,
    p_bandits
):
    empirical_rewards = []
    for action in actions:
        if random.uniform(0, 1) < p_bandits[action]:
            empirical_reward = 1
        else:
            empirical_reward = 0
        empirical_rewards.append(empirical_reward)
    
    return np.array(empirical_rewards)


def get_expected_returns_given_actions_bernoulli_bandit_one_run(
    actions,
    p_bandits
):
    expected_rewards_per_run = p_bandits[actions]
    expected_return_per_run = np.cumsum(expected_rewards_per_run)

    return expected_return_per_run, expected_rewards_per_run


###############################################################################################
# Oracle bandit
###############################################################################################
def get_bayes_optimal_states_oracle_bandit_given_ref_one_run(
    actions,
    rewards,
    num_bandits=11
):
    '''
    for a given run (trajectory of actions and rewards) in oracle bandits,
    convert to bayes optimal states:
    per timestep, (ct_a_oracle, reward_a_oracle),
    always starting from t_0
    '''

    bayes_optimal_states = np.zeros(
        (len(actions)+1, num_bandits-1)
    )
    # 1 extra timestep for t_0; (timestep+1, num_bandits-1)
    oracle_chosen = False
    for timestep in range(len(actions)):
        if timestep == 0:
            bayes_optimal_states[timestep] = np.full(
                (num_bandits-1, ),
                1.0/ (num_bandits-1)
            )
        else:
            action = actions[timestep-1]
            reward = rewards[timestep-1]
            if oracle_chosen:
                bayes_optimal_states[timestep] = bayes_optimal_states[timestep-1]
            else:
                if action == num_bandits-1:  # if choose the oracle arm
                    state = np.zeros((num_bandits-1, ))
                    # if reward == 0.7:
                    #     print(f'int(10*(reward-0.1)) {int(10*(reward-0.1))}')
                    state[int(round(reward*10-1))] = 1  # update the state to the chosen arm
                    bayes_optimal_states[timestep] = state
                    oracle_chosen = True  # mark the oracle arm as chosen
                else:
                    if reward == 5:
                        state = np.zeros((num_bandits-1, ))
                        state[action] = 1
                        bayes_optimal_states[timestep] = state
                    elif reward == 1:
                        prev_nontarget = np.where(bayes_optimal_states[timestep-1]==0)[0].tolist()
                        if action not in prev_nontarget:
                            total_nontarget = prev_nontarget + [action]
                        else:
                            total_nontarget = prev_nontarget
                        # print(total_nontarget)
                        state = np.full(
                            (num_bandits-1, ),
                            1.0/ (num_bandits-1-len(total_nontarget))
                        )
                        state[total_nontarget] = 0
                        bayes_optimal_states[timestep] = state
                    else:
                        raise ValueError("Invalid reward value")
        
    return bayes_optimal_states


def get_bayes_optimal_actions_given_states_oracle_bandit(
    bayes_optimal_states,
    num_bandits=11
):
    bayes_actions = []

    oracle_solver = OracleBanditSolver(num_bandits=num_bandits)

    for step in range(len(bayes_optimal_states)):
        bayes_state = bayes_optimal_states[step]
        
        # bayes_action
        bayes_action = oracle_solver.get_action_from_state(bayes_state)
        bayes_actions.append(bayes_action)

    bayes_actions = np.array(bayes_actions)        
    
    return bayes_actions


###############################################################################################
# DiscreteMarkovian bandit
###############################################################################################
def get_bayes_optimal_states_dis_markovian_bandit_given_ref_one_run(
    actions,
    rewards,
    optimal_solver
):
    '''
    for a given run (trajectory of actions and rewards) in discrete markovian bandits,
    convert to bayes optimal states:
    per timestep, (b_0(s=0), b_1(s=0)),
    always starting from t_0
    '''
    bayes_optimal_states = np.zeros(
        (len(actions)+1, 2)
    )  # 1 extra timestep for t_0; (timestep+1, 2)
    # per timestep, (b_0(s=0), b_1(s=0))

    optimal_solver.reset()
    for timestep in range(len(actions)):
        if timestep == 0:
            bayes_optimal_states[timestep] = np.array(
                [optimal_solver.current_belief[0, 0], optimal_solver.current_belief[1, 0]]
            )
        else:
            action = actions[timestep-1]
            reward = rewards[timestep-1]
            optimal_solver.update_belief(action, reward)
            bayes_optimal_states[timestep] = np.array(
                [optimal_solver.current_belief[0, 0], optimal_solver.current_belief[1, 0]]
            )
    
    return bayes_optimal_states


def get_bayes_optimal_actions_given_states_dis_markovian_bandit(
    bayes_optimal_states,
    optimal_solver
):
    bayes_actions = []

    for step in range(len(bayes_optimal_states)):
        bayes_state = bayes_optimal_states[step]

        # bayes_action
        bayes_action = optimal_solver.get_action_from_belief_state(bayes_state)
        bayes_actions.append(bayes_action)

    bayes_actions = np.array(bayes_actions)        
    
    return bayes_actions
    

###############################################################################################
# Tiger
###############################################################################################
def get_bayes_optimal_states_tiger_given_ref_one_run(
    actions,
    rewards,
    observations,
    optimal_solver
):
    '''
    for a given run (trajectory of actions and rewards) in Tiger,
    convert to bayes optimal states:
    per timestep, b(tiger on left),
    always starting from t_0
    '''
    bayes_optimal_states = np.zeros(
        (len(actions)+1, 1)
    )  # 1 extra timestep for t_0; (timestep+1, 2)
    # per timestep, b(tiger on left)

    optimal_solver.reset()
    for timestep in range(len(actions)):
        if timestep == 0:
            bayes_optimal_states[timestep] = np.array(
                [optimal_solver.current_belief]
            )
        else:
            action = actions[timestep-1]
            reward = rewards[timestep-1]
            observation = observations[timestep]
            optimal_solver.update_current_belief(action, observation)
            bayes_optimal_states[timestep] = np.array(
                [optimal_solver.current_belief]
            )
    
    return bayes_optimal_states


def get_bayes_optimal_actions_given_states_tiger(
    bayes_optimal_states,
    optimal_solver
):
    bayes_actions = []

    for step in range(len(bayes_optimal_states)):
        bayes_state = bayes_optimal_states[step]

        # bayes_action
        bayes_action = optimal_solver.get_optimal_action(bayes_state)
        bayes_actions.append(bayes_action)

    bayes_actions = np.array(bayes_actions)        
    
    return bayes_actions


###############################################################################################
# Latent goal cart
###############################################################################################
def get_bayes_optimal_actions_given_states_latent_goal_cart(
    bayes_optimal_states,
    ref_observations,
    optimal_solver
):
    bayes_actions = []

    for step in range(len(bayes_optimal_states)):
        bayes_state = bayes_optimal_states[step]
        ref_observation = ref_observations[step]

        # bayes_action
        bayes_action = optimal_solver.solver.get_action(bayes_state, ref_observation)
        bayes_actions.append(bayes_action)

    bayes_actions = np.array(bayes_actions)        
    
    return bayes_actions


###############################################################################################
# state space mapping
###############################################################################################
def gen_ref_trajectories_by_bayes(
    env_name,
    total_trials,
    biased_beta_prior,
    num_envs=3000,
    num_bandits=2,
    dis_markovian_bandit_solver=None,  # None if not discrete markovian bandit
    tiger_solver=None, # None if not tiger
    latent_goal_cart_solver=None,  # None if not latent goal cart
    args=None  # None if not tiger
):
    '''
    generate reference trajectories using the Bayes agent
    '''
    ref_trajectories = {}
    if 'StatBernoulliBandit' in env_name:
        ref_trajectories['p_bandits'] = []
    elif 'OracleBanditDeterministic' in env_name:
        ref_trajectories['target_arm'] = []
        ref_trajectories['r_bandits'] = []
    elif env_name.split('-')[0] in ['DisSymmetricStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
        ref_trajectories['reward_probs_t'] = []
    elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        ref_trajectories['bayes_observations'] = []
        ref_trajectories['rewards_t'] = []
    elif env_name.split('-')[0] in ['LatentGoalCart']:
        ref_trajectories['bayes_observations'] = []
        ref_trajectories['goal_positions_t'] = []
    else:
        raise ValueError(f'unsupported env: {env_name}')
    ref_trajectories['bayes_states'] = []
    ref_trajectories['bayes_actions'] = []
    ref_trajectories['bayes_rewards'] = []

    for env_id in range(num_envs):
        env = gym.make(
            f'{env_name}',
            total_trials=total_trials
        )

        if 'StatBernoulliBandit' in env_name:
            ref_trajectories['p_bandits'].append(env.unwrapped.p_bandits)
        elif 'OracleBanditDeterministic' in env_name:
            ref_trajectories['target_arm'].append(env.unwrapped.target_arm)
            ref_trajectories['r_bandits'].append(env.unwrapped.r_bandits)

        # get (reference) rollout with Bayes agent
        if 'StatBernoulliBandit' in env_name:
            bayes_states_run, bayes_actions_run, bayes_rewards_run, bayes_gittins_indices_run = \
            rollout_one_episode_bernoulli_gittins_agent_given_p_bandits(
                env.unwrapped.p_bandits,
                total_trials+1,
                biased_beta_prior=biased_beta_prior,
                arms=num_bandits
            )
            # 1 extra trial to roll out one final step
            # (ct_unreward, ct_reward) * bandit
        elif 'OracleBanditDeterministic' in env_name:
            bayes_states_run, bayes_actions_run, bayes_rewards_run = \
            rollout_one_episode_oracle_bandit_solver_given_r_bandits(
                env.unwrapped.r_bandits,
                total_trials+1,
                num_bandits=num_bandits
            )
            # 1 extra trial to roll out one final step
            # (n_a_11, r_a_11)
        
        elif env_name.split('-')[0] in ['DisSymmetricStickyMarkovBandit2State2Arm', 
                                        'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                        'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
            bayes_states_run, bayes_actions_run, bayes_rewards_run, reward_probs_t_run = \
            rollout_one_episode_dis_markovian_bandit_solver_given_env(
                dis_markovian_bandit_solver=dis_markovian_bandit_solver,
                env=env,
                total_trials=total_trials+1
            )
            # 1 extra trial to roll out one final step
            # (b_0(s=0), b_1(s=0))
            ref_trajectories['reward_probs_t'].append(reward_probs_t_run)
        
        elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
            bayes_states_run, bayes_actions_run, bayes_rewards_run, bayes_observations_run, rewards_t_run = \
            rollout_one_episode_tiger_solver_given_env(
                tiger_solver=tiger_solver,
                env=env,
                args=args,
                total_trials=total_trials+1
            )
            # 1 extra trial to roll out one final step
            # (b_tiger_on_left)
            ref_trajectories['bayes_observations'].append(bayes_observations_run)
            ref_trajectories['rewards_t'].append(rewards_t_run)

        elif env_name.split('-')[0] in ['LatentGoalCart']:
            bayes_states_run, bayes_actions_run, bayes_rewards_run, bayes_observations_run, goal_positions_t_run = \
            rollout_one_episode_latent_goal_cart_solver_given_env(
                latent_goal_cart_solver=latent_goal_cart_solver,
                env=env,
                total_trials=total_trials+1
            )
            # 1 extra trial to roll out one final step
            # (b_tiger_on_left)
            ref_trajectories['bayes_observations'].append(bayes_observations_run)
            ref_trajectories['goal_positions_t'].append(goal_positions_t_run)
        
        ref_trajectories['bayes_states'].append(bayes_states_run)
        ref_trajectories['bayes_actions'].append(bayes_actions_run)
        ref_trajectories['bayes_rewards'].append(bayes_rewards_run)
        

    # make np array
    if env_name.split('-')[0] not in ['StatTiger', 'MarkovTiger']:
        for data_name, data in ref_trajectories.items():
            ref_trajectories[data_name] = np.array(data)  # (n_runs, n_trials+1, n_feature)

    return ref_trajectories


def gen_metaRL_trajectories_given_ref(
    encoder,
    policy_network,
    args,
    ref_actions,
    ref_rewards,
    ref_observations=None,  # None if not tiger
):
    if args.env_name.split('-')[0] in ['DisSymmetricStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
        deterministic_policy = True
    elif args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        deterministic_policy = True
    elif args.env_name.split('-')[0] in ['LatentGoalCart']:
        deterministic_policy = True
    else:
        deterministic_policy = args.deterministic_policy

    conditioned_trajectories = {}
    conditioned_trajectories['metaRL_all_action_logits'] = []
    conditioned_trajectories['metaRL_all_action_probs'] = []
    conditioned_trajectories['metaRL_actions'] = []
    conditioned_trajectories['metaRL_rnn_states'] = []
    conditioned_trajectories['metaRL_belief_states'] = []
    conditioned_trajectories['metaRL_belief_states_mean_only'] = []
    conditioned_trajectories['metaRL_bottleneck_states'] = []
    
    if args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        n_episodes = len(ref_actions)
    else:
        n_episodes = ref_actions.shape[0]

    for episode_id in range(n_episodes):
        ref_actions_episode = ref_actions[episode_id]
        ref_rewards_episode = ref_rewards[episode_id]
        if ref_observations is not None:
            ref_observations_episode = ref_observations[episode_id]

        metaRL_all_action_logits_episode = []
        metaRL_actions_episode = []
        metaRL_rnn_states_episode = []
        metaRL_belief_states_episode = []
        metaRL_belief_states_mean_only_episode = []
        metaRL_bottleneck_states_episode = []
        
        if args.exp_label == 'rl2':
            prev_action = torch.zeros(1, 1, args.action_dim).to(device)
            prev_reward = torch.zeros(1, 1, 1).to(device)
            # initialize ActorCriticRNN hidden states
            # TODO: currently only supports shared actor-critic RNN
            rnn_prev_hidden_state = torch.zeros(1, 1, args.rnn_hidden_dim).to(device)

            for trial in range(len(ref_actions_episode)):
                # input_state_for_policy: already accounted for time
                if args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
                    curr_state = torch.from_numpy(
                        ref_observations_episode[trial]
                    ).float().reshape((1, 1, args.input_state_dim_for_policy)).to(device)
                elif args.env_name.split('-')[0] in ['LatentGoalCart']:
                    curr_state = torch.from_numpy(
                        np.array([ref_observations_episode[trial]])
                    ).float().reshape((1, 1, args.input_state_dim_for_policy)).to(device)
                else:
                    curr_state = torch.from_numpy(np.array([])).float().\
                        reshape((1, 1, args.input_state_dim_for_policy)).to(device)
                    
                with torch.no_grad():
                    # act
                    action_categorical, action_log_prob, entropy, state_value, \
                        rnn_hidden_state = \
                            policy_network.act(
                                curr_states=curr_state,
                                prev_actions=prev_action,
                                prev_rewards=prev_reward,
                                rnn_prev_hidden_states=rnn_prev_hidden_state,
                                return_prior=False, 
                                deterministic=deterministic_policy
                            )
                    # get action_prob
                    all_action_logit, _, _, = policy_network(
                        curr_states=curr_state, 
                        prev_actions=prev_action, 
                        prev_rewards=prev_reward,
                        rnn_prev_hidden_states=rnn_prev_hidden_state,
                        return_prior=False
                    )

                    # use ref trajectory for the next input
                    if args.env_name.split('-')[0] in ['LatentGoalCart']:
                        prev_action = torch.from_numpy(np.array([ref_actions_episode[trial]])).float().reshape((1, 1, args.action_dim)).to(device)
                        prev_reward = torch.from_numpy(np.array([ref_rewards_episode[trial]])).float().reshape(1, 1, 1).to(device)
                    else:
                        prev_action = F.one_hot(
                            torch.from_numpy(np.array([ref_actions_episode[trial]])), 
                            num_classes=args.action_dim
                        ).float().reshape((1, 1, args.action_dim)).to(device)
                    prev_reward = torch.from_numpy(
                        np.array([ref_rewards_episode[trial]])
                    ).float().reshape(1, 1, 1).to(device)
                    rnn_prev_hidden_state = rnn_hidden_state.to(device)

                    # save conditioned metaRL rollout
                    metaRL_rnn_states_episode.append(
                        rnn_hidden_state.squeeze().detach().cpu().numpy())
                    metaRL_actions_episode.append(
                        action_categorical.squeeze().detach().cpu().numpy())
                    if args.env_name.split('-')[0] in ['LatentGoalCart']:
                        metaRL_all_action_logits_episode.append(
                            torch.cat(all_action_logit).squeeze().detach().cpu().numpy())
                    else:
                        metaRL_all_action_logits_episode.append(
                            all_action_logit.squeeze().detach().cpu().numpy())
                    if len(policy_network.fc_after_rnn) > 0:  # if bottleneck layer exists
                        rnn_bottleneck_state = policy_network.fc_after_rnn[0](rnn_hidden_state.to(device))
                        rnn_bottleneck_state = policy_network.activation_function(rnn_bottleneck_state)
                        metaRL_bottleneck_states_episode.append(
                            rnn_bottleneck_state.squeeze().detach().cpu().numpy())
                    

        elif args.exp_label == 'mpc':
            prev_action = torch.zeros(1, 1, args.action_dim).to(device)
            prev_reward = torch.zeros(1, 1, 1).to(device)
            # initialize RNNEncoder hidden states
            encoder_rnn_prev_hidden_state = torch.zeros(1, 1, args.encoder_rnn_hidden_dim).to(device)

            for trial in range(len(ref_actions_episode)):
                # input_state_for_policy: already accounted for time
                if args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
                    curr_state = torch.from_numpy(
                        ref_observations_episode[trial]
                    ).float().reshape((1, 1, args.input_state_dim_for_policy)).to(device)
                elif args.env_name.split('-')[0] in ['LatentGoalCart']:
                    curr_state = torch.from_numpy(
                        np.array([ref_observations_episode[trial]])
                    ).float().reshape((1, 1, args.input_state_dim_for_policy)).to(device)
                else:
                    curr_state = torch.from_numpy(np.array([])).float().\
                        reshape((1, 1, args.input_state_dim_for_policy)).to(device)

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
                            deterministic=deterministic_policy
                        )
                    # get action_prob
                    all_action_logit, _ = policy_network(
                        curr_states=curr_state, 
                        curr_latent_means=curr_latent_mean, 
                        curr_latent_logvars=curr_latent_logvar
                    )

                    # use ref trajectory for the next input
                    if args.env_name.split('-')[0] in ['LatentGoalCart']:
                        prev_action = torch.from_numpy(np.array([ref_actions_episode[trial]])).float().reshape((1, 1, args.action_dim)).to(device)
                    else:
                        prev_action = F.one_hot(
                            torch.from_numpy(np.array([ref_actions_episode[trial]])), 
                            num_classes=args.action_dim
                        ).float().reshape((1, 1, args.action_dim)).to(device)
                    prev_reward = torch.from_numpy(
                        np.array([ref_rewards_episode[trial]])
                    ).float().reshape(1, 1, 1).to(device)
                    encoder_rnn_prev_hidden_state = encoder_rnn_hidden_state.to(device)

                    # save conditioned metaRL rollout
                    metaRL_actions_episode.append(
                        action_categorical.squeeze().detach().cpu().numpy())
                    if args.env_name.split('-')[0] in ['LatentGoalCart']:
                        metaRL_all_action_logits_episode.append(
                            torch.cat(all_action_logit).squeeze().detach().cpu().numpy())
                    else:
                        metaRL_all_action_logits_episode.append(
                            all_action_logit.squeeze().detach().cpu().numpy())
                    metaRL_rnn_states_episode.append(
                        encoder_rnn_hidden_state.squeeze().detach().cpu().numpy())
                    metaRL_belief_states_episode.append(
                        np.concatenate(
                            (curr_latent_mean.squeeze().detach().cpu().numpy(), 
                             curr_latent_logvar.squeeze().detach().cpu().numpy())))
                    metaRL_belief_states_mean_only_episode.append(
                            curr_latent_mean.squeeze().detach().cpu().numpy())
                    
        metaRL_actions_episode = np.array(metaRL_actions_episode)
        metaRL_all_action_logits_episode = np.array(metaRL_all_action_logits_episode)
        metaRL_rnn_states_episode = np.array(metaRL_rnn_states_episode)
        metaRL_belief_states_episode = np.array(metaRL_belief_states_episode)
        metaRL_belief_states_mean_only_episode = np.array(metaRL_belief_states_mean_only_episode)
        metaRL_bottleneck_states_episode = np.array(metaRL_bottleneck_states_episode)
        # process action_logits to action_prob
        shifted_logits = metaRL_all_action_logits_episode - np.max(metaRL_all_action_logits_episode)  # shift to avoid numerical overfloat
        norm_factors = np.sum(np.exp(shifted_logits), axis=1)
        metaRL_all_action_probs_episode = np.exp(shifted_logits)/ np.repeat(norm_factors, shifted_logits.shape[1]).reshape(shifted_logits.shape)
        
        conditioned_trajectories['metaRL_all_action_logits'].append(metaRL_all_action_logits_episode)
        conditioned_trajectories['metaRL_all_action_probs'].append(metaRL_all_action_probs_episode)
        conditioned_trajectories['metaRL_actions'].append(metaRL_actions_episode)
        conditioned_trajectories['metaRL_rnn_states'].append(metaRL_rnn_states_episode)
        conditioned_trajectories['metaRL_belief_states'].append(metaRL_belief_states_episode)
        conditioned_trajectories['metaRL_belief_states_mean_only'].append(metaRL_belief_states_mean_only_episode)
        conditioned_trajectories['metaRL_bottleneck_states'].append(metaRL_bottleneck_states_episode)

    # make np array
    if args.env_name.split('-')[0] not in ['StatTiger', 'MarkovTiger']:
        for data_name, data in conditioned_trajectories.items():
            conditioned_trajectories[data_name] = np.array(data)  # (n_runs * n_trials * n_feature)
    
    return conditioned_trajectories


def gen_state_mapper_training_dataset(
    flattened_metaRL_states,
    flattened_bayes_states
):
    '''
    state machine analysis works only for
    stationary bandits for now
    '''
    # input states should be of dimensions:
    # num_samples (episodes*trials) X num_features

    # pack as Tensor dataset
    metaRL2bayes_dataset = TensorDataset(
        torch.from_numpy(flattened_metaRL_states), 
        torch.from_numpy(flattened_bayes_states)
    )
    bayes2metaRL_dataset = TensorDataset(
        torch.from_numpy(flattened_bayes_states),
        torch.from_numpy(flattened_metaRL_states)
    )

    return metaRL2bayes_dataset, bayes2metaRL_dataset


def train_state_space_mapper(
    source_dim,
    target_dim,
    dataset,
    num_epochs,
    batch_size=32,
    validation_split=0.2,
    patience=10,
    min_delta=0.001,
    min_training_epochs=100
):
    """
    Train a state space mapper with early stopping.
    
    Args:
        source_dim: Dimension of the source state space
        target_dim: Dimension of the target state space
        dataset: Dataset containing source and target states
        num_epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        validation_split: Proportion of data to use for validation
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to be considered an improvement
        
    Returns:
        state_space_mapper: Trained state space mapper
        losses_mse: Training loss history
    """
    # Create training and validation datasets
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # initialize mapper
    state_space_mapper = StateSpaceMapper(
        source_dim=source_dim,
        hidden_layers=[64, 128, 64],
        activation_function='relu',  # tanh, relu, leaky-relu
        initialization_method='orthogonal', # orthogonal, normc
        target_dim=target_dim
    )

    # initialize trainer
    state_mapper_trainer = StateMapperTrainer(
        state_space_mapper,
        lr=0.001,
        eps=1e-8,
        anneal_lr=False,
        train_steps=num_epochs,
        patience=patience,
        min_delta=min_delta,
        min_training_epochs=min_training_epochs
    )

    # training with early stopping
    train_losses, val_losses, best_epoch = state_mapper_trainer.train(
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size=batch_size
    )
    
    print(f"Training completed at epoch {best_epoch}/{num_epochs} due to early stopping.")

    return state_space_mapper, train_losses


def get_mapped_bayes_states_and_actions(
    metaRL2bayes_mapper,
    metaRL_states,
    env_name,
    total_trials,
    num_bandits=None,  # total number of bandits
    is_pcaed=False,
    pca_model_bayes_for_dataset=None,
    dis_markovian_bandit_solver=None,  # None if not discrete markovian bandit
    tiger_solver=None,  # None if not tiger
    latent_goal_cart_solver=None,  # None if not latent goal cart
    ref_observations=None,  # None if not latent goal cart
):
    # metaRL_states: (n_runs, n_trials, n_features)

    mapped_bayes_states = []
    mapped_bayes_actions = []

    if env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        n_episodes = len(metaRL_states)
    else:
        n_episodes = metaRL_states.shape[0]

    for run_id in range(n_episodes):
        metaRL_states_per_run = metaRL_states[run_id]
        # state mappping: metaRL to bayes
        mapped_bayes_states_per_run = metaRL2bayes_mapper(
            torch.from_numpy(metaRL_states_per_run).to(device))
        mapped_bayes_states_per_run = mapped_bayes_states_per_run.detach().cpu().numpy()
        # inverse transform if needed
        if is_pcaed:
            mapped_bayes_states_per_run = pca_model_bayes_for_dataset.inverse_transform(
                mapped_bayes_states_per_run)
        
        # (ct_unrewarded, ct_rewarded) * num_bandits
        
        if 'StatBernoulliBandit' in env_name:
            # rounding to int
            mapped_bayes_states_per_run = np.round(mapped_bayes_states_per_run).astype(int)

            mapped_bayes_gittins_indices_per_run, mapped_bayes_actions_per_run = \
            get_bayes_optimal_actions_given_states_bernoulli_bandit(
                mapped_bayes_states_per_run,
                total_trials=total_trials
            )

        elif 'OracleBanditDeterministic' in env_name:
            # note: rounding is done within the OracleBanditSolver
            mapped_bayes_actions_per_run = get_bayes_optimal_actions_given_states_oracle_bandit(
                mapped_bayes_states_per_run,
                num_bandits=num_bandits
            )
        
        elif env_name.split('-')[0] in ['DisSymmetricStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
            mapped_bayes_actions_per_run = get_bayes_optimal_actions_given_states_dis_markovian_bandit(
                mapped_bayes_states_per_run,
                optimal_solver=dis_markovian_bandit_solver
            )
        
        elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
            mapped_bayes_actions_per_run = get_bayes_optimal_actions_given_states_tiger(
                mapped_bayes_states_per_run,
                optimal_solver=tiger_solver
            )
        
        elif env_name.split('-')[0] in ['LatentGoalCart']:
            ref_observations_per_run = ref_observations[run_id]
            mapped_bayes_actions_per_run = get_bayes_optimal_actions_given_states_latent_goal_cart(
                mapped_bayes_states_per_run,
                ref_observations=ref_observations_per_run,
                optimal_solver=latent_goal_cart_solver
            )

        else:
            raise ValueError(f'unsupported env: {env_name}')
        
        mapped_bayes_states.append(mapped_bayes_states_per_run)
        mapped_bayes_actions.append(mapped_bayes_actions_per_run)
    
    # make np array
    if env_name.split('-')[0] not in ['StatTiger', 'MarkovTiger']:
        mapped_bayes_states = np.array(mapped_bayes_states)
        mapped_bayes_actions = np.array(mapped_bayes_actions)

    return mapped_bayes_states, mapped_bayes_actions


def get_mapped_metaRL_states_and_actions(
    bayes2metaRL_mapper,
    bayes_states,
    args,
    encoder,
    policy_network,
    mapped_metaRL_state_type,  # 'rnn', 'belief'
    is_pcaed=False,
    pca_model_metaRL_for_dataset=None,
    pcaed_metaRL_states_mean=None,
    bayes_state_dim=4,
    ref_observations=None,  # None if not tiger
):
    # bayes_states: (n_runs, n_trials, n_features)
    
    mapped_metaRL_states = []
    mapped_metaRL_all_action_logits = []
    mapped_metaRL_actions = []

    if args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        n_episodes = len(bayes_states)
    else:
        n_episodes = bayes_states.shape[0]

    for run_id in range(n_episodes):
        bayes_states_per_run = bayes_states[run_id]
        if ref_observations is not None:
            ref_observations_per_run = ref_observations[run_id]

        # state mappping: bayes to metaRL
        mapped_metaRL_states_per_run = bayes2metaRL_mapper(
            torch.from_numpy(bayes_states_per_run).type(torch.float).to(device)
        ).detach().cpu().numpy()

        # inverse transform if needed
        if is_pcaed:
            # the first bayes_state_dim PCs are the mapped_metaRL_states
            # the rest of the dimensions are mean PC values
            pcaed_metaRL_states_mean_broadcasted = np.tile(
                pcaed_metaRL_states_mean, 
                (mapped_metaRL_states_per_run.shape[0], 1))

            pcaed_metaRL_states_mean_broadcasted[:, :bayes_state_dim] = mapped_metaRL_states_per_run
            mapped_metaRL_states_per_run = pcaed_metaRL_states_mean_broadcasted

            mapped_metaRL_states_per_run = pca_model_metaRL_for_dataset.inverse_transform(
                mapped_metaRL_states_per_run)

        # get mapped metaRL actions
        mapped_metaRL_all_action_logits_per_run = []
        mapped_metaRL_actions_per_run = []

        for step in range(len(bayes_states_per_run)):
            if args.exp_label == 'mpc':
                # input_state_for_policy: already accounted for time
                if args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
                    curr_state = torch.from_numpy(
                        ref_observations_per_run[step]).float().to(device)
                elif args.env_name.split('-')[0] in ['LatentGoalCart']:
                    curr_state = torch.from_numpy(
                        np.array([ref_observations_per_run[step]])).float().to(device)
                else:
                    curr_state = torch.from_numpy(np.array([])).float().to(device)
                    
                if mapped_metaRL_state_type == 'belief':
                    mapped_action_logits, _ = policy_network(
                        curr_states=curr_state, 
                        curr_latent_means=torch.from_numpy(mapped_metaRL_states_per_run[step, :args.latent_dim]).type(torch.float).to(device), 
                        curr_latent_logvars=torch.from_numpy(mapped_metaRL_states_per_run[step, args.latent_dim:]).type(torch.float).to(device)
                    )
                elif mapped_metaRL_state_type == 'rnn':
                    latent_means_step = encoder.encoder_mu(torch.from_numpy(mapped_metaRL_states_per_run[step, :]).type(torch.float).to(device))
                    latent_logvars_step = encoder.encoder_logvar(torch.from_numpy(mapped_metaRL_states_per_run[step, :]).type(torch.float).to(device))
                    mapped_action_logits, _ = policy_network(
                        curr_states=curr_state, 
                        curr_latent_means=latent_means_step, 
                        curr_latent_logvars=latent_logvars_step
                    )
                else:
                    raise ValueError(f'incompatible state type for mpc: {mapped_metaRL_state_type}')
                
            elif args.exp_label == 'rl2':
                if mapped_metaRL_state_type == 'rnn':
                    # NOTE: now only supports sharedActorCritcRNN
                    if len(policy_network.fc_after_rnn) == 0:
                        mapped_action_logits = policy_network.actor_output(
                            torch.from_numpy(mapped_metaRL_states_per_run[step, :]).type(torch.float).unsqueeze(0).to(device)
                        )
                    else:
                        h = torch.from_numpy(mapped_metaRL_states_per_run[step, :]).type(torch.float).unsqueeze(0).to(device)
                        for i in range(len(policy_network.fc_after_rnn)):
                            mapped_h = policy_network.fc_after_rnn[i](h)
                            mapped_h = policy_network.activation_function(mapped_h)
                            h = mapped_h
                        if policy_network.action_space_type == 'discrete': 
                            mapped_action_logits = policy_network.actor_output(mapped_h)
                        else:  # Box
                            mapped_action_means = policy_network.actor_mean(mapped_h)
                            mapped_action_stds = torch.exp(policy_network.actor_log_std(mapped_h))
                            mapped_action_logits = (mapped_action_means, mapped_action_stds)
                        
                elif mapped_metaRL_state_type == 'bottleneck':
                    if len(policy_network.fc_after_rnn) == 1:
                        mapped_action_logits = policy_network.actor_output(
                                torch.from_numpy(mapped_metaRL_states_per_run[step, :]).type(torch.float).unsqueeze(0).to(device)
                            )
                    elif len(policy_network.fc_after_rnn) == 2:
                        h = torch.from_numpy(mapped_metaRL_states_per_run[step, :]).type(torch.float).unsqueeze(0).to(device)
                        mapped_h = policy_network.fc_after_rnn[1](h)
                        mapped_h = policy_network.activation_function(mapped_h)
                        if policy_network.action_space_type == 'discrete': 
                            mapped_action_logits = policy_network.actor_output(mapped_h)
                        else:  # Box
                            mapped_action_means = policy_network.actor_mean(mapped_h)
                            mapped_action_stds = torch.exp(policy_network.actor_log_std(mapped_h))
                            mapped_action_logits = (mapped_action_means, mapped_action_stds)
                    else:
                        raise ValueError('Unsupported number of fc layers_after_rnn in rl2')
                    
                else:
                    raise ValueError(f'incompatible state type for rl2: {mapped_metaRL_state_type}')

            # sample action
            if policy_network.action_space_type == 'discrete': 
                mapped_action_pd = policy_network.policy_dist(logits=mapped_action_logits)
            else:  # Box
                mapped_action_means, mapped_action_stds = mapped_action_logits
                mapped_action_pd = policy_network.policy_dist(mapped_action_means, mapped_action_stds)

            if args.env_name.split('-')[0] in ['DisSymmetricStickyMarkovBandit2State2Arm',
                                    'DisAsymmetricRewardStickyMarkovBandit2State2Arm',
                                    'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
                deterministic_policy = True
            elif args.env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
                deterministic_policy = True
            elif args.env_name.split('-')[0] in ['LatentGoalCart']:
                deterministic_policy = True
            else:
                deterministic_policy = args.deterministic_policy

            if deterministic_policy:
                if isinstance(mapped_action_pd, torch.distributions.Categorical):
                    mapped_action = mapped_action_pd.mode
                elif isinstance(mapped_action_pd, torch.distributions.Normal):
                    mapped_action = mapped_action_pd.mean
                else:
                    mapped_action = mapped_action_pd.mean
            else:
                mapped_action = mapped_action_pd.sample()
            
            mapped_metaRL_actions_per_run.append(mapped_action.squeeze().detach().cpu().numpy())
            mapped_metaRL_all_action_logits_per_run.append(
                torch.cat(mapped_action_logits).squeeze().detach().cpu().numpy()
            )

        mapped_metaRL_all_action_logits_per_run = np.array(mapped_metaRL_all_action_logits_per_run)
        
        mapped_metaRL_states.append(mapped_metaRL_states_per_run)
        mapped_metaRL_all_action_logits.append(mapped_metaRL_all_action_logits_per_run)
        mapped_metaRL_actions.append(mapped_metaRL_actions_per_run)

    # make np array
    if args.env_name.split('-')[0] not in ['StatTiger', 'MarkovTiger']:
        mapped_metaRL_states = np.array(mapped_metaRL_states)
        mapped_metaRL_all_action_logits = np.array(mapped_metaRL_all_action_logits)
        mapped_metaRL_actions = np.array(mapped_metaRL_actions)

    return mapped_metaRL_states, mapped_metaRL_all_action_logits, mapped_metaRL_actions
    

def get_state_dissimilarity(
    true_states,
    mapped_states,
    env_name
):
    
    if env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        
        n_runs = len(true_states)
        state_mapping_se_total = 0
        true_states_var_total = 0
        total_trials = 0

        for run_id in range(n_runs):
            true_states_per_run = true_states[run_id]
            mapped_states_per_run = mapped_states[run_id]
            
            state_mapping_se_per_run = np.sum(np.square(true_states_per_run - mapped_states_per_run), axis=1)
            true_states_var_per_run = np.sum(np.square(true_states_per_run), axis=1)
            
            state_mapping_se_total += np.sum(state_mapping_se_per_run)
            true_states_var_total += np.sum(true_states_var_per_run)
            total_trials += state_mapping_se_per_run.shape[0]
        
        state_mapping_mse = state_mapping_se_total/ float(total_trials)
        true_states_avg_var = true_states_var_total/ float(total_trials)
        normalized_state_mapping_mse = state_mapping_mse/ true_states_avg_var
            
    else:
        # states shape: episodes X trials X features
        state_mapping_mse = np.average(
            np.sum(
                np.square(true_states - mapped_states), axis=2))
        true_states_avg_var = np.average(
            np.sum(
                np.square(true_states), axis=2))
        normalized_state_mapping_mse = state_mapping_mse/ true_states_avg_var

    return state_mapping_mse, true_states_avg_var, normalized_state_mapping_mse

def get_output_dissimilarity(
    true_actions,
    mapped_actions,
    env_name,
    p_bandits=None,
    r_bandits=None,
    reward_probs_t=None,  # None if not discrete markovian bandit
    rewards_t=None,  # None if not tiger
    goal_positions_t=None,  # None if not LatentGoalCart
    positions_t=None,  # None if not LatentGoalCart
):
    # actions shape: episodes X trials
    
    expected_return_diff = 0
    total_timesteps = 0

    if 'StatBernoulliBandit' in env_name:
        n_runs = p_bandits.shape[0]
    elif 'OracleBanditDeterministic' in env_name:
        n_runs = r_bandits.shape[0]
    elif env_name.split('-')[0] in ['DisSymmetricStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
        n_runs = reward_probs_t.shape[0]
    elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        n_runs = len(rewards_t)
    elif env_name.split('-')[0] in ['LatentGoalCart']:
        n_runs = goal_positions_t.shape[0]
    else:
        raise ValueError(f'unsupported env: {env_name}')
    
    for run_id in range(n_runs):
        true_actions_per_run = true_actions[run_id]
        mapped_actions_per_run = mapped_actions[run_id]

        if 'StatBernoulliBandit' in env_name:
            true_expected_rewards_per_run = p_bandits[run_id][true_actions_per_run]
            mapped_expected_rewards_per_run = p_bandits[run_id][mapped_actions_per_run]
        
        elif 'OracleBanditDeterministic' in env_name:
            true_expected_rewards_per_run = r_bandits[run_id][true_actions_per_run]
            mapped_expected_rewards_per_run = r_bandits[run_id][mapped_actions_per_run]
        
        elif env_name.split('-')[0] in ['DisSymmetricStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                    'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
            true_expected_rewards_per_run = []
            mapped_expected_rewards_per_run = []
            for step in range(len(true_actions_per_run)):
                true_expected_rewards_per_run.append(reward_probs_t[run_id][step][true_actions_per_run[step]])
                mapped_expected_rewards_per_run.append(reward_probs_t[run_id][step][mapped_actions_per_run[step]])
            true_expected_rewards_per_run = np.array(true_expected_rewards_per_run)
            mapped_expected_rewards_per_run = np.array(mapped_expected_rewards_per_run)
        
        elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
            true_expected_rewards_per_run = []
            mapped_expected_rewards_per_run = []

            for step in range(len(true_actions_per_run)):
                true_expected_rewards_per_run.append(rewards_t[run_id][step][true_actions_per_run[step]])
                mapped_expected_rewards_per_run.append(rewards_t[run_id][step][mapped_actions_per_run[step]])
            
            true_expected_rewards_per_run = np.array(true_expected_rewards_per_run)
            mapped_expected_rewards_per_run = np.array(mapped_expected_rewards_per_run)

        elif env_name.split('-')[0] in ['LatentGoalCart']:
            goal_positions_t_per_run = goal_positions_t[run_id]
            positions_t_per_run = positions_t[run_id]

            true_expected_rewards_per_run = []
            mapped_expected_rewards_per_run = []
            
            for step in range(len(true_actions_per_run)):
                # compute true and mapped expected rewards
                true_velocity = np.clip(true_actions_per_run[step], -1.0, 1.0)
                true_new_position = positions_t_per_run[step] + true_velocity * 0.1
                true_new_position = np.clip(true_new_position, -2.0, 2.0)  # clip to valid range

                mapped_velocity = np.clip(mapped_actions_per_run[step], -1.0, 1.0)
                mapped_new_position = positions_t_per_run[step] + mapped_velocity * 0.1
                mapped_new_position = np.clip(mapped_new_position, -2.0, 2.0)  # clip to valid range

                true_expected_reward = -((true_new_position - goal_positions_t_per_run)**2) - 0.01 * (true_actions_per_run[step]**2)
                mapped_expected_reward = -((mapped_new_position - goal_positions_t_per_run)**2) - 0.01 * (mapped_actions_per_run[step]**2)
                 
                true_expected_rewards_per_run.append(true_expected_reward)
                mapped_expected_rewards_per_run.append(mapped_expected_reward)
            
            true_expected_rewards_per_run = np.array(true_expected_rewards_per_run)
            mapped_expected_rewards_per_run = np.array(mapped_expected_rewards_per_run)

        else:
            raise ValueError(f'unsupported env: {env_name}')
            
        expected_rewards_diff_per_run = true_expected_rewards_per_run - mapped_expected_rewards_per_run
        total_timesteps += len(true_actions_per_run)

        if env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
            # expected_return_diff += np.sum(expected_rewards_diff_per_run)
            expected_return_diff += np.sum(np.abs(expected_rewards_diff_per_run))

        else:
            expected_return_diff += np.abs(np.sum(expected_rewards_diff_per_run))
    
    if env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        avg_expected_return_diff = np.abs(expected_return_diff)/ float(total_timesteps)
    elif env_name.split('-')[0] in ['LatentGoalCart']:
        avg_expected_return_diff = np.abs(expected_return_diff)/ float(total_timesteps)
    else:
        avg_expected_return_diff = np.abs(expected_return_diff)/ float(n_runs)
    
    return avg_expected_return_diff


def dissimilarity_analysis(
    true_bayes_states,
    mapped_bayes_states,
    true_metaRL_actions,
    mapped_bayes_actions,
    true_metaRL_states,
    mapped_metaRL_states,
    true_bayes_actions,
    mapped_metaRL_actions,
    env_name,
    p_bandits=None,
    r_bandits=None,
    reward_probs_t=None,  # None if not discrete markovian bandit
    rewards_t=None,  # None if not tiger
    goal_positions_t=None,  # None if not latent goal cart
    positions_t=None  # None if not latent goal cart
):
    '''
    compute state dissimilarity and 
    output dissimilarity (expected return difference)
    '''
    # states shape: episodes X trials X features
    
    # metaRL to bayes
    print('metaRL to bayes:')
    ## state
    mapped_bayes_mse, bayes_avg_var, normalized_mapped_bayes_mse = \
    get_state_dissimilarity(
        true_bayes_states, 
        mapped_bayes_states,
        env_name
    )

    print(f' mapped_bayes_mse: {mapped_bayes_mse}')
    print(f' bayes_avg_var: {bayes_avg_var}')
    print(f' normalized_mapped_bayes_mse: {normalized_mapped_bayes_mse}')

    ## output
    avg_expected_return_diff_metaRL_mapped_bayes = get_output_dissimilarity(
        true_actions=true_metaRL_actions,
        mapped_actions=mapped_bayes_actions,
        env_name=env_name,
        p_bandits=p_bandits,
        r_bandits=r_bandits,
        reward_probs_t=reward_probs_t,
        rewards_t=rewards_t,
        goal_positions_t=goal_positions_t,
        positions_t=positions_t
    )
    print(f' return_diff (metaRL - mapped_bayes): {avg_expected_return_diff_metaRL_mapped_bayes}')

    # bayes to metaRL
    print('\nbayes to metaRL:')
    ## state
    mapped_metaRL_mse, metaRL_avg_var, normalized_mapped_metaRL_mse =\
    get_state_dissimilarity(
        true_metaRL_states, 
        mapped_metaRL_states,
        env_name
    )

    print(f' mapped_metaRL_mse: {mapped_metaRL_mse}')
    print(f' metaRL_avg_var: {metaRL_avg_var}')
    print(f' normalized_mapped_metaRL_mse: {normalized_mapped_metaRL_mse}')

    # output
    avg_expected_return_diff_bayes_mapped_metaRL = get_output_dissimilarity(
        true_actions=true_bayes_actions,
        mapped_actions=mapped_metaRL_actions,
        env_name=env_name,
        p_bandits=p_bandits,
        r_bandits=r_bandits,
        reward_probs_t=reward_probs_t,
        rewards_t=rewards_t
    )
    print(f' return_diffs (bayes - mapped_metaRL): {avg_expected_return_diff_bayes_mapped_metaRL}')

    return normalized_mapped_bayes_mse, avg_expected_return_diff_metaRL_mapped_bayes, \
        normalized_mapped_metaRL_mse, avg_expected_return_diff_bayes_mapped_metaRL



###############################################################################################
# Plotting
###############################################################################################
def get_PCA_model(flattened_states, num_pc_components):
    # input state dimensions: (n_samples, n_feature_dim)
    pca_model = PCA(n_components=num_pc_components)
    pca_model.fit(flattened_states)

    return pca_model


def plot_PCA_bayes_and_metaRL(
    pca_model_bayes, 
    pca_model_metaRL,
    true_bayes_states,
    mapped_bayes_states,
    true_metaRL_states,
    mapped_metaRL_states,
    true_bayes_actions,
    mapped_bayes_actions,
    env_name,
    # coloring: BernoulliBandit
    true_metaRL_a1_probs=None,
    mapped_metaRL_a1_probs=None,
    # coloring: OracleBandit
    true_metaRL_all_action_logits=None,
    mapped_metaRL_all_action_logits=None,
    # coloring: LatentGoalCart
    true_metaRL_actions=None,
    mapped_metaRL_actions=None,
    total_trials=None,
    num_bandits=None
):
    size = 18
    params = {
        'legend.fontsize': 'large',
        'axes.labelsize': size,
        'axes.titlesize': size*1.15,
        'xtick.labelsize': size*0.8,
        'ytick.labelsize': size*0.8,
        'axes.titlepad': 15
    }
    plt.rcParams.update(params)

    transformed_true_bayes_states = pca_model_bayes.transform(true_bayes_states)
    transformed_true_metaRL_states = pca_model_metaRL.transform(true_metaRL_states)
    if mapped_bayes_states is not None:
        transformed_mapped_bayes_states = pca_model_bayes.transform(mapped_bayes_states)
    if mapped_metaRL_states is not None:
        transformed_mapped_metaRL_states = pca_model_metaRL.transform(mapped_metaRL_states)

    # figure setup
    fig, axs = plt.subplots(
        nrows=2, ncols=2, 
        figsize=(16, 12), dpi=300
    )
    
    if  env_name.split('-')[0] in ['StatBernoulliBandit2ArmIndependent',
                                   'DisSymmetricStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
        color_map = cm.bwr
        scatter_ms = 8
        plot_ms = 5
        alpha = 0.7
        plot_example_rollout_len = 20
    elif 'OracleBanditDeterministic' in env_name:
        import matplotlib
        color_map = list(matplotlib.colors.TABLEAU_COLORS.keys()) + ['k']
        scatter_ms = 30
        oracle_scatter_ms = 50
        alpha = 0.7
    elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        import matplotlib
        color_map = list(matplotlib.colors.TABLEAU_COLORS.keys()) + ['k']
        scatter_ms = 15
        plot_ms = 5
        alpha = 0.7
    elif env_name.split('-')[0] in ['LatentGoalCart']:
        color_map = cm.bwr
        scatter_ms = 8
        plot_ms = 5
        alpha = 0.7
        plot_example_rollout_len = 30
    else:
        raise ValueError(f'unsupported env: {env_name}')

    # plot 0: true bayes_states
    ax = axs[0, 0]
    ax.set_title('Bayes-optimal states')
    if  env_name.split('-')[0] in ['StatBernoulliBandit2ArmIndependent',
                                   'DisSymmetricStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
        scatter = ax.scatter(
            transformed_true_bayes_states[:, 0], 
            transformed_true_bayes_states[:, 1],
            # c=metaRL_a1_probs_train, cmap=cm.bwr,
            c=true_bayes_actions, cmap=color_map,
            vmin=0, vmax=1,
            s=scatter_ms,
            alpha=alpha
        )
        ax.plot(
            transformed_true_bayes_states[0:plot_example_rollout_len-1, 0], 
            transformed_true_bayes_states[0:plot_example_rollout_len-1, 1],
            color='k', marker='o', 
            ms=plot_ms
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        fig.colorbar(scatter, ax=ax)

    elif 'OracleBanditDeterministic' in env_name:
        for arm_id in range(num_bandits):
            selected_idx = np.where(true_bayes_actions==arm_id)[0]
            if arm_id == 10:
                scatter = ax.scatter(
                    transformed_true_bayes_states[selected_idx, 0], 
                    transformed_true_bayes_states[selected_idx, 1],
                    c=color_map[arm_id],
                    s=oracle_scatter_ms,
                    label=f'arm {arm_id}'
                )
            else:
                scatter = ax.scatter(
                    transformed_true_bayes_states[selected_idx, 0], 
                    transformed_true_bayes_states[selected_idx, 1],
                    c=color_map[arm_id],
                    s=scatter_ms,
                    label=f'arm {arm_id}',
                    alpha=alpha
                )
        ax.plot(
            transformed_true_bayes_states[0:total_trials-1, 0], 
            transformed_true_bayes_states[0:total_trials-1, 1],
            color='k'
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        for action in range(3):
            selected_idx = np.where(true_bayes_actions==action)[0]
            if action == 2:
                scatter = ax.scatter(
                    transformed_true_bayes_states[selected_idx], 
                    np.ones(len(selected_idx)),
                    c='k',
                    s=scatter_ms,
                    label=f'listen'
                )
            elif action == 0:
                scatter = ax.scatter(
                    transformed_true_bayes_states[selected_idx], 
                    np.ones(len(selected_idx)),
                    c='r',
                    label=f'open L',
                    alpha=alpha
                )
            elif action == 1:
                scatter = ax.scatter(
                    transformed_true_bayes_states[selected_idx], 
                    np.ones(len(selected_idx)),
                    c='b',
                    label=f'open R',
                    alpha=alpha
                )
        plot_traj_trial_end = np.where(true_bayes_actions!=2)[0][0]+1
        ax.plot(
            transformed_true_bayes_states[0:plot_traj_trial_end], 
            np.ones(plot_traj_trial_end),
            color='gray'
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    elif env_name.split('-')[0] in ['LatentGoalCart']:
        scatter = ax.scatter(
            transformed_true_bayes_states[:, 0], 
            np.zeros(transformed_true_bayes_states.shape[0]),  # 1D
            c=true_bayes_actions, cmap=color_map,
            vmin=true_bayes_actions.min(), vmax=true_bayes_actions.max(),
            s=scatter_ms,
            alpha=alpha
        )
        ax.plot(
            transformed_true_bayes_states[0:plot_example_rollout_len-1, 0], 
            np.zeros(plot_example_rollout_len-1),  # 1D
            color='k', marker='o', 
            ms=plot_ms
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        fig.colorbar(scatter, ax=ax)


    # plot 1: mapped bayes_states
    ax = axs[0, 1]
    ax.set_title('Mapped Bayes-optimal states meta-RL')
    if mapped_bayes_states is not None:
        if  env_name.split('-')[0] in ['StatBernoulliBandit2ArmIndependent',
                                   'DisSymmetricStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
            scatter = ax.scatter(
                transformed_mapped_bayes_states[:, 0], 
                transformed_mapped_bayes_states[:, 1],
                # c=metaRL_a1_probs_test, cmap=cm.bwr,
                c=mapped_bayes_actions, cmap=color_map,
                vmin=0, vmax=1,
                s=scatter_ms,
                alpha=alpha
            )
            ax.plot(
                transformed_mapped_bayes_states[0:plot_example_rollout_len-1, 0], 
                transformed_mapped_bayes_states[0:plot_example_rollout_len-1, 1],
                color='k', marker='o', 
                ms=plot_ms
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            fig.colorbar(scatter, ax=ax)
        elif 'OracleBanditDeterministic' in env_name:
            for arm_id in range(num_bandits):
                selected_idx = np.where(mapped_bayes_actions==arm_id)[0]
                if arm_id == 10:
                    scatter = ax.scatter(
                        transformed_mapped_bayes_states[selected_idx, 0], 
                        transformed_mapped_bayes_states[selected_idx, 1],
                        c=color_map[arm_id],
                        s=oracle_scatter_ms,
                        label=f'arm {arm_id}'
                    )
                else:
                    scatter = ax.scatter(
                        transformed_mapped_bayes_states[selected_idx, 0], 
                        transformed_mapped_bayes_states[selected_idx, 1],
                        c=color_map[arm_id],
                        s=scatter_ms,
                        label=f'arm {arm_id}',
                        alpha=alpha
                    )
            ax.plot(
                transformed_mapped_bayes_states[0:total_trials-1, 0], 
                transformed_mapped_bayes_states[0:total_trials-1, 1],
                color='k'
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
            for action in range(3):
                selected_idx = np.where(mapped_bayes_actions==action)[0]
                if action == 2:
                    scatter = ax.scatter(
                        transformed_mapped_bayes_states[selected_idx], 
                        np.ones(len(selected_idx)),
                        c='k',
                        s=scatter_ms,
                        label=f'listen'
                    )
                elif action == 0:
                    scatter = ax.scatter(
                        transformed_mapped_bayes_states[selected_idx], 
                        np.ones(len(selected_idx)),
                        c='r',
                        label=f'open L',
                        alpha=alpha
                    )
                elif action == 1:
                    scatter = ax.scatter(
                        transformed_mapped_bayes_states[selected_idx], 
                        np.ones(len(selected_idx)),
                        c='b',
                        label=f'open R',
                        alpha=alpha
                    )
            # plot_traj_trial_end = np.where(mapped_bayes_actions!=2)[0][0]+1
            plot_traj_trial_end = np.where(true_bayes_actions!=2)[0][0]+1
            ax.plot(
                transformed_mapped_bayes_states[0:plot_traj_trial_end], 
                np.ones(plot_traj_trial_end),
                color='gray'
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        elif env_name.split('-')[0] in ['LatentGoalCart']:
            scatter = ax.scatter(
                transformed_mapped_bayes_states[:, 0], 
                np.zeros(transformed_mapped_bayes_states.shape[0]),  # 1D
                c=mapped_bayes_actions, cmap=color_map,
                vmin=mapped_bayes_actions.min(), vmax=mapped_bayes_actions.max(),
                s=scatter_ms,
                alpha=alpha
            )
            ax.plot(
                transformed_mapped_bayes_states[0:plot_example_rollout_len-1, 0], 
                np.zeros(plot_example_rollout_len-1),  # 1D
                color='k', marker='o', 
                ms=plot_ms
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            fig.colorbar(scatter, ax=ax)


    # plot 2: mapped metaRL_states
    ax = axs[1, 0]
    ax.set_title('Mapped meta-RL states from Bayes')
    if mapped_metaRL_states is not None:
        if  env_name.split('-')[0] in ['StatBernoulliBandit2ArmIndependent',
                                   'DisSymmetricStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
            scatter = ax.scatter(
                transformed_mapped_metaRL_states[:, 0], 
                transformed_mapped_metaRL_states[:, 1],
                # c=metaRL_a1_probs_test, cmap=cm.bwr,
                c=mapped_metaRL_a1_probs, cmap=color_map,
                vmin=0, vmax=1,
                s=scatter_ms,
                alpha=alpha
            )
            ax.plot(
                transformed_mapped_metaRL_states[0:plot_example_rollout_len-1, 0], 
                transformed_mapped_metaRL_states[0:plot_example_rollout_len-1, 1],
                color='k', marker='o', 
                ms=plot_ms
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            fig.colorbar(scatter, ax=ax)
        elif 'OracleBanditDeterministic' in env_name:
            for arm_id in range(num_bandits):
                selected_idx = np.where(np.argmax(mapped_metaRL_all_action_logits, axis=-1)==arm_id)[0]
                if arm_id == 10:
                    scatter = ax.scatter(
                        transformed_mapped_metaRL_states[selected_idx, 0], 
                        transformed_mapped_metaRL_states[selected_idx, 1],
                        c=color_map[arm_id],
                        s=oracle_scatter_ms,
                        label=f'arm {arm_id}'
                    )
                else:
                    scatter = ax.scatter(
                        transformed_mapped_metaRL_states[selected_idx, 0], 
                        transformed_mapped_metaRL_states[selected_idx, 1],
                        c=color_map[arm_id],
                        s=scatter_ms,
                        label=f'arm {arm_id}',
                        alpha=alpha
                    )
            ax.plot(
                transformed_mapped_metaRL_states[0:total_trials-1, 0], 
                transformed_mapped_metaRL_states[0:total_trials-1, 1],
                color='k'
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
            for action in range(3):
                selected_idx = np.where(np.argmax(mapped_metaRL_all_action_logits, axis=-1)==action)[0]
                if action == 2:
                    scatter = ax.scatter(
                        transformed_mapped_metaRL_states[selected_idx, 0], 
                        transformed_mapped_metaRL_states[selected_idx, 1],
                        c='k',
                        s=scatter_ms,
                        label=f'listen'
                    )
                elif action == 0:
                    scatter = ax.scatter(
                        transformed_mapped_metaRL_states[selected_idx, 0], 
                        transformed_mapped_metaRL_states[selected_idx, 1],
                        c='r',
                        label=f'open L',
                        alpha=alpha
                    )
                elif action == 1:
                    scatter = ax.scatter(
                        transformed_mapped_metaRL_states[selected_idx, 0], 
                        transformed_mapped_metaRL_states[selected_idx, 1],
                        c='b',
                        label=f'open R',
                        alpha=alpha
                    )
            # plot_traj_trial_end = np.where(np.argmax(mapped_metaRL_all_action_logits, axis=-1)!=2)[0][0]+1
            plot_traj_trial_end = np.where(true_bayes_actions!=2)[0][0]+1
            ax.plot(
                transformed_mapped_metaRL_states[:plot_traj_trial_end, 0], 
                transformed_mapped_metaRL_states[:plot_traj_trial_end, 1],
                color='gray'
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        elif env_name.split('-')[0] in ['LatentGoalCart']:
            scatter = ax.scatter(
                transformed_mapped_metaRL_states[:, 0], 
                transformed_mapped_metaRL_states[:, 1], 
                c=mapped_metaRL_actions, cmap=color_map,
                vmin=mapped_metaRL_actions.min(), vmax=mapped_metaRL_actions.max(),
                s=scatter_ms,
                alpha=alpha
            )
            ax.plot(
                transformed_mapped_metaRL_states[0:plot_example_rollout_len-1, 0], 
                transformed_mapped_metaRL_states[0:plot_example_rollout_len-1, 1],
                color='k', marker='o', 
                ms=plot_ms
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            fig.colorbar(scatter, ax=ax)


    # plot 3: true_metaRL_states
    ax = axs[1, 1]
    ax.set_title('MetaRL states')
    if  env_name.split('-')[0] in ['StatBernoulliBandit2ArmIndependent',
                                   'DisSymmetricStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricRewardStickyMarkovBandit2State2Arm', 
                                   'DisAsymmetricTransitionStickyMarkovBandit2State2Arm']:
        scatter = ax.scatter(
            transformed_true_metaRL_states[:, 0], 
            transformed_true_metaRL_states[:, 1],
            # c=metaRL_a1_probs_test, cmap=cm.bwr,
            c=true_metaRL_a1_probs, cmap=color_map,
            vmin=0, vmax=1,
            s=scatter_ms,
            alpha=alpha
        )
        ax.plot(
            transformed_true_metaRL_states[0:plot_example_rollout_len-1, 0], 
            transformed_true_metaRL_states[0:plot_example_rollout_len-1, 1],
            color='k', marker='o', 
            ms=plot_ms
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        fig.colorbar(scatter, ax=ax)

    elif 'OracleBanditDeterministic' in env_name:
        for arm_id in range(num_bandits):
            selected_idx = np.where(np.argmax(true_metaRL_all_action_logits, axis=-1)==arm_id)[0]
            if arm_id == 10:
                scatter = ax.scatter(
                    transformed_true_metaRL_states[selected_idx, 0], 
                    transformed_true_metaRL_states[selected_idx, 1],
                    c=color_map[arm_id],
                    s=oracle_scatter_ms,
                    label=f'arm {arm_id}'
                )
            else:
                scatter = ax.scatter(
                    transformed_true_metaRL_states[selected_idx, 0], 
                    transformed_true_metaRL_states[selected_idx, 1],
                    c=color_map[arm_id],
                    s=scatter_ms,
                    label=f'arm {arm_id}',
                    alpha=alpha
                )
        ax.plot(
            transformed_true_metaRL_states[0:total_trials-1, 0], 
            transformed_true_metaRL_states[0:total_trials-1, 1],
            color='k'
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    elif env_name.split('-')[0] in ['StatTiger', 'MarkovTiger']:
        for action in range(3):
            selected_idx = np.where(np.argmax(true_metaRL_all_action_logits, axis=-1)==action)[0]
            if action == 2:
                scatter = ax.scatter(
                    transformed_true_metaRL_states[selected_idx, 0], 
                    transformed_true_metaRL_states[selected_idx, 1],
                    c='k',
                    s=scatter_ms,
                    label=f'listen'
                )
            elif action == 0:
                scatter = ax.scatter(
                    transformed_true_metaRL_states[selected_idx, 0], 
                    transformed_true_metaRL_states[selected_idx, 1],
                    c='r',
                    label=f'open L',
                    alpha=alpha
                )
            elif action == 1:
                scatter = ax.scatter(
                    transformed_true_metaRL_states[selected_idx, 0], 
                    transformed_true_metaRL_states[selected_idx, 1],
                    c='b',
                    label=f'open R',
                    alpha=alpha
                )
        # plot_traj_trial_end = np.where(np.argmax(true_metaRL_all_action_logits, axis=-1)!=2)[0][0]+1
        plot_traj_trial_end = np.where(true_bayes_actions!=2)[0][0]+1
        ax.plot(
            transformed_true_metaRL_states[:plot_traj_trial_end, 0], 
            transformed_true_metaRL_states[:plot_traj_trial_end, 1], 
            color='gray'
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    elif env_name.split('-')[0] in ['LatentGoalCart']:
        scatter = ax.scatter(
            transformed_true_metaRL_states[:, 0], 
            transformed_true_metaRL_states[:, 1], 
            c=true_metaRL_actions, cmap=color_map,
            vmin=true_metaRL_actions.min(), vmax=true_metaRL_actions.max(),
            s=scatter_ms,
            alpha=alpha
        )
        ax.plot(
            transformed_true_metaRL_states[0:plot_example_rollout_len-1, 0], 
            transformed_true_metaRL_states[0:plot_example_rollout_len-1, 1],
            color='k', marker='o', 
            ms=plot_ms
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        fig.colorbar(scatter, ax=ax)
    

    # fig setup
    fig.tight_layout()

    return fig


def plot_expected_return_comparison(
    # args, 
    out_dir,
    eval_epoch_ids,
    metaRL_expected_returns_stat_bernoulli,
    generative_bayes_expected_returns_stat_bernoulli,
    generative_expected_return_diff_mean_stat_bernoulli,
    generative_expected_return_diff_std_stat_bernoulli,
    conditioned_bayes_expected_returns_stat_bernoulli,
    conditioned_expected_return_diff_mean_stat_bernoulli,
    conditioned_expected_return_diff_std_stat_bernoulli,
    num_eval_runs,
    use_absolute_value
):

    def colorFader(c1,c2,mix=0): 
        # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1=np.array(mpl.colors.to_rgb(c1))
        c2=np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

    metaRL_expected_returns_stat_bernoulli = np.array(metaRL_expected_returns_stat_bernoulli)
    generative_bayes_expected_returns_stat_bernoulli = np.array(generative_bayes_expected_returns_stat_bernoulli)
    generative_expected_return_diff_mean_stat_bernoulli = np.array(generative_expected_return_diff_mean_stat_bernoulli)
    generative_expected_return_diff_std_stat_bernoulli = np.array(generative_expected_return_diff_std_stat_bernoulli)
    conditioned_bayes_expected_returns_stat_bernoulli = np.array(conditioned_bayes_expected_returns_stat_bernoulli)
    conditioned_expected_return_diff_mean_stat_bernoulli = np.array(conditioned_expected_return_diff_mean_stat_bernoulli)
    conditioned_expected_return_diff_std_stat_bernoulli = np.array(conditioned_expected_return_diff_std_stat_bernoulli)
    
    # correct the first epoch_id (-1) to be 0
    eval_epoch_ids[0] = 0

    # use absolute value
    if use_absolute_value:
        generative_expected_return_diff_mean_stat_bernoulli = np.abs(generative_expected_return_diff_mean_stat_bernoulli)
        conditioned_expected_return_diff_mean_stat_bernoulli = np.abs(conditioned_expected_return_diff_mean_stat_bernoulli)

    c1 = 'blue'
    c2 = 'red'

    fig, axs = plt.subplots(
        nrows=2, ncols=2, 
        figsize=(16, 12),
        dpi=300
    )
    fig.suptitle(
        f"Expected return comparison"
    )
    ax = axs[0,0]
    ax.set_title('Conditioned return diff')
    for eval_epoch in range(conditioned_expected_return_diff_mean_stat_bernoulli.shape[0]):
        conditioned_expected_return_diff_mean = conditioned_expected_return_diff_mean_stat_bernoulli[eval_epoch]
        conditioned_expected_return_diff_std = conditioned_expected_return_diff_std_stat_bernoulli[eval_epoch]
        ax.plot(
            np.arange(len(conditioned_expected_return_diff_mean)),
            conditioned_expected_return_diff_mean,
            color=colorFader(c1, c2, eval_epoch/conditioned_expected_return_diff_mean_stat_bernoulli.shape[0])
        )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Expected return diff")

    ax = axs[0,1]
    ax.set_title('Conditioned return diff evolution')
    ax.plot(eval_epoch_ids, conditioned_expected_return_diff_mean_stat_bernoulli[:, -1])
    for eval_epoch in range(conditioned_expected_return_diff_mean_stat_bernoulli.shape[0]):
        conditioned_expected_return_diff_mean = conditioned_expected_return_diff_mean_stat_bernoulli[eval_epoch]
        eval_epoch_id = eval_epoch_ids[eval_epoch]
        ax.scatter(
            eval_epoch_id,
            conditioned_expected_return_diff_mean[-1],
            color=colorFader(c1, c2, eval_epoch/conditioned_expected_return_diff_mean_stat_bernoulli.shape[0])
        )
    ax.plot(
        eval_epoch_ids,
        conditioned_expected_return_diff_mean_stat_bernoulli[:, -1],
        color='k'
    )
    ax.set_xlabel("Epoch")
    ax.set_xscale('symlog')
    ax.set_ylabel("Expected return diff")

    ax = axs[1,0]
    ax.set_title('Generative return diff')
    for eval_epoch in range(generative_expected_return_diff_mean_stat_bernoulli.shape[0]):
        generative_expected_return_diff_mean = generative_expected_return_diff_mean_stat_bernoulli[eval_epoch]
        generative_expected_return_diff_std = generative_expected_return_diff_std_stat_bernoulli[eval_epoch]
        ax.plot(
            np.arange(len(generative_expected_return_diff_mean)),
            generative_expected_return_diff_mean,
            color=colorFader(c1, c2, eval_epoch/generative_expected_return_diff_mean_stat_bernoulli.shape[0])
        )
        # ax.errorbar(
        #     np.arange(len(generative_expected_return_diff_mean)),
        #     generative_expected_return_diff_mean,
        #     yerr=generative_expected_return_diff_std/np.sqrt(num_eval_runs),
        #     color=colorFader(c1, c2, eval_epoch/conditioned_expected_return_diff_mean_stat_bernoulli.shape[0])
        # )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Expected return diff")

    ax = axs[1,1]
    ax.set_title('Generative return diff evolution')
    ax.plot(eval_epoch_ids, generative_expected_return_diff_mean_stat_bernoulli[:, -1])
    for eval_epoch in range(generative_expected_return_diff_mean_stat_bernoulli.shape[0]):
        generative_expected_return_diff_mean = generative_expected_return_diff_mean_stat_bernoulli[eval_epoch]
        eval_epoch_id = eval_epoch_ids[eval_epoch]
        ax.scatter(
            eval_epoch_id,
            generative_expected_return_diff_mean[-1],
            color=colorFader(c1, c2, eval_epoch/generative_expected_return_diff_mean_stat_bernoulli.shape[0])
        )
    ax.plot(
        eval_epoch_ids,
        generative_expected_return_diff_mean_stat_bernoulli[:, -1],
        color='k'
    )
    ax.set_xlabel("Epoch")
    ax.set_xscale('symlog')
    ax.set_ylabel("Expected return diff")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eval_expected_return_comparison.png'))
