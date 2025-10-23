import random

import numpy as np
import gymnasium as gym

from utils import helpers as utl


###############################################################################################
# Bernoulli bandit
###############################################################################################
# Gittins index for bernoulli bandit
class BernoulliGittinsAgent():
    def __init__(
        self, 
        lookahead_window, 
        discount_factor,
        biased_beta_prior
        ):
    
        self.lookahead_window = lookahead_window
        self.discount_factor = discount_factor
        self.biased_beta_prior = biased_beta_prior

        self.get_idx = None
        
        self.ub = 1 ## upper bound for index
        self.lb = 0 ## lower bound for index
        self.eps = 0.01 ## epsilon for index calculations
    
    def init_actions(self, n_actions):
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._total_pulls = 0

        if self.biased_beta_prior:
            self._successes[0] += 1  # beta_prior_a0=[2,1]
            self._failures[1] += 1  # beta_prior_a1=[1,2]

        if self.get_idx is None:
            self.get_idx = np.zeros(len(self._successes))
            for arm in range(n_actions):
                p = self._successes[arm] + 1
                q = self._failures[arm] + 1
                self.get_idx[arm] = self.calc_gittins_index(p,q)
        
    def get_v(self,p,q,lamb):

        L = self.lookahead_window
        beta = self.discount_factor

        v = np.zeros([L+1,L+1])

        p_ = np.arange(0,L+1)
        q_ = L-p_
        v[p_,q_] = beta**(L-1) / (1-beta) * np.maximum((p+p_) / (p+p_+q+q_)-lamb,0)

        for i in range(L-1,-1,-1):
            p_ = np.arange(0,i+1)
            q_ = i - p_
            v[p_,q_] = (p + p_) / (p + p_ + q + q_) - lamb + \
                       (p + p_) / (p + p_ + q + q_) * beta *  v[p_+1,q_] + \
                       (q + q_) / (p + p_ + q + q_) * beta *  v[p_,q_+1]
            v[p_,q_] = np.maximum(v[p_,q_],0)

        return v[0,0]

    def calc_gittins_index(self,p,q):
        if p <0 or q<0:
            print(f'WARNING: negative p or q: p {p}, q {q}')

        up = self.ub
        lb = self.lb
        eps = self.eps
        while up-lb>eps:
            lambd = (up+lb) / 2
            v = self.get_v(p,q,lambd)
            if v>0:
                lb = lambd
            else:
                up = lambd

        return (up+lb) * 0.5

    def get_action(self):
        best_action = np.argmax(self.get_idx)
        return best_action
          
    def update(self, action, reward):
        self._total_pulls += 1
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1

        ## smoothing agents that doesn't have successes or failures    
        p = self._successes[action] + 1
        q = self._failures[action] + 1
        
        ## update gittins index 
        self.get_idx[action] = self.calc_gittins_index(p,q)
            
    @property
    def name(self):
        return self.__class__.__name__ + "(Gittins={})".format(self.discount_factor)


def rollout_one_episode_bernoulli_gittins_agent(
    env,
    bernoulli_gittins_agent,
    arms=2
):
    '''
    rollout gitiins index agent in the given environment    
    '''

    bayes_states = []
    gittins_indices = []
    actions = []
    rewards = []
    
    # reset the env
    curr_state_dict, info = env.reset()
    # initialize the agent
    bernoulli_gittins_agent.init_actions(n_actions=2)

    # rollout
    done = False
    while not done:
        bayes_state = np.concatenate(
            [bernoulli_gittins_agent._successes, 
             bernoulli_gittins_agent._failures]
        )
        bayes_states.append(bayes_state)
        # NOTE: (ct_reward, ct_unreward) here

        gittins_indices.append(
            [bernoulli_gittins_agent.get_idx[i] for i in range(arms)])
        
        action = bernoulli_gittins_agent.get_action()
        actions.append(action)

        next_state_dict, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        # update for next step
        bernoulli_gittins_agent.update(action=action, reward=reward)
        
        # update if the environment is done
        done = terminated or truncated
    env.close()

    bayes_states = np.array(bayes_states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    gittins_indices = np.array(gittins_indices)

    return info, bayes_states, actions, rewards, gittins_indices


def rollout_one_episode_bernoulli_gittins_agent_given_p_bandits(
    p_bandits,
    total_trials,
    biased_beta_prior,
    arms=2
):
    '''
    rollout gitiins index agent in the given environment    
    '''

    bayes_states = []
    gittins_indices = []
    actions = []
    rewards = []
    
    # initialize the agent
    bernoulli_gittins_agent = BernoulliGittinsAgent(
        lookahead_window=total_trials, 
        discount_factor=0.95,
        biased_beta_prior=biased_beta_prior
    )
    bernoulli_gittins_agent.init_actions(n_actions=2)

    # rollout
    for trial in range(total_trials):
        bayes_state = np.concatenate(
            [bernoulli_gittins_agent._failures, bernoulli_gittins_agent._successes])
        bayes_states.append(bayes_state)
        # NOTE: (ct_unreward, ct_reward)

        gittins_indices.append(
            [bernoulli_gittins_agent.get_idx[i] for i in range(arms)])
        
        action = bernoulli_gittins_agent.get_action()
        actions.append(action)

        if random.uniform(0, 1) < p_bandits[action]:
            reward = 1
        else:
            reward = 0
        rewards.append(reward)

        # update for next step
        bernoulli_gittins_agent.update(action=action, reward=reward)

    bayes_states = np.array(bayes_states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    gittins_indices = np.array(gittins_indices)

    return bayes_states, actions, rewards, gittins_indices


def get_expected_regret_one_run_bernoulli_bandit(
    test_env,
    actions
):
    '''
    calculate (empirical) regret for a given run
    '''

    optimal_value = np.max(test_env.unwrapped.p_bandits)

    regret_per_timestep = []
    for timestep in range(len(actions)):
        gap = optimal_value - test_env.unwrapped.p_bandits[actions[timestep]]
        if timestep == 0:
            regret_per_timestep.append(gap)
        else:
            regret_per_timestep.append(gap + regret_per_timestep[timestep-1])
    
    regret_per_timestep = np.array(regret_per_timestep)
    
    return regret_per_timestep


def cumulative_expected_regret_bernoulli_bandit_Gittins_agent(
    env_name,
    gittins_look_ahead_window=20,
    gittins_discounting=0.95,
    biased_beta_prior=False,
    num_test_envs=10,
    num_runs_per_test_env=5,
    total_trials=20
):
    '''
    calculate cumulative expected regret 
    for a gittins index agent
    in stationary multi-armed bernoulli bandit env
    '''

    regret_per_timestep_per_run = []
    for test_env_id in range(num_test_envs):
        test_env = gym.make(
            f'environments.bandit:{env_name}',
            total_trials=total_trials
        )
        for run_id in range(num_runs_per_test_env):
            gittins_agent = BernoulliGittinsAgent(
                lookahead_window=gittins_look_ahead_window, 
                discount_factor=gittins_discounting,
                biased_beta_prior=biased_beta_prior
            )
            _, _, actions, _, _ = rollout_one_episode_bernoulli_gittins_agent(
                test_env, 
                gittins_agent
            )

            regret_per_timestep = get_expected_regret_one_run_bernoulli_bandit(
                test_env, actions)
            regret_per_timestep_per_run.append(regret_per_timestep)
        
    regret_per_timestep_per_run = np.array(regret_per_timestep_per_run)
    cumulative_expected_regret_mean = np.mean(regret_per_timestep_per_run, axis=0)
    cumulative_expected_regret_std = np.std(regret_per_timestep_per_run, axis=0)
    
    return cumulative_expected_regret_mean, cumulative_expected_regret_std



###############################################################################################
# Oracle bandit
###############################################################################################
class OracleBanditSolver():
    def __init__(
        self,
        num_bandits=11
    ):
        self.num_bandits = num_bandits
        # uniform distribution over all arms
        self.state = np.full(
            (num_bandits-1, ),
            1.0/ (num_bandits-1)
        )
        self.oracle_chosen = False

    def get_action_from_state(self, state):

        if np.max(state) > 0.499:
            best_action = np.argmax(state)    
        else:
            best_action = self.num_bandits-1

        return best_action

    def get_action(self):
        if self.oracle_chosen:  # if the oracle arm has been chosen
            best_action = np.argmax(self.state)

        else:  # if the oracle arm hasn't been chosen yet
            best_action = self.num_bandits-1

        return best_action

    def update(self, action, reward):
        if self.oracle_chosen:  # if the oracle arm has been chosen
            pass  # keep the state as it is

        else:  # if the oracle arm hasn't been chosen yet
            if action == self.num_bandits-1:  # if choose the oracle arm
                self.state = np.zeros((self.num_bandits-1, ))
                self.state[int(round(reward*10-1))] = 1  # update the state to the chosen arm
                self.oracle_chosen = True  # mark the oracle arm as chosen
            else:
                if reward == 5:
                    self.state = np.zeros((self.num_bandits-1, ))
                    self.state[action] = 1
                elif reward == 1:
                    prev_nontarget = np.where(self.state==0)[0].tolist()
                    total_nontarget = prev_nontarget + [action]
                    self.state = np.full(
                        (self.num_bandits-1, ),
                        1.0/ (self.num_bandits-1-len(total_nontarget))
                    )
                    self.state[total_nontarget] = 0
                else:
                    raise ValueError("Invalid reward value")


def rollout_one_episode_oracle_bandit_solver_given_r_bandits(
    r_bandits,
    total_trials,
    num_bandits=11
):
    '''
    rollout oracle bandit solver in the given environment    
    '''

    bayes_states = []
    actions = []
    rewards = []
    
    # initialize the agent
    oracle_solver = OracleBanditSolver(num_bandits=num_bandits)

    # rollout
    for trial in range(total_trials):
        bayes_state = oracle_solver.state
        bayes_states.append(bayes_state)
        # NOTE: (n_a_11, r_a_11)
        
        action = oracle_solver.get_action()
        actions.append(action)

        reward = r_bandits[action]
        rewards.append(reward)

        # update for next step
        oracle_solver.update(action=action, reward=reward)

    bayes_states = np.array(bayes_states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    return bayes_states, actions, rewards



###############################################################################################
# Discrete Markovian Bernoulli Bandit
###############################################################################################
# for dynamic multi-armed bandit environments
class OptimalDisMarkovianBanditSolver:
    """
    Optimal solver for the discrete Markovian multi-armed bandit problem.
    Implements value iteration on the belief-MDP.
    
    For the 2-arm, 2-state case, we represent beliefs as:
        b = (b₀(0), b₁(0))
    where bₐ(s) is the probability that arm a is in state s.
    Note that b₀(1) = 1 - b₀(0) and b₁(1) = 1 - b₁(0).
    """
    
    def __init__(
        self, 
        num_arms, 
        num_states, 
        reward_states, 
        transition_matrices, 
        discount_factor=0.95, 
        grid_size=50, 
        max_iterations=100, 
        tolerance=1e-4
    ):
        """
        Initialize the optimal solver.
        
        Args:
            num_arms (int): Number of arms
            num_states (int): Number of states per arm
            reward_states (numpy array): Reward probabilities for each state of each arm.
                                         Shape: (num_arms, num_states)
            transition_matrices (numpy array): Transition matrices for each arm.
                                              Shape: (num_arms, num_states, num_states)
            discount_factor (float): Discount factor for future rewards
            grid_size (int): Size of grid for discretizing the belief space
            max_iterations (int): Maximum number of value iteration steps
            tolerance (float): Convergence tolerance for value iteration
        """
        assert num_arms == 2 and num_states == 2, "This implementation supports only 2 arms with 2 states each"
        
        self.num_arms = num_arms
        self.num_states = num_states
        self.reward_states = reward_states
        self.transition_matrices = transition_matrices
        self.discount_factor = discount_factor
        self.grid_size = grid_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Discretize the belief space for value iteration
        self.b0_grid = np.linspace(0, 1, grid_size)
        self.b1_grid = np.linspace(0, 1, grid_size)
        
        # Initialize value function and policy
        self.value_function = np.zeros((grid_size, grid_size))
        self.policy = np.zeros((grid_size, grid_size), dtype=int)
        self.q_values = np.zeros((grid_size, grid_size, num_arms))
        
        # Initialize current belief
        self.current_belief = np.ones((num_arms, num_states)) / num_states
        
        # Run value iteration to find the optimal policy
        self._value_iteration()

    def _expected_immediate_reward(self, belief, arm):
        """
        Calculate the expected immediate reward for pulling an arm given a belief state.
        
        Args:
            belief (numpy array): Belief state for both arms. Shape: (num_arms, num_states)
            arm (int): The arm to pull
            
        Returns:
            float: Expected immediate reward
        """
        return np.sum(belief[arm] * self.reward_states[arm])
    
    def _update_belief_simulate(self, belief, arm, reward):
        """
        Simulate a belief update if arm is pulled and reward is observed.
        Does not modify the current belief state.
        
        Args:
            belief (numpy array): Current belief state. Shape: (num_arms, num_states)
            arm (int): The arm pulled
            reward (int): The observed reward (0 or 1)
            
        Returns:
            numpy array: Updated belief state
        """
        updated_belief = belief.copy()
        
        # Step 1: Bayesian update for the pulled arm based on observation
        likelihood = np.zeros(self.num_states)
        for s in range(self.num_states):
            if reward == 1:
                likelihood[s] = self.reward_states[arm, s]
            else:
                likelihood[s] = 1 - self.reward_states[arm, s]
        
        posterior = likelihood * updated_belief[arm]
        if np.sum(posterior) > 0:
            posterior = posterior / np.sum(posterior)
        updated_belief[arm] = posterior
        
        # Step 2: Transition update for all arms
        for a in range(self.num_arms):
            updated_belief[a] = np.dot(updated_belief[a], self.transition_matrices[a])
        
        return updated_belief
    
    def _probability_of_reward(self, belief, arm, reward):
        """
        Calculate P(reward | belief, arm) - the probability of observing a reward
        given a belief state and an action.
        
        Args:
            belief (numpy array): Belief state. Shape: (num_arms, num_states)
            arm (int): The arm to pull
            reward (int): The possible reward (0 or 1)
            
        Returns:
            float: The probability of observing the reward
        """
        prob = 0
        for s in range(self.num_states):
            # P(reward | state s) * P(state s)
            if reward == 1:
                prob += self.reward_states[arm, s] * belief[arm, s]
            else:
                prob += (1 - self.reward_states[arm, s]) * belief[arm, s]
        return prob
    
    def _interpolate_value(self, belief):
        """
        Interpolate the value function for a belief state that may not be on the grid.
        Uses bilinear interpolation.
        
        Args:
            belief (numpy array): Belief state. Shape: (num_arms, num_states)
            
        Returns:
            float: Interpolated value
        """
        b0 = belief[0, 0]  # Prob that arm 0 is in state 0
        b1 = belief[1, 0]  # Prob that arm 1 is in state 0
        
        # Find the grid indices for interpolation
        i0 = max(0, min(self.grid_size - 2, int(b0 * (self.grid_size - 1))))
        i1 = max(0, min(self.grid_size - 2, int(b1 * (self.grid_size - 1))))
        
        # Calculate interpolation weights
        w0 = b0 * (self.grid_size - 1) - i0
        w1 = b1 * (self.grid_size - 1) - i1
        
        # Bilinear interpolation
        v00 = self.value_function[i0, i1]
        v01 = self.value_function[i0, i1+1]
        v10 = self.value_function[i0+1, i1]
        v11 = self.value_function[i0+1, i1+1]
        
        return (1-w0)*(1-w1)*v00 + (1-w0)*w1*v01 + w0*(1-w1)*v10 + w0*w1*v11
    
    def _bellman_update(self, belief):
        """
        Perform a Bellman update for a given belief state.
        
        Args:
            belief (numpy array): Belief state. Shape: (num_arms, num_states)
            
        Returns:
            tuple: (new_value, best_action)
        """
        q_values = np.zeros(self.num_arms)
        
        for arm in range(self.num_arms):
            # Expected immediate reward
            immediate_reward = self._expected_immediate_reward(belief, arm)
            
            # Expected future reward
            future_reward = 0
            for reward in [0, 1]:
                # Calculate P(reward | belief, arm)
                prob_reward = self._probability_of_reward(belief, arm, reward)
                
                if prob_reward > 0:
                    # Calculate the next belief state
                    next_belief = self._update_belief_simulate(belief, arm, reward)
                    
                    # Value of the next belief state (interpolated)
                    next_value = self._interpolate_value(next_belief)
                    
                    # Add to expected future reward
                    future_reward += prob_reward * next_value
            
            # Q(belief, arm) = immediate_reward + γ * E[V(next_belief)]
            q_values[arm] = immediate_reward + self.discount_factor * future_reward
        
        # Find the best action
        best_action = np.argmax(q_values)
        max_q_value = q_values[best_action]
        
        return max_q_value, best_action, q_values
    
    def _value_iteration(self):
        """
        Perform value iteration on the discretized belief space.
        Updates the value function and policy.
        """
        for iteration in range(self.max_iterations):
            max_diff = 0
            
            # For each belief point in the grid
            for i, b0 in enumerate(self.b0_grid):
                for j, b1 in enumerate(self.b1_grid):
                    # Construct the belief state
                    belief = np.array([[b0, 1-b0], [b1, 1-b1]])
                    
                    # Perform Bellman update
                    new_value, best_action, q_values = self._bellman_update(belief)
                    
                    # Update max difference for convergence check
                    diff = abs(new_value - self.value_function[i, j])
                    if diff > max_diff:
                        max_diff = diff
                    
                    # Update value function and policy
                    self.value_function[i, j] = new_value
                    self.policy[i, j] = best_action
                    self.q_values[i, j] = q_values
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration+1}, max diff: {max_diff}")
            
            # Check for convergence
            if max_diff < self.tolerance:
                print(f"Value iteration converged after {iteration+1} iterations.")
                break
        
        if iteration == self.max_iterations - 1:
            print(f"Value iteration reached maximum iterations ({self.max_iterations}).")

    def update_belief(self, arm, reward):
        """
        Update the current belief state based on action and reward.
        
        Args:
            arm (int): The arm that was pulled
            reward (int): The observed reward (0 or 1)
        """
        self.current_belief = self._update_belief_simulate(self.current_belief, arm, reward)

    def choose_action(self):
        """
        Choose the optimal action according to the learned policy.
        
        Returns:
            int: The arm to pull
        """
        # Get the indices for the current belief
        b0 = self.current_belief[0, 0]
        b1 = self.current_belief[1, 0]
        
        i0 = max(0, min(self.grid_size - 1, int(b0 * (self.grid_size - 1) + 0.5)))
        i1 = max(0, min(self.grid_size - 1, int(b1 * (self.grid_size - 1) + 0.5)))
        
        return self.policy[i0, i1]
    
    def get_action_from_belief_state(self, belief_state):
        """
        Choose the optimal action depending on the given belief state,
        according to the learned policy.

        note: input (b_0(s=0), b_1(s=0))
        
        Returns:
            int: The arm to pull
        """
        # Get the indices for the current belief
        b0, b1 = belief_state
        
        i0 = max(0, min(self.grid_size - 1, int(b0 * (self.grid_size - 1) + 0.5)))
        i1 = max(0, min(self.grid_size - 1, int(b1 * (self.grid_size - 1) + 0.5)))
        
        return self.policy[i0, i1]

    def reset(self):
        """Reset the current belief to uniform distribution."""
        self.current_belief = np.ones((self.num_arms, self.num_states)) / self.num_states


def rollout_one_episode_dis_markovian_bandit_solver_given_env(
    dis_markovian_bandit_solver,
    env,
    total_trials
):
    '''
    rollout oracle bandit solver in the given environment    
    '''

    bayes_states = []
    actions = []
    rewards = []
    reward_probs_t = []

    curr_state_dict, info = env.reset()
    dis_markovian_bandit_solver.reset()
    
    # rollout
    for trial in range(total_trials):

        reward_probs_t.append(info['reward_probs_t'])
        
        bayes_state = [
            dis_markovian_bandit_solver.current_belief[0, 0], 
            dis_markovian_bandit_solver.current_belief[1, 0]
        ]
        bayes_states.append(bayes_state)
        
        action = dis_markovian_bandit_solver.choose_action()
        actions.append(action)

        next_state_dict, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        # update for next step
        dis_markovian_bandit_solver.update_belief(arm=action, reward=reward)

    bayes_states = np.array(bayes_states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    reward_probs_t = np.array(reward_probs_t)

    return bayes_states, actions, rewards, reward_probs_t


###############################################################################################
# Tiger
###############################################################################################
import numpy as np
from typing import Tuple


class TigerPOMDPSolver:
    """
    Solver for the Tiger Problem POMDP using value iteration over the belief space.
    """
    def __init__(
        self,
        tiger_reward: float = -100.0,
        treasure_reward: float = 10.0,
        listen_reward: float = -1.0,
        obs_accuracy: float = 0.85,
        tiger_stay_prob: float = 0.8,
        gamma: float = 0.95,
        grid_size: int = 10001,
        max_iterations: int = 200,
        tolerance: float = 1e-5
    ):
        """
        Initialize the Dynamic Tiger POMDP solver.
        
        Args:
            listen_reward: Reward for the listen action
            tiger_reward: Reward for opening the door with the tiger
            treasure_reward: Reward for opening the door with the treasure
            obs_accuracy: Probability of correct observation when listening
            tiger_stay_prob: Probability the tiger stays in the same location
            gamma: Discount factor
            grid_size: Number of points in the belief space discretization
        """
        # POMDP parameters
        self.tiger_reward = tiger_reward
        self.treasure_reward = treasure_reward
        self.listen_reward = listen_reward
        self.obs_accuracy = obs_accuracy
        self.tiger_stay_prob = tiger_stay_prob
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Define transition dynamics: P(s'|s)
        self.transition_matrix = np.array([
            [tiger_stay_prob, 1 - tiger_stay_prob],  # Tiger at the left
            [1 - tiger_stay_prob, tiger_stay_prob]   # Tiger at the right
        ])

        # Action mappings
        self.actions = {2: "listen", 0: "open-left", 1: "open-right"}
        
        # Discretization of belief space
        self.grid_size = grid_size
        self.belief_grid = np.linspace(0, 1, grid_size)
        
        # Value function and policy
        self.value_function = np.zeros(grid_size)
        self.policy = np.zeros(grid_size, dtype=int)
        self.q_values = np.zeros((grid_size, 3))  # Q(b,a) for all belief points and actions
        
        # Convergence diagnostics
        self.convergence_history = []

        # Initialize current belief
        self.current_belief = 0.5  # Start with uniform belief

        # Run value iteration to find the optimal policy
        self.value_iteration()
        
    def update_belief_with_observation(self, b: float, obs: int) -> float:
        """
        Update belief state using Bayes rule after an observation.
        
        Args:
            b: Current belief (probability tiger is on left)
            obs: Observation (0=hear-left, 1=hear-right)
            
        Returns:
            Updated belief state
        """
        if np.all(obs == np.array([1,0])):  # Heard tiger on left
            p_obs_given_tiger_left = self.obs_accuracy
            p_obs_given_tiger_right = 1 - self.obs_accuracy
            # Bayes rule: p(tiger_left | obs) = p(obs | tiger_left) * p(tiger_left) / p(obs)
            p_obs = (p_obs_given_tiger_left * b + p_obs_given_tiger_right * (1 - b))
        
        elif np.all(obs == np.array([0,1])):  # Heard tiger on right
            p_obs_given_tiger_left = 1 - self.obs_accuracy
            p_obs_given_tiger_right = self.obs_accuracy
            # Bayes rule: p(tiger_left | obs) = p(obs | tiger_left) * p(tiger_left) / p(obs)
            p_obs = (p_obs_given_tiger_left * b + p_obs_given_tiger_right * (1 - b))
        
        else:
            return 0.5    
        
        if p_obs == 0:  # Avoid division by zero
            return 0.5  # Reset to uniform belief
            
        updated_belief = (p_obs_given_tiger_left * b) / p_obs
        return updated_belief
    
    def update_belief_with_transition(self, b: float) -> float:
        """
        Update belief state to account for tiger movement.
        
        Args:
            b: Current belief (probability tiger is on left)
            
        Returns:
            Updated belief state after transition
        """
        # Apply transition dynamics to belief state
        # p(s_t+1 = left) = p(s_t = left) * p(stay left) + p(s_t = right) * p(move left)
        new_belief_left = (b * self.transition_matrix[0, 0] + 
                          (1 - b) * self.transition_matrix[1, 0])
        
        return new_belief_left
    
    def update_current_belief(self, action, observation):
        if action == 2:  # Listen
            # Update belief based on observation first
            self.current_belief = self.update_belief_with_observation(self.current_belief, observation)
            # Then update for transition (as tiger moves after listen action)
            self.current_belief = self.update_belief_with_transition(self.current_belief)

        # For open door actions, no update needed (episode ends)
        
    def q_value_listen(self, b: float, value_function: np.ndarray) -> float:
        """
        Compute Q-value of the listen action from belief state b using the Bellman equation.
        
        Q(b, listen) = R(listen) + γ * ∑ P(o|b) * V(τ(b,listen,o))
        
        Args:
            b: Belief state (probability tiger is on left)
            value_function: Current value function
            
        Returns:
            Q-value of listen action
        """
        # Immediate reward for listening
        immediate_reward = self.listen_reward
        
        # Probability of observations given CURRENT belief (before transition)
        p_hear_left = (b * self.obs_accuracy + (1 - b) * (1 - self.obs_accuracy))
        p_hear_right = (b * (1 - self.obs_accuracy) + (1 - b) * self.obs_accuracy)
        
        # Updated beliefs after observations
        b_hear_left = self.update_belief_with_observation(b, np.array([1,0]))
        b_hear_right = self.update_belief_with_observation(b, np.array([0,1]))
        
        # THEN apply transition to these updated beliefs
        b_hear_left_after_transition = self.update_belief_with_transition(b_hear_left)
        b_hear_right_after_transition = self.update_belief_with_transition(b_hear_right)
        
        # Find indices in discretized belief space
        idx_hear_left = np.abs(self.belief_grid - b_hear_left_after_transition).argmin()
        idx_hear_right = np.abs(self.belief_grid - b_hear_right_after_transition).argmin()
        
        # Expected future value
        expected_future_value = (p_hear_left * value_function[idx_hear_left] + 
                            p_hear_right * value_function[idx_hear_right])
        
        return immediate_reward + self.gamma * expected_future_value
    
    def q_value_open_left(self, b: float) -> float:
        """
        Compute Q-value of opening left door from belief state b.
        
        Q(b, open-left) = b * R(tiger) + (1-b) * R(treasure)
        
        Args:
            b: Belief state (probability tiger is on left)
            
        Returns:
            Q-value of open-left action
        """
        # If tiger is on left (with probability b), get tiger_reward
        # If tiger is on right (with probability 1-b), get treasure_reward
        return b * self.tiger_reward + (1 - b) * self.treasure_reward
    
    def q_value_open_right(self, b: float) -> float:
        """
        Compute Q-value of opening right door from belief state b.
        
        Q(b, open-right) = (1-b) * R(tiger) + b * R(treasure)
        
        Args:
            b: Belief state (probability tiger is on left)
            
        Returns:
            Q-value of open-right action
        """
        # If tiger is on right (with probability 1-b), get tiger_reward
        # If tiger is on left (with probability b), get treasure_reward
        return (1 - b) * self.tiger_reward + b * self.treasure_reward
    
    def bellman_update(self, b: float, value_function: np.ndarray) -> Tuple[float, int, np.ndarray]:
        """
        Apply the Bellman optimality equation to compute the optimal value and action.
        
        V(b) = max_a Q(b,a)
        
        Args:
            b: Belief state
            value_function: Current value function
            
        Returns:
            Tuple of (optimal value, optimal action, q-values)
        """
        # Compute Q-values for all actions
        q_open_left = self.q_value_open_left(b)
        q_open_right = self.q_value_open_right(b)
        q_listen = self.q_value_listen(b, value_function)
        
        q_values = np.array([q_open_left, q_open_right, q_listen])
        
        # Apply Bellman optimality equation: V(b) = max_a Q(b,a)
        optimal_value = np.max(q_values)
        optimal_action = np.argmax(q_values)
        
        return optimal_value, optimal_action, q_values
    
    def value_iteration(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform value iteration to find optimal value function and policy.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple of (value_function, policy, q_values)
        """
        # Initialize value function
        value_function = np.zeros(self.grid_size)
        policy = np.zeros(self.grid_size, dtype=int)
        q_values = np.zeros((self.grid_size, 3))
        
        # Reset convergence history
        self.convergence_history = []
        
        for iter_num in range(self.max_iterations):
            prev_value = np.copy(value_function)
            
            for i, b in enumerate(self.belief_grid):
                # Apply Bellman update
                value_function[i], policy[i], q_values[i] = self.bellman_update(b, prev_value)
            
            # Track maximum change for convergence check
            max_diff = np.max(np.abs(value_function - prev_value))
            self.convergence_history.append(max_diff)
            
            # Check convergence
            if max_diff < self.tolerance:
                print(f"Value iteration converged after {iter_num+1} iterations")
                break
                
        self.value_function = value_function
        self.policy = policy
        self.q_values = q_values
        
    def get_optimal_action(self, belief: float) -> int:
        """Get optimal action for a given belief state."""
        idx = np.abs(self.belief_grid - belief).argmin()
        return int(self.policy[idx])
    
    def choose_action(self):
        """Choose action based on current belief state."""
        action = self.get_optimal_action(self.current_belief)
        return action
    
    def reset(self) -> None:
        """Reset the current belief to uniform belief."""
        self.current_belief = 0.5


def rollout_one_episode_tiger_solver_given_env(
    tiger_solver,
    env,
    args,
    total_trials
):
    '''
    rollout oracle bandit solver in the given environment    
    '''
    rewards_t = []
    bayes_states = []
    observations = []
    actions = []
    rewards = []
    

    curr_state_dict, info = env.reset()
    curr_state = utl.get_states_from_state_dicts(
        curr_state_dict, args, args.time_as_state
    )
    observations.append(curr_state)
    tiger_solver.reset()
    
    # rollout
    for trial in range(total_trials):
        
        rewards_t.append(info['rewards_t'])

        bayes_states.append(tiger_solver.current_belief)
        
        action = tiger_solver.choose_action()
        actions.append(action)

        next_state_dict, reward, terminated, truncated, info = env.step(action)
        next_state = utl.get_states_from_state_dicts(
            next_state_dict, args, args.time_as_state
        )
        observations.append(next_state)
        rewards.append(reward)

        # update for next step
        tiger_solver.update_current_belief(action, next_state)

        if info['episode_terminated']:
            break

    bayes_states = np.array(bayes_states).reshape(-1, 1)
    actions = np.array(actions)
    rewards = np.array(rewards)
    observations = np.array(observations)
    rewards_t = np.array(rewards_t)

    return bayes_states, actions, rewards, observations, rewards_t


###########################################################################################
# LatentGoalCart Solver
###########################################################################################
from scipy.stats import norm

class LatentGoalCartPOMDPSolver:
    """Optimal POMDP solver using value iteration on belief space."""
    
    def __init__(self, env, belief_resolution=101, pos_resolution=41, gamma=0.95):
        self.env = env
        self.gamma = gamma
        
        # Discretize belief space (probability of goal 1)
        self.belief_points = np.linspace(0, 1, belief_resolution)
        self.n_beliefs = len(self.belief_points)
        
        # Discretize state space (just position now)
        self.pos_grid = np.linspace(-2, 2, pos_resolution)
        
        # Discretize action space for value iteration
        self.action_grid = np.linspace(-2, 2, 21)
        
        # Value function: V[belief_idx, pos_idx]
        self.V = np.zeros((self.n_beliefs, len(self.pos_grid)))
        self.policy = {}  # Store optimal actions
        
    def belief_update(self, belief, state, action, observed_reward):
        """Update belief given observed reward."""
        # Prior
        p_goal_1 = belief
        p_goal_2 = 1 - belief
        
        # Likelihoods
        r_1 = self.env.get_expected_reward(state, action, 0)
        r_2 = self.env.get_expected_reward(state, action, 1)
        
        likelihood_1 = norm.pdf(observed_reward, r_1, self.env.reward_noise_std)
        likelihood_2 = norm.pdf(observed_reward, r_2, self.env.reward_noise_std)
        
        # Posterior
        denominator = p_goal_1 * likelihood_1 + p_goal_2 * likelihood_2
        if denominator > 0:
            new_belief = (p_goal_1 * likelihood_1) / denominator
        else:
            new_belief = belief
            
        return np.clip(new_belief, 0, 1)
    
    def get_state_index(self, state):
        """Get nearest grid index for position."""
        pos = state[0] if isinstance(state, np.ndarray) else state
        return np.argmin(np.abs(self.pos_grid - pos))
    
    def get_belief_index(self, belief):
        """Get nearest belief grid index."""
        return np.argmin(np.abs(self.belief_points - belief))
    
    def expected_future_value(self, belief, state, action):
        """Compute expected future value after taking action and observing reward."""
        pos = state[0] if isinstance(state, np.ndarray) else state
        velocity = np.clip(action, -self.env.max_velocity, self.env.max_velocity)
        
        # Predict next state
        next_pos = pos + self.env.dt * velocity
        next_pos = np.clip(next_pos, -self.env.max_position, self.env.max_position)
        
        # Expected rewards for each goal
        r_1 = self.env.get_expected_reward(state, action, 0)
        r_2 = self.env.get_expected_reward(state, action, 1)
        
        # Compute expected value over possible observations
        # We'll use Gaussian quadrature for more accuracy
        n_samples = 10
        expected_value = 0.0
        
        # Standard Gaussian quadrature points and weights
        z_points = np.linspace(-3, 3, n_samples)
        weights = norm.pdf(z_points)
        weights = weights / weights.sum()
        
        for goal_idx in range(2):
            goal_prob = belief if goal_idx == 0 else (1 - belief)
            true_reward = [r_1, r_2][goal_idx]
            
            for i, z in enumerate(z_points):
                # Convert standard normal to reward observation
                reward_obs = true_reward + z * self.env.reward_noise_std
                
                # Update belief
                new_belief = self.belief_update(belief, state, action, reward_obs)
                
                # Get value at new belief and state
                belief_idx = self.get_belief_index(new_belief)
                pos_idx = self.get_state_index(next_pos)
                
                future_value = self.V[belief_idx, pos_idx]
                expected_value += goal_prob * weights[i] * future_value
        
        return expected_value
    
    def value_iteration(self, n_iterations=100, verbose=True):
        """Perform value iteration on belief-state space."""
        if verbose:
            print("Starting value iteration...")
        
        for iteration in range(n_iterations):
            V_new = np.zeros_like(self.V)
            
            # Iterate over all belief-state pairs
            for b_idx, belief in enumerate(self.belief_points):
                for p_idx, pos in enumerate(self.pos_grid):
                    state = np.array([pos])
                    
                    # Find optimal action
                    best_value = -np.inf
                    best_action = 0.0
                    
                    # Search over actions
                    for action in self.action_grid:
                        # Immediate expected reward
                        immediate_reward = (belief * self.env.get_expected_reward(state, action, 0) + 
                                          (1 - belief) * self.env.get_expected_reward(state, action, 1))
                        
                        # Future value
                        future_value = self.expected_future_value(belief, state, action)
                        
                        total_value = immediate_reward + self.gamma * future_value
                        
                        if total_value > best_value:
                            best_value = total_value
                            best_action = action
                    
                    V_new[b_idx, p_idx] = best_value
                    self.policy[(b_idx, p_idx)] = best_action
            
            # Check convergence
            max_diff = np.max(np.abs(V_new - self.V))
            self.V = V_new
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, Max value change: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
    
    def get_action(self, belief, state):
        """Get optimal action for given belief and state."""
        belief_idx = self.get_belief_index(belief)
        pos_idx = self.get_state_index(state)
        
        if (belief_idx, pos_idx) in self.policy:
            return self.policy[(belief_idx, pos_idx)]
        else:
            # use nearest neighbor
            return 0.0  # Default action if not found in policy
    

def train_latent_goal_cart_solver(solver):
    """Train the POMDP solver and evaluate performance."""
    # Create environment
    env = gym.make('environments.latent_goal_cart.latent_goal_cart:LatentGoalCart-v1')
    
    # Create and train solver
    # solver = LatentGoalCartPOMDPSolver(env, belief_resolution=51, pos_resolution=41, gamma=0.95)
    solver.value_iteration(n_iterations=50)
    
    return solver


class LatentGoalCartBeliefBasedAgent:
    """Agent that maintains belief and acts using the POMDP policy."""
    
    def __init__(self, solver):
        self.solver = solver
        self.reset()
    
    def reset(self):
        self.belief = 0.5  # Uniform prior
        self.history = {
            'beliefs': [self.belief], 
            'rewards': [], 
            'positions': [], 
            'actions': []
        }
    
    def act(self, state):
        action = self.solver.get_action(self.belief, state)
        self.history['positions'].append(state)
        self.history['actions'].append(action)
        return action
    
    def update(self, state, action, reward):
        self.belief = self.solver.belief_update(self.belief, state, action, reward)
        self.history['beliefs'].append(self.belief)
        self.history['rewards'].append(reward)


def rollout_one_episode_latent_goal_cart_solver_given_env(
    latent_goal_cart_solver,
    env,
    total_trials
):
    '''
    rollout oracle bandit solver in the given environment    
    '''
    goal_positions_t = []
    bayes_states = []
    observations = []
    actions = []
    rewards = []
    

    curr_state_dict, info = env.reset()
    curr_state = curr_state_dict['position'][0]
    observations.append(curr_state)
    latent_goal_cart_solver.reset()
    
    # rollout
    for trial in range(total_trials):

        goal_positions_t.append(info['goal_position_t'])

        bayes_states.append(latent_goal_cart_solver.belief)

        action = latent_goal_cart_solver.act(curr_state)
        actions.append(action)

        next_state_dict, reward, terminated, truncated, info = env.step(action)
        next_state = next_state_dict['position'][0]
        observations.append(next_state)
        rewards.append(reward)

        # update for next step
        latent_goal_cart_solver.update(curr_state, action, reward)

        curr_state = next_state


    bayes_states = np.array(bayes_states).reshape(-1, 1)
    actions = np.array(actions)
    rewards = np.array(rewards)
    observations = np.array(observations)
    goal_positions_t = np.array(goal_positions_t)

    return bayes_states, actions, rewards, observations, goal_positions_t