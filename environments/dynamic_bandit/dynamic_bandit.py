import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class DiscreteBernoulliMarkovianBanditEnv(gym.Env):
    """
    A dynamic multi-armed bandit task with discrete hidden states 
    governed by a Markov process.
    """

    def __init__(
        self, 
        num_arms=2, 
        reward_states=[[0.9, 0.1], [0.9, 0.1]], 
        transition_matrix=[
            [[0.9, 0.1], 
             [0.1, 0.9]],
            [[0.9, 0.1], 
             [0.1, 0.9]]
        ],
        total_trials=300
    ):
        super().__init__()

        # Define
        self.num_arms = num_arms
        self.reward_states = np.array(reward_states)
        self.transition_matrix = np.array(transition_matrix)
        self.max_episode_steps = total_trials

        # Check dimensions of the transition matrix
        assert self.transition_matrix.shape == (
            num_arms, len(self.reward_states), len(self.reward_states)), \
            "Transition matrix dimensions must match the number of reward states."

        # Observation space
        self.observation_space = spaces.Dict({
            "timestep": spaces.Box(low=0, high=self.max_episode_steps, dtype=np.int64),  # timestep
        })
        self.timestep = np.array([0])

        # Action space: choose one of the arms
        self.action_space = spaces.Discrete(self.num_arms)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _get_obs(self):
        return {"timestep": self.timestep}
    
    def _get_info(self):
        return {
            "current_states": np.copy(self.current_states),
            "reward_probs_t": [self.reward_states[0, self.current_states[0]], self.reward_states[1, self.current_states[1]]],
            "reward_states": self.reward_states,
            "transition_matrix": self.transition_matrix
        }

    def reset(self, seed=None, options={}):
        """
        The reset method will be called to initiate a new episode. 
        You may assume that the step method will not be called before reset has been called. 
        Moreover, reset should be called whenever a done signal has been issued.
        This should *NOT* automatically reset the task! Resetting the task is 
        handled in the wrapper.

        Args:
            seed (int): Random seed for reproducibility.

        Returns:
            observation (None): No initial observation.
            info (dict): Additional info (e.g., true hidden states).
        """
        super().reset(seed=seed)

        # reset timestep
        self.timestep = np.array([0])

        # Reset the hidden states uniformly for each arm
        self.current_states = np.array([
            random.choice(np.arange(len(self.reward_states[arm_id]))) 
            for arm_id in range(self.num_arms)
        ])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """
        Take an action (choose an arm) and return the outcome.

        Args:
            action (int): The index of the arm to pull.
        """
        # action should be type integer in [0, k_bandits-1]
        assert self.action_space.contains(action), "Invalid action."

        # time counter
        self.timestep += 1
        # An episode is done iff max_episode_steps is reached
        terminated = bool((self.timestep >= self.max_episode_steps))

        # Get the current hidden state of the chosen arm
        current_state_chosen_arm = self.current_states[action]

        # Generate a binary reward based on the reward probability
        reward = 0
        if random.uniform(0, 1) < self.reward_states[action][current_state_chosen_arm]:
            reward = 1

        # Update the hidden states of all arms based on the transition matrix
        for arm_id in range(self.num_arms):
            self.current_states[arm_id] = random.choices(
                np.arange(len(self.reward_states[arm_id])), 
                weights=self.transition_matrix[arm_id][self.current_states[arm_id]],
                k=1
            )[0]

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info


class DisBerMarkovBandit2ArmIndependent(DiscreteBernoulliMarkovianBanditEnv):
    def __init__(
        self, 
        num_arms=2, 
        reward_states=[[0.9, 0.1], [0.9, 0.1]], 
        transition_matrix=[
            [[0.9, 0.1], 
             [0.1, 0.9]],
            [[0.9, 0.1], 
             [0.1, 0.9]]
        ],
        total_trials=300
    ):
        DiscreteBernoulliMarkovianBanditEnv.__init__(
            self,
            num_arms=num_arms, 
            reward_states=reward_states, 
            transition_matrix=transition_matrix, 
            total_trials=total_trials
        )