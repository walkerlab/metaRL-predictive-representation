import random

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class LatentGoalCartEnv(gym.Env):
    """Cart environment with latent goal position (either -1 or +1).
    Action directly controls velocity."""
    
    def __init__(
        self, 
        goal_positions=[-1.0, 1.0], 
        reward_noise_std=0.1, 
        total_trials=30
    ):
        super().__init__()
        
        self.goal_positions = np.array(goal_positions)
        self.n_goals = len(goal_positions)
        self.reward_noise_std = reward_noise_std
        self.max_episode_steps = total_trials
        
        # Environment parameters
        self.dt = 0.1
        self.max_velocity = 1.0
        self.max_position = 2.0
        
        # Spaces
        self.observation_space = spaces.Dict({
            'position': spaces.Box(
                low=np.array([-self.max_position]),
                high=np.array([self.max_position]),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Box(
            low=-self.max_velocity,
            high=self.max_velocity,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize step counter
        self.timestep = np.array([0])
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _get_obs(self):
        return {
            'position': np.array([self.position], dtype=np.float32)
        }

    def _get_info(self):
        return {
            'goal_position': self.goal_position,
            'latent_goal_idx': self.latent_goal_idx,
            'goal_position_t': np.array([self.goal_position], dtype=np.float32), # for state machine analysis
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset step counter
        self.timestep = np.array([0])
        
        # Sample latent goal
        self.latent_goal_idx = random.randint(0, self.n_goals - 1)
        self.goal_position = self.goal_positions[self.latent_goal_idx]
        
        # Initialize position near origin
        self.position = random.uniform(-0.1, 0.1)

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # Increment timestep
        self.timestep += 1

        # An episode is done iff max_episode_steps is reached
        terminated = bool((self.timestep >= self.max_episode_steps))

        # Action directly sets velocity
        velocity = np.clip(action, -self.max_velocity, self.max_velocity)
        
        # Update position
        self.position += self.dt * velocity
        self.position = np.clip(self.position, -self.max_position, self.max_position)
        
        # Calculate reward with observation noise
        ## v0
        # true_reward = -((self.position - self.goal_position)**2) - 0.01 * (velocity**2)
        ## v1: fix for correctly penalizing large actions
        true_reward = -((self.position - self.goal_position)**2) - 0.01 * (action**2)
        
        observed_reward = true_reward + random.gauss(0, self.reward_noise_std)

        obs = self._get_obs()
        info = self._get_info()

        return obs, observed_reward, terminated, False, info
    
    def get_expected_reward(self, state, action, goal_idx):
        """Get expected reward for a given state, action, and goal."""
        pos = state[0] if isinstance(state, np.ndarray) else state
        goal_pos = self.goal_positions[goal_idx]
        
        # Apply the same physics as in step(): action -> velocity -> new position
        velocity = np.clip(action, -self.max_velocity, self.max_velocity)
        new_position = pos + self.dt * velocity
        new_position = np.clip(new_position, -self.max_position, self.max_position)
        
        # Calculate reward based on new position and raw action (not clipped velocity)
        return -((new_position - goal_pos)**2) - 0.01 * (action**2)