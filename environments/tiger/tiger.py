import random
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


#######################################################################################
# Tiger Problem
#######################################################################################
class TigerEnv(gym.Env):
    """
    The Tiger Problem environment.
    
    In this POMDP:
    - The tiger is behind one of two doors (left=0 or right=1)
    - The agent can take three actions: listen (0), open left door (1), open right door (2)
    - Observations after listening are: hear-left (0) or hear-right (1) with some noise
    - Rewards: 
        * listen: small negative reward
        * open door with tiger: large negative reward
        * open door with treasure: positive reward
    - Actions:
        0: Open left door
        1: Open right door
        2: Listen (gathers noisy observation, costs -1)
    - Observations:
        [1, 0]: Hear tiger on left
        [0, 1]: Hear tiger on right
        [0, 0]: No observation (used at reset or terminal state)
    """
    
    def __init__(
        self, 
        tiger_reward=-100.0,
        treasure_reward=10.0,
        listen_reward=-1.0,
        obs_accuracy=0.85,
        tiger_loc=None,
        total_trials=30
    ):
        super().__init__()

        # Parameters
        self.tiger_reward = tiger_reward
        self.treasure_reward = treasure_reward
        self.listen_reward = listen_reward
        self.obs_accuracy = obs_accuracy  # probability of correct observation when listening
        self.max_episode_steps = total_trials

        # Observation space: hear tiger location (None, left=0, right=1)
        self.observation_space = spaces.Dict({
            "hear": spaces.MultiBinary(2)  # hear-left or hear-right
        })
        self.timestep = np.array([0])
        
        # Action space: open-left, open-right, listen
        self.action_space = spaces.Discrete(3) 
        
        # State
        self.tiger_location = None
        self.fixed_tiger_loc = tiger_loc

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def _get_obs(self):
        return {"hear": self.obs}

    def _gen_rewards_t(self):
        # generate rewards for the current time step for all three possible actions
        rewards_t = np.zeros(3)
        rewards_t[0] = self.tiger_reward if self.tiger_location == 0 else self.treasure_reward
        rewards_t[1] = self.tiger_reward if self.tiger_location == 1 else self.treasure_reward
        rewards_t[2] = self.listen_reward
        return rewards_t
    
    def _get_info(self):
        # Return the true tiger location and current belief state
        return {
            "episode_terminated": self.episode_terminated,
            "tiger_location": self.tiger_location,
            "rewards_t": self._gen_rewards_t(),
        }
    
    def reset(self, seed=None, options=None):
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
        
        # Reset tiger location (random unless fixed)
        if self.fixed_tiger_loc is not None:
            self.tiger_location = self.fixed_tiger_loc
        else:
            self.tiger_location = self.np_random.integers(0, 2)  # 0=left, 1=right
        
        # reset timestep
        self.timestep = np.array([0])
        # No observation at the beginning
        self.obs = np.array([0,0])
        # keep track of termination
        self.episode_terminated = False

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        
        # time counter
        self.timestep += 1
        terminated = bool((self.timestep >= self.max_episode_steps))

        # observation & reward
        self.obs = np.array([0,0])
        if action == 0:  # Open left door
            if self.tiger_location == 0:  # Tiger on left
                reward = self.tiger_reward
            else:  # Treasure on left
                reward = self.treasure_reward
            self.episode_terminated = True
            
        elif action == 1:  # Open right door
            if self.tiger_location == 1:  # Tiger on right
                reward = self.tiger_reward
            else:  # Treasure on right
                reward = self.treasure_reward
            self.episode_terminated = True
            
        elif action == 2:  # Listen
            reward = self.listen_reward
            # Return observation based on listen action
            # This is a noisy observation about the tiger's location
            if random.random() < self.obs_accuracy:
                # Correct observation
                self.obs[self.tiger_location] += 1
            else:
                # Incorrect observation
                self.obs[1 - self.tiger_location] += 1
            
            # An episode is done if max_episode_steps is reached
            if not self.episode_terminated:
                self.episode_terminated = bool((self.timestep >= self.max_episode_steps))

        observation = self._get_obs()
        info = self._get_info()
            
        return observation, reward, terminated, False, info


class StatTiger(TigerEnv):
    def __init__(
        self, 
        tiger_reward=-100.0,
        treasure_reward=10.0,
        listen_reward=-1.0,
        obs_accuracy=0.85,
        tiger_loc=None,
        total_trials=30
    ):
        TigerEnv.__init__(
            self,
            tiger_reward=tiger_reward,
            treasure_reward=treasure_reward,
            listen_reward=listen_reward,
            obs_accuracy=obs_accuracy,
            tiger_loc=tiger_loc,
            total_trials=total_trials
        )


class MarkovianTigerEnv(gym.Env):
    """
    Dynamic Tiger Problem environment with Markovian state transitions.
    
    In this POMDP:
    - The tiger is behind one of two doors (left=0 or right=1)
    - The agent can take three actions: listen (0), open left door (1), open right door (2)
    - Observations after listening are: hear-left (0) or hear-right (1) with some noise
    - Rewards: 
        * listen: small negative reward
        * open door with tiger: large negative reward
        * open door with treasure: positive reward
    - Actions:
        0: Open left door
        1: Open right door
        2: Listen (gathers noisy observation, costs -1)
    - Observations:
        [1, 0]: Hear tiger on left
        [0, 1]: Hear tiger on right
        [0, 0]: No observation (used at reset or terminal state)
    """
    
    def __init__(
        self, 
        tiger_reward: float = -100.0,
        treasure_reward: float = 10.0,
        listen_reward: float = -1.0,
        obs_accuracy: float = 0.85,
        tiger_stay_prob: float = 0.9,
        tiger_loc: Optional[int] = None,
        total_trials: Optional[int] = 30
    ):
        super().__init__()

        # Parameters
        self.tiger_reward = tiger_reward
        self.treasure_reward = treasure_reward
        self.listen_reward = listen_reward
        self.obs_accuracy = obs_accuracy  # probability of correct observation when listening
        self.tiger_stay_prob = tiger_stay_prob
        self.max_episode_steps = total_trials

        # Define transition dynamics: P(s'|s)
        self.transition_matrix = np.array([
            [tiger_stay_prob, 1 - tiger_stay_prob],  # Tiger starts left -> [Pr(stay left), Pr(move right)]
            [1 - tiger_stay_prob, tiger_stay_prob]   # Tiger starts right -> [Pr(move left), Pr(stay right)]
        ])

        # Observation space: hear tiger location (None, left=0, right=1)
        self.observation_space = spaces.Dict({
            "hear": spaces.MultiBinary(2)  # hear-left or hear-right
        })
        self.timestep = np.array([0])
        
        # Action space: open-left, open-right, listen
        self.action_space = spaces.Discrete(3) 
        
        # State
        self.tiger_location = None
        self.fixed_tiger_loc = tiger_loc

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def _get_obs(self):
        return {"hear": self.obs}
    
    def _gen_rewards_t(self):
        # generate rewards for the current time step for all three possible actions
        rewards_t = np.zeros(3)
        rewards_t[0] = self.tiger_reward if self.tiger_location == 0 else self.treasure_reward
        rewards_t[1] = self.tiger_reward if self.tiger_location == 1 else self.treasure_reward
        rewards_t[2] = self.listen_reward
        return rewards_t

    def _get_info(self):
        # Return the true tiger location and current belief state
        return {
            "episode_terminated": self.episode_terminated,
            "tiger_location": self.tiger_location,
            "rewards_t": self._gen_rewards_t(),
        }
    
    def _transition_tiger(self):
        """Transition the tiger's location according to Markovian dynamics."""
        # Sample next state based on current state and transition matrix
        next_state_probs = self.transition_matrix[self.tiger_location]
        self.tiger_location = random.choices([0, 1], weights=next_state_probs, k=1)[0]

    def reset(self, seed=None, options=None):
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
        
        # Reset tiger location (random unless fixed)
        if self.fixed_tiger_loc is not None:
            self.tiger_location = self.fixed_tiger_loc
        else:
            self.tiger_location = self.np_random.integers(0, 2)  # 0=left, 1=right
        
        # reset timestep
        self.timestep = np.array([0])
        # No observation at the beginning
        self.obs = np.array([0,0])
        # keep track of termination
        self.episode_terminated = False

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        
        # time counter
        self.timestep += 1
        terminated = bool((self.timestep >= self.max_episode_steps))

        # observation & reward
        self.obs = np.array([0,0])
        if action == 0:  # Open left door
            if self.tiger_location == 0:  # Tiger on left
                reward = self.tiger_reward
            else:  # Treasure on left
                reward = self.treasure_reward
            self.episode_terminated = True
            
        elif action == 1:  # Open right door
            if self.tiger_location == 1:  # Tiger on right
                reward = self.tiger_reward
            else:  # Treasure on right
                reward = self.treasure_reward
            self.episode_terminated = True
            
        elif action == 2:  # Listen
            reward = self.listen_reward
            # Return observation based on listen action
            # This is a noisy observation about the tiger's location
            if random.random() < self.obs_accuracy:
                # Correct observation
                self.obs[self.tiger_location] += 1
            else:
                # Incorrect observation
                self.obs[1 - self.tiger_location] += 1         

            # An episode is done if max_episode_steps is reached
            if not self.episode_terminated:
                self.episode_terminated = bool((self.timestep >= self.max_episode_steps))    

        observation = self._get_obs()

        # Unless the episode is terminated, transition the tiger
        if not self.episode_terminated:
            self._transition_tiger()

        info = self._get_info()
            
        return observation, reward, terminated, False, info
    

class MarkovTiger(MarkovianTigerEnv):
    def __init__(
        self, 
        tiger_reward=-100.0,
        treasure_reward=10.0,
        listen_reward=-1.0,
        obs_accuracy=0.85,
        tiger_stay_prob=0.9,
        tiger_loc=None,
        total_trials=30
    ):
        MarkovianTigerEnv.__init__(
            self,
            tiger_reward=tiger_reward,
            treasure_reward=treasure_reward,
            listen_reward=listen_reward,
            obs_accuracy=obs_accuracy,
            tiger_stay_prob=tiger_stay_prob,
            tiger_loc=tiger_loc,
            total_trials=total_trials
        )