import random

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


#################################################################################################
# Stationary bandit
class StatBernoulliBanditEnv(gym.Env):
    """
    Base bandit environment for stationary (time-aware) 
    k-armed bandit tasks.
    ---
    p_bandits:
        A list of reward probabilities for each bandit
    r_bandits:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout for each bandit
    info:
        Info about the environment that the agents is not supposed to know. 
        For instance, info can reveal the index of the optimal arm, 
        or the value of prior parameter.
        Can be useful to evaluate the agent's perfomance
    """
    def __init__(
        self, 
        p_bandits, 
        r_bandits, 
        total_trials,
        nonnegative_payout=True
    ):

        # sanity checks
        if len(p_bandits) != len(r_bandits):
            raise ValueError("Probability and Reward distribution must be of the same length")
        if min(p_bandits) < 0 or max(p_bandits) > 1:
            raise ValueError("All probabilities must be between 0 and 1")
        for reward in r_bandits:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.p_bandits = p_bandits
        self.r_bandits = r_bandits
        self.max_episode_steps = total_trials
        self.nonnegative_payout = nonnegative_payout
        
        # state space
        self.observation_space = spaces.Dict({
            "timestep": spaces.Box(low=0, high=self.max_episode_steps, dtype=np.int64),  # timestep
        })
        self.timestep = np.array([0])
        
        # action space
        self.k_bandits = len(p_bandits)
        self.action_space = spaces.Discrete(self.k_bandits)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _get_obs(self):
        return {"timestep": self.timestep}
    
    def _get_info(self):
        return {"reward_prob": self.p_bandits,
                "reward_size": self.r_bandits}
    
    def reset(self, seed=None, options={}):
        """
        The reset method will be called to initiate a new episode. 
        You may assume that the step method will not be called before reset has been called. 
        Moreover, reset should be called whenever a done signal has been issued.
        This should *NOT* automatically reset the task! Resetting the task is 
        handled in the wrapper.
        """
        # seed self.np_random
        # pass an integer for RHG right after the environment has been initialized 
        # and then never again
        super().reset(seed=seed)

        # reset timestep
        self.timestep = np.array([0])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: (observation, reward, terminated, truncated, info)
        If terminated or truncated is true, the user needs to call reset().
        """
        # action should be type integer in [0, k_bandits-1]
        assert self.action_space.contains(action)

        # state transition
        self.timestep += 1

        # An episode is done iff max_episode_steps is reached
        terminated = bool((self.timestep >= self.max_episode_steps))

        # compute reward
        reward = 0
        if random.uniform(0, 1) < self.p_bandits[action]:
            if not isinstance(self.r_bandits[action], list):
                reward = self.r_bandits[action]
            else:
                if self.nonnegative_payout:
                    reward = max(0, random.gauss(self.r_bandits[action][0], self.r_bandits[action][1]))
                else:
                    reward = random.gauss(self.r_bandits[action][0], self.r_bandits[action][1])

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info


class StatBernoulliBandit2ArmIndependent(StatBernoulliBanditEnv):
    """
    Stochastic version with independent reward probabilities
    """
    def __init__(self, total_trials=20):
        
        p_bandits = [random.random() for _ in range(2)]

        StatBernoulliBanditEnv.__init__(
            self, 
            p_bandits=p_bandits, 
            r_bandits=[1, 1],
            total_trials=total_trials
        )

    def reset(self, seed=None, options={}):
        """
        over-riding such that p_bandits is reset
        """
        # seed self.np_random
        # pass an integer for RHG right after the environment 
        # has been initialized and then never again
        if self.timestep != np.array([0]):
            super().reset(seed=seed)

            # reset timestep
            self.timestep = np.array([0])

            # reset probability
            self.p_bandits = [random.random() for _ in range(2)]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


#################################################################################################
# Oracle bandit
class OracleBanditDeterministic(StatBernoulliBanditEnv):
    """
    11-arm bandit with the last arm being informative
    """
    def __init__(
        self, 
        r_target_arm=5,
        r_nontarget_arm=1,
        total_trials=20):

        self.r_target_arm = r_target_arm
        self.r_nontarget_arm = r_nontarget_arm

        self.target_arm = random.randint(0, 9)
        self.p_bandits = [1. for _ in range(11)]
        
        
        self.r_bandits = []
        for arm in range(11):
            if arm == 10:
                self.r_bandits.append(float(self.target_arm+1)/ 10)
            else:
                if arm == self.target_arm:
                    self.r_bandits.append(self.r_target_arm)
                else:
                    self.r_bandits.append(self.r_nontarget_arm)
            
        StatBernoulliBanditEnv.__init__(
            self, 
            p_bandits=self.p_bandits, 
            r_bandits=self.r_bandits,
            total_trials=total_trials
        )

    def reset(self, seed=None, options={}):
        """
        over-riding such that p_bandits is reset
        """
        # seed self.np_random
        # pass an integer for RHG right after the environment 
        # has been initialized and then never again
        if self.timestep != np.array([0]):
            super().reset(seed=seed)

            # reset timestep
            self.timestep = np.array([0])

            # reset target arm
            target_arm = random.randint(0, 9)
            r_bandits = []
            for arm in range(11):
                if arm == 10:
                    r_bandits.append(float(target_arm+1)/ 10)
                else:
                    if arm == target_arm:
                        r_bandits.append(self.r_target_arm)
                    else:
                        r_bandits.append(self.r_nontarget_arm)
            self.r_bandits = r_bandits

        observation = self._get_obs()
        info = self._get_info()

        return observation, info