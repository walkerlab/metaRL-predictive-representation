from gymnasium.envs.registration import register

##################################################
# Bandit
register(
     "StatBernoulliBandit2ArmIndependent-v0",
     entry_point="environments.bandit.bandit:StatBernoulliBandit2ArmIndependent",
     max_episode_steps=40,
     kwargs={
          'total_trials': 40
     }
)


##################################################
# OracleBanditDeterministic
register(
     "OracleBanditDeterministic-v0",
     entry_point="environments.bandit.bandit:OracleBanditDeterministic",
     max_episode_steps=6,
     kwargs={
          'r_target_arm': 5,
          'r_nontarget_arm': 1,
          'total_trials': 6
     }
)


##################################################
# Dynamic bandit
register(
     "DisSymmetricStickyMarkovBandit2State2Arm-v0",
     entry_point="environments.dynamic_bandit.dynamic_bandit:DisBerMarkovBandit2ArmIndependent",
     max_episode_steps=300,
     kwargs={
          'total_trials': 300,
          'reward_states': [[0.1, 0.9], [0.1, 0.9]],
          'transition_matrix': [
               [[0.9, 0.1], 
                [0.1, 0.9]],
               [[0.9, 0.1], 
                [0.1, 0.9]]
          ],
     }
)

register(
     "DisAsymmetricRewardStickyMarkovBandit2State2Arm-v0",
     entry_point="environments.dynamic_bandit.dynamic_bandit:DisBerMarkovBandit2ArmIndependent",
     max_episode_steps=300,
     kwargs={
          'total_trials': 300,
          'reward_states': [[0.1, 0.9], [0.4, 0.6]],
          'transition_matrix': [
               [[0.9, 0.1], 
                [0.1, 0.9]],
               [[0.9, 0.1], 
                [0.1, 0.9]]
          ],
     }
)

register(
     "DisAsymmetricTransitionStickyMarkovBandit2State2Arm-v0",
     entry_point="environments.dynamic_bandit.dynamic_bandit:DisBerMarkovBandit2ArmIndependent",
     max_episode_steps=300,
     kwargs={
          'total_trials': 300,
          'reward_states': [[0.1, 0.9], [0.1, 0.9]],
          'transition_matrix': [
               [[0.9, 0.1], 
                [0.1, 0.9]],
               [[0.5, 0.5], 
                [0.5, 0.5]]
          ],
     }
)


##################################################
# Tiger Env
register(
     "StatTiger-v0",
     entry_point="environments.tiger.tiger:StatTiger",
     max_episode_steps=1000,
     kwargs={
          'tiger_reward': -100.0,
          'treasure_reward': 10.0,
          'listen_reward': -1.0,
          'obs_accuracy': 0.8,
          'total_trials': 20
     }
)

register(
     "StatTiger-v1",
     entry_point="environments.tiger.tiger:StatTiger",
     max_episode_steps=100,
     kwargs={
          'tiger_reward': -100.0,
          'treasure_reward': 10.0,
          'listen_reward': -1.0,
          'obs_accuracy': 0.7,
          'total_trials': 30
     }
)

register(
     "MarkovTiger-v0",
     entry_point="environments.tiger.tiger:MarkovTiger",
     max_episode_steps=100,
     kwargs={
          'tiger_reward': -100.0,
          'treasure_reward': 10.0,
          'listen_reward': -1.0,
          'obs_accuracy': 0.8,
          'tiger_stay_prob': 0.9,
          'total_trials': 30
     }
)

register(
     "MarkovTiger-v1",
     entry_point="environments.tiger.tiger:MarkovTiger",
     max_episode_steps=100,
     kwargs={
          'tiger_reward': -100.0,
          'treasure_reward': 10.0,
          'listen_reward': -1.0,
          'obs_accuracy': 0.7,
          'tiger_stay_prob': 0.9,
          'total_trials': 40
     }
)


######################################################
# Continuous task
register(
     "LatentGoalCart-v0",
     entry_point="environments.latent_goal_cart.latent_goal_cart:LatentGoalCartEnv",
     max_episode_steps=30,
     kwargs={
          'goal_positions': [-1.0, 1.0], 
          'reward_noise_std': 0.1,
          'total_trials': 30
     }
)