import os
import argparse
import pickle

import numpy as np
import torch
import gymnasium as gym

from config.bandit import args_bandit_rl2, args_bandit_mpc
from metalearner import MetaLearner


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    # -- ARGS --
    parser_for_exp_label = argparse.ArgumentParser()
    parser_for_exp_label.add_argument('--exp_label')
    args_for_exp_label, extra = parser_for_exp_label.parse_known_args()

    if args_for_exp_label.exp_label == 'mpc':
        # set model type: mpc
        parser = args_bandit_mpc.get_args()
        args = parser.parse_args()
        args.exp_label = 'mpc'
    elif args_for_exp_label.exp_label == 'rl2':
        # set model type: mpc
        parser = args_bandit_rl2.get_args()
        args = parser.parse_args()
        args.exp_label = 'rl2'
    else:
        raise ValueError(f'incorrect exp_label: {args_for_exp_label.exp_label}')
    
    # set env
    env = gym.make(args.env_name)


    # -- shared parameters for rl2 and mpc
    args.max_episode_steps = env.unwrapped.max_episode_steps
    args.policy_num_steps_per_update = env.unwrapped.max_episode_steps
    args.time_as_state = False
    args.policy_algorithm = 'a2c'
    args.policy_net_activation_function = 'tanh'
    # feature extractor
    embed_dim = 0
    args.action_embed_dim = embed_dim
    args.state_embed_dim = embed_dim
    args.reward_embed_dim = embed_dim
    # logging
    num_evals = 10
    num_saves = 5
    eval_ids = [-1]
    eval_ids_train = np.geomspace(
        1, args.num_updates, num_evals, 
        endpoint=True, dtype=int).tolist()
    args.eval_ids = eval_ids + eval_ids_train
    args.save_intermediate_models = True
    args.save_interval = args.num_updates / num_saves

    # -- parameters for rl2
    if args.exp_label == 'rl2':
        shared_rnn = True
        args.shared_rnn = shared_rnn

    # -- parameters for mpc
    elif args.exp_label == 'mpc':
        args.vae_storage_max_num_steps = env.unwrapped.max_episode_steps

    
    # -- TRAINING --
    # initialize metalearner
    metalearner = MetaLearner(args)
    out_dir = metalearner.logger.full_output_folder
    
    # training
    train_stats, evaluation_stats = metalearner.train()

    # save training history
    with open(os.path.join(out_dir, 'train_stats.pickle'), 'wb') as f:
        pickle.dump(train_stats, f)
    with open(os.path.join(out_dir, 'evaluation_stats.pickle'), 'wb') as f:
        pickle.dump(evaluation_stats, f)


if __name__ == "__main__":
    main()