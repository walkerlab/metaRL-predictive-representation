import os
import pickle
import json
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.evaluation import load_trained_network

from utils.state_machine_analysis import gen_ref_trajectories_by_bayes, gen_metaRL_trajectories_given_ref
from utils.state_machine_analysis import gen_state_mapper_training_dataset
from utils.state_machine_analysis import get_PCA_model, plot_PCA_bayes_and_metaRL
from utils.state_machine_analysis import train_state_space_mapper
from utils.state_machine_analysis import get_mapped_bayes_states_and_actions
from utils.state_machine_analysis import get_mapped_metaRL_states_and_actions
from utils.state_machine_analysis import dissimilarity_analysis



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model_paths = [
        # add the paths to the trained models you want to analyze
    ]

    for trained_model_path in trained_model_paths:
        print(f'Trained_model_path: {trained_model_path}')

        # for document
        lines = []
        lines.append(f'trained_model_path: {trained_model_path}')
        
        # for saving ana result
        results = {}


        # -- LOAD --
        # load args
        args_json_path = os.path.join(trained_model_path, 'config.json')
        with open(args_json_path, 'rt') as f_json:
            args_dict = json.load(f_json)
        args = Namespace(**args_dict)

        # load trained model
        if args.exp_label == 'rl2':
            load_a2crnn_path = os.path.join(trained_model_path, 'actor_critic_weights.h5')
            a2crnn = load_trained_network(
                network_type='a2crnn',
                path_to_trained_network_state_dict=load_a2crnn_path,
                args=args,
                device=device
            )
        elif args.exp_label == 'mpc':
            load_a2cmlp_path = os.path.join(trained_model_path, 'actor_critic_weights.h5')
            a2cmlp = load_trained_network(
                network_type='a2cmlp',
                path_to_trained_network_state_dict=load_a2cmlp_path,
                args=args,
                device=device
            )
            load_encoder_path = os.path.join(trained_model_path, 'encoder_weights.h5')
            rnn_encoder = load_trained_network(
                network_type='rnn_encoder',
                path_to_trained_network_state_dict=load_encoder_path,
                args=args,
                device=device
            )
            load_reward_decoder_path = os.path.join(trained_model_path, 'reward_decoder_weights.h5')
            reward_decoder = load_trained_network(
                network_type='reward_decoder',
                path_to_trained_network_state_dict=load_reward_decoder_path,
                args=args,
                device=device
            )

        # load untrained model
        if args.exp_label == 'rl2':
            load_a2crnn_path = os.path.join(trained_model_path, 'actor_critic_weights-1.h5')
            untrained_a2crnn = load_trained_network(
                network_type='a2crnn',
                path_to_trained_network_state_dict=load_a2crnn_path,
                args=args,
                device=device
            )
        elif args.exp_label == 'mpc':
            load_a2cmlp_path = os.path.join(trained_model_path, 'actor_critic_weights-1.h5')
            untrained_a2cmlp = load_trained_network(
                network_type='a2cmlp',
                path_to_trained_network_state_dict=load_a2cmlp_path,
                args=args,
                device=device
            )
            load_encoder_path = os.path.join(trained_model_path, 'encoder_weights-1.h5')
            untrained_rnn_encoder = load_trained_network(
                network_type='rnn_encoder',
                path_to_trained_network_state_dict=load_encoder_path,
                args=args,
                device=device
            )
            load_reward_decoder_path = os.path.join(trained_model_path, 'reward_decoder_weights-1.h5')
            untrained_reward_decoder = load_trained_network(
                network_type='reward_decoder',
                path_to_trained_network_state_dict=load_reward_decoder_path,
                args=args,
                device=device
            )


        # ------------------------------------------------------------------
        # -- STATE MACHINE ANALYSIS  --
        # ------------------------------------------------------------------
        save_dir = trained_model_path

        # hyper param
        sma_total_trials = 40
        sma_env_name = args.env_name
        
        sma_num_training_envs = 500
        sma_num_testing_envs = 200

        sma_num_training_epochs = 500
        sma_training_batch_size = 64
        sma_validation_split = 0.2
        sma_patience = 10
        sma_min_delta = 0.01
        sma_min_training_epochs = 0

        sma_num_bandits = 2  # action_dim
        biased_beta_prior = None

        bayes_state_dim = sma_num_bandits * 2  # dim of bayes belief state, depending on the task
        
            
        if args.exp_label == 'mpc':
            encoder = rnn_encoder
            policy_network = a2cmlp
            untrained_encoder = untrained_rnn_encoder
            untrained_policy_network = untrained_a2cmlp
            metaRL_rnn_state_dim = args.encoder_rnn_hidden_dim
            metaRL_belief_state_dim = args.latent_dim * 2

        elif args.exp_label == 'rl2':
            encoder = None
            policy_network = a2crnn
            untrained_encoder = None
            untrained_policy_network = untrained_a2crnn
            metaRL_rnn_state_dim = args.rnn_hidden_dim
            if len(args.layers_after_rnn) > 0:
                metaRL_bottleneck_state_dim = args.layers_after_rnn[0]

        else:
            raise ValueError(f'incompatible model type: {args.exp_label}')
        

        # get optimal solver
        # if latent goal cart
        from utils.bayes_optimal_agents import LatentGoalCartPOMDPSolver, LatentGoalCartBeliefBasedAgent
        from utils.bayes_optimal_agents import train_latent_goal_cart_solver
        solver = LatentGoalCartPOMDPSolver(env=None, belief_resolution=101, pos_resolution=41, gamma=0.95)
        solver = train_latent_goal_cart_solver(solver)
        latent_cart_optimal_agent = LatentGoalCartBeliefBasedAgent(solver)


        # generate training dataset for state mapping
        ref_trajectories_bayes_train_bayes_sampled = gen_ref_trajectories_by_bayes(
            env_name=sma_env_name,
            total_trials=sma_total_trials,
            biased_beta_prior=biased_beta_prior,
            num_envs=sma_num_training_envs,
            num_bandits=sma_num_bandits
        )  # 1 extra trial to roll out one final step
        conditioned_trajectories_metaRL_train_bayes_sampled = gen_metaRL_trajectories_given_ref(
            encoder=encoder,
            policy_network=policy_network,
            args=args,
            ref_actions=ref_trajectories_bayes_train_bayes_sampled['bayes_actions'][:, :-1],
            ref_rewards=ref_trajectories_bayes_train_bayes_sampled['bayes_rewards'][:, :-1]
        )
        conditioned_trajectories_untrained_metaRL_train_bayes_sampled = gen_metaRL_trajectories_given_ref(
            encoder=untrained_encoder,
            policy_network=untrained_policy_network,
            args=args,
            ref_actions=ref_trajectories_bayes_train_bayes_sampled['bayes_actions'][:, :-1],
            ref_rewards=ref_trajectories_bayes_train_bayes_sampled['bayes_rewards'][:, :-1]
        )

        # save training dataset
        with open(os.path.join(trained_model_path, 'sma_ref_trajectories_bayes_train_bayes_sampled.pickle'), 'wb') as fo:
            pickle.dump(ref_trajectories_bayes_train_bayes_sampled, fo)
        with open(os.path.join(trained_model_path, 'sma_conditioned_trajectories_metaRL_train_bayes_sampled.pickle'), 'wb') as fo:
            pickle.dump(conditioned_trajectories_metaRL_train_bayes_sampled, fo)
        with open(os.path.join(trained_model_path, 'sma_conditioned_trajectories_untrained_metaRL_train_bayes_sampled.pickle'), 'wb') as fo:
            pickle.dump(conditioned_trajectories_untrained_metaRL_train_bayes_sampled, fo)
        
        # flatten for state space
        flattened_bayes_states_train_bayes_sampled = ref_trajectories_bayes_train_bayes_sampled['bayes_states'][:, :-1, :].\
            reshape(-1, bayes_state_dim)
        flattened_metaRL_rnn_states_train_bayes_sampled = conditioned_trajectories_metaRL_train_bayes_sampled['metaRL_rnn_states'].\
            reshape(-1, metaRL_rnn_state_dim)
        flattened_untrained_metaRL_rnn_states_train_bayes_sampled = conditioned_trajectories_untrained_metaRL_train_bayes_sampled['metaRL_rnn_states'].\
            reshape(-1, metaRL_rnn_state_dim)

        flattened_bayes_actions_train_bayes_sampled = ref_trajectories_bayes_train_bayes_sampled['bayes_actions'][:, :-1].\
            reshape(-1, 1)
        flattened_metaRL_all_action_logits_train_bayes_sampled = conditioned_trajectories_metaRL_train_bayes_sampled['metaRL_all_action_logits'].\
            reshape(-1, sma_num_bandits)
        flattened_metaRL_a1_probs_train_bayes_sampled = conditioned_trajectories_metaRL_train_bayes_sampled['metaRL_all_action_probs'][:, :, 1].\
            reshape(-1, 1)
        flattened_untrained_metaRL_all_action_logits_train_bayes_sampled = conditioned_trajectories_untrained_metaRL_train_bayes_sampled['metaRL_all_action_logits'].\
            reshape(-1, sma_num_bandits)
        flattened_untrained_metaRL_a1_probs_train_bayes_sampled = conditioned_trajectories_untrained_metaRL_train_bayes_sampled['metaRL_all_action_probs'][:, :, 1].\
            reshape(-1, 1)
        
        if args.exp_label == 'rl2':
            if len(args.layers_after_rnn) > 0:  # if bottleneck layer exists
                flattened_metaRL_bottleneck_states_train_bayes_sampled = conditioned_trajectories_metaRL_train_bayes_sampled['metaRL_bottleneck_states'].\
                    reshape(-1, metaRL_bottleneck_state_dim)
                flattened_untrained_metaRL_bottleneck_states_train_bayes_sampled = conditioned_trajectories_untrained_metaRL_train_bayes_sampled['metaRL_bottleneck_states'].\
                    reshape(-1, metaRL_bottleneck_state_dim)

        if args.exp_label == 'mpc':
            flattened_metaRL_belief_states_train_bayes_sampled = conditioned_trajectories_metaRL_train_bayes_sampled['metaRL_belief_states'].\
                reshape(-1, metaRL_belief_state_dim)
            flattened_metaRL_belief_states_mean_only_train_bayes_sampled = conditioned_trajectories_metaRL_train_bayes_sampled['metaRL_belief_states_mean_only'].\
                reshape(-1, round(metaRL_belief_state_dim/2))
            flattened_untrained_metaRL_belief_states_train_bayes_sampled = conditioned_trajectories_untrained_metaRL_train_bayes_sampled['metaRL_belief_states'].\
                reshape(-1, metaRL_belief_state_dim)
            flattened_untrained_metaRL_belief_states_mean_only_train_bayes_sampled = conditioned_trajectories_untrained_metaRL_train_bayes_sampled['metaRL_belief_states_mean_only'].\
                reshape(-1, round(metaRL_belief_state_dim/2))
            
        
        # get PCA models
        pca_model_bayes_states_bayes_sampled = get_PCA_model(flattened_bayes_states_train_bayes_sampled, bayes_state_dim)
        pca_model_metaRL_rnn_states_bayes_sampled = get_PCA_model(flattened_metaRL_rnn_states_train_bayes_sampled, metaRL_rnn_state_dim)
        pca_model_untrained_metaRL_rnn_states_bayes_sampled = get_PCA_model(flattened_untrained_metaRL_rnn_states_train_bayes_sampled, metaRL_rnn_state_dim)
        if args.exp_label == 'rl2':
            if len(args.layers_after_rnn) > 0:  # if bottleneck layer exists
                pca_model_metaRL_bottleneck_states_bayes_sampled = get_PCA_model(flattened_metaRL_bottleneck_states_train_bayes_sampled, metaRL_bottleneck_state_dim)
                pca_model_untrained_metaRL_bottleneck_states_bayes_sampled = get_PCA_model(flattened_untrained_metaRL_bottleneck_states_train_bayes_sampled, metaRL_bottleneck_state_dim)
        if args.exp_label == 'mpc':
            pca_model_metaRL_belief_states_bayes_sampled = get_PCA_model(flattened_metaRL_belief_states_train_bayes_sampled, metaRL_belief_state_dim)
            pca_model_untrained_metaRL_belief_states_bayes_sampled = get_PCA_model(flattened_untrained_metaRL_belief_states_train_bayes_sampled, metaRL_belief_state_dim)
            pca_model_metaRL_belief_states_mean_only_bayes_sampled = get_PCA_model(flattened_metaRL_belief_states_mean_only_train_bayes_sampled, round(metaRL_belief_state_dim/2))
            pca_model_untrained_metaRL_belief_states_mean_only_bayes_sampled = get_PCA_model(flattened_untrained_metaRL_belief_states_mean_only_train_bayes_sampled, round(metaRL_belief_state_dim/2))


        # plot PCA training set
        fig_sma_pca_train_metaRL_rnn_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
            pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
            pca_model_metaRL=pca_model_metaRL_rnn_states_bayes_sampled,
            true_bayes_states=flattened_bayes_states_train_bayes_sampled,
            mapped_bayes_states=None,
            true_metaRL_states=flattened_metaRL_rnn_states_train_bayes_sampled,
            mapped_metaRL_states=None,
            true_bayes_actions=flattened_bayes_actions_train_bayes_sampled,
            mapped_bayes_actions=None,
            env_name=sma_env_name,
            true_metaRL_a1_probs=flattened_metaRL_a1_probs_train_bayes_sampled,
            mapped_metaRL_a1_probs=None,
            true_metaRL_all_action_logits=flattened_metaRL_all_action_logits_train_bayes_sampled,
            mapped_metaRL_all_action_logits=None,
            total_trials=sma_total_trials,
            num_bandits=sma_num_bandits
        )
        fig_sma_pca_train_metaRL_rnn_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_train_metaRL_rnn_states_bayes_sampled.png'))

        fig_sma_pca_train_untrained_metaRL_rnn_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
            pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
            pca_model_metaRL=pca_model_untrained_metaRL_rnn_states_bayes_sampled,
            true_bayes_states=flattened_bayes_states_train_bayes_sampled,
            mapped_bayes_states=None,
            true_metaRL_states=flattened_untrained_metaRL_rnn_states_train_bayes_sampled,
            mapped_metaRL_states=None,
            true_bayes_actions=flattened_bayes_actions_train_bayes_sampled,
            mapped_bayes_actions=None,
            env_name=sma_env_name,
            true_metaRL_a1_probs=flattened_untrained_metaRL_a1_probs_train_bayes_sampled,
            mapped_metaRL_a1_probs=None,
            true_metaRL_all_action_logits=flattened_untrained_metaRL_all_action_logits_train_bayes_sampled,
            mapped_metaRL_all_action_logits=None,
            total_trials=sma_total_trials,
            num_bandits=sma_num_bandits
        )
        fig_sma_pca_train_untrained_metaRL_rnn_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_train_untrained_metaRL_rnn_states_bayes_sampled.png'))

        if args.exp_label == 'rl2':
            if len(args.layers_after_rnn) > 0:  # if bottleneck layer exists
                fig_sma_pca_train_metaRL_bottleneck_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
                    pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
                    pca_model_metaRL=pca_model_metaRL_bottleneck_states_bayes_sampled,
                    true_bayes_states=flattened_bayes_states_train_bayes_sampled,
                    mapped_bayes_states=None,
                    true_metaRL_states=flattened_metaRL_bottleneck_states_train_bayes_sampled,
                    mapped_metaRL_states=None,
                    true_bayes_actions=flattened_bayes_actions_train_bayes_sampled,
                    mapped_bayes_actions=None,
                    env_name=sma_env_name,
                    true_metaRL_a1_probs=flattened_metaRL_a1_probs_train_bayes_sampled,
                    mapped_metaRL_a1_probs=None,
                    true_metaRL_all_action_logits=flattened_metaRL_all_action_logits_train_bayes_sampled,
                    mapped_metaRL_all_action_logits=None,
                    total_trials=sma_total_trials,
                    num_bandits=sma_num_bandits
                )
                fig_sma_pca_train_metaRL_bottleneck_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_train_metaRL_bottleneck_states_bayes_sampled.png'))

                fig_sma_pca_train_untrained_metaRL_bottleneck_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
                    pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
                    pca_model_metaRL=pca_model_untrained_metaRL_bottleneck_states_bayes_sampled,
                    true_bayes_states=flattened_bayes_states_train_bayes_sampled,
                    mapped_bayes_states=None,
                    true_metaRL_states=flattened_untrained_metaRL_bottleneck_states_train_bayes_sampled,
                    mapped_metaRL_states=None,
                    true_bayes_actions=flattened_bayes_actions_train_bayes_sampled,
                    mapped_bayes_actions=None,
                    env_name=sma_env_name,
                    true_metaRL_a1_probs=flattened_untrained_metaRL_a1_probs_train_bayes_sampled,
                    mapped_metaRL_a1_probs=None,
                    true_metaRL_all_action_logits=flattened_untrained_metaRL_all_action_logits_train_bayes_sampled,
                    mapped_metaRL_all_action_logits=None,
                    total_trials=sma_total_trials,
                    num_bandits=sma_num_bandits
                )
                fig_sma_pca_train_untrained_metaRL_bottleneck_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_train_untrained_metaRL_bottleneck_states_bayes_sampled.png'))

        if args.exp_label == 'mpc':
            fig_sma_pca_train_metaRL_belief_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
                pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
                pca_model_metaRL=pca_model_metaRL_belief_states_mean_only_bayes_sampled,
                true_bayes_states=flattened_bayes_states_train_bayes_sampled,
                mapped_bayes_states=None,
                true_metaRL_states=flattened_metaRL_belief_states_mean_only_train_bayes_sampled,
                mapped_metaRL_states=None,
                true_bayes_actions=flattened_bayes_actions_train_bayes_sampled,
                mapped_bayes_actions=None,
                env_name=sma_env_name,
                true_metaRL_a1_probs=flattened_metaRL_a1_probs_train_bayes_sampled,
                mapped_metaRL_a1_probs=None,
                true_metaRL_all_action_logits=flattened_metaRL_all_action_logits_train_bayes_sampled,
                mapped_metaRL_all_action_logits=None,
                total_trials=sma_total_trials,
                num_bandits=sma_num_bandits
            )
            fig_sma_pca_train_metaRL_belief_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_train_metaRL_belief_states_bayes_sampled.png'))

            fig_sma_pca_train_untrained_metaRL_belief_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
                pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
                pca_model_metaRL=pca_model_untrained_metaRL_belief_states_mean_only_bayes_sampled,
                true_bayes_states=flattened_bayes_states_train_bayes_sampled,
                mapped_bayes_states=None,
                true_metaRL_states=flattened_untrained_metaRL_belief_states_mean_only_train_bayes_sampled,
                mapped_metaRL_states=None,
                true_bayes_actions=flattened_bayes_actions_train_bayes_sampled,
                mapped_bayes_actions=None,
                env_name=sma_env_name,
                true_metaRL_a1_probs=flattened_untrained_metaRL_a1_probs_train_bayes_sampled,
                mapped_metaRL_a1_probs=None,
                true_metaRL_all_action_logits=flattened_untrained_metaRL_all_action_logits_train_bayes_sampled,
                mapped_metaRL_all_action_logits=None,
                total_trials=sma_total_trials,
                num_bandits=sma_num_bandits
            )
            fig_sma_pca_train_untrained_metaRL_belief_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_train_untrained_metaRL_belief_states_bayes_sampled.png'))

        
        # pack dataset 1: without first PCA'ed
        metaRL_rnn2bayes_dataset_bayes_sampled, bayes2metaRL_rnn_dataset_bayes_sampled = \
        gen_state_mapper_training_dataset(
            flattened_metaRL_states=flattened_metaRL_rnn_states_train_bayes_sampled,
            flattened_bayes_states=flattened_bayes_states_train_bayes_sampled
        )
        untrained_metaRL_rnn2bayes_dataset_bayes_sampled, bayes2untrained_metaRL_rnn_dataset_bayes_sampled = \
        gen_state_mapper_training_dataset(
            flattened_metaRL_states=flattened_untrained_metaRL_rnn_states_train_bayes_sampled,
            flattened_bayes_states=flattened_bayes_states_train_bayes_sampled
        )

        if args.exp_label == 'rl2':
            if len(args.layers_after_rnn) > 0:  # if bottleneck layer exists
                metaRL_bottleneck2bayes_dataset_bayes_sampled, bayes2metaRL_bottleneck_dataset_bayes_sampled = \
                gen_state_mapper_training_dataset(
                    flattened_metaRL_states=flattened_metaRL_bottleneck_states_train_bayes_sampled,
                    flattened_bayes_states=flattened_bayes_states_train_bayes_sampled
                )
                untrained_metaRL_bottleneck2bayes_dataset_bayes_sampled, bayes2untrained_metaRL_bottleneck_dataset_bayes_sampled = \
                gen_state_mapper_training_dataset(
                    flattened_metaRL_states=flattened_untrained_metaRL_bottleneck_states_train_bayes_sampled,
                    flattened_bayes_states=flattened_bayes_states_train_bayes_sampled
                )

        if args.exp_label == 'mpc':
            metaRL_belief2bayes_dataset_bayes_sampled, bayes2metaRL_belief_dataset_bayes_sampled = \
            gen_state_mapper_training_dataset(
                flattened_metaRL_states=flattened_metaRL_belief_states_train_bayes_sampled,
                flattened_bayes_states=flattened_bayes_states_train_bayes_sampled
            )
            untrained_metaRL_belief2bayes_dataset_bayes_sampled, bayes2untrained_metaRL_belief_dataset_bayes_sampled = \
            gen_state_mapper_training_dataset(
                flattened_metaRL_states=flattened_untrained_metaRL_belief_states_train_bayes_sampled,
                flattened_bayes_states=flattened_bayes_states_train_bayes_sampled
            )


        # ------------------------------------------------------------------
        # training state space mapper: for dataset 1, without first PCA'ed
        # ------------------------------------------------------------------

        # for trained_metaRL_rnn
        ## metaRL_rnn2bayes_bayes_sampled
        source_dim = metaRL_rnn_state_dim
        target_dim = bayes_state_dim
        dataset = metaRL_rnn2bayes_dataset_bayes_sampled
        metaRL_rnn2bayes_mapper_bayes_sampled, losses_mse_metaRL_rnn2bayes_bayes_sampled = \
        train_state_space_mapper(
            source_dim,
            target_dim,
            dataset,
            sma_num_training_epochs,
            batch_size=sma_training_batch_size,
            validation_split=sma_validation_split,
            patience=sma_patience,
            min_delta=sma_min_delta,
            min_training_epochs=sma_min_training_epochs
        )
        metaRL_rnn2bayes_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_metaRL_rnn2bayes_bayes_sampled_weights.h5')
        torch.save(metaRL_rnn2bayes_mapper_bayes_sampled.state_dict(), metaRL_rnn2bayes_mapper_bayes_sampled_save_path)

        bayes_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_bayes_states_train_bayes_sampled), axis=1))
        mse_metaRL_rnn2bayes_bayes_sampled = losses_mse_metaRL_rnn2bayes_bayes_sampled[-1]
        normalized_mse_metaRL_rnn2bayes_bayes_sampled = mse_metaRL_rnn2bayes_bayes_sampled/ bayes_avg_var_bayes_sampled
        print(f'sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
        print(f'sma_mse_metaRL_rnn2bayes: {mse_metaRL_rnn2bayes_bayes_sampled}')
        print(f'sma_normalized_mse_metaRL_rnn2bayes: {normalized_mse_metaRL_rnn2bayes_bayes_sampled}')
        lines.append('state mapper training: trained_metaRL_rnn')
        lines.append(f' sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
        lines.append(f' sma_mse_metaRL_rnn2bayes: {mse_metaRL_rnn2bayes_bayes_sampled}')
        lines.append(f' sma_normalized_mse_metaRL_rnn2bayes: {normalized_mse_metaRL_rnn2bayes_bayes_sampled}')

        ## bayes2metaRL_rnn_bayes_sampled
        source_dim = bayes_state_dim
        target_dim = metaRL_rnn_state_dim
        dataset = bayes2metaRL_rnn_dataset_bayes_sampled
        bayes2metaRL_rnn_mapper_bayes_sampled, losses_mse_bayes2metaRL_rnn_bayes_sampled = \
        train_state_space_mapper(
            source_dim,
            target_dim,
            dataset,
            sma_num_training_epochs,
            batch_size=sma_training_batch_size,
            validation_split=sma_validation_split,
            patience=sma_patience,
            min_delta=sma_min_delta,
            min_training_epochs=sma_min_training_epochs
        )
        bayes2metaRL_rnn_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_bayes2metaRL_rnn_bayes_sampled_weights.h5')
        torch.save(bayes2metaRL_rnn_mapper_bayes_sampled.state_dict(), bayes2metaRL_rnn_mapper_bayes_sampled_save_path)

        metaRL_rnn_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_metaRL_rnn_states_train_bayes_sampled), axis=1))
        mse_bayes2metaRL_rnn_bayes_sampled = losses_mse_bayes2metaRL_rnn_bayes_sampled[-1]
        normalized_mse_bayes2metaRL_rnn_bayes_sampled = mse_bayes2metaRL_rnn_bayes_sampled/ metaRL_rnn_avg_var_bayes_sampled
        print(f'sma_metaRL_rnn_avg_var: {metaRL_rnn_avg_var_bayes_sampled}')
        print(f'sma_mse_bayes2metaRL_rnn: {mse_bayes2metaRL_rnn_bayes_sampled}')
        print(f'sma_normalized_mse_bayes2metaRL_rnn: {normalized_mse_bayes2metaRL_rnn_bayes_sampled}')
        lines.append(f' sma_metaRL_rnn_avg_var: {metaRL_rnn_avg_var_bayes_sampled}')
        lines.append(f' sma_mse_bayes2metaRL_rnn: {mse_bayes2metaRL_rnn_bayes_sampled}')
        lines.append(f' sma_normalized_mse_bayes2metaRL_rnn: {normalized_mse_bayes2metaRL_rnn_bayes_sampled}')

        # for untrained_metaRL_rnn
        ## untrained_metaRL_rnn2bayes_bayes_sampled
        source_dim = metaRL_rnn_state_dim
        target_dim = bayes_state_dim
        dataset = untrained_metaRL_rnn2bayes_dataset_bayes_sampled
        untrained_metaRL_rnn2bayes_mapper_bayes_sampled, losses_mse_untrained_metaRL_rnn2bayes_bayes_sampled = train_state_space_mapper(
            source_dim,
            target_dim,
            dataset,
            sma_num_training_epochs,
            batch_size=sma_training_batch_size,
            validation_split=sma_validation_split,
            patience=sma_patience,
            min_delta=sma_min_delta,
            min_training_epochs=sma_min_training_epochs
        )
        untrained_metaRL_rnn2bayes_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_untrained_metaRL_rnn2bayes_bayes_sampled_weights.h5')
        torch.save(untrained_metaRL_rnn2bayes_mapper_bayes_sampled.state_dict(), untrained_metaRL_rnn2bayes_mapper_bayes_sampled_save_path)

        bayes_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_bayes_states_train_bayes_sampled), axis=1))
        mse_untrained_metaRL_rnn2bayes_bayes_sampled = losses_mse_untrained_metaRL_rnn2bayes_bayes_sampled[-1]
        normalized_mse_untrained_metaRL_rnn2bayes_bayes_sampled = mse_untrained_metaRL_rnn2bayes_bayes_sampled/ bayes_avg_var_bayes_sampled
        print(f'sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
        print(f'sma_mse_untrained_metaRL_rnn2bayes: {mse_untrained_metaRL_rnn2bayes_bayes_sampled}')
        print(f'sma_normalized_mse_untrained_metaRL_rnn2bayes: {normalized_mse_untrained_metaRL_rnn2bayes_bayes_sampled}')
        lines.append('state mapper training: untrained_metaRL_rnn')
        lines.append(f' sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
        lines.append(f' sma_mse_untrained_metaRL_rnn2bayes: {mse_untrained_metaRL_rnn2bayes_bayes_sampled}')
        lines.append(f' sma_normalized_mse_untrained_metaRL_rnn2bayes: {normalized_mse_untrained_metaRL_rnn2bayes_bayes_sampled}')

        ## bayes2untrained_metaRL_rnn
        source_dim = bayes_state_dim
        target_dim = metaRL_rnn_state_dim
        dataset = bayes2untrained_metaRL_rnn_dataset_bayes_sampled
        bayes2untrained_metaRL_rnn_mapper_bayes_sampled, losses_mse_bayes2untrained_metaRL_rnn_bayes_sampled = train_state_space_mapper(
            source_dim,
            target_dim,
            dataset,
            sma_num_training_epochs,
            batch_size=sma_training_batch_size,
            validation_split=sma_validation_split,
            patience=sma_patience,
            min_delta=sma_min_delta,
            min_training_epochs=sma_min_training_epochs
        )

        bayes2untrained_metaRL_rnn_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_bayes2untrained_metaRL_rnn_bayes_sampled_weights.h5')
        torch.save(bayes2untrained_metaRL_rnn_mapper_bayes_sampled.state_dict(), bayes2untrained_metaRL_rnn_mapper_bayes_sampled_save_path)

        untrained_metaRL_rnn_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_untrained_metaRL_rnn_states_train_bayes_sampled), axis=1))
        mse_bayes2untrained_metaRL_rnn_bayes_sampled = losses_mse_bayes2untrained_metaRL_rnn_bayes_sampled[-1]
        normalized_mse_bayes2untrained_metaRL_rnn_bayes_sampled = mse_bayes2untrained_metaRL_rnn_bayes_sampled/ untrained_metaRL_rnn_avg_var_bayes_sampled
        print(f'sma_untrained_metaRL_rnn_avg_var: {untrained_metaRL_rnn_avg_var_bayes_sampled}')
        print(f'sma_mse_bayes2untrained_metaRL_rnn: {mse_bayes2untrained_metaRL_rnn_bayes_sampled}')
        print(f'sma_normalized_mse_bayes2untrained_metaRL_rnn: {normalized_mse_bayes2untrained_metaRL_rnn_bayes_sampled}')
        lines.append(f' sma_untrained_metaRL_rnn_avg_var: {untrained_metaRL_rnn_avg_var_bayes_sampled}')
        lines.append(f' sma_mse_bayes2untrained_metaRL_rnn: {mse_bayes2untrained_metaRL_rnn_bayes_sampled}')
        lines.append(f' sma_normalized_mse_bayes2untrained_metaRL_rnn: {normalized_mse_bayes2untrained_metaRL_rnn_bayes_sampled}')


        if args.exp_label == 'rl2':
            if len(args.layers_after_rnn) > 0:  # if bottleneck layer exists
                # for trained_metaRL_bottleneck
                ## metaRL_bottleneck2bayes_bayes_sampled
                source_dim = metaRL_bottleneck_state_dim
                target_dim = bayes_state_dim
                dataset = metaRL_bottleneck2bayes_dataset_bayes_sampled
                metaRL_bottleneck2bayes_mapper_bayes_sampled, losses_mse_metaRL_bottleneck2bayes_bayes_sampled = \
                train_state_space_mapper(
                    source_dim,
                    target_dim,
                    dataset,
                    sma_num_training_epochs,
                    batch_size=sma_training_batch_size,
                    validation_split=sma_validation_split,
                    patience=sma_patience,
                    min_delta=sma_min_delta,
                    min_training_epochs=sma_min_training_epochs
                )
                metaRL_bottleneck2bayes_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_metaRL_bottleneck2bayes_bayes_sampled_weights.h5')
                torch.save(metaRL_bottleneck2bayes_mapper_bayes_sampled.state_dict(), metaRL_bottleneck2bayes_mapper_bayes_sampled_save_path)

                bayes_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_bayes_states_train_bayes_sampled), axis=1))
                mse_metaRL_bottleneck2bayes_bayes_sampled = losses_mse_metaRL_bottleneck2bayes_bayes_sampled[-1]
                normalized_mse_metaRL_bottleneck2bayes_bayes_sampled = mse_metaRL_bottleneck2bayes_bayes_sampled/ bayes_avg_var_bayes_sampled
                print(f'sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
                print(f'sma_mse_metaRL_bottleneck2bayes: {mse_metaRL_bottleneck2bayes_bayes_sampled}')
                print(f'sma_normalized_mse_metaRL_bottleneck2bayes: {normalized_mse_metaRL_bottleneck2bayes_bayes_sampled}')
                lines.append('state mapper training: trained_metaRL_bottleneck')
                lines.append(f' sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
                lines.append(f' sma_mse_metaRL_bottleneck2bayes: {mse_metaRL_bottleneck2bayes_bayes_sampled}')
                lines.append(f' sma_normalized_mse_metaRL_bottleneck2bayes: {normalized_mse_metaRL_bottleneck2bayes_bayes_sampled}')

                ## bayes2metaRL_bottleneck_bayes_sampled
                source_dim = bayes_state_dim
                target_dim = metaRL_bottleneck_state_dim
                dataset = bayes2metaRL_bottleneck_dataset_bayes_sampled
                bayes2metaRL_bottleneck_mapper_bayes_sampled, losses_mse_bayes2metaRL_bottleneck_bayes_sampled = \
                train_state_space_mapper(
                    source_dim,
                    target_dim,
                    dataset,
                    sma_num_training_epochs,
                    batch_size=sma_training_batch_size,
                    validation_split=sma_validation_split,
                    patience=sma_patience,
                    min_delta=sma_min_delta,
                    min_training_epochs=sma_min_training_epochs
                )
                bayes2metaRL_bottleneck_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_bayes2metaRL_bottleneck_bayes_sampled_weights.h5')
                torch.save(bayes2metaRL_bottleneck_mapper_bayes_sampled.state_dict(), bayes2metaRL_bottleneck_mapper_bayes_sampled_save_path)

                metaRL_bottleneck_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_metaRL_bottleneck_states_train_bayes_sampled), axis=1))
                mse_bayes2metaRL_bottleneck_bayes_sampled = losses_mse_bayes2metaRL_bottleneck_bayes_sampled[-1]
                normalized_mse_bayes2metaRL_bottleneck_bayes_sampled = mse_bayes2metaRL_bottleneck_bayes_sampled/ metaRL_bottleneck_avg_var_bayes_sampled
                print(f'sma_metaRL_bottleneck_avg_var: {metaRL_bottleneck_avg_var_bayes_sampled}')
                print(f'sma_mse_bayes2metaRL_bottleneck: {mse_bayes2metaRL_bottleneck_bayes_sampled}')
                print(f'sma_normalized_mse_bayes2metaRL_bottleneck: {normalized_mse_bayes2metaRL_bottleneck_bayes_sampled}')
                lines.append(f' sma_metaRL_bottleneck_avg_var: {metaRL_bottleneck_avg_var_bayes_sampled}')
                lines.append(f' sma_mse_bayes2metaRL_bottleneck: {mse_bayes2metaRL_bottleneck_bayes_sampled}')
                lines.append(f' sma_normalized_mse_bayes2metaRL_bottleneck: {normalized_mse_bayes2metaRL_bottleneck_bayes_sampled}')

                # for untrained_metaRL_bottleneck
                ## untrained_metaRL_bottleneck2bayes_bayes_sampled
                source_dim = metaRL_bottleneck_state_dim
                target_dim = bayes_state_dim
                dataset = untrained_metaRL_bottleneck2bayes_dataset_bayes_sampled
                untrained_metaRL_bottleneck2bayes_mapper_bayes_sampled, losses_mse_untrained_metaRL_bottleneck2bayes_bayes_sampled = train_state_space_mapper(
                    source_dim,
                    target_dim,
                    dataset,
                    sma_num_training_epochs,
                    batch_size=sma_training_batch_size,
                    validation_split=sma_validation_split,
                    patience=sma_patience,
                    min_delta=sma_min_delta,
                    min_training_epochs=sma_min_training_epochs
                )
                untrained_metaRL_bottleneck2bayes_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_untrained_metaRL_bottleneck2bayes_bayes_sampled_weights.h5')
                torch.save(untrained_metaRL_bottleneck2bayes_mapper_bayes_sampled.state_dict(), untrained_metaRL_bottleneck2bayes_mapper_bayes_sampled_save_path)

                bayes_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_bayes_states_train_bayes_sampled), axis=1))
                mse_untrained_metaRL_bottleneck2bayes_bayes_sampled = losses_mse_untrained_metaRL_bottleneck2bayes_bayes_sampled[-1]
                normalized_mse_untrained_metaRL_bottleneck2bayes_bayes_sampled = mse_untrained_metaRL_bottleneck2bayes_bayes_sampled/ bayes_avg_var_bayes_sampled
                print(f'sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
                print(f'sma_mse_untrained_metaRL_bottleneck2bayes: {mse_untrained_metaRL_bottleneck2bayes_bayes_sampled}')
                print(f'sma_normalized_mse_untrained_metaRL_bottleneck2bayes: {normalized_mse_untrained_metaRL_bottleneck2bayes_bayes_sampled}')
                lines.append('state mapper training: untrained_metaRL_bottleneck')
                lines.append(f' sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
                lines.append(f' sma_mse_untrained_metaRL_bottleneck2bayes: {mse_untrained_metaRL_bottleneck2bayes_bayes_sampled}')
                lines.append(f' sma_normalized_mse_untrained_metaRL_bottleneck2bayes: {normalized_mse_untrained_metaRL_bottleneck2bayes_bayes_sampled}')

                ## bayes2untrained_metaRL_bottleneck
                source_dim = bayes_state_dim
                target_dim = metaRL_bottleneck_state_dim
                dataset = bayes2untrained_metaRL_bottleneck_dataset_bayes_sampled
                bayes2untrained_metaRL_bottleneck_mapper_bayes_sampled, losses_mse_bayes2untrained_metaRL_bottleneck_bayes_sampled = train_state_space_mapper(
                    source_dim,
                    target_dim,
                    dataset,
                    sma_num_training_epochs,
                    batch_size=sma_training_batch_size,
                    validation_split=sma_validation_split,
                    patience=sma_patience,
                    min_delta=sma_min_delta,
                    min_training_epochs=sma_min_training_epochs
                )
                bayes2untrained_metaRL_bottleneck_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_bayes2untrained_metaRL_bottleneck_bayes_sampled_weights.h5')
                torch.save(bayes2untrained_metaRL_bottleneck_mapper_bayes_sampled.state_dict(), bayes2untrained_metaRL_bottleneck_mapper_bayes_sampled_save_path)

                untrained_metaRL_bottleneck_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_untrained_metaRL_bottleneck_states_train_bayes_sampled), axis=1))
                mse_bayes2untrained_metaRL_bottleneck_bayes_sampled = losses_mse_bayes2untrained_metaRL_bottleneck_bayes_sampled[-1]
                normalized_mse_bayes2untrained_metaRL_bottleneck_bayes_sampled = mse_bayes2untrained_metaRL_bottleneck_bayes_sampled/ untrained_metaRL_bottleneck_avg_var_bayes_sampled
                print(f'sma_untrained_metaRL_bottleneck_avg_var: {untrained_metaRL_bottleneck_avg_var_bayes_sampled}')
                print(f'sma_mse_bayes2untrained_metaRL_bottleneck: {mse_bayes2untrained_metaRL_bottleneck_bayes_sampled}')
                print(f'sma_normalized_mse_bayes2untrained_metaRL_bottleneck: {normalized_mse_bayes2untrained_metaRL_bottleneck_bayes_sampled}')
                lines.append(f' sma_untrained_metaRL_bottleneck_avg_var: {untrained_metaRL_bottleneck_avg_var_bayes_sampled}')
                lines.append(f' sma_mse_bayes2untrained_metaRL_bottleneck: {mse_bayes2untrained_metaRL_bottleneck_bayes_sampled}')
                lines.append(f' sma_normalized_mse_bayes2untrained_metaRL_bottleneck: {normalized_mse_bayes2untrained_metaRL_bottleneck_bayes_sampled}')


        if args.exp_label == 'mpc':
            # for trained_metaRL_belief
            ## metaRL_belief2bayes_bayes_sampled
            source_dim = metaRL_belief_state_dim
            target_dim = bayes_state_dim
            dataset = metaRL_belief2bayes_dataset_bayes_sampled
            metaRL_belief2bayes_mapper_bayes_sampled, losses_mse_metaRL_belief2bayes_bayes_sampled = \
            train_state_space_mapper(
                source_dim,
                target_dim,
                dataset,
                sma_num_training_epochs,
                batch_size=sma_training_batch_size,
                validation_split=sma_validation_split,
                patience=sma_patience,
                min_delta=sma_min_delta,
                min_training_epochs=sma_min_training_epochs
            )
            metaRL_belief2bayes_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_metaRL_belief2bayes_bayes_sampled_weights.h5')
            torch.save(metaRL_belief2bayes_mapper_bayes_sampled.state_dict(), metaRL_belief2bayes_mapper_bayes_sampled_save_path)

            bayes_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_bayes_states_train_bayes_sampled), axis=1))
            mse_metaRL_belief2bayes_bayes_sampled = losses_mse_metaRL_belief2bayes_bayes_sampled[-1]
            normalized_mse_metaRL_belief2bayes_bayes_sampled = mse_metaRL_belief2bayes_bayes_sampled/ bayes_avg_var_bayes_sampled
            print(f'sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
            print(f'sma_mse_metaRL_belief2bayes: {mse_metaRL_belief2bayes_bayes_sampled}')
            print(f'sma_normalized_mse_metaRL_belief2bayes: {normalized_mse_metaRL_belief2bayes_bayes_sampled}')
            lines.append('state mapper training: trained_metaRL_belief')
            lines.append(f' sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
            lines.append(f' sma_mse_metaRL_belief2bayes: {mse_metaRL_belief2bayes_bayes_sampled}')
            lines.append(f' sma_normalized_mse_metaRL_belief2bayes: {normalized_mse_metaRL_belief2bayes_bayes_sampled}')

            ## bayes2metaRL_belief_bayes_sampled
            source_dim = bayes_state_dim
            target_dim = metaRL_belief_state_dim
            dataset = bayes2metaRL_belief_dataset_bayes_sampled
            bayes2metaRL_belief_mapper_bayes_sampled, losses_mse_bayes2metaRL_belief_bayes_sampled = \
            train_state_space_mapper(
                source_dim,
                target_dim,
                dataset,
                sma_num_training_epochs,
                batch_size=sma_training_batch_size,
                validation_split=sma_validation_split,
                patience=sma_patience,
                min_delta=sma_min_delta,
                min_training_epochs=sma_min_training_epochs
            )
            bayes2metaRL_belief_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_bayes2metaRL_belief_bayes_sampled_weights.h5')
            torch.save(bayes2metaRL_belief_mapper_bayes_sampled.state_dict(), bayes2metaRL_belief_mapper_bayes_sampled_save_path)

            metaRL_belief_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_metaRL_belief_states_train_bayes_sampled), axis=1))
            mse_bayes2metaRL_belief_bayes_sampled = losses_mse_bayes2metaRL_belief_bayes_sampled[-1]
            normalized_mse_bayes2metaRL_belief_bayes_sampled = mse_bayes2metaRL_belief_bayes_sampled/ metaRL_belief_avg_var_bayes_sampled
            print(f'sma_metaRL_belief_avg_var: {metaRL_belief_avg_var_bayes_sampled}')
            print(f'sma_mse_bayes2metaRL_belief: {mse_bayes2metaRL_belief_bayes_sampled}')
            print(f'sma_normalized_mse_bayes2metaRL_belief: {normalized_mse_bayes2metaRL_belief_bayes_sampled}')
            lines.append(f' sma_metaRL_belief_avg_var: {metaRL_belief_avg_var_bayes_sampled}')
            lines.append(f' sma_mse_bayes2metaRL_belief: {mse_bayes2metaRL_belief_bayes_sampled}')
            lines.append(f' sma_normalized_mse_bayes2metaRL_belief: {normalized_mse_bayes2metaRL_belief_bayes_sampled}')

            # for untrained_metaRL_belief
            ## untrained_metaRL_belief2bayes_bayes_sampled
            source_dim = metaRL_belief_state_dim
            target_dim = bayes_state_dim
            dataset = untrained_metaRL_belief2bayes_dataset_bayes_sampled
            untrained_metaRL_belief2bayes_mapper_bayes_sampled, losses_mse_untrained_metaRL_belief2bayes_bayes_sampled = train_state_space_mapper(
                source_dim,
                target_dim,
                dataset,
                sma_num_training_epochs,
                batch_size=sma_training_batch_size,
                validation_split=sma_validation_split,
                patience=sma_patience,
                min_delta=sma_min_delta,
                min_training_epochs=sma_min_training_epochs
            )
            untrained_metaRL_belief2bayes_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_untrained_metaRL_belief2bayes_bayes_sampled_weights.h5')
            torch.save(untrained_metaRL_belief2bayes_mapper_bayes_sampled.state_dict(), untrained_metaRL_belief2bayes_mapper_bayes_sampled_save_path)

            bayes_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_bayes_states_train_bayes_sampled), axis=1))
            mse_untrained_metaRL_belief2bayes_bayes_sampled = losses_mse_untrained_metaRL_belief2bayes_bayes_sampled[-1]
            normalized_mse_untrained_metaRL_belief2bayes_bayes_sampled = mse_untrained_metaRL_belief2bayes_bayes_sampled/ bayes_avg_var_bayes_sampled
            print(f'sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
            print(f'sma_mse_untrained_metaRL_belief2bayes: {mse_untrained_metaRL_belief2bayes_bayes_sampled}')
            print(f'sma_normalized_mse_untrained_metaRL_belief2bayes: {normalized_mse_untrained_metaRL_belief2bayes_bayes_sampled}')
            lines.append('state mapper training: untrained_metaRL_belief')
            lines.append(f' sma_bayes_avg_var: {bayes_avg_var_bayes_sampled}')
            lines.append(f' sma_mse_untrained_metaRL_belief2bayes: {mse_untrained_metaRL_belief2bayes_bayes_sampled}')
            lines.append(f' sma_normalized_mse_untrained_metaRL_belief2bayes: {normalized_mse_untrained_metaRL_belief2bayes_bayes_sampled}')

            ## bayes2untrained_metaRL_belief
            source_dim = bayes_state_dim
            target_dim = metaRL_belief_state_dim
            dataset = bayes2untrained_metaRL_belief_dataset_bayes_sampled
            bayes2untrained_metaRL_belief_mapper_bayes_sampled, losses_mse_bayes2untrained_metaRL_belief_bayes_sampled = train_state_space_mapper(
                source_dim,
                target_dim,
                dataset,
                sma_num_training_epochs,
                batch_size=sma_training_batch_size,
                validation_split=sma_validation_split,
                patience=sma_patience,
                min_delta=sma_min_delta,
                min_training_epochs=sma_min_training_epochs
            )

            bayes2untrained_metaRL_belief_mapper_bayes_sampled_save_path = os.path.join(trained_model_path, 'sma_mapper_bayes2untrained_metaRL_belief_bayes_sampled_weights.h5')
            torch.save(bayes2untrained_metaRL_belief_mapper_bayes_sampled.state_dict(), bayes2untrained_metaRL_belief_mapper_bayes_sampled_save_path)

            untrained_metaRL_belief_avg_var_bayes_sampled = np.average(np.sum(np.square(flattened_untrained_metaRL_belief_states_train_bayes_sampled), axis=1))
            mse_bayes2untrained_metaRL_belief_bayes_sampled = losses_mse_bayes2untrained_metaRL_belief_bayes_sampled[-1]
            normalized_mse_bayes2untrained_metaRL_belief_bayes_sampled = mse_bayes2untrained_metaRL_belief_bayes_sampled/ untrained_metaRL_belief_avg_var_bayes_sampled
            print(f'sma_untrained_metaRL_belief_avg_var: {untrained_metaRL_belief_avg_var_bayes_sampled}')
            print(f'sma_mse_bayes2untrained_metaRL_belief: {mse_bayes2untrained_metaRL_belief_bayes_sampled}')
            print(f'sma_normalized_mse_bayes2untrained_metaRL_belief: {normalized_mse_bayes2untrained_metaRL_belief_bayes_sampled}')
            lines.append(f' sma_untrained_metaRL_belief_avg_var: {untrained_metaRL_belief_avg_var_bayes_sampled}')
            lines.append(f' sma_mse_bayes2untrained_metaRL_belief: {mse_bayes2untrained_metaRL_belief_bayes_sampled}')
            lines.append(f' sma_normalized_mse_bayes2untrained_metaRL_belief: {normalized_mse_bayes2untrained_metaRL_belief_bayes_sampled}')


        # ------------------------------------------------------------------
        # testing state space mapper
        # ------------------------------------------------------------------

        # generate testing dataset for state mapping
        ref_trajectories_bayes_test_bayes_sampled = gen_ref_trajectories_by_bayes(
            env_name=sma_env_name,
            total_trials=sma_total_trials,
            biased_beta_prior=biased_beta_prior,
            num_envs=sma_num_testing_envs,
            num_bandits=sma_num_bandits,
            latent_goal_cart_solver=latent_cart_optimal_agent,
        )  # 1 extra trial to roll out one final step
        conditioned_trajectories_metaRL_test_bayes_sampled = gen_metaRL_trajectories_given_ref(
            encoder=encoder,
            policy_network=policy_network,
            args=args,
            ref_actions=ref_trajectories_bayes_test_bayes_sampled['bayes_actions'][:, :-1],
            ref_rewards=ref_trajectories_bayes_test_bayes_sampled['bayes_rewards'][:, :-1],
            ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
        )
        conditioned_trajectories_untrained_metaRL_test_bayes_sampled = gen_metaRL_trajectories_given_ref(
            encoder=untrained_encoder,
            policy_network=untrained_policy_network,
            args=args,
            ref_actions=ref_trajectories_bayes_test_bayes_sampled['bayes_actions'][:, :-1],
            ref_rewards=ref_trajectories_bayes_test_bayes_sampled['bayes_rewards'][:, :-1],
            ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
        )

        # save testing dataset
        with open(os.path.join(trained_model_path, 'sma_ref_trajectories_bayes_test_bayes_sampled.pickle'), 'wb') as fo:
            pickle.dump(ref_trajectories_bayes_test_bayes_sampled, fo)
        with open(os.path.join(trained_model_path, 'sma_conditioned_trajectories_metaRL_test_bayes_sampled.pickle'), 'wb') as fo:
            pickle.dump(conditioned_trajectories_metaRL_test_bayes_sampled, fo)
        with open(os.path.join(trained_model_path, 'sma_conditioned_trajectories_untrained_metaRL_test_bayes_sampled.pickle'), 'wb') as fo:
            pickle.dump(conditioned_trajectories_untrained_metaRL_test_bayes_sampled, fo)
        

        # get traj for state space analysis: no flattening needed here
        bayes_states_test_bayes_sampled = ref_trajectories_bayes_test_bayes_sampled['bayes_states'][:, :-1, :]
        metaRL_rnn_states_test_bayes_sampled = conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_rnn_states']
        untrained_metaRL_rnn_states_test_bayes_sampled = conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_rnn_states']

        bayes_actions_test_bayes_sampled = ref_trajectories_bayes_test_bayes_sampled['bayes_actions'][:, :-1]
        metaRL_actions_test_bayes_sampled = conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_actions'][:, :]
        metaRL_all_action_logits_test_bayes_sampled = conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_all_action_logits']
        metaRL_a1_probs_test_bayes_sampled = conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_all_action_probs'][:, :, 1]
        untrained_metaRL_actions_test_bayes_sampled = conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_actions'][:, :]
        untrained_metaRL_all_action_logits_test_bayes_sampled = conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_all_action_logits']
        untrained_metaRL_a1_probs_test_bayes_sampled = conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_all_action_probs'][:, :, 1]

        transformed_bayes_states_test_bayes_sampled = pca_model_bayes_states_bayes_sampled.transform(bayes_states_test_bayes_sampled.reshape(-1, bayes_state_dim)).\
            reshape(sma_num_testing_envs, sma_total_trials, bayes_state_dim)
        transformed_metaRL_rnn_states_test_bayes_sampled = pca_model_metaRL_rnn_states_bayes_sampled.transform(metaRL_rnn_states_test_bayes_sampled.reshape(-1, metaRL_rnn_state_dim)).\
            reshape(sma_num_testing_envs, sma_total_trials, metaRL_rnn_state_dim)
        transformed_metaRL_rnn_states_test_mean_bayes_sampled = np.average(transformed_metaRL_rnn_states_test_bayes_sampled, axis=(0,1))
        transformed_untrained_metaRL_rnn_states_test_bayes_sampled = pca_model_untrained_metaRL_rnn_states_bayes_sampled.transform(untrained_metaRL_rnn_states_test_bayes_sampled.reshape(-1, metaRL_rnn_state_dim)).\
            reshape(sma_num_testing_envs, sma_total_trials, metaRL_rnn_state_dim)
        transformed_untrained_metaRL_rnn_states_test_mean_bayes_sampled = np.average(transformed_untrained_metaRL_rnn_states_test_bayes_sampled, axis=(0,1))

        if args.exp_label == 'rl2':
            if len(args.layers_after_rnn) > 0:  # if bottleneck layer exists
                metaRL_bottleneck_states_test_bayes_sampled = conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_bottleneck_states']
                untrained_metaRL_bottleneck_states_test_bayes_sampled = conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_bottleneck_states']

                transformed_metaRL_bottleneck_states_test_bayes_sampled = pca_model_metaRL_bottleneck_states_bayes_sampled.transform(metaRL_bottleneck_states_test_bayes_sampled.reshape(-1, metaRL_bottleneck_state_dim)).\
                    reshape(sma_num_testing_envs, sma_total_trials, metaRL_bottleneck_state_dim)
                transformed_untrained_metaRL_bottleneck_states_test_bayes_sampled = pca_model_untrained_metaRL_bottleneck_states_bayes_sampled.transform(untrained_metaRL_bottleneck_states_test_bayes_sampled.reshape(-1, metaRL_bottleneck_state_dim)).\
                    reshape(sma_num_testing_envs, sma_total_trials, metaRL_bottleneck_state_dim)

        if args.exp_label == 'mpc':
            metaRL_belief_states_test_bayes_sampled = conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_belief_states']
            metaRL_belief_states_mean_only_test_bayes_sampled = conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_belief_states_mean_only']
            untrained_metaRL_belief_states_test_bayes_sampled = conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_belief_states']
            untrained_metaRL_belief_states_mean_only_test_bayes_sampled = conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_belief_states_mean_only']

            transformed_metaRL_belief_states_test_bayes_sampled = pca_model_metaRL_belief_states_bayes_sampled.transform(metaRL_belief_states_test_bayes_sampled.reshape(-1, metaRL_belief_state_dim)).\
                reshape(sma_num_testing_envs, sma_total_trials, metaRL_belief_state_dim)
            transformed_untrained_metaRL_belief_states_test_bayes_sampled = pca_model_untrained_metaRL_belief_states_bayes_sampled.transform(untrained_metaRL_belief_states_test_bayes_sampled.reshape(-1, metaRL_belief_state_dim)).\
                reshape(sma_num_testing_envs, sma_total_trials, metaRL_belief_state_dim)
            transformed_metaRL_belief_states_test_mean_bayes_sampled = np.average(transformed_metaRL_belief_states_test_bayes_sampled, axis=(0,1))
            transformed_untrained_metaRL_belief_states_test_mean_bayes_sampled = np.average(transformed_untrained_metaRL_belief_states_test_bayes_sampled, axis=(0,1))


        # ------------------------------------------------------------------
        # get state mapping: for dataset 1, without first PCA'ed
        # ------------------------------------------------------------------

        # for trained_metaRL_rnn
        print('\nState mapping testing: trained_metaRL_rnn')
        mapped_bayes_states_test_from_metaRL_rnn_bayes_sampled, mapped_bayes_actions_test_from_metaRL_rnn_bayes_sampled = \
        get_mapped_bayes_states_and_actions(
            metaRL2bayes_mapper=metaRL_rnn2bayes_mapper_bayes_sampled,
            metaRL_states=metaRL_rnn_states_test_bayes_sampled,
            env_name=sma_env_name,
            total_trials=sma_total_trials,
            num_bandits=sma_num_bandits,
            is_pcaed=False
        )
        mapped_metaRL_rnn_states_test_from_bayes_bayes_sampled, mapped_metaRL_rnn_all_action_logits_test_from_bayes_bayes_sampled, \
        mapped_metaRL_rnn_actions_test_from_bayes_bayes_sampled = get_mapped_metaRL_states_and_actions(
            bayes2metaRL_mapper=bayes2metaRL_rnn_mapper_bayes_sampled,
            bayes_states=bayes_states_test_bayes_sampled,
            args=args,
            encoder=encoder,
            policy_network=policy_network,
            mapped_metaRL_state_type='rnn',
            is_pcaed=False,
            pca_model_metaRL_for_dataset=None,
            pcaed_metaRL_states_mean=None,
            bayes_state_dim=bayes_state_dim,
            ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
        )
        ## dissimilarity analysis
        normalized_mse_mapped_bayes_from_metaRL_rnn_bayes_sampled, \
        avg_expected_return_diff_mapped_bayes_from_metaRL_rnn_bayes_sampled, \
        normalized_mse_mapped_metaRL_rnn_from_bayes_bayes_sampled, \
        avg_expected_return_diff_mapped_metaRL_rnn_from_bayes_bayes_sampled = \
        dissimilarity_analysis(
            true_bayes_states=bayes_states_test_bayes_sampled,
            mapped_bayes_states=mapped_bayes_states_test_from_metaRL_rnn_bayes_sampled,
            true_metaRL_actions=conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_actions'],
            mapped_bayes_actions=mapped_bayes_actions_test_from_metaRL_rnn_bayes_sampled,
            true_metaRL_states=metaRL_rnn_states_test_bayes_sampled,
            mapped_metaRL_states=mapped_metaRL_rnn_states_test_from_bayes_bayes_sampled,
            true_bayes_actions=bayes_actions_test_bayes_sampled,
            mapped_metaRL_actions=mapped_metaRL_rnn_actions_test_from_bayes_bayes_sampled,
            env_name=sma_env_name,
            p_bandits=ref_trajectories_bayes_test_bayes_sampled['p_bandits'],
            r_bandits=None,
            goal_positions_t=ref_trajectories_bayes_test_bayes_sampled['goal_positions_t'][:, :-1],
            positions_t=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
        )
        lines.append('\nState mapping testing: trained_metaRL_rnn')
        lines.append('metaRL_rnn to bayes:')
        lines.append(f' normalized_mse_mapped_bayes_from_metaRL_rnn_bayes_sampled: {normalized_mse_mapped_bayes_from_metaRL_rnn_bayes_sampled}')
        lines.append(f' return_diff (metaRL_rnn - mapped_bayes): {avg_expected_return_diff_mapped_bayes_from_metaRL_rnn_bayes_sampled}')
        lines.append('bayes to metaRL_rnn:')
        lines.append(f' normalized_mse_mapped_metaRL_rnn_from_bayes_bayes_sampled: {normalized_mse_mapped_metaRL_rnn_from_bayes_bayes_sampled}')
        lines.append(f' return_diffs (bayes - mapped_metaRL_rnn): {avg_expected_return_diff_mapped_metaRL_rnn_from_bayes_bayes_sampled}')
        results['normalized_mse_mapped_bayes_from_metaRL_rnn_bayes_sampled'] = normalized_mse_mapped_bayes_from_metaRL_rnn_bayes_sampled
        results['avg_expected_return_diff_mapped_bayes_from_metaRL_rnn_bayes_sampled'] = avg_expected_return_diff_mapped_bayes_from_metaRL_rnn_bayes_sampled
        results['normalized_mse_mapped_metaRL_rnn_from_bayes_bayes_sampled'] = normalized_mse_mapped_metaRL_rnn_from_bayes_bayes_sampled
        results['avg_expected_return_diff_mapped_metaRL_rnn_from_bayes_bayes_sampled'] = avg_expected_return_diff_mapped_metaRL_rnn_from_bayes_bayes_sampled

        ## PCA on testing dataset
        mapped_metaRL_rnn_a1_probs_test_from_bayes_bayes_sampled = np.exp(mapped_metaRL_rnn_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2)[:, 1]/ (np.exp(mapped_metaRL_rnn_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2).sum(axis=1))
        fig_sma_pca_test_metaRL_rnn_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
            pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
            pca_model_metaRL=pca_model_metaRL_rnn_states_bayes_sampled,
            true_bayes_states=bayes_states_test_bayes_sampled.reshape(-1, bayes_state_dim),
            mapped_bayes_states=mapped_bayes_states_test_from_metaRL_rnn_bayes_sampled.reshape(-1, bayes_state_dim),
            true_metaRL_states=metaRL_rnn_states_test_bayes_sampled.reshape(-1, metaRL_rnn_state_dim),
            mapped_metaRL_states=mapped_metaRL_rnn_states_test_from_bayes_bayes_sampled.reshape(-1, metaRL_rnn_state_dim),
            true_bayes_actions=bayes_actions_test_bayes_sampled.reshape(-1,1),
            mapped_bayes_actions=mapped_bayes_actions_test_from_metaRL_rnn_bayes_sampled.reshape(-1,1),
            env_name=sma_env_name,
            true_metaRL_a1_probs=metaRL_a1_probs_test_bayes_sampled.reshape(-1,1),
            mapped_metaRL_a1_probs=mapped_metaRL_rnn_a1_probs_test_from_bayes_bayes_sampled.reshape(-1,1),
            true_metaRL_all_action_logits=metaRL_all_action_logits_test_bayes_sampled.reshape(-1, sma_num_bandits),
            mapped_metaRL_all_action_logits=mapped_metaRL_rnn_all_action_logits_test_from_bayes_bayes_sampled.reshape(-1, sma_num_bandits),
            total_trials=sma_total_trials,
            num_bandits=sma_num_bandits
        )
        fig_sma_pca_test_metaRL_rnn_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_test_metaRL_rnn_states_bayes_sampled.png'))

        # for untrained_metaRL_rnn
        print('\nState mapping testing: untrained_metaRL_rnn')
        mapped_bayes_states_test_from_untrained_metaRL_rnn_bayes_sampled, mapped_bayes_actions_test_from_untrained_metaRL_rnn_bayes_sampled = \
        get_mapped_bayes_states_and_actions(
            metaRL2bayes_mapper=untrained_metaRL_rnn2bayes_mapper_bayes_sampled,
            metaRL_states=untrained_metaRL_rnn_states_test_bayes_sampled,
            env_name=sma_env_name,
            total_trials=sma_total_trials,
            num_bandits=sma_num_bandits,
            is_pcaed=False,
            latent_goal_cart_solver=latent_cart_optimal_agent,
            ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
        )
        mapped_untrained_metaRL_rnn_states_test_from_bayes_bayes_sampled, mapped_untrained_metaRL_rnn_all_action_logits_test_from_bayes_bayes_sampled, \
        mapped_untrained_metaRL_rnn_actions_test_from_bayes_bayes_sampled = get_mapped_metaRL_states_and_actions(
            bayes2metaRL_mapper=bayes2untrained_metaRL_rnn_mapper_bayes_sampled,
            bayes_states=bayes_states_test_bayes_sampled,
            args=args,
            encoder=encoder,
            policy_network=untrained_policy_network,
            mapped_metaRL_state_type='rnn',
            is_pcaed=False,
            pca_model_metaRL_for_dataset=None,
            pcaed_metaRL_states_mean=None,
            bayes_state_dim=bayes_state_dim,
            ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
        )
        ## dissimilarity ananlysis
        normalized_mse_mapped_bayes_from_untrained_metaRL_rnn_bayes_sampled, \
        avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_rnn_bayes_sampled, \
        normalized_mse_mapped_untrained_metaRL_rnn_from_bayes_bayes_sampled, \
        avg_expected_return_diff_mapped_untrained_metaRL_rnn_from_bayes_bayes_sampled = \
        dissimilarity_analysis(
            true_bayes_states=bayes_states_test_bayes_sampled,
            mapped_bayes_states=mapped_bayes_states_test_from_untrained_metaRL_rnn_bayes_sampled,
            true_metaRL_actions=conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_actions'],
            mapped_bayes_actions=mapped_bayes_actions_test_from_untrained_metaRL_rnn_bayes_sampled,
            true_metaRL_states=untrained_metaRL_rnn_states_test_bayes_sampled,
            mapped_metaRL_states=mapped_untrained_metaRL_rnn_states_test_from_bayes_bayes_sampled,
            true_bayes_actions=bayes_actions_test_bayes_sampled,
            mapped_metaRL_actions=mapped_untrained_metaRL_rnn_actions_test_from_bayes_bayes_sampled,
            env_name=sma_env_name,
            p_bandits=ref_trajectories_bayes_test_bayes_sampled['p_bandits'],
            r_bandits=None,
            goal_positions_t=ref_trajectories_bayes_test_bayes_sampled['goal_positions_t'][:, :-1],
            positions_t=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
        )
        lines.append('\nState mapping testing: untrained_metaRL_rnn')
        lines.append('untrained_metaRL_rnn to bayes:')
        lines.append(f' normalized_mse_mapped_bayes_from_untrained_metaRL_rnnbayes_sampled: {normalized_mse_mapped_bayes_from_untrained_metaRL_rnn_bayes_sampled}')
        lines.append(f' return_diff (untrained_metaRL_rnn - mapped_bayes): {avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_rnn_bayes_sampled}')
        lines.append('bayes to untrained_metaRL_rnn:')
        lines.append(f' normalized_mse_mapped_untrained_metaRL_rnn_from_bayes_bayes_sampled: {normalized_mse_mapped_untrained_metaRL_rnn_from_bayes_bayes_sampled}')
        lines.append(f' return_diffs (bayes - untrained_mapped_metaRL_rnn): {avg_expected_return_diff_mapped_untrained_metaRL_rnn_from_bayes_bayes_sampled}')
        results['normalized_mse_mapped_bayes_from_untrained_metaRL_rnn_bayes_sampled'] = normalized_mse_mapped_bayes_from_untrained_metaRL_rnn_bayes_sampled
        results['avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_rnn_bayes_sampled'] = avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_rnn_bayes_sampled
        results['normalized_mse_mapped_untrained_metaRL_rnn_from_bayes_bayes_sampled'] = normalized_mse_mapped_untrained_metaRL_rnn_from_bayes_bayes_sampled
        results['avg_expected_return_diff_mapped_untrained_metaRL_rnn_from_bayes_bayes_sampled'] = avg_expected_return_diff_mapped_untrained_metaRL_rnn_from_bayes_bayes_sampled

        ## PCA on testing dataset
        mapped_untrained_metaRL_rnn_a1_probs_test_bayes_sampled = np.exp(mapped_untrained_metaRL_rnn_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2)[:, 1]/ (np.exp(mapped_untrained_metaRL_rnn_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2).sum(axis=1))
        fig_sma_pca_test_untrained_metaRL_rnn_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
            pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
            pca_model_metaRL=pca_model_untrained_metaRL_rnn_states_bayes_sampled,
            true_bayes_states=bayes_states_test_bayes_sampled.reshape(-1, bayes_state_dim),
            mapped_bayes_states=mapped_bayes_states_test_from_untrained_metaRL_rnn_bayes_sampled.reshape(-1, bayes_state_dim),
            true_metaRL_states=untrained_metaRL_rnn_states_test_bayes_sampled.reshape(-1, metaRL_rnn_state_dim),
            mapped_metaRL_states=mapped_untrained_metaRL_rnn_states_test_from_bayes_bayes_sampled.reshape(-1, metaRL_rnn_state_dim),
            true_bayes_actions=bayes_actions_test_bayes_sampled.reshape(-1,1),
            mapped_bayes_actions=mapped_bayes_actions_test_from_untrained_metaRL_rnn_bayes_sampled.reshape(-1,1),
            env_name=sma_env_name,
            true_metaRL_a1_probs=untrained_metaRL_a1_probs_test_bayes_sampled.reshape(-1,1),
            mapped_metaRL_a1_probs=mapped_untrained_metaRL_rnn_a1_probs_test_bayes_sampled.reshape(-1,1),
            true_metaRL_all_action_logits=untrained_metaRL_all_action_logits_test_bayes_sampled.reshape(-1, sma_num_bandits),
            mapped_metaRL_all_action_logits=mapped_untrained_metaRL_rnn_all_action_logits_test_from_bayes_bayes_sampled.reshape(-1, sma_num_bandits),
            total_trials=sma_total_trials,
            num_bandits=sma_num_bandits
        )
        fig_sma_pca_test_untrained_metaRL_rnn_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_test_untrained_metaRL_rnn_states_bayes_sampled.png'))


        if args.exp_label == 'rl2':
            if len(args.layers_after_rnn) > 0:  # if bottleneck layer exists
                # for trained_metaRL_bottleneck
                print('\nState mapping testing: trained_metaRL_bottleneck')
                mapped_bayes_states_test_from_metaRL_bottleneck_bayes_sampled, mapped_bayes_actions_test_from_metaRL_bottleneck_bayes_sampled = \
                get_mapped_bayes_states_and_actions(
                    metaRL2bayes_mapper=metaRL_bottleneck2bayes_mapper_bayes_sampled,
                    metaRL_states=metaRL_bottleneck_states_test_bayes_sampled,
                    env_name=sma_env_name,
                    total_trials=sma_total_trials,
                    num_bandits=sma_num_bandits,
                    is_pcaed=False,
                    latent_goal_cart_solver=latent_cart_optimal_agent,
                    ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
                )
                mapped_metaRL_bottleneck_states_test_from_bayes_bayes_sampled, mapped_metaRL_bottleneck_all_action_logits_test_from_bayes_bayes_sampled, \
                mapped_metaRL_bottleneck_actions_test_from_bayes_bayes_sampled = get_mapped_metaRL_states_and_actions(
                    bayes2metaRL_mapper=bayes2metaRL_bottleneck_mapper_bayes_sampled,
                    bayes_states=bayes_states_test_bayes_sampled,
                    args=args,
                    encoder=encoder,
                    policy_network=policy_network,
                    mapped_metaRL_state_type='bottleneck',
                    is_pcaed=False,
                    pca_model_metaRL_for_dataset=None,
                    pcaed_metaRL_states_mean=None,
                    bayes_state_dim=bayes_state_dim,
                    ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
                )
                ## dissimilarity analysis
                normalized_mse_mapped_bayes_from_metaRL_bottleneck_bayes_sampled, \
                avg_expected_return_diff_mapped_bayes_from_metaRL_bottleneck_bayes_sampled, \
                normalized_mse_mapped_metaRL_bottleneck_from_bayes_bayes_sampled, \
                avg_expected_return_diff_mapped_metaRL_bottleneck_from_bayes_bayes_sampled = \
                dissimilarity_analysis(
                    true_bayes_states=bayes_states_test_bayes_sampled,
                    mapped_bayes_states=mapped_bayes_states_test_from_metaRL_bottleneck_bayes_sampled,
                    true_metaRL_actions=conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_actions'],
                    mapped_bayes_actions=mapped_bayes_actions_test_from_metaRL_bottleneck_bayes_sampled,
                    true_metaRL_states=metaRL_bottleneck_states_test_bayes_sampled,
                    mapped_metaRL_states=mapped_metaRL_bottleneck_states_test_from_bayes_bayes_sampled,
                    true_bayes_actions=bayes_actions_test_bayes_sampled,
                    mapped_metaRL_actions=mapped_metaRL_bottleneck_actions_test_from_bayes_bayes_sampled,
                    env_name=sma_env_name,
                    p_bandits=ref_trajectories_bayes_test_bayes_sampled['p_bandits'],
                    r_bandits=None,
                    goal_positions_t=ref_trajectories_bayes_test_bayes_sampled['goal_positions_t'][:, :-1],
                    positions_t=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
                )
                lines.append('\nState mapping testing: trained_metaRL_bottleneck')
                lines.append('metaRL_bottleneck to bayes:')
                lines.append(f' normalized_mse_mapped_bayes_from_metaRL_bottleneck_bayes_sampled: {normalized_mse_mapped_bayes_from_metaRL_bottleneck_bayes_sampled}')
                lines.append(f' return_diff (metaRL_bottleneck - mapped_bayes): {avg_expected_return_diff_mapped_bayes_from_metaRL_bottleneck_bayes_sampled}')
                lines.append('bayes to metaRL_bottleneck:')
                lines.append(f' normalized_mse_mapped_metaRL_bottleneck_from_bayes_bayes_sampled: {normalized_mse_mapped_metaRL_bottleneck_from_bayes_bayes_sampled}')
                lines.append(f' return_diffs (bayes - mapped_metaRL_bottleneck): {avg_expected_return_diff_mapped_metaRL_bottleneck_from_bayes_bayes_sampled}')
                results['normalized_mse_mapped_bayes_from_metaRL_bottleneck_bayes_sampled'] = normalized_mse_mapped_bayes_from_metaRL_bottleneck_bayes_sampled
                results['avg_expected_return_diff_mapped_bayes_from_metaRL_bottleneck_bayes_sampled'] = avg_expected_return_diff_mapped_bayes_from_metaRL_bottleneck_bayes_sampled
                results['normalized_mse_mapped_metaRL_bottleneck_from_bayes_bayes_sampled'] = normalized_mse_mapped_metaRL_bottleneck_from_bayes_bayes_sampled
                results['avg_expected_return_diff_mapped_metaRL_bottleneck_from_bayes_bayes_sampled'] = avg_expected_return_diff_mapped_metaRL_bottleneck_from_bayes_bayes_sampled

                ## PCA on testing dataset
                mapped_metaRL_bottleneck_a1_probs_test_from_bayes_bayes_sampled = np.exp(mapped_metaRL_bottleneck_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2)[:, 1]/ (np.exp(mapped_metaRL_bottleneck_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2).sum(axis=1))
                fig_sma_pca_test_metaRL_bottleneck_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
                    pca_model_bayes=pca_model_bayes_states_bayes_sampled,
                    pca_model_metaRL=pca_model_metaRL_bottleneck_states_bayes_sampled,
                    true_bayes_states=bayes_states_test_bayes_sampled.reshape(-1, bayes_state_dim),
                    mapped_bayes_states=mapped_bayes_states_test_from_metaRL_bottleneck_bayes_sampled.reshape(-1, bayes_state_dim),
                    true_metaRL_states=metaRL_bottleneck_states_test_bayes_sampled.reshape(-1, metaRL_bottleneck_state_dim),
                    mapped_metaRL_states=mapped_metaRL_bottleneck_states_test_from_bayes_bayes_sampled.reshape(-1, metaRL_bottleneck_state_dim),
                    true_bayes_actions=bayes_actions_test_bayes_sampled.reshape(-1,1),
                    mapped_bayes_actions=mapped_bayes_actions_test_from_metaRL_bottleneck_bayes_sampled.reshape(-1,1),
                    env_name=sma_env_name,
                    true_metaRL_a1_probs=metaRL_a1_probs_test_bayes_sampled.reshape(-1,1),
                    mapped_metaRL_a1_probs=mapped_metaRL_bottleneck_a1_probs_test_from_bayes_bayes_sampled.reshape(-1,1),
                    true_metaRL_all_action_logits=metaRL_all_action_logits_test_bayes_sampled.reshape(-1, sma_num_bandits),
                    mapped_metaRL_all_action_logits=mapped_metaRL_bottleneck_all_action_logits_test_from_bayes_bayes_sampled.reshape(-1, sma_num_bandits),
                    total_trials=sma_total_trials,
                    num_bandits=sma_num_bandits
                )
                fig_sma_pca_test_metaRL_bottleneck_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_test_metaRL_bottleneck_states_bayes_sampled.png'))

                # for untrained_metaRL_bottleneck
                print('\nState mapping testing: untrained_metaRL_bottleneck')
                mapped_bayes_states_test_from_untrained_metaRL_bottleneck_bayes_sampled, mapped_bayes_actions_test_from_untrained_metaRL_bottleneck_bayes_sampled = \
                get_mapped_bayes_states_and_actions(
                    metaRL2bayes_mapper=untrained_metaRL_bottleneck2bayes_mapper_bayes_sampled,
                    metaRL_states=untrained_metaRL_bottleneck_states_test_bayes_sampled,
                    env_name=sma_env_name,
                    total_trials=sma_total_trials,
                    num_bandits=sma_num_bandits,
                    is_pcaed=False,
                    latent_goal_cart_solver=latent_cart_optimal_agent,
                    ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
                )
                mapped_untrained_metaRL_bottleneck_states_test_from_bayes_bayes_sampled, mapped_untrained_metaRL_bottleneck_all_action_logits_test_from_bayes_bayes_sampled, \
                mapped_untrained_metaRL_bottleneck_actions_test_from_bayes_bayes_sampled = get_mapped_metaRL_states_and_actions(
                    bayes2metaRL_mapper=bayes2untrained_metaRL_bottleneck_mapper_bayes_sampled,
                    bayes_states=bayes_states_test_bayes_sampled,
                    args=args,
                    encoder=encoder,
                    policy_network=untrained_policy_network,
                    mapped_metaRL_state_type='bottleneck',
                    is_pcaed=False,
                    pca_model_metaRL_for_dataset=None,
                    pcaed_metaRL_states_mean=None,
                    bayes_state_dim=bayes_state_dim,
                    ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
                )
                ## dissimilarity analysis
                normalized_mse_mapped_bayes_from_untrained_metaRL_bottleneck_bayes_sampled, \
                avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_bottleneck_bayes_sampled, \
                normalized_mse_mapped_untrained_metaRL_bottleneck_from_bayes_bayes_sampled, \
                avg_expected_return_diff_mapped_untrained_metaRL_bottleneck_from_bayes_bayes_sampled = \
                dissimilarity_analysis(
                    true_bayes_states=bayes_states_test_bayes_sampled,
                    mapped_bayes_states=mapped_bayes_states_test_from_untrained_metaRL_bottleneck_bayes_sampled,
                    true_metaRL_actions=conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_actions'],
                    mapped_bayes_actions=mapped_bayes_actions_test_from_untrained_metaRL_bottleneck_bayes_sampled,
                    true_metaRL_states=untrained_metaRL_bottleneck_states_test_bayes_sampled,
                    mapped_metaRL_states=mapped_untrained_metaRL_bottleneck_states_test_from_bayes_bayes_sampled,
                    true_bayes_actions=bayes_actions_test_bayes_sampled,
                    mapped_metaRL_actions=mapped_untrained_metaRL_bottleneck_actions_test_from_bayes_bayes_sampled,
                    env_name=sma_env_name,
                    p_bandits=ref_trajectories_bayes_test_bayes_sampled['p_bandits'],
                    r_bandits=None,
                    goal_positions_t=ref_trajectories_bayes_test_bayes_sampled['goal_positions_t'][:, :-1],
                    positions_t=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
                )
                lines.append('\nState mapping testing: untrained_metaRL_bottleneck')
                lines.append('untrained_metaRL_bottleneck to bayes:')
                lines.append(f' normalized_mse_mapped_bayes_from_untrained_metaRL_bottleneck_bayes_sampled: {normalized_mse_mapped_bayes_from_untrained_metaRL_bottleneck_bayes_sampled}')
                lines.append(f' return_diff (untrained_metaRL_bottleneck - mapped_bayes): {avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_bottleneck_bayes_sampled}')
                lines.append('bayes to untrained_metaRL_bottleneck:')
                lines.append(f' normalized_mse_mapped_untrained_metaRL_bottleneck_from_bayes_bayes_sampled: {normalized_mse_mapped_untrained_metaRL_bottleneck_from_bayes_bayes_sampled}')
                lines.append(f' return_diffs (bayes - untrained_mapped_metaRL_bottleneck): {avg_expected_return_diff_mapped_untrained_metaRL_bottleneck_from_bayes_bayes_sampled}')
                results['normalized_mse_mapped_bayes_from_untrained_metaRL_bottleneck_bayes_sampled'] = normalized_mse_mapped_bayes_from_untrained_metaRL_bottleneck_bayes_sampled
                results['avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_bottleneck_bayes_sampled'] = avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_bottleneck_bayes_sampled
                results['normalized_mse_mapped_untrained_metaRL_bottleneck_from_bayes_bayes_sampled'] = normalized_mse_mapped_untrained_metaRL_bottleneck_from_bayes_bayes_sampled
                results['avg_expected_return_diff_mapped_untrained_metaRL_bottleneck_from_bayes_bayes_sampled'] = avg_expected_return_diff_mapped_untrained_metaRL_bottleneck_from_bayes_bayes_sampled

                ## PCA on testing dataset
                mapped_untrained_metaRL_bottleneck_a1_probs_test_bayes_sampled = np.exp(mapped_untrained_metaRL_bottleneck_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2)[:, 1]/ (np.exp(mapped_untrained_metaRL_bottleneck_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2).sum(axis=1))
                fig_sma_pca_test_untrained_metaRL_bottleneck_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
                    pca_model_bayes=pca_model_bayes_states_bayes_sampled,
                    pca_model_metaRL=pca_model_untrained_metaRL_bottleneck_states_bayes_sampled,
                    true_bayes_states=bayes_states_test_bayes_sampled.reshape(-1, bayes_state_dim),
                    mapped_bayes_states=mapped_bayes_states_test_from_untrained_metaRL_bottleneck_bayes_sampled.reshape(-1, bayes_state_dim),
                    true_metaRL_states=untrained_metaRL_bottleneck_states_test_bayes_sampled.reshape(-1, metaRL_bottleneck_state_dim),
                    mapped_metaRL_states=mapped_untrained_metaRL_bottleneck_states_test_from_bayes_bayes_sampled.reshape(-1, metaRL_bottleneck_state_dim),
                    true_bayes_actions=bayes_actions_test_bayes_sampled.reshape(-1,1),
                    mapped_bayes_actions=mapped_bayes_actions_test_from_untrained_metaRL_bottleneck_bayes_sampled.reshape(-1,1),
                    env_name=sma_env_name,
                    true_metaRL_a1_probs=untrained_metaRL_a1_probs_test_bayes_sampled.reshape(-1,1),
                    mapped_metaRL_a1_probs=mapped_untrained_metaRL_bottleneck_a1_probs_test_bayes_sampled.reshape(-1,1),
                    true_metaRL_all_action_logits=untrained_metaRL_all_action_logits_test_bayes_sampled.reshape(-1, sma_num_bandits),
                    mapped_metaRL_all_action_logits=mapped_untrained_metaRL_bottleneck_all_action_logits_test_from_bayes_bayes_sampled.reshape(-1, sma_num_bandits),
                    total_trials=sma_total_trials,
                    num_bandits=sma_num_bandits
                )
                fig_sma_pca_test_untrained_metaRL_bottleneck_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_test_untrained_metaRL_bottleneck_states_bayes_sampled.png'))


        if args.exp_label == 'mpc':
            # for trained_metaRL_belief
            print('\nState mapping testing: trained_metaRL_belief')
            mapped_bayes_states_test_from_metaRL_belief_bayes_sampled, mapped_bayes_actions_test_from_metaRL_belief_bayes_sampled = \
            get_mapped_bayes_states_and_actions(
                metaRL2bayes_mapper=metaRL_belief2bayes_mapper_bayes_sampled,
                metaRL_states=metaRL_belief_states_test_bayes_sampled,
                env_name=sma_env_name,
                total_trials=sma_total_trials,
                num_bandits=sma_num_bandits,
                is_pcaed=False,
                latent_goal_cart_solver=latent_cart_optimal_agent,
                ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
            )
            mapped_metaRL_belief_states_test_from_bayes_bayes_sampled, mapped_metaRL_belief_all_action_logits_test_from_bayes_bayes_sampled, \
            mapped_metaRL_belief_actions_test_from_bayes_bayes_sampled = get_mapped_metaRL_states_and_actions(
                bayes2metaRL_mapper=bayes2metaRL_belief_mapper_bayes_sampled,
                bayes_states=bayes_states_test_bayes_sampled,
                args=args,
                encoder=encoder,
                policy_network=policy_network,
                mapped_metaRL_state_type='belief',
                is_pcaed=False,
                pca_model_metaRL_for_dataset=None,
                pcaed_metaRL_states_mean=None,
                bayes_state_dim=bayes_state_dim,
                ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
            )
            ## dissimilarity analysis
            normalized_mse_mapped_bayes_from_metaRL_belief_bayes_sampled, \
            avg_expected_return_diff_mapped_bayes_from_metaRL_belief_bayes_sampled, \
            normalized_mse_mapped_metaRL_belief_from_bayes_bayes_sampled, \
            avg_expected_return_diff_mapped_metaRL_belief_from_bayes_bayes_sampled = \
            dissimilarity_analysis(
                true_bayes_states=bayes_states_test_bayes_sampled,
                mapped_bayes_states=mapped_bayes_states_test_from_metaRL_belief_bayes_sampled,
                true_metaRL_actions=conditioned_trajectories_metaRL_test_bayes_sampled['metaRL_actions'],
                mapped_bayes_actions=mapped_bayes_actions_test_from_metaRL_belief_bayes_sampled,
                true_metaRL_states=metaRL_belief_states_test_bayes_sampled,
                mapped_metaRL_states=mapped_metaRL_belief_states_test_from_bayes_bayes_sampled,
                true_bayes_actions=bayes_actions_test_bayes_sampled,
                mapped_metaRL_actions=mapped_metaRL_belief_actions_test_from_bayes_bayes_sampled,
                env_name=sma_env_name,
                p_bandits=ref_trajectories_bayes_test_bayes_sampled['p_bandits'],
                r_bandits=None,
                goal_positions_t=ref_trajectories_bayes_test_bayes_sampled['goal_positions_t'][:, :-1],
                positions_t=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
            )
            lines.append('\nState mapping testing: trained_metaRL_belief')
            lines.append('metaRL_belief to bayes:')
            lines.append(f' normalized_mse_mapped_bayes_from_metaRL_belief_bayes_sampled: {normalized_mse_mapped_bayes_from_metaRL_belief_bayes_sampled}')
            lines.append(f' return_diff (metaRL_belief - mapped_bayes): {avg_expected_return_diff_mapped_bayes_from_metaRL_belief_bayes_sampled}')
            lines.append('bayes to metaRL_belief:')
            lines.append(f' normalized_mse_mapped_metaRL_belief_from_bayes_bayes_sampled: {normalized_mse_mapped_metaRL_belief_from_bayes_bayes_sampled}')
            lines.append(f' return_diffs (bayes - mapped_metaRL_belief): {avg_expected_return_diff_mapped_metaRL_belief_from_bayes_bayes_sampled}')
            results['normalized_mse_mapped_bayes_from_metaRL_belief_bayes_sampled'] = normalized_mse_mapped_bayes_from_metaRL_belief_bayes_sampled
            results['avg_expected_return_diff_mapped_bayes_from_metaRL_belief_bayes_sampled'] = avg_expected_return_diff_mapped_bayes_from_metaRL_belief_bayes_sampled
            results['normalized_mse_mapped_metaRL_belief_from_bayes_bayes_sampled'] = normalized_mse_mapped_metaRL_belief_from_bayes_bayes_sampled
            results['avg_expected_return_diff_mapped_metaRL_belief_from_bayes_bayes_sampled'] = avg_expected_return_diff_mapped_metaRL_belief_from_bayes_bayes_sampled

            ## PCA on testing dataset
            mapped_metaRL_belief_a1_probs_test_from_bayes_bayes_sampled = np.exp(mapped_metaRL_belief_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2)[:, 1]/ (np.exp(mapped_metaRL_belief_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2).sum(axis=1))
            mapped_metaRL_belief_states_mean_only_test_from_bayes_bayes_sampled = mapped_metaRL_belief_states_test_from_bayes_bayes_sampled[:, : , :round(metaRL_belief_state_dim/2)]
            fig_sma_pca_test_metaRL_belief_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
                pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
                pca_model_metaRL=pca_model_metaRL_belief_states_mean_only_bayes_sampled,
                true_bayes_states=bayes_states_test_bayes_sampled.reshape(-1, bayes_state_dim),
                mapped_bayes_states=mapped_bayes_states_test_from_metaRL_belief_bayes_sampled.reshape(-1, bayes_state_dim),
                true_metaRL_states=metaRL_belief_states_mean_only_test_bayes_sampled.reshape(-1, round(metaRL_belief_state_dim/2)),
                mapped_metaRL_states=mapped_metaRL_belief_states_mean_only_test_from_bayes_bayes_sampled.reshape(-1, round(metaRL_belief_state_dim/2)),
                true_bayes_actions=bayes_actions_test_bayes_sampled.reshape(-1,1),
                mapped_bayes_actions=mapped_bayes_actions_test_from_metaRL_belief_bayes_sampled.reshape(-1,1),
                env_name=sma_env_name,
                true_metaRL_a1_probs=metaRL_a1_probs_test_bayes_sampled.reshape(-1,1),
                mapped_metaRL_a1_probs=mapped_metaRL_belief_a1_probs_test_from_bayes_bayes_sampled.reshape(-1,1),
                true_metaRL_all_action_logits=metaRL_all_action_logits_test_bayes_sampled.reshape(-1, sma_num_bandits),
                mapped_metaRL_all_action_logits=mapped_metaRL_belief_all_action_logits_test_from_bayes_bayes_sampled.reshape(-1, sma_num_bandits),
                total_trials=sma_total_trials,
                num_bandits=sma_num_bandits
            )
            fig_sma_pca_test_metaRL_belief_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_test_metaRL_belief_states_bayes_sampled.png'))

            # for untrained_metaRL_belief
            print('\nState mapping testing: untrained_metaRL_belief')
            mapped_bayes_states_test_from_untrained_metaRL_belief_bayes_sampled, mapped_bayes_actions_test_from_untrained_metaRL_belief_bayes_sampled = \
            get_mapped_bayes_states_and_actions(
                metaRL2bayes_mapper=untrained_metaRL_belief2bayes_mapper_bayes_sampled,
                metaRL_states=untrained_metaRL_belief_states_test_bayes_sampled,
                env_name=sma_env_name,
                total_trials=sma_total_trials,
                num_bandits=sma_num_bandits,
                is_pcaed=False,
                latent_goal_cart_solver=latent_cart_optimal_agent,
                ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
            )
            mapped_untrained_metaRL_belief_states_test_from_bayes_bayes_sampled, mapped_untrained_metaRL_belief_all_action_logits_test_from_bayes_bayes_sampled, \
            mapped_untrained_metaRL_belief_actions_test_from_bayes_bayes_sampled = get_mapped_metaRL_states_and_actions(
                bayes2metaRL_mapper=bayes2untrained_metaRL_belief_mapper_bayes_sampled,
                bayes_states=bayes_states_test_bayes_sampled,
                args=args,
                encoder=encoder,
                policy_network=untrained_policy_network,
                mapped_metaRL_state_type='belief',
                is_pcaed=False,
                pca_model_metaRL_for_dataset=None,
                pcaed_metaRL_states_mean=None,
                bayes_state_dim=bayes_state_dim,
                ref_observations=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
            )
            ## dissimilarity ananlysis
            normalized_mse_mapped_bayes_from_untrained_metaRL_belief_bayes_sampled, \
            avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_belief_bayes_sampled, \
            normalized_mse_mapped_untrained_metaRL_belief_from_bayes_bayes_sampled, \
            avg_expected_return_diff_mapped_untrained_metaRL_belief_from_bayes_bayes_sampled = \
            dissimilarity_analysis(
                true_bayes_states=bayes_states_test_bayes_sampled,
                mapped_bayes_states=mapped_bayes_states_test_from_untrained_metaRL_belief_bayes_sampled,
                true_metaRL_actions=conditioned_trajectories_untrained_metaRL_test_bayes_sampled['metaRL_actions'],
                mapped_bayes_actions=mapped_bayes_actions_test_from_untrained_metaRL_belief_bayes_sampled,
                true_metaRL_states=untrained_metaRL_belief_states_test_bayes_sampled,
                mapped_metaRL_states=mapped_untrained_metaRL_belief_states_test_from_bayes_bayes_sampled,
                true_bayes_actions=bayes_actions_test_bayes_sampled,
                mapped_metaRL_actions=mapped_untrained_metaRL_belief_actions_test_from_bayes_bayes_sampled,
                env_name=sma_env_name,
                p_bandits=ref_trajectories_bayes_test_bayes_sampled['p_bandits'],
                r_bandits=None,
                goal_positions_t=ref_trajectories_bayes_test_bayes_sampled['goal_positions_t'][:, :-1],
                positions_t=ref_trajectories_bayes_test_bayes_sampled['bayes_observations'][:, :-1]
            )
            lines.append('\nState mapping testing: untrained_metaRL_belief')
            lines.append('untrained_metaRL_belief to bayes:')
            lines.append(f' normalized_mse_mapped_bayes_from_untrained_metaRL_belief_bayes_sampled: {normalized_mse_mapped_bayes_from_untrained_metaRL_belief_bayes_sampled}')
            lines.append(f' return_diff (untrained_metaRL_belief - mapped_bayes): {avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_belief_bayes_sampled}')
            lines.append('bayes to untrained_metaRL_belief:')
            lines.append(f' normalized_mse_mapped_untrained_metaRL_belief_from_bayes_bayes_sampled: {normalized_mse_mapped_untrained_metaRL_belief_from_bayes_bayes_sampled}')
            lines.append(f' return_diffs (bayes - untrained_mapped_metaRL_belief): {avg_expected_return_diff_mapped_untrained_metaRL_belief_from_bayes_bayes_sampled}')
            results['normalized_mse_mapped_bayes_from_untrained_metaRL_belief_bayes_sampled'] = normalized_mse_mapped_bayes_from_untrained_metaRL_belief_bayes_sampled
            results['avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_belief_bayes_sampled'] = avg_expected_return_diff_mapped_bayes_from_untrained_metaRL_belief_bayes_sampled
            results['normalized_mse_mapped_untrained_metaRL_belief_from_bayes_bayes_sampled'] = normalized_mse_mapped_untrained_metaRL_belief_from_bayes_bayes_sampled
            results['avg_expected_return_diff_mapped_untrained_metaRL_belief_from_bayes_bayes_sampled'] = avg_expected_return_diff_mapped_untrained_metaRL_belief_from_bayes_bayes_sampled
            
            ## PCA on testing dataset
            mapped_untrained_metaRL_belief_a1_probs_test_bayes_sampled = np.exp(mapped_untrained_metaRL_belief_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2)[:, 1]/ (np.exp(mapped_untrained_metaRL_belief_all_action_logits_test_from_bayes_bayes_sampled).reshape(-1,2).sum(axis=1))
            mapped_untrained_metaRL_belief_states_mean_only_test_from_bayes_bayes_sampled = mapped_untrained_metaRL_belief_states_test_from_bayes_bayes_sampled[:, : , :round(metaRL_belief_state_dim/2)]
            fig_sma_pca_test_untrained_metaRL_belief_states_bayes_sampled = plot_PCA_bayes_and_metaRL(
                pca_model_bayes=pca_model_bayes_states_bayes_sampled, 
                pca_model_metaRL=pca_model_untrained_metaRL_belief_states_mean_only_bayes_sampled,
                true_bayes_states=bayes_states_test_bayes_sampled.reshape(-1, bayes_state_dim),
                mapped_bayes_states=mapped_bayes_states_test_from_untrained_metaRL_belief_bayes_sampled.reshape(-1, bayes_state_dim),
                true_metaRL_states=untrained_metaRL_belief_states_mean_only_test_bayes_sampled.reshape(-1, round(metaRL_belief_state_dim/2)),
                mapped_metaRL_states=mapped_untrained_metaRL_belief_states_mean_only_test_from_bayes_bayes_sampled.reshape(-1, round(metaRL_belief_state_dim/2)),
                true_bayes_actions=bayes_actions_test_bayes_sampled.reshape(-1,1),
                mapped_bayes_actions=mapped_bayes_actions_test_from_untrained_metaRL_belief_bayes_sampled.reshape(-1,1),
                env_name=sma_env_name,
                true_metaRL_a1_probs=untrained_metaRL_a1_probs_test_bayes_sampled.reshape(-1,1),
                mapped_metaRL_a1_probs=mapped_untrained_metaRL_belief_a1_probs_test_bayes_sampled.reshape(-1,1),
                true_metaRL_all_action_logits=untrained_metaRL_all_action_logits_test_bayes_sampled.reshape(-1, sma_num_bandits),
                mapped_metaRL_all_action_logits=mapped_untrained_metaRL_belief_all_action_logits_test_from_bayes_bayes_sampled.reshape(-1, sma_num_bandits),
                total_trials=sma_total_trials,
                num_bandits=sma_num_bandits
            )
            fig_sma_pca_test_untrained_metaRL_belief_states_bayes_sampled.savefig(os.path.join(trained_model_path, 'fig_sma_pca_test_untrained_metaRL_belief_states_bayes_sampled.png'))

        
        with open(os.path.join(trained_model_path, 'state_machine_ana.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        
        with open(os.path.join(trained_model_path, 'sma_results.pickle'), 'wb') as fo:
            pickle.dump(results, fo)

        plt.close('all')

    print('###### all analyses completed ######')