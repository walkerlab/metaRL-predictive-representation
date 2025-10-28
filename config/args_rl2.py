import argparse
from utils.helpers import boolean_argument


def get_args():
    parser = argparse.ArgumentParser()

    # -- ENVIRONMENT --
    parser.add_argument('--env_name', default='StatBernoulliBandit2ArmIndependent-v0',
                        help='environment to train on')
    parser.add_argument('--exp_label', default='rl2', 
                        help='name of method')
    parser.add_argument('--max_episode_steps', type=int, default=60)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--time_as_state', type=boolean_argument, default=False,
                        help='whether to use timestep as state for inputs to the policy')
    parser.add_argument('--deterministic_execution', type=boolean_argument, default=False,
                        help='?')

    # -- LOGGING --
    parser.add_argument('--results_log_dir', default=None, help='directory to save results (None uses ./logs)')
    parser.add_argument('--run_id', type=int, default=99, help='to prevent simulaneously launced jobs having the same log_dir')
    parser.add_argument('--log_interval', type=int, default=500, help='log interval, one log per n updates')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval, one save per n updates')
    parser.add_argument('--save_intermediate_models', type=boolean_argument, default=False, help='save all models')
    parser.add_argument('--eval_interval', type=int, default=500, help='eval interval, one eval per n updates')
    parser.add_argument('--eval_ids', nargs='+', type=int, default=[],
                         help='epoch ids for evaluation')
    parser.add_argument('--num_eval_envs', type=int, default=100, help='number of environments for evaluation during training')
    parser.add_argument('--vis_interval', type=int, default=500, help='visualisation interval, one eval per n updates')
    
    # -- POLICY --
    # general
    parser.add_argument('--num_updates', type=int, default=1e3,
                        help='number of updates to train')
    parser.add_argument('--policy_num_steps_per_update', type=int, default=60,
                        help='number of env steps to do (per process) before updating')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='how many training CPU processes / parallel environments to use (default: 16)')
    parser.add_argument('--deterministic_policy', type=boolean_argument, default=False,
                        help='if false, sample from policy distribution')
    # RNN setup
    parser.add_argument('--shared_rnn', type=boolean_argument, default=True, 
                        help='whether to use a shared RNN for both actor and critic')
    parser.add_argument('--layers_before_rnn', nargs='+', type=int, default=[])
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, 
                        help='dimensionality of RNN hidden state')
    parser.add_argument('--layers_after_rnn', nargs='+', action='extend', type=int, default=[])
    parser.add_argument('--rnn_cell_type', type=str, default='vanilla', 
                        help='choose: vanilla, gru')
    parser.add_argument('--action_embed_dim', type=int, default=8)
    parser.add_argument('--state_embed_dim', type=int, default=8)
    parser.add_argument('--reward_embed_dim', type=int, default=8)
    parser.add_argument('--policy_net_activation_function', type=str, default='tanh',
                        help='choose: tanh, relu, leaky-relu')
    parser.add_argument('--policy_net_initialization_method', type=str, default='normc',
                        help='choose: orthogonal, normc')
    parser.add_argument('--action_pred_type', type=str, default='bernoulli',
                        help='choose: '
                             'bernoulli (predict p(r=1|s))'
                             'categorical (predict p(r=1|s) but use softmax instead of sigmoid)'
                             'deterministic (treat as regression problem)')
    # RL algorithm
    parser.add_argument('--policy_gamma', type=float, default=0.95, help='discount factor for rewards')
    parser.add_argument('--policy_use_gae', type=boolean_argument, default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--policy_lambda', type=float, default=0.95, help='gae parameter')
    parser.add_argument('--policy_algorithm', type=str, default='ppo', help='choose: a2c, ppo')
    parser.add_argument('--policy_critic_loss_coeff', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--policy_entropy_loss_coeff', type=float, default=0.01, help='entropy term coefficient')
    parser.add_argument('--policy_optimizer', type=str, default='adam', help='choose: rmsprop, adam')
    parser.add_argument('--policy_eps', type=float, default=1e-8, help='optimizer epsilon (1e-8 for ppo, 1e-5 for a2c)')
    parser.add_argument('--policy_lr', type=float, default=0.0007, help='learning rate (default: 7e-4)')
    parser.add_argument('--policy_anneal_lr', type=boolean_argument, default=False)
    parser.add_argument('--policy_max_grad_norm', type=float, default=0.5, help='max norm of gradients')
    parser.add_argument('--normalize_advantages', type=boolean_argument, default=False,
                        help='normalize advantages')
    parser.add_argument('--normalize_rew_for_policy', type=boolean_argument, default=False, 
                        help='normalize rewards for policy')

    return parser
