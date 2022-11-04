import stable_baselines3


def add_rl_args(parser):
    # RL algo params
    parser.add_argument('--rl_algo', type=str, default='A2C')
    parser.add_argument('--n_steps', type=int, default=-1,
        help='Number of transitions to collect from one env (-1 to use default).'
    )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=str, default="3e-4")
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=1.0)

    # Training params
    parser.add_argument('--total_timesteps', type=float, default=1e6)
    parser.add_argument('--normalize', type=eval, default=False)
    parser.add_argument('--normalize_reward', type=eval, default=True)
    parser.add_argument('--normalize_obs', type=eval, default=True)
    parser.add_argument('--gamma_norm_env', type=float, default=0.99)
    parser.add_argument('--policy_save_interval', type=int, default=0)
    parser.add_argument('--policy_eval_interval', type=int, default=0)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--format_str', type=str, default='tensorboard_stdout')
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)

    # Evaluation params
    parser.add_argument('--n_eval_episodes', type=int, default=10)

    # Log params
    parser.add_argument('--log_wandb', type=eval, default=False)
    parser.add_argument('--log_interval', type=int, default=-1)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument('--project', type=str, default='constrained_rl')
    parser.add_argument('--run_notes', type=str, default='')

def get_rl_id(args):
    return args.rl_algo

def get_rl_algo(args):
    if args.rl_algo == "A2C":
        return stable_baselines3.A2C
    elif args.rl_algo == "PPO":
        return stable_baselines3.PPO
    else:
        raise NotImplementedError()
