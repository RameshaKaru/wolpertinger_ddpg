from json import loads


def add_env_args(parser):
    # Env params
    parser.add_argument('--env_id', type=str, default='LunarLander-v2')
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--n_eval_envs', type=int, default=1)
    parser.add_argument('--max_episode_steps', type=int, default=None)
    parser.add_argument('--vec_env_type', type=str, default='dummy', choices=['dummy', 'subproc'])
    tmp_args, _ = parser.parse_known_args()

    if "ERSEnv" in tmp_args.env_id:
        import gym_ERSLE
        parser.add_argument('--convert_act', type=eval, default=False,)
    elif "SeqSSG" in tmp_args.env_id:
        from . import gym_seqssg
        parser.add_argument('--conf', type=str, default='automatic', 
                            choices=['automatic', 'manual', 'a', 'm'])
        parser.add_argument('--num_target', type=int, default=10)
        parser.add_argument('--graph_type', type=str, default='random_scale_free')
        parser.add_argument('--num_res', type=int, default=5)
        parser.add_argument('--groups', type=int, default=3, help='number of groups')
        parser.add_argument('--payoff_matrix', type=loads, default=None)
        parser.add_argument('--adj_matrix', type=loads, default=None)
        parser.add_argument('--norm_adj_matrix', type=loads, default=None)
        parser.add_argument('--def_constraints', type=loads, default=None)
        parser.add_argument('--no_constraint', type=eval, default=True)

def get_env_id(args):
    return args.env_id
