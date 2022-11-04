#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import logging
from train_test import train, test
import warnings
from arg_parser import init_parser
from setproctitle import setproctitle as ptitle
from normalized_env import NormalizedEnv
import gym

from gym import spaces
from envs import gym_seqssg

def set_gym_space_attr(gym_space):
    '''Set missing gym space attributes for standardization'''
    if isinstance(gym_space, spaces.Box):
        print("Continuous")
        # setattr(gym_space, 'is_discrete', False)
    elif isinstance(gym_space, spaces.Discrete):
        print("Discrete")
        # setattr(gym_space, 'is_discrete', True)
        # setattr(gym_space, 'low', 0)
        # setattr(gym_space, 'high', gym_space.n)
    # elif isinstance(gym_space, spaces.MultiBinary):
    #     setattr(gym_space, 'is_discrete', True)
    #     setattr(gym_space, 'low', np.full(gym_space.n, 0))
    #     setattr(gym_space, 'high', np.full(gym_space.n, 2))
    elif isinstance(gym_space, spaces.MultiDiscrete):
        print("MultiDiscrete")
        # setattr(gym_space, 'is_discrete', True)
        # setattr(gym_space, 'low', np.zeros_like(gym_space.nvec))
        # setattr(gym_space, 'high', np.array(gym_space.nvec))
    else:
        raise ValueError('gym_space not recognized')

if __name__ == "__main__":
    ptitle('WOLP_DDPG')
    warnings.filterwarnings('ignore')
    parser = init_parser('WOLP_DDPG')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]

    from util import get_output_folder, setup_logger
    from wolp_agent import WolpertingerAgent

    args.save_model_dir = get_output_folder('../output', args.env)
    env = gym.make(args.env)
    print("action space", env.action_space)
    print("obs space", env.observation_space)

    # print("max_actions", env.action_space.n)


    # env_idd = "SeqSSG-no-cstr-v0"
    # args.save_model_dir = get_output_folder('../output', env_idd)
    # env = gym.make(env_idd)
    # print("action space", env.action_space)
    # print("obs space", env.observation_space)

    # set_gym_space_attr(env.action_space)
    # print("env", env)
    # print(env.action_space.nvec)
    # print('low', np.zeros_like(env.action_space.nvec))
    # print('high', np.array(env.action_space.nvec))
    # print(env.action_space.shape[0])
    # print("act space high", env.action_space.high)
    # print("max_actions", env.action_space.n)


    continuous = None
    multi_discrete = None
    if isinstance(env.action_space, spaces.Box):
        print("Continuous")
        print("act space low", env.action_space.low)
        print("act space high", env.action_space.high)
        nb_states = env.observation_space.shape[0]
        nb_actions = env.action_space.shape[0]
        action_high = env.action_space.high
        action_low = env.action_space.low
        continuous = True
        env = NormalizedEnv(env)
    elif isinstance(env.action_space, spaces.Discrete):     #discrete action for 1 dimension
        print("Discrete")
        nb_states = env.observation_space.shape[0]
        nb_actions = 1  # the dimension of actions, usually it is 1. Depend on the environment.
        max_actions = env.action_space.n
        continuous = False
        multi_discrete = False
    elif isinstance(env.action_space, spaces.MultiDiscrete):
        print("MultiDiscrete")
        nb_states = env.observation_space.shape[0]
        nb_actions = env.action_space.shape[0]
        # max_actions = env.action_space.shape[0] #sum or product 5+5+5 or 5*5*5
        continuous = False
        multi_discrete = True
        action_high = np.array(env.action_space.nvec) - 1
        print("action_high", action_high)
        action_low = np.zeros_like(env.action_space.nvec)
        max_actions = np.prod(env.action_space.nvec)
        print(max_actions)
    else:
        raise ValueError('gym_space not recognized')


    # try:
    #     # continuous action
    #     nb_states = env.observation_space.shape[0]
    #     nb_actions = env.action_space.shape[0]
    #     action_high = env.action_space.high
    #     action_low = env.action_space.low
    #     continuous = True
    #     env = NormalizedEnv(env)
    # except IndexError:
    #     # discrete action for 1 dimension
    #     nb_states = env.observation_space.shape[0]
    #     nb_actions = 1  # the dimension of actions, usually it is 1. Depend on the environment.
    #     max_actions = env.action_space.n
    #     continuous = False


    # nb_states = env.observation_space.shape[0]
    # nb_actions = env.action_space.shape[0]
    # max_actions = env.action_space.shape[0] #sum or product 5+5+5 or 5*5*5
    # continuous = False


    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    if continuous:
        agent_args = {
            'continuous': continuous,
            'multi_discrete': multi_discrete,
            'max_actions': None,
            'action_low': action_low,
            'action_high': action_high,
            'nb_states': nb_states,
            'nb_actions': nb_actions,
            'args': args,
        }
    else:
        if not multi_discrete: #1-d discrete
            agent_args = {
                'continuous': continuous,
                'multi_discrete': multi_discrete,
                'max_actions': max_actions,
                'action_low': None,
                'action_high': None,
                'nb_states': nb_states,
                'nb_actions': nb_actions,
                'args': args,
            }
        else: #multi-discrete
            agent_args = {
                'continuous': continuous,
                'multi_discrete': multi_discrete,
                'max_actions': max_actions,
                'action_low': action_low,
                'action_high': action_high,
                'nb_states': nb_states,
                'nb_actions': nb_actions,
                'args': args,
            }


    agent = WolpertingerAgent(**agent_args)

    if args.load:
        agent.load_weights(args.load_model_dir)

    if args.gpu_ids[0] >= 0 and args.gpu_nums > 0:
        agent.cuda_convert()

    # set logger, log args here
    log = {}
    if args.mode == 'train':
        setup_logger('RS_log', r'{}/RS_train_log'.format(args.save_model_dir))
    elif args.mode == 'test':
        setup_logger('RS_log', r'{}/RS_test_log'.format(args.save_model_dir))
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
    log['RS_log'] = logging.getLogger('RS_log')
    d_args = vars(args)
    d_args['max_actions'] = args.max_actions
    for key in agent_args.keys():
        if key == 'args':
            continue
        d_args[key] = agent_args[key]
    for k in d_args.keys():
        log['RS_log'].info('{0}: {1}'.format(k, d_args[k]))

    if args.mode == 'train':

        train_args = {
            'continuous': continuous,
            'multi_discrete': multi_discrete,
            'env': env,
            'agent': agent,
            'max_episode': args.max_episode,
            'warmup': args.warmup,
            'save_model_dir': args.save_model_dir,
            'max_episode_length': args.max_episode_length,
            'logger': log['RS_log'],
            'save_per_epochs': args.save_per_epochs
        }

        train(**train_args)

    elif args.mode == 'test':

        test_args = {
            'env': env,
            'agent': agent,
            'model_path': args.load_model_dir,
            'test_episode': args.test_episode,
            'max_episode_length': args.max_episode_length,
            'logger': log['RS_log'],
            'save_per_epochs': args.save_per_epochs
        }

        test(**test_args)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
