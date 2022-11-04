#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stable_baselines3 import DDPG
import numpy as np

"""
    This class is an intermediary between stable baselines 3 and the Wolpertinger implementation
"""


class DDPGSB3(DDPG):

    def __init__(self, args, nb_states, nb_actions):
        super().__init__(policy="MlpPolicy",
                         env=args.env,
                         learning_rate=1e-3,        #args.p-lr     
                         buffer_size=1_000_000,     # 1e6
                         learning_starts=100,
                         batch_size=100,            #args.bsize
                         tau=0.005,                 #args.tau_update
                         gamma=0.99,                #args.gamma
                         train_freq=(1, "episode"),
                         gradient_steps=-1,
                         action_noise=None,
                         replay_buffer_class=None,
                         replay_buffer_kwargs=None,
                         optimize_memory_usage=False,
                         tensorboard_log=None,
                         create_eval_env=False,
                         policy_kwargs=None,
                         verbose=0,
                         seed=None,                 #agrs.seed
                         device="auto",
                         _init_setup_model=True
                         )

        self.nb_actions = nb_actions
        self.is_training = True
        # self.memory
        # self.random_process

    #select the next action given the observation
    def select_action(self, s_t, decay_epsilon):
        action, _ = super.predict()
        return action

    #select a random action
    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        return action


    ### Below functions might be not necessary if DDPG.learn is used directly?
    def observe(self, r_t, s_t1, done):
        pass

    def reset(self, s_t):
        # self.s_t = s_t
        # self.random_process.reset_states()
        pass

    def load_weights(self, dir):
        pass

    def eval(self):
        pass

    def cuda_convert():
        pass
