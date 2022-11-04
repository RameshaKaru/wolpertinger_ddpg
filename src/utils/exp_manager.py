import argparse
import os
from collections import OrderedDict
from typing import Any, List, Mapping, Optional, Type

import gym
import yaml
from envs.util import ActMapVecEnvWrapper
from envs.util import SubprocVecEnv_v2 as SubprocVecEnv
from envs.util import assginment_rsc_to_stat_2d
from gym.wrappers import TimeLimit
from policies.act_mask.base import ActorCriticActMaskPolicy
from policies.flow_policy.base import ActorCriticFlowPolicy
from policies.flow_policy.callbacks import (ActCorrCallback,
                                            LogFlowNetDistCallback,
                                            LogModelStructureCallback,
                                            ModelSummaryCallback,
                                            TrackModelGradCallback,
                                            UpdateFlowNetCallback)
from stable_baselines3.common import base_class, policies
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import (
    is_image_space, is_image_space_channels_first)
from stable_baselines3.common.vec_env import (DummyVecEnv, VecEnv,
                                              VecFrameStack, VecNormalize,
                                              VecTransposeImage,
                                              is_vecenv_wrapped)

from utils.callbacks import (SaveVecNormalizeCallback, SeqSSGCallback,
                             UpdMaskFnCallback)
from utils.logger import setup_logging
from utils.util import (REPALCE_PAIR, clean_dict, format_str_to_strs,
                        no_log_configs)


class ExperimentManager:
    """
    Experiment manager: read the hyperparameters,
    preprocess them, create the environment and the RL model.
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        total_timesteps: int,
        rl_cls: Type[base_class.BaseAlgorithm],
        rl_kwargs: Mapping[str, Any],
        policy_cls: Type[policies.BasePolicy],
        policy_kwargs: Mapping[str, Any],
        n_envs: int,
        env_kwargs: Mapping[str, Any],
        normalize: bool,
        normalize_kwargs: Mapping[str, Any],
        policy_save_interval: int,
        policy_eval_interval: bool,
        n_eval_episodes: int,
        seed: int,
        env_id: str, 
        rl_id: str, 
        policy_id: str,
        wandb_kwargs: Mapping[str, Any],
        log_interval: int,
        n_eval_envs: int=1,
        max_episode_steps: int=None,
        vec_env_type: str = "dummy",
        convert_act: bool=False,
        format_str: Optional[str]=None,
        log_flow_dist: bool=False,
        log_weight_grad: bool=False,
        log_model_structure: bool=False,
        model_summary : bool=False,
        has_act_corr: bool=False,
        act_corr_prot: str=None,
        verbose: int=1,
    ) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps
        self.rl_cls = rl_cls
        self.rl_kwargs = rl_kwargs
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs
        self.n_envs = n_envs 
        self.env_kwargs = env_kwargs
        self.normalize = normalize 
        self.normalize_kwargs = normalize_kwargs
        self.policy_save_interval = policy_save_interval
        self.policy_eval_interval = policy_eval_interval
        self.n_eval_envs = n_eval_envs
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.env_id = env_id
        self.rl_id = rl_id
        self.policy_id = policy_id
        self.wandb_kwargs = wandb_kwargs
        self.log_interval = log_interval
        self.max_episode_steps = max_episode_steps
        self.vec_env_type = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]
        self.convert_act = convert_act
        self.format_str = format_str
        self.log_flow_dist = log_flow_dist
        self.log_weight_grad = log_weight_grad
        self.log_model_structure = log_model_structure
        self.model_summary = model_summary
        self.continue_training = False # TODO: continue training - implement continue training
        self.verbose = verbose
        self.deterministic_eval = not self.is_atari(env_id)
        self.frame_stack = None  # TODO: add args
        self.has_act_corr = has_act_corr
        self.act_corr_prot = act_corr_prot

        self._setup_experiment(
            OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
        )
    
    def create_envs(self, n_envs: int, eval_env: bool = False) -> VecEnv:
        """
        Create the environment and wrap it if necessary.

        Args:
            n_envs:
            eval_env: Whether is it an environment used for evaluation or not
        
        Return: 
            the vectorized environment, with appropriate wrappers
        """
        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env else os.path.join(self.log_dir, "log")

        monitor_kwargs = {}
        # Special case for GoalEnvs: log success rate too
        if "Neck" in self.env_id or self.is_robotics_env(self.env_id) or "parking-v0" in self.env_id:
            monitor_kwargs = dict(info_keywords=("is_success",))

        # On most env, SubprocVecEnv does not help and is quite memory hungry
        # therefore we use DummyVecEnv by default
        venv = make_vec_env(
            env_id=self.env_id,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=self.env_kwargs,
            monitor_dir=log_dir,
            vec_env_cls=self.vec_env_type,
            monitor_kwargs=monitor_kwargs,
        )

        # Wrap the environment if necessary
        venv = self._wrap_env(venv)

        # Wrap the env into a VecNormalize wrapper if needed
        # and load saved statistics when present
        venv = self._maybe_normalize(venv, eval_env)

        # Optional Frame-stacking
        if self.frame_stack is not None:
            n_stack = self.frame_stack
            venv = VecFrameStack(venv, n_stack)
            if self.verbose > 0:
                print(f"Stacking {n_stack} frames")

        if not is_vecenv_wrapped(venv, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(venv.observation_space, gym.spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in venv.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)
                    )
            else:
                wrap_with_vectranspose = is_image_space(venv.observation_space) and not is_image_space_channels_first(
                    venv.observation_space
                )

            if wrap_with_vectranspose:
                if self.verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                venv = VecTransposeImage(venv)

        return venv
        
    def create_callbacks(self) -> List[BaseCallback]:
        callback_objs = []
        # callback of saving policy
        if self.policy_save_interval > 0:
            self.policy_save_interval = max(1, self.policy_save_interval//self.n_envs)
            callback_objs.append(
                CheckpointCallback(
                    save_freq=self.policy_save_interval, save_path=self.policy_dir,
                    name_prefix='rl_model',verbose=1))
        # callback of eval policy
        if self.policy_eval_interval > 0:
            self.policy_eval_interval = max(1, self.policy_eval_interval//self.n_envs)
            if self.verbose > 0: print("Creating test environment")

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.policy_dir)
            callback_objs.append(
                EvalCallback(
                    self.create_envs(self.n_eval_envs, eval_env=True),
                    callback_on_new_best=save_vec_normalize,
                    best_model_save_path=self.policy_dir,
                    n_eval_episodes=self.n_eval_episodes,
                    log_path=os.path.join(self.log_dir, "log"),
                    eval_freq=self.policy_eval_interval,
                    deterministic=self.deterministic_eval,
                ))
        # callback of correct actions
        if self.has_act_corr:
            callback_objs.append(ActCorrCallback(corr_prot=self.act_corr_prot))
        # callback of updating flow net
        if self.policy_cls is ActorCriticFlowPolicy:
            callback_objs.append(UpdateFlowNetCallback())
        # callback of logging flow distribution
        if self.log_flow_dist: callback_objs.append(LogFlowNetDistCallback())
        # callback of logging weights and gradients
        if self.log_weight_grad: callback_objs.append(
            TrackModelGradCallback(n_steps_track=3800, pairs4replace=REPALCE_PAIR))
        # callback of logging model structure
        if self.log_model_structure: callback_objs.append(LogModelStructureCallback())
        # callback of print model summary
        if self.model_summary: callback_objs.append(ModelSummaryCallback())
        # callback of pass mask function to the policy
        if self.policy_cls is ActorCriticActMaskPolicy:
            callback_objs.append(UpdMaskFnCallback())
        # callback of logging the status of SeqSSG env
        if 'SeqSSG' in self.env_id:
            callback_objs.append(SeqSSGCallback())

        return callback_objs

    def learn(self) -> None:
        kwargs = {}
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}

        if len(self.callback) > 0:
            kwargs["callback"] = self.callback

        try:
            self.rl_algo.learn(self.total_timesteps, **kwargs)
        except KeyboardInterrupt:
            # this allows to save the model when interrupting training
            pass
        finally:
            # Release resources
            try:
                self.rl_algo.env.close()
            except EOFError:
                pass

    def save_trained_model(self) -> None:
        print(f"Saving to {self.policy_dir}")
        self.rl_algo.save(f"{self.policy_dir}/{self.env_id.lower()}")

        if self.normalize:
            # Important: save the running average, for testing the agent we need that normalization
            self.rl_algo.get_vec_normalize_env().save(os.path.join(self.policy_dir, "vecnormalize_final.pkl"))

    @staticmethod
    def is_atari(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "AtariEnv" in str(entry_point)

    @staticmethod
    def is_bullet(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "pybullet_envs" in str(entry_point)

    @staticmethod
    def is_robotics_env(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "gym.envs.robotics" in str(entry_point) or "panda_gym.envs" in str(entry_point)

    def _setup_experiment(self, hyperparams: Mapping[str, Any]) -> None:
        """
        Set up the experiment.
        """
        # setup logger and log_dir
        self.custom_logger, self.log_dir = setup_logging(
            self.env_id.lower(), self.rl_id.lower(), self.policy_id.lower(), 
            format_strs=format_str_to_strs(self.format_str),
            **self.wandb_kwargs
        )
        self.policy_dir = os.path.join(self.log_dir, "policies")
        os.makedirs(self.policy_dir, exist_ok=True)
        
        # Save the hyperparameters
        self._save_config(os.path.join(self.log_dir, 'config'), hyperparams)

        # setup environment
        venv = self.create_envs(self.n_envs)

        # setup callbacks
        self.callback = self.create_callbacks()

        # setup RL algorithm
        if self.continue_training:
            self.rl_algo = self._load_trained_agent() # TODO: continue training - implement _load_trained_agent
                                                      # Think about if set_logger is needed
        else:
            # Train an agent from scratch
            self.rl_algo = self.rl_cls(
                env = venv,
                policy = self.policy_cls,
                policy_kwargs = self.policy_kwargs,
                seed = self.seed,
                verbose=self.verbose,
                **self.rl_kwargs
            )
            self.rl_algo.set_logger(self.custom_logger)

    def _wrap_env(self, venv) -> None:
        """
        Wrap the environment if necessary.

        Excluding VecNormalize wrapper, which is wrapped separately.
        """
        if self.max_episode_steps is not None:
            venv = TimeLimit(venv, self.max_episode_steps)
        if self.convert_act: venv = ActMapVecEnvWrapper(venv, assginment_rsc_to_stat_2d, 
                                                        n_stations = venv.action_space.nvec[0])
        return venv
    
    def _maybe_normalize(self, venv: VecEnv, eval_env: bool) -> VecEnv:
        """
        Wrap the env into a VecNormalize wrapper if needed
        and load saved statistics when present.

        Args:
            venv
            eval_env
        """
        # ----------- Todo Start -----------
        # TODO: continue training - handle the pretrained case 
        # Pretrained model, load normalization 
        # path_ = os.path.join(os.path.dirname(self.trained_agent), self.env_id)
        # path_ = os.path.join(path_, "vecnormalize.pkl")
        path_ = '/vecnormalize.pkl' # Use a fake path for now

        if os.path.exists(path_):
            raise RuntimeError("vecnormalize.pkl found, but continue training is not implemented")
            print("Loading saved VecNormalize stats")
            venv = VecNormalize.load(path_, venv)
            # Deactivate training and reward normalization
            if eval_env:
                venv.training = False
                venv.norm_reward = False
        # ----------- Todo End -----------

        elif self.normalize:
            # Copy to avoid changing default values by reference
            local_normalize_kwargs = self.normalize_kwargs.copy()
            # Do not normalize reward for env used for evaluation
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs["norm_reward"] = False
                else:
                    local_normalize_kwargs = {"norm_reward": False}

            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            venv = VecNormalize(venv, **local_normalize_kwargs)
        return venv

    def _save_config(
        self,
        path: str, 
        args: dict,
    ):
        config_dict = clean_dict(args, keys=no_log_configs)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.yml'), 'w') as f:
            yaml.dump(config_dict, f)
