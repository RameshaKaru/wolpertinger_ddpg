import itertools
import os
from typing import Optional

import numpy as np
from envs.gym_seqssg.base import SeqSSG
from stable_baselines3.common.callbacks import BaseCallback


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True

class UpdMaskFnCallback(BaseCallback):
    """Callback for updating the mask_fn of the policy at the beginning of training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        self.model.policy.upd_mask_fn(self.training_env.envs[0].gen_mask_from_obs)

    def _on_step(self) -> bool:
        return True

class SeqSSGCallback(BaseCallback):
    """
    Callback for SeqSSG env to retrieve game status, such as
        `protected`, `valid`, and `def_util_ori`.

    Another function is to check if the environment is stuck 
        (has no feasible actions and stuck at certain state).
    """
    def __init__(
        self, 
        verbose: int = 0, 
        eval_freq: int = 2000, 
        check_freq: int = int(4e5),
        buf_size: int = 300,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.check_freq = check_freq
        self.buf_size = buf_size
        self.prot_rate = []
        self.val_rate = []
        self.rets = []
        self.eval_timesteps = 0
        self.check_timesteps = 0
        self.mask_fn = None
        self.cand_act = None

    def _init_callback(self) -> None:
        assert isinstance(self.training_env.envs[0].unwrapped, SeqSSG)
        n_envs = self.model.env.num_envs
        self.protecteds = -np.ones((n_envs, self.buf_size), dtype=np.float32)
        self.valids = -np.ones((n_envs, self.buf_size), dtype=np.float32)
        self.rewards = -np.ones((n_envs, self.buf_size), dtype=np.float32)*np.inf
        self.idx_pos = np.zeros(n_envs, dtype=np.int32)

        # Get mask_fn and cand_act
        self.mask_fn = self.training_env.envs[0].gen_mask_from_obs
        self.cand_act = np.array(list(itertools.product(*[
            range(dim) for dim in self.training_env.action_space.nvec])))

    def _on_step(self) -> bool:
        # Collect game status
        infos = self.training_env.buf_infos
        dones = self.training_env.buf_dones
        for i, info in enumerate(infos):
            i_pos = self.idx_pos[i]
            self.protecteds[i, i_pos] = info['protected']
            self.valids[i, i_pos] = info['valid']
            self.rewards[i, i_pos] = info['rew_ori']
            self.idx_pos[i] += 1
            if dones[i]:
                prot = self.protecteds[i, :i_pos+1]
                val = self.valids[i, :i_pos+1]
                rew = self.rewards[i, :i_pos+1]
                self.prot_rate.append(prot.sum()/prot.shape[0])
                self.val_rate.append(val.sum()/val.shape[0])
                self.rets.append(rew.sum())
                self.protecteds[i] = -np.ones_like(self.protecteds[i], dtype=np.float32)
                self.valids[i] = -np.ones_like(self.valids[i], dtype=np.float32)
                self.rewards[i] = -np.ones_like(self.rewards[i], dtype=np.float32)*np.inf
                self.idx_pos[i] = 0

        # Evaluate and log
        if self.num_timesteps-self.eval_timesteps >= self.eval_freq:
            prot_rate = np.mean(self.prot_rate)
            val_rate = np.mean(self.val_rate)
            ret = np.mean(self.rets)
            self.logger.record("rollout/prot_rate_mean", prot_rate)
            self.logger.record("rollout/val_rate_mean", val_rate)
            self.logger.record("rollout/ep_ori_rew_mean", ret)
            if self.verbose > 1:
                print(f"prot_rate: {prot_rate:.4f}, val_rate: {val_rate:.4f}, ret: {ret:.4f}")
            self.prot_rate = []
            self.val_rate = []
            self.rets = []
            self.eval_timesteps = self.num_timesteps
        
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps-self.check_timesteps>=self.check_freq:
            # Obtain current observation
            obs = self.model._last_obs

            # Check if valid actions exist 
            masks = np.array([self.mask_fn(o, self.cand_act) for o in obs])
            if np.any(masks.sum(axis=1)==0):
                raise RuntimeError("No valid actions exist, the environment is stuck.")

            self.check_timesteps = self.num_timesteps
