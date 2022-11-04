import argparse
import copy
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from einops import rearrange
from sb3_contrib import ARS, QRDQN, TQC, TRPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.utils import constant_fn

from .torch_layers import GnnExtractor

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "ars": ARS,
    "qrdqn": QRDQN,
    "tqc": TQC,
    "trpo": TRPO,
}

no_log_configs = [ 'wandb_kwargs', 'policy_save_interval',
                'policy_eval_interval', 'device', 'n_eval_episodes','format_str',
                'log_flow_dist', 'log_weight_grad', 'log_model_structure',
                'custom_logger', 'log_dir', 'run_notes']

REPALCE_PAIR = [
    ('lstm_model', 'lstm'),
    ('posterior', 'q'),
    ('weight_ih_l0_reverse', 'w_ih_l0_r'),
    ('context_net', 'ctx_net'),
    ('weight', ''),
    ('weight_hh_l1_reverse', 'w_hh_l1_r'),
    ('weight_ih_l1_reverse', 'w_ih_l1_r'),
    ('flow_net', 'flow'),
    ('encoder', 'en'),
    ('weight_ih_l0', 'w_ih_l0'),
    ('weight_ih_l1', 'w_ih_l1'),
    ('obs_embedding', 'obs_emb'),
    ('context_init', 'ctx_init'),
    ('time_embed_layer', 'time_emb'),
    ('transforms', 't'),
    ('mixture_scaling_factor', 'mix_sf'),
    ('weight_hh_l0_reverse', 'w_hh_l0_r'),
    ('act_embedding', 'act_emb'),
    ('weight_hh_l0', 'w_hh_l0'),
    ('log_scale', ''),
    ('weight_hh_l1', 'w_hh_l1'),
 ]

def set_seeds(seed, cuda_deterministic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

def get_exp_idx(log_parent_path, num_digit=3):
    if not os.path.exists(log_parent_path):
        return 0
    else:
        subfolders = [sf for sf in os.listdir(log_parent_path) if \
            os.path.isdir(os.path.join(log_parent_path,sf))]
        
        if len(subfolders) == 0:
            return 0
        else:
            return max([int(sf[:num_digit]) for sf in subfolders])+1

def clean_dict(d, keys):
    d2 = copy.deepcopy(d)
    for key in keys:
        if key in d2:
            del d2[key]
    return d2

def format_str_to_strs(format_str: Optional[str]=None) -> Sequence[str]:
    valid_format_str = ["stdout", "csv", "log", "tensorboard", "json"]

    if not format_str: return ["tensorboard"]

    format_strs = format_str.split('_')
    for f in format_strs:
        assert f in valid_format_str, f"Invalid format_str found: {f}."
    
    return format_strs

def one_hot(
    idx:np.ndarray, 
    n_classes:np.ndarray,
    dtype=np.int32)->np.ndarray:
    return np.eye(n_classes, dtype=dtype)[idx]

def one_hot_w_padding(x: np.ndarray, n_cls:int, constant_values=0):
    """
    One hot encoding with padding.
        Encode an 1d array of integers into one-hot vectors, padding with zeros.
        e.g. [1, 2] -> [0, 1, 0, 0, 0, 1, 0, 0, 0] (n_cls=3)
    """
    assert x.ndim == 1
    return np.pad(
        rearrange(one_hot(x, n_cls), '... f1 f2 -> ... (f1 f2)'), 
        (0, n_cls*(n_cls-x.shape[0])), 'constant', constant_values=constant_values)

def sample_cartesian_product(arrays: List[np.ndarray]) -> np.ndarray:
    """
    Sample from a cartesian product of the given arrays.
    """
    return np.array([np.random.choice(a) for a in arrays]).flatten()

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x,
    i.e. perform a softmax transformation on the last dimension.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def args_parser(args):
    # Process schedules
    args = preprocess_schedules(args)
    
    # ========== rl_kwargs ==========
    if args.rl_algo == 'PPO':
        rl_kwargs = dict(
            # For recommended PPO hyperparams in each environment, see:
            # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
            learning_rate=args.learning_rate,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            gae_lambda=args.gae_lambda,
            gamma=args.gamma,
        )
        if args.n_steps > 0: rl_kwargs['n_steps'] = args.n_steps
    elif args.rl_algo == 'A2C':
        rl_kwargs = dict(
            # For recommended A2C hyperparams in each environment, see:
            # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef
        )
        if args.n_steps > 0: rl_kwargs['n_steps'] = args.n_steps
    
    # ========== policy_kwargs ==========
    if args.policy == 'flow':
        policy_kwargs = dict(
            flow_type = args.flow_type,
            flow_net_hidden_size = args.flow_net_hidden_size,
            context_size = args.context_size,
            num_flow_layers = args.num_flow_layers,
            batch_size_flow_updating = args.batch_size_flow_updating,
            n_iters_flow_pretraining = args.n_iters_flow_pretraining,
            val_steps = args.val_steps,
            val_batch_size = args.val_batch_size,
            elbo_steps = args.elbo_steps,
            n_samples_prob_est = args.n_samples_prob_est,
            flow_base_dist = args.flow_base_dist,
            act_encoding_scheme = args.act_encoding_scheme,
            pol_grad_G = args.pol_grad_G,
            elbo_Q = args.elbo_Q,
            has_act_corr = args.has_act_corr,
            act_corr_prot = args.act_corr_prot,
        )
    else: policy_kwargs = dict()
    if args.features_extractor_class == 'gnn':
        assert args.embedding_vars_gnn_extractor is not None
        assert all(isinstance(e, int) for e in args.embedding_vars_gnn_extractor), (
            "[--embedding_vars_gnn_extractor] only accept a list of integers,\n"
            "E.g. --embedding_vars_gnn_extractor 5 2 4"
        )
        assert len(args.embedding_vars_gnn_extractor)%2==0, (
            "[--embedding_vars_gnn_extractor] only accept a list of integers with even length,\n"
            "E.g. --embedding_vars_gnn_extractor 2 1 3 3"
        )
        evge = args.embedding_vars_gnn_extractor
        embedding_vars = []
        for i in range(len(evge)//2):
            embedding_vars.append((evge[2*i], evge[2*i+1]))

        policy_kwargs['features_extractor_class'] = GnnExtractor
        policy_kwargs['features_extractor_kwargs'] = dict(
            embedding_vars = embedding_vars,
            apply_bn = args.apply_bn_gnn_extractor,
        )

    # ========== env_kwargs ==========
    if args.env_id is not None: env_kwargs = dict()

    # ========== normalize_kwargs ==========
    if args.normalize: 
        normalize_kwargs = dict( 
            norm_obs=args.normalize_obs,
            norm_reward=args.normalize_reward,
            gamma = args.gamma_norm_env,
        )
    else: normalize_kwargs = dict()

    # ========== log_kwargs ==========
    if args.log_wandb:
        wandb_kwargs = dict(
            args = args,
            log_wandb = args.log_wandb,
            project = args.project,
            run_notes = args.run_notes,
        )
    else: wandb_kwargs = dict()

    return rl_kwargs, policy_kwargs, env_kwargs, normalize_kwargs, wandb_kwargs

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Args:
        initial_value: (float or str)
    Return: 
        (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

def preprocess_schedules(args):
    args = copy.deepcopy(args)
    # Create schedules
    for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
        if not hasattr(args, key):
            continue
        try: 
            setattr(args, key, float(getattr(args, key)))
        except: pass
        if isinstance(getattr(args, key), str):
            schedule, initial_value = getattr(args, key).split("_")
            initial_value = float(initial_value)
            setattr(args, key, linear_schedule(initial_value))
        elif isinstance(getattr(args, key), (float, int)):
            # Negative value: ignore (ex: for clipping)
            if getattr(args, key) < 0:
                continue
            setattr(args, key, constant_fn(float(getattr(args, key)))) 
        else:
            raise ValueError(f"Invalid value for {key}: {getattr(args, key)}")
    return args

def viz_weight_norm(
    model: torch.nn.Module, 
    norm_type: float = 2.0,
    lay_names: Optional[Sequence[str]] = None,
    pairs4replace: List[Tuple]=[],
    style_use: str = 'seaborn-deep',
):
    """Visualize weight norms of a model.
    Plot the norms, where x-axis is the name of the layer and y-axis is the norm. 

    Returns:
        (matplotlib.pyplot.figure)
    """
    MAX_LAYER = 80

    if lay_names is not None:
        lay_names = [lay_names] if isinstance(lay_names, str) else lay_names
        assert isinstance(lay_names, list)

    names = []
    norms = []
    for n, p in model.named_parameters():
        if 'bias' in n:
            continue
        if lay_names is not None and all((lay_name not in n) for lay_name in lay_names):
            continue
        p_flat = p.data.flatten()
        # npe: norm per element
        npe = torch.linalg.norm(p_flat, norm_type).item()/np.power(p_flat.numel(), 1/norm_type)
        names.append(n)
        norms.append(npe)
    
    names = [replace_pairs(n, pairs4replace) for n in names] # simplify names

    if len(names) > MAX_LAYER//2: 
        fig = plt.figure(figsize=(10, 12))
    else: 
        fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.barh(names[:MAX_LAYER], norms[:MAX_LAYER])
    plt.style.use(style_use)
    plt.title('Weight Norms')
    plt.ylabel('Layer')
    plt.xlabel('Norm')
    plt.tight_layout()

    return fig

def viz_weight_grad_norm(
    model: torch.nn.Module, 
    norm_type: float = 2.0,
    lay_names: Optional[Sequence[str]] = None,
    pairs4replace: List[Tuple]=[],
    style_use: str = 'seaborn-deep',
):
    """Visualize weight gradient norms of a model.
    Plot the norms, where x-axis is the name of the layer and y-axis is the norm. 

    Returns:
        (matplotlib.pyplot.figure)
    """
    MAX_LAYER = 80

    if lay_names is not None:
        lay_names = [lay_names] if isinstance(lay_names, str) else lay_names
        assert isinstance(lay_names, list)

    names = []
    grad_norms = []
    for n, p in model.named_parameters():
        if 'bias' in n:
            continue
        if lay_names is not None and all((lay_name not in n) for lay_name in lay_names):
            continue
        p_grad_flat = p.grad.data.flatten()
        # npe: norm per element
        npe = torch.linalg.norm(p_grad_flat, norm_type).item()/np.power(p_grad_flat.numel(), 1/norm_type)
        names.append(n)
        grad_norms.append(npe)
    
    names = [replace_pairs(n, pairs4replace) for n in names] # simplify names

    if len(names) > MAX_LAYER//2: 
        fig = plt.figure(figsize=(10, 12))
    else: 
        fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.barh(names[:MAX_LAYER], grad_norms[:MAX_LAYER])
    plt.style.use(style_use)
    plt.title('Weight Gradient Norms')
    plt.ylabel('Layer')
    plt.xlabel('Norm')
    plt.tight_layout()

    return fig

def replace_pairs(string: str, pairs: List[Tuple[str, str]]):
    for old, new in pairs:
        string = string.replace(old, new)
    return string

def get_saved_hyperparams(
    stats_path: str,
    norm_reward: bool = False,
    test_mode: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    :param stats_path:
    :param norm_reward:
    :param test_mode:
    :return:
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, "config.yml")) as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path

class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)

class cus_obj:      
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
   
def dict2obj(dict1):
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=cus_obj)
