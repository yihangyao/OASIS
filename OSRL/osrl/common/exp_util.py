import os
import os.path as osp
import random
import uuid
from typing import Dict, Optional, Sequence
import matplotlib.pyplot as plt

import numpy as np
import torch
import yaml

from osrl.common.net import apply_conditioning, to_torch


def seed_all(seed=1029, others: Optional[list] = None):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if others is not None:
        if hasattr(others, "seed"):
            others.seed(seed)
            return True
        try:
            for item in others:
                if hasattr(item, "seed"):
                    item.seed(seed)
        except:
            pass


def get_cfg_value(config, key):
    if key in config:
        value = config[key]
        if isinstance(value, list):
            suffix = ""
            for i in value:
                suffix += str(i)
            return suffix
        return str(value)
    for k in config.keys():
        if isinstance(config[k], dict):
            res = get_cfg_value(config[k], key)
            if res is not None:
                return res
    return "None"


def load_config_and_model(path: str, best: bool = False):
    '''
    Load the configuration and trained model from a specified directory.

    :param path: the directory path where the configuration and trained model are stored.
    :param best: whether to load the best-performing model or the most recent one. Defaults to False.

    :return: a tuple containing the configuration dictionary and the trained model.
    :raises ValueError: if the specified directory does not exist.
    '''
    if osp.exists(path):
        config_file = osp.join(path, "config.yaml")
        print(f"load config from {config_file}")
        with open(config_file) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        model_file = "model.pt"
        if best:
            model_file = "model_best.pt"
        model_path = osp.join(path, "checkpoint/" + model_file)
        print(f"load model from {model_path}")
        model = torch.load(model_path)
        return config, model
    else:
        raise ValueError(f"{path} doesn't exist!")


def to_string(values):
    '''
    Recursively convert a sequence or dictionary of values to a string representation.
    :param values: the sequence or dictionary of values to be converted to a string.
    :return: a string representation of the input values.
    '''
    name = ""
    if isinstance(values, Sequence) and not isinstance(values, str):
        for i, v in enumerate(values):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(v)
        return name
    elif isinstance(values, Dict):
        for i, k in enumerate(sorted(values.keys())):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(values[k])
        return name
    else:
        return str(values)


DEFAULT_SKIP_KEY = [
    "task", "reward_threshold", "logdir", "worker", "project", "group", "name", "prefix",
    "suffix", "save_interval", "render", "verbose", "save_ckpt", "training_num",
    "testing_num", "epoch", "device", "thread"
]

DEFAULT_KEY_ABBRE = {
    "cost_limit": "cost",
    "mstep_iter_num": "mnum",
    "estep_iter_num": "enum",
    "estep_kl": "ekl",
    "mstep_kl_mu": "kl_mu",
    "mstep_kl_std": "kl_std",
    "mstep_dual_lr": "mlr",
    "estep_dual_lr": "elr",
    "update_per_step": "update"
}


def auto_name(default_cfg: dict,
              current_cfg: dict,
              prefix: str = "",
              suffix: str = "",
              skip_keys: list = DEFAULT_SKIP_KEY,
              key_abbre: dict = DEFAULT_KEY_ABBRE) -> str:
    '''
    Automatic generate the experiment name by comparing the current config with the default one.

    :param dict default_cfg: a dictionary containing the default configuration values.
    :param dict current_cfg: a dictionary containing the current configuration values.
    :param str prefix: (optional) a string to be added at the beginning of the generated name.
    :param str suffix: (optional) a string to be added at the end of the generated name.
    :param list skip_keys: (optional) a list of keys to be skipped when generating the name.
    :param dict key_abbre: (optional) a dictionary containing abbreviations for keys in the generated name.

    :return str: a string representing the generated experiment name.
    '''
    name = prefix
    for i, k in enumerate(sorted(default_cfg.keys())):
        if default_cfg[k] == current_cfg[k] or k in skip_keys:
            continue
        prefix = "_" if len(name) else ""
        value = to_string(current_cfg[k])
        # replace the name with abbreviation if key has abbreviation in key_abbre
        if k in key_abbre:
            k = key_abbre[k]
        # Add the key-value pair to the name variable with the prefix
        name += prefix + k + value
    if len(suffix):
        name = name + "_" + suffix if len(name) else suffix

    name = "default" if not len(name) else name
    name = f"{name}-{str(uuid.uuid4())[:4]}"
    return name

def visualization(obs_seq, name="", path = "", model = None, step = 0, returns = None, test_list = None):
    """
        For Circle tasks only
        obs_seq: [horizon, obs_dim]
        model: a diffuser model
    """
    plt.figure(3)
    tnp = np.array([0, 1, 
                    int(model.n_timesteps * 1 // 4),
                    int(model.n_timesteps * 2 // 4),
                    int(model.n_timesteps * 3 // 4),
                    model.n_timesteps-1,
                    ])
    fig, ax = plt.subplots(ncols=tnp.shape[0], nrows=2, figsize=(30,7))
    
    horizon, dim = obs_seq.shape[0], obs_seq.shape[1]

    x_seq, y_seq = obs_seq[:, 0], obs_seq[:, 1]
    x_seq = x_seq.cpu().numpy()
    y_seq = y_seq.cpu().numpy()

    x_seq *= 1
    y_seq *= 1

    init_x, init_y = x_seq[:1], y_seq[:1]

    

    if model is not None:
        x_start = obs_seq.unsqueeze(0) # [1, Horizon, dim]

        test_list = [0.2, 0.5, 0.7, 0.9]
        
        cond = {0: x_start[:, 0]} # 
        noise = torch.randn(size=(1, horizon, dim), device=obs_seq.device) #  noise: [Batch_size, Horizon, dim]
         # model.n_timesteps-1, model.n_timesteps // 2, 
        t = torch.tensor(tnp, device=obs_seq.device)
        x_noisy = model.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0) 
        x_noisy = x_noisy.cpu().numpy()
        x_seq_noisy, y_seq_noisy = x_noisy[:, :, 0], x_noisy[:, :, 1]

        conditions = {0: to_torch(x_start[:, 0], device=model.betas.device)}
        

        for noise_step in range(x_seq_noisy.shape[0]):
            # forward
            ax[0][noise_step].set_xlim([-2, 2])
            ax[0][noise_step].set_ylim([-2, 2])
            ax[0][noise_step].scatter(x_seq, y_seq, label="original dataset")
            ax[0][noise_step].scatter(init_x, init_y, color='r', label="initial point", marker="*", linewidths=1)
            t_step = tnp[noise_step]
            ax[0][noise_step].scatter(x_seq_noisy[noise_step] , 
                        y_seq_noisy[noise_step]  , 
                        label="noise step {}".format(t_step),
                        alpha=0.35)
            ax[0][noise_step].scatter(x_seq_noisy[noise_step, 0]  , 
                        y_seq_noisy[noise_step, 0]  , 
                        label="noise step {}, init".format(t_step),
                        alpha=0.35, marker="X")
            ax[0][noise_step].set_title(f"forward t={t_step}")
            ax[0][noise_step].legend()

            # inverse
            

            ax[1][noise_step].set_xlim([-2, 2])
            ax[1][noise_step].set_ylim([-2, 2])
            ax[1][noise_step].scatter(x_seq, y_seq, label="original dataset")
            ax[1][noise_step].scatter(init_x, init_y, color='r', label="initial point", marker="*", linewidths=1)

            for test_cond in test_list:
                test_ret = test_cond
                samples, diffusion = model.conditional_sample(conditions,
                                                        return_diffusion=True,
                                                        returns=0.75,
                                                        cost_returns=test_ret) # Why return is None?
                
                diffusion_np = diffusion.squeeze(axis=0).cpu().numpy()
                diffusion_np = np.flip(diffusion_np, 0)
                
                data = diffusion_np[t_step]
                denoise_x, denoise_y = data[:, 0], data[:, 1]
                ax[1][noise_step].scatter(denoise_x, 
                            denoise_y, 
                            label="return {}, denoise step {}".format(test_ret, t_step),
                            alpha=0.35)
                # ax[1][noise_step].scatter(denoise_x[0], 
                #             denoise_y[0], 
                #             label="denoise step {}, init".format(t_step),
                #             alpha=0.35, marker="X")
            ax[1][noise_step].set_title(f"inverse t={t_step}")
            ax[1][noise_step].legend()

    plt.savefig(path+"Step-"+str(step)+"-"+name+".png", dpi=400)
    # print(model.betas)

    return

def visualization_seqlen(obs_seq, 
                         name="", 
                         path = "", 
                         model = None, 
                         step = 0, 
                         returns = None, 
                         seq_len_list = None):
    """
        For Circle tasks only
        obs_seq: [horizon, obs_dim]
        model: a diffuser model
    """
    plt.figure(3)
    tnp = np.array([0, 1, 
                    int(model.n_timesteps * 1 // 4),
                    int(model.n_timesteps * 2 // 4),
                    int(model.n_timesteps * 3 // 4),
                    model.n_timesteps-1,
                    ])
    fig, ax = plt.subplots(ncols=tnp.shape[0], nrows=2, figsize=(30,7))
    
    horizon, dim = obs_seq.shape[0], obs_seq.shape[1]

    x_seq, y_seq = obs_seq[:, 0], obs_seq[:, 1]
    x_seq = x_seq.cpu().numpy()
    y_seq = y_seq.cpu().numpy()

    x_seq *= 1
    y_seq *= 1

    init_x, init_y = x_seq[:1], y_seq[:1]


    if model is not None:
        x_start = obs_seq.unsqueeze(0) # [1, Horizon, dim]

        test_list = [0.7]
        
        cond = {0: x_start[:, 0]} # 
        noise = torch.randn(size=(1, horizon, dim), device=obs_seq.device) #  noise: [Batch_size, Horizon, dim]
        
        t = torch.tensor(tnp, device=obs_seq.device)
        x_noisy = model.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0) 
        x_noisy = x_noisy.cpu().numpy()
        x_seq_noisy, y_seq_noisy = x_noisy[:, :, 0], x_noisy[:, :, 1]

        conditions = {0: to_torch(x_start[:, 0], device=model.betas.device)}
        

        for noise_step in range(x_seq_noisy.shape[0]):
            # forward
            ax[0][noise_step].set_xlim([-2, 2])
            ax[0][noise_step].set_ylim([-2, 2])
            ax[0][noise_step].scatter(x_seq, y_seq, label="original dataset")
            ax[0][noise_step].scatter(init_x, init_y, color='r', label="initial point", marker="*", linewidths=1)
            t_step = tnp[noise_step]
            ax[0][noise_step].scatter(x_seq_noisy[noise_step] , 
                        y_seq_noisy[noise_step]  , 
                        label="noise step {}".format(t_step),
                        alpha=0.35)
            ax[0][noise_step].scatter(x_seq_noisy[noise_step, 0]  , 
                        y_seq_noisy[noise_step, 0]  , 
                        label="noise step {}, init".format(t_step),
                        alpha=0.35, marker="X")
            ax[0][noise_step].set_title(f"forward t={t_step}")
            ax[0][noise_step].legend()

            # inverse
            

            ax[1][noise_step].set_xlim([-2, 2])
            ax[1][noise_step].set_ylim([-2, 2])
            ax[1][noise_step].scatter(x_seq, y_seq, label="original dataset")
            ax[1][noise_step].scatter(init_x, init_y, color='r', label="initial point", marker="*", linewidths=1)

            for test_cond in test_list:
                test_ret = test_cond
                samples, diffusion = model.conditional_sample(conditions,
                                                        return_diffusion=True,
                                                        returns=test_ret,
                                                        cost_returns=test_ret) # Why return is None?
                
                diffusion_np = diffusion.squeeze(axis=0).cpu().numpy()
                diffusion_np = np.flip(diffusion_np, 0)
                
                data = diffusion_np[t_step]
                denoise_x, denoise_y = data[:, 0], data[:, 1]
                ax[1][noise_step].scatter(denoise_x, 
                            denoise_y, 
                            label="return {}, denoise step {}".format(test_ret, t_step),
                            alpha=0.35)
                # ax[1][noise_step].scatter(denoise_x[0], 
                #             denoise_y[0], 
                #             label="denoise step {}, init".format(t_step),
                #             alpha=0.35, marker="X")
            ax[1][noise_step].set_title(f"inverse t={t_step}")
            ax[1][noise_step].legend()

    plt.savefig(path+"Step-"+str(step)+"-"+name+".png", dpi=400)
    # print(model.betas)

    return
