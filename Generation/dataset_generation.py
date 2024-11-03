import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym  # noqa
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import wrap_env # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from osrl.configs.oasis_configs import OASISTrainConfig, DD_DEFAULT_CONFIG
from osrl.common import TransitionDataset
from osrl.common.exp_util import auto_name, seed_all

from osrl.common.net import  to_torch
from osrl.algorithms import OASIS
import matplotlib.pyplot as plt
# from osrl.common import SequenceDataset
# import seaborn as sns

from osrl.common.plot import *

from osrl.common.exp_util import visualization

import os
import h5py
from tianshou.data.utils.converter import to_hdf5
from tianshou.data import Batch
from osrl.common.function import save

from causal_model import CausalDecomposition

Short_name_map = {
    # bullet_safety_gym
    "OfflineBallRun-v0": "BR",
    "OfflineBallCircle-v0": "BC",
    "OfflineCarRun-v0": "CR",
    "OfflineCarCircle-v0": "CC",
    "OfflineDroneRun-v0": "DR",
    "OfflineDroneCircle-v0": "DC",
}

@pyrallis.wrap()
def main(args: OASISTrainConfig):
    # cfg, old_cfg = asdict(args), asdict(OASISTrainConfig())
    
    cfg, old_cfg = asdict(args), asdict(OASISTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(DD_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    if "Metadrive" in args.task:
        import gym
    else: 
        import gymnasium as gym
    env = gym.make(args.task)

    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]

    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb,
                                removed_r_min=args.removed_r_min,
                                removed_r_max=args.removed_r_max,
                                removed_c_min=args.removed_c_min,
                                removed_c_max=args.removed_c_max,
                                removed_ratio=args.removed_ratio
                                )
    
    print("removed ratio:", args.removed_ratio)
    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    # env = NormalizationEnvWrapper(env)

    # Normalize the original data

    dataset = TransitionDataset(data,
                                reward_scale=args.reward_scale,
                                cost_scale=args.cost_scale)

    model = OASIS(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=args.embedding_dim,
        seq_len=args.seq_len,
        episode_len=args.episode_len,
        cost_transform=args.cost_transform,
        add_cost_feat=args.add_cost_feat,
        mul_cost_feat=args.mul_cost_feat,
        cat_cost_feat=args.cat_cost_feat,
        action_head_layers=args.action_head_layers,
        cost_prefix=args.cost_prefix,
        stochastic=args.stochastic,
        init_temperature=args.init_temperature,
        target_entropy=-env.action_space.shape[0],
        n_timesteps = args.n_timesteps,
        returns_condition = args.returns_condition,
        condition_guidance_w = args.condition_guidance_w
    )
    
    short_name = Short_name_map[args.task]
    path = args.generator_loading_path + "{}_diffusion.pt".format(short_name)

    model_state = torch.load(path) #  model.load_state_dict
    model.load_state_dict(model_state['model_state'])
    model.to(args.device)
    
    if "Run" in args.task:
        dim_times = 2
    else:
        dim_times = 1
    
    reward_model = CausalDecomposition(
        input_dim=env.observation_space.shape[0] * dim_times, 
        output_dim=2,
        hidden_dim=128,
        causal_dim=10,
        num_hidden=10,
        sparse_weight=0.1,
        sim_weight=0.5,
        norm_sim_weight=0.05,
        sparse_norm=0.25,
        mse_weight=20,
        device=args.device,
        learning_mode="reward"
    )

    cost_model = CausalDecomposition(
        input_dim=env.observation_space.shape[0] * dim_times, 
        output_dim=2,
        hidden_dim=128,
        causal_dim=10,
        num_hidden=10,
        sparse_weight=0.1,
        sim_weight=0.5,
        norm_sim_weight=0.05,
        sparse_norm=0.25,
        mse_weight=20,
        device=args.device,
        learning_mode="cost"
    )
    
    reward_model_path = args.labeling_model_path + "{}_reward.pt".format(short_name)
    cost_model_path = args.labeling_model_path + "{}_cost.pt".format(short_name)

    reward_model.load_model(path=reward_model_path)
    cost_model.load_model(path=cost_model_path)

    reward_model.eval()
    cost_model.eval()
    
    cost_scale = env.cost_scale_max
    reward_scale = env.reward_scale_max
    # exit()
    
    condition = [[round(cost / cost_scale, 2), round(reward / reward_scale, 2)] for (reward, cost) in args.generation_conditions]
    
    curate_dataset(
        dataset=dataset,
        model = model,
        seq_len=args.seq_len,
        device=args.device,
        name=args.task,
        reward_model=reward_model,
        cost_model=cost_model,
        clip_len=16,
        data_saving_path=args.data_saving_path,
        generation_condition=condition
    )


def curate_dataset(
        dataset, 
        model=None, 
        seq_len = None,
        device = "cuda:3",
        name = "",
        zero_rc = False,
        reward_model = None,
        cost_model = None,
        clip_len = 16,
        data_saving_path = "",
        generation_condition = None
    ):
    """
        model is a diffusion model
    """
    # observations, next_observations, actions, rewards, costs, done
    # seq_len += 1
    model.eval()
    batch_size = 4000  # dataset.dataset_size // seq_len
    s, _, _, _, _, _ = dataset.random_sample(batch_size)

    # s: [Batch_size, s_dim]

    s = dataset.normalize_obs(s)
    s = to_torch(s, device=device)
    s_ori = s

    model.seq_len = seq_len
    model.condition_guidance_w = 2.0

    clip_len = min(clip_len+1, seq_len)
    
    test_condition_list = generation_condition
    
    with torch.no_grad():
        s_list = None
        s_next_list = None
        a_list = None
        r_pred_list = None
        c_pred_list = None
        terminal_list = None
        timeout_list = None

        for test_i in test_condition_list:

            test_c_i, test_r_i = test_i[0], test_i[1]
        
            s, s_next, a, r_pred, c_pred = conditional_generation(
                device=device,
                model=model,
                test_ret=test_r_i,
                cost_ret=test_c_i,
                dataset=dataset,
                s=s_ori,
                reward_model=reward_model,
                cost_model=cost_model,
                clip_len=clip_len,
                name = name
            )

            r_pred = r_pred.reshape(-1, clip_len-1) * 10
            c_pred = c_pred.reshape(-1, clip_len-1)
            batch_size = r_pred.shape[0]
            
            if zero_rc:
                r_pred = np.zeros(r_pred.shape)
                c_pred = np.zeros(c_pred.shape)

            s = s.reshape(batch_size, clip_len-1, -1)
            s_next = s_next.reshape(batch_size, clip_len-1, -1)
            a = a.reshape(batch_size, clip_len-1, -1)

            terminals = [[False] * r_pred.shape[1]] * r_pred.shape[0]
            timeouts = [[False] * r_pred.shape[1]] * r_pred.shape[0]

            if s_list is None:
                s_list = s
                s_next_list =s_next
                a_list = a
                r_pred_list = r_pred
                c_pred_list = c_pred
                terminal_list = terminals
                timeout_list = timeouts
            else:
                s_list = np.concatenate((s_list, s), axis=0)
                s_next_list = np.concatenate((s_next_list, s_next), axis=0)
                a_list = np.concatenate((a_list, a), axis=0)
                r_pred_list = np.concatenate((r_pred_list, r_pred), axis=0)
                c_pred_list = np.concatenate((c_pred_list, c_pred), axis=0)
                terminal_list = np.concatenate((terminal_list, terminals), axis=0)
                timeout_list = np.concatenate((timeout_list, timeouts), axis=0)
        
        traj_data = Batch(
            observations=s_list,
            next_observations=s_next_list,
            actions=a_list,
            rewards=r_pred_list,
            costs=c_pred_list,
            terminals=terminal_list,
            timeouts=timeout_list
        )
            
        batch_size = traj_data["rewards"].shape[0]

        num = s_list.shape[0] * s_list.shape[1] 

    suffix = "-from_tempting" 
    
    if zero_rc:
        suffix += "-zero_rc"

    save(
        log_dir=data_saving_path,
        dataset_name= name + suffix + \
                  ".hdf5",
        data=traj_data
    )

    return
    
def conditional_generation(device, 
                           model, 
                           test_ret,
                           cost_ret,
                           dataset,
                           s,
                           reward_model = None,
                           cost_model = None, 
                           clip_len = 8,
                           name = ""
                           ):
    obs = s
    conditions = {0: to_torch(obs, device=device)}
    samples, diffusion = model.conditional_sample(conditions,
                                                    return_diffusion=True,
                                                    returns = test_ret, 
                                                    cost_returns = cost_ret)
    samples = samples[:, :clip_len, :]
    s = samples[:, :-1, :]
    s_next = samples[:, 1:, :]

    s = torch.flatten(s, start_dim=0, end_dim=1) # [Batch_size, s_dim]
    s_next = torch.flatten(s_next, start_dim=0, end_dim=1) # [Batch_size, s_dim]
    x_comb_t = torch.cat([s, s_next], dim=-1)
    x_comb_t = x_comb_t.reshape(-1, 2 * s_next.shape[-1])
    a = model.inv_model(x_comb_t)

    s = s.detach().cpu().numpy()
    s_next = s_next.detach().cpu().numpy()
    a = a.detach().cpu().numpy()

    s = dataset.denormalize_obs(s)
    s_next = dataset.denormalize_obs(s_next)
    a = dataset.denormalize_action(a)
    
    s_cuda = torch.tensor(s, device=device)
    s_next_cuda = torch.tensor(s_next, device=device)
    a_cuda = torch.tensor(a, device=device)

    if reward_model is not None and cost_model is not None:
        if "Run" in name:
            input_val = torch.cat((s_cuda, s_next_cuda), axis=-1)
        else:
            input_val = s_next_cuda
        r_pred = reward_model(input_val)
        c_pred = cost_model(input_val)
        r_pred = r_pred.detach().cpu().numpy() * 0.1
        c_pred = c_pred.detach().cpu().numpy()
    
    return s, s_next, a, r_pred, c_pred

if __name__ == "__main__":
    main()
    