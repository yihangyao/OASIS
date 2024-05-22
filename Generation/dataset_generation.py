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
from dsrl.offline_env import OfflineEnvWrapper, wrap_env, NormalizationEnvWrapper  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from osrl.configs.oasis_configs import OASISTrainConfig
from osrl.common import TransitionDataset
from osrl.common.exp_util import auto_name, seed_all

from osrl.common.net import cosine_beta_schedule, extract, apply_conditioning, Losses, TemporalUnet, to_torch
from osrl.algorithms import OASIS
import matplotlib.pyplot as plt
from osrl.common import SequenceDataset
import seaborn as sns

from osrl.common.plot import *

from osrl.common.exp_util import visualization

import os
import h5py
from tianshou.data.utils.converter import to_hdf5
from tianshou.data import Batch
from osrl.common.function import save

from causal_model import CausalDecomposition
import torch.nn as nn

@pyrallis.wrap()
def main(args: OASISTrainConfig):
    cfg, old_cfg = asdict(args), asdict(OASISTrainConfig())
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

    path = "../models/BallCircle.pt"

    model_state = torch.load(path) #  model.load_state_dict
    model.load_state_dict(model_state['model_state'])
    model.to(args.device)
    
    reward_model = CausalDecomposition(
        input_dim=env.observation_space.shape[0], 
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
        input_dim=env.observation_space.shape[0], 
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

    reward_model.load_model(path="../models/BC_reward.pt")
    cost_model.load_model(path="../models/BC_cost.pt")

    reward_model.eval()
    cost_model.eval()

    conditional_generation(
        dataset=dataset,
        model = model,
        seq_len=args.seq_len,
        device=args.device,
        name=args.task,
        reward_model=reward_model,
        cost_model=cost_model,
        clip_len=16
    )


def conditional_generation(dataset, 
                        model=None, 
                        seq_len = None,
                        device = "cuda:3",
                        name = "",
                        zero_rc = False,
                        reward_model = None,
                        cost_model = None,
                        clip_len = 16
                        ):
    """
        model is a diffusion model
    """
    # observations, next_observations, actions, rewards, costs, done
    # seq_len += 1
    model.eval()
    batch_size = 50  # dataset.dataset_size // seq_len
    s, next_observations, actions, rewards, costs, done = dataset.random_sample(batch_size)

    s = dataset.normalize_obs(s)
    s = to_torch(s, device=device)
    s_ori = s

    model.seq_len = seq_len
    model.condition_guidance_w = 2.0

    clip_len = min(clip_len+1, seq_len)
    
    test_condition_list = [[0.2, 0.8]] # Correspong to cost = 20
    
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
                clip_len=clip_len
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

    suffix = "_BC_" + "-batch_size-" + str(batch_size)
    if zero_rc:
        suffix += "-zero_rc"

    save(
        log_dir="../dataset/",
        dataset_name= name +  "-condition-" + str(test_condition_list) + "-num-" + str(num) + suffix + \
                  ".hdf5",
        data=traj_data
    )

    return

if __name__ == "__main__":
    main()
    
    
