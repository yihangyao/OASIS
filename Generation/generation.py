import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm

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

from examples.configs.dd_configs import DD_DEFAULT_CONFIG, DDTrainConfig
from osrl.algorithms import BCQL, BCQLTrainer
from osrl.common import TransitionDataset
from osrl.common.exp_util import auto_name, seed_all

from osrl.common.function import RewardTruthModel, CostTruthModel
from osrl.common.net import cosine_beta_schedule, extract, apply_conditioning, Losses, TemporalUnet, to_torch
from osrl.algorithms import DecisionDiffuser
import matplotlib.pyplot as plt
from osrl.common import SequenceDataset
import seaborn as sns

from osrl.common.plot import *

from osrl.common.exp_util import visualization

@pyrallis.wrap()
def main(args: DDTrainConfig):
    seed_all(args.seed)
    cfg, old_cfg = asdict(args), asdict(DDTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(DD_DEFAULT_CONFIG[args.task]())
    # cfg.update(differing_values)

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
                                min_npb=min_npb)

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

    model = DecisionDiffuser(
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

    
    # path = "/home/yihang/code/OSRL_DD/OSRL/examples/train/logs/OfflineCarCircle-v0-test_ret-0.9-w-1.2-MM/DD-142b/DD-142b/checkpoint/model_-100000.pt"
    # path = "/home/yihang/code/OSRL_DD/OSRL/examples/train/logs/Diffusion-OfflineCarCircle-v0-test_ret-(0.2, 0.75)-w-2.0-n_step-20-seq_len-64-emb-64-remove_c-30.0-ratio-None/DD-b96d/DD-b96d/checkpoint/model_-160000.pt"
    # path = "/home/yihang/code/OSRL_DD/OSRL/examples/train/logs/Diffusion-OfflineBallCircle-v0-test_ret-(0.15, 0.8)-w-2.0-n_step-10-seq_len-32-emb-32-remove_c-30.0-ratio-None/DD-04d4/DD-04d4/checkpoint/model_-160000.pt"
    path = "/home/yihang/code/OSRL_DD/OSRL/examples/train/logs/Diffusion-OfflineCarCircle-v0-test_ret-(0.15, 0.5)-w-2.0-n_step-20-seq_len-64-emb-64-remove_c-30.0-ratio-0.9/DD-1cc7/DD-1cc7/checkpoint/model_-100000.pt"
    
    model_state = torch.load(path) #  model.load_state_dict
    model.load_state_dict(model_state['model_state'])
    model.to(args.device)
    
    # visualized_traj(
    #     dataset=dataset,
    #     model = model,
    #     seq_len=args.seq_len,
    #     device=args.device,
    #     test_condition = [0.1, 0.5]
    # )

    eval_generation_mean(
        dataset=dataset,
        model=model,
        seq_len=args.seq_len,
        device=args.device,
    )

def visualized_traj(
        dataset, 
        model=None, 
        seq_len = None,
        device = "cuda:1",
        test_condition = None
    ):
    # observations, next_observations, actions, rewards, costs, done
    # seq_len += 1
    model.eval()
    batch_size = 5000 # dataset.dataset_size // seq_len
    s, next_observations, actions, rewards, costs, done = dataset.random_sample(batch_size)

    # s: [Batch_size, s_dim]

    # s = np.repeat(s, 10, axis=0)

    s_raw = s

    s = dataset.normalize_obs(s)

    s = to_torch(s, device=device)
    s_ori = s

    model.seq_len = seq_len
    model.condition_guidance_w = 2
    test_c_i, test_r_i = test_condition[0], test_condition[1]
    
    plt.figure(2, figsize=(6, 6))
    print("eval: (c: {}, r: {})".format(test_c_i, test_r_i))
    s, s_next, a, r_pred, c_pred = conditional_generation(
        device=device,
        model=model,
        test_ret=test_r_i,
        cost_ret=test_c_i,
        dataset=dataset,
        s=s_ori
    )

    plot_state = s[::1]
    x, y = plot_state[:, 0], plot_state[:, 1]
    plt.scatter(x[::100], y[::100])
    plt.savefig("Figures/density_exp_new/state_visualization_scatter_r-{}_c-{}.png".format(test_r_i, test_c_i), dpi=400)

    plt.figure(3, figsize=(6, 6))
    sigma = 18
    bins = 1000

    img, extent = myplot(x, y, sigma, bins=bins)

    # img = np.clip(img, 0.0, 0.8)
    plt.figure()

    from matplotlib.colors import LinearSegmentedColormap

    # Define hex colors
    hex_colors = ['#ffffff', '#3883B9', '#104D93']

    # Create the colormap
    custom_cmap = LinearSegmentedColormap.from_list('CustomMap', hex_colors, N=256)
    
    plt.imshow(img, extent=extent, origin='lower', cmap=custom_cmap) # cm.viridis cm.Oranges
    
    # plt.title("Smoothing with  $\sigma$ = %d" % sigma)
    plt.xticks([])
    plt.yticks([])

    plt.savefig("Figures/density_exp_new/custom_visualization_scatter_r-{}_c-{}_sigma-{}_bin-{}.png".format(test_r_i, test_c_i, sigma, bins), dpi=400)
    
    plt.figure(4, figsize=(6, 6))
    sigma = 15
    bins = 1000
    
    x, y = s_raw[:, 0], s_raw[:, 1]
    img, extent = myplot(x, y, sigma, bins=bins)
    plt.imshow(img, extent=extent, origin='lower', cmap=cm.gist_heat_r) # cm.viridis cm.Oranges
    plt.xticks([])
    plt.yticks([])

    plt.savefig("Figures/density_exp_new/custom_dataset_sigma-{}_bin-{}.png".format(sigma, bins), dpi=400)

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-1, 1], [-1,1]])
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent




def eval_generation_mean(dataset, 
                        model=None, 
                        seq_len = None,
                        device = "cuda:1",
                        cost_returns = None,
                        reward_returns = None
                        ):
    """
        model is a diffusion model
    """
    # observations, next_observations, actions, rewards, costs, done
    # seq_len += 1
    model.eval()
    batch_size = 2000 # dataset.dataset_size // seq_len
    s, next_observations, actions, rewards, costs, done = dataset.random_sample(batch_size)

    # s: [Batch_size, s_dim]

    s = dataset.normalize_obs(s)
    s = to_torch(s, device=device)
    s_ori = s

    model.seq_len = seq_len
    model.condition_guidance_w = 2


    test_c_condition_list = [0.1, 0.25, 0.5, 0.75, 0.9]#  [0.1, 0.25, 0.5, 0.75, 0.9] # 
    test_r_condition_list = [0.1, 0.25, 0.5, 0.75, 0.9] #  [0.1, 0.25, 0.5, 0.75, 0.9] [0.1, 0.2, 0.3, 0.4, 0.5]
    fig_index = 4
    plt.figure(fig_index)
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,5))

    
    expectation_r = np.zeros((len(test_c_condition_list), len(test_r_condition_list)))
    expectation_c = np.zeros((len(test_c_condition_list), len(test_r_condition_list)))
    
    with torch.no_grad():
        for i in range(expectation_r.shape[0]):
            for j in range(expectation_r.shape[1]):
                test_c_i, test_r_i = test_c_condition_list[i], test_r_condition_list[j]
                print("eval: (c: {}, r: {})".format(test_c_i, test_r_i))

                # shield = Relabelling(
                #     x=test_c_i, cost_returns=cost_returns, reward_returns=reward_returns, 
                # )
                # print("eval: (c: {}, r: {}, shield: {})".format(test_c_i, test_r_i, shield))
                # test_r_i = test_r_i * 53 / 100 # min(test_r_i * 53 / 100, shield)

                s, s_next, a, r_pred, c_pred = conditional_generation(
                    device=device,
                    model=model,
                    test_ret=test_r_i,
                    cost_ret=test_c_i,
                    dataset=dataset,
                    s=s_ori
                )

                expectation_r[i, j] = r_pred.mean() * 300 / 60
                expectation_c[i, j] = c_pred.mean() * 300 / 100

                traj_r_i = r_pred.reshape(-1, seq_len-1)
                traj_c_i = c_pred.reshape(-1, seq_len-1)
                
                # del s, s_next, a, r_pred, c_pred
    cost_condition_label = ["{}".format(c_i) for c_i in test_c_condition_list ]
    reward_condition_label = ["{}".format(r_i) for r_i in test_r_condition_list]

    expectation_r = expectation_r[::-1, :]
    expectation_c = expectation_c[::-1, :]
    cost_condition_label = cost_condition_label[::-1]

    im, _ = heatmap(expectation_r, cost_condition_label, reward_condition_label, ax=ax[0],
                cbarlabel="reward")
    annotate_heatmap(im, valfmt="{x:.2f}", size=14)
    im, _ = heatmap(expectation_c, cost_condition_label, reward_condition_label, ax=ax[1],
                cbarlabel="cost")
    annotate_heatmap(im, valfmt="{x:.2f}", size=14)

    
    plt.savefig("Figures/condition_exp/customed-w-{}-CarCircle-5.png".format(model.condition_guidance_w), dpi=400)


def Relabelling(x, cost_returns, reward_returns, alpha=0.5, epsilon=0.1):
    # Ensure cost_returns and reward_returns are numpy arrays
    cost_returns = np.array(cost_returns).flatten()
    reward_returns = np.array(reward_returns).flatten()
    
    # Filter reward_returns where cost_returns are within the interval [x - epsilon, x + epsilon]
    filtered_rewards = reward_returns[(cost_returns >= x - epsilon) & (cost_returns <= x + epsilon)]
    
    # If there are no matches, return NaN
    if len(filtered_rewards) == 0:
        print("len(filtered_rewards) == 0")
        return x
    
    # Sort the filtered_rewards in descending order
    sorted_rewards = np.sort(filtered_rewards)[::-1]
    
    # Determine the number of top-alpha rewards to consider
    top_alpha_count = max(1, int(np.ceil(alpha * len(sorted_rewards))))
    
    # Compute the mean of the top-alpha rewards
    top_alpha_mean = np.min(sorted_rewards[:top_alpha_count])
    
    return top_alpha_mean

def conditional_generation(device, 
                           model, 
                           test_ret,
                           cost_ret,
                           dataset,
                           s):
    obs = s
    conditions = {0: to_torch(obs, device=device)}
    samples, diffusion = model.conditional_sample(conditions,
                                                    return_diffusion=True,
                                                    returns = test_ret, 
                                                    cost_returns = cost_ret)
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

    reward_model = RewardTruthModel(radius=7.0)
    cost_model = CostTruthModel(xlim=6.0, dataset=dataset)
    r_pred = reward_model.forward(s, a, s_next)
    c_pred = cost_model.forward(s, a, s_next)
    s_tensor = torch.tensor(s, device="cuda:3")
    grad = cost_model.c_gradient(s_tensor)

    return s, s_next, a, r_pred, c_pred

if __name__ == "__main__":
    main()
    
    
