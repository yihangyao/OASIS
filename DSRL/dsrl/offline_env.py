

import os
import urllib.request
from collections import defaultdict
from random import sample
from typing import Tuple, Union

import gymnasium as gym
import h5py
import numpy as np
from numpy.random import PCG64, Generator
from tqdm import tqdm
import random


def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)


set_dataset_path(
    os.environ.get('DSRL_DATASET_DIR', os.path.expanduser('~/.dsrl/datasets'))
)


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    print(f"Loading dataset from {dataset_filepath}")
    return dataset_filepath


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:

    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def grid_filter(
    x,
    y,
    xmin=-np.inf,
    xmax=np.inf,
    ymin=-np.inf,
    ymax=np.inf,
    xbins=10,
    ybins=10,
    max_num_per_bin=10,
    min_num_per_bin=1
):
    xmin, xmax = max(min(x), xmin), min(max(x), xmax)
    ymin, ymax = max(min(y), ymin), min(max(y), ymax)
    xbin_step = (xmax - xmin) / xbins
    ybin_step = (ymax - ymin) / ybins
    # the key is x y bin index, the value is a list of indices
    bin_hashmap = defaultdict(list)
    for i in range(len(x)):
        if x[i] < xmin or x[i] > xmax or y[i] < ymin or y[i] > ymax:
            continue
        x_bin_idx = (x[i] - xmin) // xbin_step
        y_bin_idx = (y[i] - ymin) // ybin_step
        bin_hashmap[(x_bin_idx, y_bin_idx)].append(i)
    # start filtering
    indices = []
    for v in bin_hashmap.values():
        if len(v) > max_num_per_bin:
            # random sample max_num_per_bin indices
            indices += sample(v, max_num_per_bin)
        elif len(v) <= min_num_per_bin:
            continue
        else:
            indices += v
    return indices


def filter_trajectory(
    cost,
    rew,
    traj,
    cost_min=-np.inf,
    cost_max=np.inf,
    rew_min=-np.inf,
    rew_max=np.inf,
    cost_bins=60,
    rew_bins=50,
    max_num_per_bin=10,
    min_num_per_bin=1
):
    indices = grid_filter(
        cost,
        rew,
        cost_min,
        cost_max,
        rew_min,
        rew_max,
        xbins=cost_bins,
        ybins=rew_bins,
        max_num_per_bin=max_num_per_bin,
        min_num_per_bin=min_num_per_bin
    )
    cost2, rew2, traj2 = [], [], []
    for i in indices:
        cost2.append(cost[i])
        rew2.append(rew[i])
        traj2.append(traj[i])
    return cost2, rew2, traj2, indices

def downsampling_area(
    cost,
    rew,
    traj,
    r_min = -np.inf,
    r_max = np.inf, 
    c_min = -np.inf,
    c_max = np.inf,
    ratio = 0.75
):
    
    inarea_index_list = []
    for i in range(len(traj)):
        if rew[i] >= r_min and rew[i] <= r_max and cost[i] >= c_min and cost[i] <= c_max:
            inarea_index_list.append(i)
    
    in_num = len(inarea_index_list)
    remove_num = int(in_num * ratio)
    removed_index = range(in_num)

    removed_index = np.array(removed_index)
    np.random.shuffle(removed_index)
    removed_index = removed_index[:remove_num]
    
    removed_index = [inarea_index_list[i] for i in removed_index]

    cost2, rew2, traj2 = [], [], []
    for idx in range(len(traj)):
        if not idx in removed_index:
            cost2.append(cost[idx])
            rew2.append(rew[idx])
            traj2.append(traj[idx])
        # else:
        #     print(cost[idx])


    return cost2, rew2, traj2


class OfflineEnv(gym.Env):
    """
    Base class for offline RL envs.

    Args:
        dataset_url: URL pointing to the dataset.
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
        deprecated: If True, will display a warning that the environment is deprecated.
    """

    def __init__(
        self,
        max_episode_reward=None,
        min_episode_reward=None,
        max_episode_cost=None,
        min_episode_cost=None,
        dataset_url=None,
        **kwargs
    ):
        super(OfflineEnv, self).__init__(**kwargs)
        self.dataset_url = dataset_url
        self.max_episode_reward = max_episode_reward
        self.min_episode_reward = min_episode_reward
        self.max_episode_cost = max_episode_cost
        self.min_episode_cost = min_episode_cost
        self.target_cost = None
        # set random number generator
        self.rng = Generator(PCG64(seed=1234))
        self.reward_returns = None
        self.cost_returns = None

    def set_target_cost(self, target_cost):
        self.target_cost = target_cost
        self.epsilon = 1 if self.target_cost == 0 else 0

    def get_normalized_score(self, reward, cost):
        if (self.max_episode_reward is None) or \
            (self.min_episode_reward is None) or \
            (self.target_cost is None):
            raise ValueError("Reference score not provided for env")
        normalized_reward = (reward - self.min_episode_reward
                             ) / (self.max_episode_reward - self.min_episode_reward)
        normalized_cost = (cost + self.epsilon) / (self.target_cost + self.epsilon)
        return normalized_reward, normalized_cost

    @property
    def dataset_filepath(self):
        return filepath_from_url(self.dataset_url)

    def get_dataset(self, h5path: str = None):
        if h5path is None:
            if self.dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        data_dict = {}
        with h5py.File(h5path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in [
            'observations', 'next_observations', 'actions', 'rewards', 'costs',
            'terminals', 'timeouts'
        ]:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (
            N_samples,
        ), 'Reward has wrong shape: %s' % (str(data_dict['rewards'].shape))
        if data_dict['costs'].shape == (N_samples, 1):
            data_dict['costs'] = data_dict['costs'][:, 0]
        assert data_dict['costs'].shape == (
            N_samples,
        ), 'Costs has wrong shape: %s' % (str(data_dict['costs'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (
            N_samples,
        ), 'Terminals has wrong shape: %s' % (str(data_dict['rewards'].shape))
        data_dict["observations"] = data_dict["observations"].astype("float32")
        data_dict["actions"] = data_dict["actions"].astype("float32")
        data_dict["next_observations"] = data_dict["next_observations"].astype("float32")
        data_dict["rewards"] = data_dict["rewards"].astype("float32")
        data_dict["costs"] = data_dict["costs"].astype("float32")
        return data_dict

    def pre_process_data(
        self,
        data_dict: dict,
        outliers_percent: float = None,
        noise_scale: float = None,
        inpaint_ranges: Tuple[Tuple[float, float]] = None,
        epsilon: float = None,
        density: float = 1.0,
        cbins: int = 10,
        rbins: int = 50,
        max_npb: int = 5,
        min_npb: int = 2,
        removed_r_min = -np.inf,
        removed_r_max = np.inf, 
        removed_c_min = -np.inf,
        removed_c_max = np.inf,
        removed_ratio = None,
        filter_len = None,
        removed_all_ratio = None
    ):
        """
        pre-process the data to add outliers and/or gaussian noise and/or inpaint part of the data
        """

        # get trajectories
        done_idx = np.where(
            (data_dict["terminals"] == 1) | (data_dict["timeouts"] == 1)
        )[0]

        trajs, cost_returns, reward_returns = [], [], []
        for i in range(done_idx.shape[0]):
            start = 0 if i == 0 else done_idx[i - 1] + 1
            end = done_idx[i] + 1
            cost_return = np.sum(data_dict["costs"][start:end])
            reward_return = np.sum(data_dict["rewards"][start:end])
            traj = {k: data_dict[k][start:end] for k in data_dict.keys()}
            trajs.append(traj)
            cost_returns.append(cost_return)
            reward_returns.append(reward_return)

        print(
            f"before filter: traj num = {len(trajs)}, transitions num = {data_dict['observations'].shape[0]}"
        )

        if density != 1.0:
            assert density < 1.0, "density should be less than 1.0"
            cost_returns, reward_returns, trajs, indices = filter_trajectory(
                cost_returns,
                reward_returns,
                trajs,
                cost_min=self.min_episode_cost,
                cost_max=self.max_episode_cost,
                rew_min=self.min_episode_reward,
                rew_max=self.max_episode_reward,
                cost_bins=cbins,
                rew_bins=rbins,
                max_num_per_bin=max_npb,
                min_num_per_bin=min_npb
            )
        print(f"after grid filter: traj num = {len(trajs)}")

        if filter_len is not None:
            cost_returns2, reward_returns2, trajs2 = [], [], []
            for i in range(len(cost_returns)):
                if len(trajs[i]['actions']) > filter_len:
                    cost_returns2.append(cost_returns[i])
                    reward_returns2.append(reward_returns[i])
                    trajs2.append(trajs[i])
            cost_returns, reward_returns, trajs = cost_returns2, reward_returns2, trajs2

        if removed_ratio is not None:
            assert 0 < removed_ratio <= 1
            cost_returns, reward_returns, trajs = downsampling_area(
                cost = cost_returns,
                rew = reward_returns,
                traj = trajs,
                r_min = removed_r_min,
                r_max = removed_r_max, 
                c_min = removed_c_min,
                c_max = removed_c_max,
                ratio = removed_ratio
            )
            print(f"after downsampling: traj num = {len(trajs)}")

        if removed_all_ratio is not None:
            cost_returns, reward_returns, trajs = downsampling_area(
                cost = cost_returns,
                rew = reward_returns,
                traj = trajs,
                r_min = 0,
                r_max = 5000, 
                c_min = 0,
                c_max = 200,
                ratio = removed_all_ratio
            )
            print(f"after removed_all_ratio downsampling: traj num = {len(trajs)}")

        n_trajs = len(trajs)
        traj_idx = np.arange(n_trajs)
        cost_returns = np.array(cost_returns)
        reward_returns = np.array(reward_returns)

        print("reward return max:", reward_returns.max())
        print("cost return max:", cost_returns.max())

        self.reward_returns = reward_returns / reward_returns.max()
        self.cost_returns = cost_returns / cost_returns.max()

        # outliers and inpaint ranges are based-on episode cost returns
        if outliers_percent is not None:
            assert self.target_cost is not None, \
            "Please set target cost using env.set_target_cost(target_cost) if you want to add outliers"
            outliers_num = np.max([int(n_trajs * outliers_percent), 1])
            mask = np.logical_and(
                cost_returns >= self.max_episode_cost / 2,
                reward_returns >= self.max_episode_reward / 2
            )
            outliers_idx = self.rng.choice(
                traj_idx[mask], size=outliers_num, replace=False
            )
            outliers_cost_returns = self.rng.choice(
                np.arange(int(self.target_cost)), size=outliers_num
            )
            # replace the original risky trajs with outliers
            for i, cost in zip(outliers_idx, outliers_cost_returns):
                len_traj = trajs[i]["observations"].shape[0]
                idx = self.rng.choice(np.arange(len_traj), cost, replace=False)
                trajs[i]["costs"] = np.zeros_like(trajs[i]["costs"])
                trajs[i]["costs"][idx] = 1
                trajs[i]["rewards"] = 1.5 * trajs[i]["rewards"]
                cost_returns[i] = cost
                reward_returns[i] = 1.5 * reward_returns[i]

        if inpaint_ranges is not None:
            inpainted_idx = []
            for inpaint_range in inpaint_ranges:
                pcmin, pcmax, prmin, prmax = inpaint_range
                cmask = np.logical_and(
                    (self.max_episode_cost - self.min_episode_cost) * pcmin + self.min_episode_cost <= cost_returns[traj_idx], 
                     cost_returns[traj_idx] <= (self.max_episode_cost - self.min_episode_cost) * pcmax + self.min_episode_cost)
                rmin2 = np.min(reward_returns[traj_idx[cmask]])
                rmax2 = np.max(reward_returns[traj_idx[cmask]])
                rmask = np.logical_and(
                    (rmax2 - rmin2) * prmin + rmin2 <= reward_returns[traj_idx],
                    reward_returns[traj_idx] <= (rmax2 - rmin2) * prmax + rmin2
                )
                mask = np.logical_and(cmask, rmask)
                inpainted_idx.append(traj_idx[mask])
                traj_idx = traj_idx[np.logical_not(mask)]
            inpainted_idx = np.array(inpainted_idx)
            # check if outliers are filtered
            if outliers_percent is not None:
                for idx in outliers_idx:
                    if idx not in traj_idx:
                        traj_idx = np.append(traj_idx, idx)

        if epsilon is not None:
            assert self.target_cost is not None, \
            "Please set target cost using env.set_target_cost(target_cost) if you want to change epsilon"
            # make it more difficult to train by filtering out
            # high reward trajectoris that satisfy target_cost
            if epsilon > 0:
                safe_idx = np.where(cost_returns <= self.target_cost)[0]
                ret = np.max(reward_returns[traj_idx[safe_idx]])
                mask = np.logical_and(
                    reward_returns[traj_idx[safe_idx]] >= ret - epsilon,
                    reward_returns[traj_idx[safe_idx]] <= ret
                )
                eps_reduce_idx = traj_idx[safe_idx[mask]]
                traj_idx = np.setdiff1d(traj_idx, eps_reduce_idx)
            # make it easier to train by filtering out
            # high reward trajectoris that violate target_cost
            if epsilon < 0:
                risk_idx = np.where(cost_returns > self.target_cost)[0]
                ret = np.max(reward_returns[traj_idx[risk_idx]])
                mask = np.logical_and(
                    reward_returns[traj_idx[risk_idx]] >= ret + epsilon,
                    reward_returns[traj_idx[risk_idx]] <= ret
                )
                eps_reduce_idx = traj_idx[risk_idx[mask]]
                traj_idx = np.setdiff1d(traj_idx, eps_reduce_idx)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        fontsize = 18
        plt.scatter(
            cost_returns[traj_idx],
            reward_returns[traj_idx],
            alpha=0.75
            # c='dodgerblue',
            # label='remained data'
        )
        if inpaint_ranges is not None:
            plt.scatter(
                cost_returns[inpainted_idx],
                reward_returns[inpainted_idx],
                c='gray',
                label='inpainted data'
            )
        if outliers_percent is not None:
            plt.scatter(
                cost_returns[outliers_idx],
                reward_returns[outliers_idx],
                c='tomato',
                label="augmented outliers"
            )
        if epsilon is not None:
            plt.scatter(
                cost_returns[eps_reduce_idx],
                reward_returns[eps_reduce_idx],
                c='grey',
                label="filtered data"
            )
        if self.target_cost is not None:
            plt.axvline(self.target_cost, linestyle='--', label="cost limit", color='k')
        plt.legend(fontsize=fontsize, loc='lower right')
        plt.xlabel("Cost return", fontsize=fontsize)
        plt.ylabel("reward return", fontsize=fontsize)
        plt.tight_layout()
        if removed_ratio is not None:
            fig_name = f"c_min-{removed_c_min}_c_max-{removed_c_max}_ratio-{removed_ratio}-num-{len(trajs)}"
        else:
            fig_name = f"no_removed-num-{len(trajs)}"
            
        # plt.savefig(
        #     fig_name+"-ilustration.png", dpi=400
        # )

        processed_data_dict = defaultdict(list)
        for k in data_dict.keys():
            for i in traj_idx:
                # print("processing: ", i)
                processed_data_dict[k].append(trajs[i][k])
        processed_data_dict = {
            k: np.concatenate(v)
            for k, v in processed_data_dict.items()
        }

        # perturbed observations
        if noise_scale is not None:
            noise_size = processed_data_dict["observations"].shape
            std = np.std(processed_data_dict["observations"])
            gaussian_noise = noise_scale * self.rng.normal(0, std, noise_size)
            processed_data_dict["observations"] += gaussian_noise
            std = np.std(processed_data_dict["next_observations"])
            gaussian_noise = noise_scale * self.rng.normal(0, std, noise_size)
            processed_data_dict["next_observations"] += gaussian_noise

        return processed_data_dict
    
    


class OfflineEnvWrapper(gym.Wrapper, OfflineEnv):
    """
    Wrapper class for offline RL envs.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.noise_scale = None

    def reset(self):
        obs, info = self.env.reset()
        if self.noise_scale is not None:
            obs += np.random.normal(0, self.noise_scale, obs.shape)
        return obs, info

    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def step(self, action):
        obs_next, reward, terminated, truncated, info = self.env.step(action)
        if self.noise_scale is not None:
            obs_next += np.random.normal(0, self.noise_scale, obs_next.shape)
        return obs_next, reward, terminated, truncated, info
    
class NormalizationEnvWrapper(gym.Wrapper, OfflineEnv):
    """
    Wrapper class for offline RL envs.
    """

    def __init__(self, env,
                 obs_mean = None,
                 obs_std = None,
                 action_mean = None,
                 action_std = None):
        gym.Wrapper.__init__(self, env)
        
        self.noise_scale = None
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std

    def reset(self):
        obs, info = self.env.reset()
        if self.noise_scale is not None:
            obs += np.random.normal(0, self.noise_scale, obs.shape)
        if self.obs_mean is not None:
            self.normalize_obs(obs)
        return obs, info
    
    def update_mean_std(self,
                        obs_mean = None,
                        obs_std = None,
                        action_mean = None,
                        action_std = None):
        
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std

    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def step(self, action):
        
        action = self.denormalize_action(action)
        obs_next, reward, terminated, truncated, info = self.env.step(action)
        if self.noise_scale is not None:
            obs_next += np.random.normal(0, self.noise_scale, obs_next.shape)
        if self.obs_mean is not None:
            obs_next = self.normalize_obs(obs_next)
        return obs_next, reward, terminated, truncated, info
    
    def normalize_obs(self, obs):
        return (obs-self.obs_mean) / self.obs_std

    def denormalize_obs(self, obs):
        return obs * self.obs_std + self.obs_mean
    
    def normalize_action(self, action):
        return (action-self.action_mean) / self.action_std

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean
