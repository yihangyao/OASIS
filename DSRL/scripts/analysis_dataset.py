import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
# from dsrl.script.utils import get_trajectory_info
from numba import njit
from numba.typed import List
@njit
def compute_cost_reward_return(
    rew: np.ndarray,
    cost: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    returns,
    costs,
    starts,
    ends,
) -> np.ndarray:
    data_num = rew.shape[0]
    rew_ret, cost_ret = 0, 0
    is_start = True
    for i in range(data_num):
        if is_start:
            starts.append(i)
            is_start = False
        rew_ret += rew[i]
        cost_ret += cost[i]
        if terminals[i] or timeouts[i]:
            returns.append(rew_ret)
            costs.append(cost_ret)
            ends.append(i)
            is_start = True
            rew_ret, cost_ret = 0, 0


def get_trajectory_info(dataset: dict):
    # we need to initialize the numba List such that it knows the item type
    returns, costs = List([0.0]), List([0.0])
    # store the start and end indexes of the trajectory in the original data
    starts, ends = List([0]), List([0])
    data_num = dataset["rewards"].shape[0]
    print(f"Total number of data points: {data_num}")
    compute_cost_reward_return(
        dataset["rewards"], dataset["costs"], dataset["terminals"], dataset["timeouts"],
        returns, costs, starts, ends
    )
    return returns[1:], costs[1:], starts[1:], ends[1:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--file_path')
    parser.add_argument('--output', '-o', type=str, default='cr-plot.png')
    parser.add_argument('--maxlen', type=int, default=50000000)
    args = parser.parse_args()

    # root_dir = args.root

    # file_paths = glob.glob(os.path.join(root_dir, '**', 'dataset.hdf5'), recursive=True)

    # for file_path in file_paths:
    file_path = "/home/yihang/code/OSRL_DD/Generation/test.hdf5"
    dir_path = os.path.dirname(file_path)
    print("reading from ... ", dir_path)
    data = h5py.File(file_path, 'r')

    keys = [
        'observations', 'next_observations', 'actions', 'rewards', 'costs',
        'terminals', 'timeouts'
    ]

    dataset_dict = {}
    for k in keys:
        combined = np.array(data[k])[:args.maxlen]
        print(k, combined.shape)
        dataset_dict[k] = combined

    rew_ret, cost_ret, start_index, end_index = get_trajectory_info(dataset_dict)

    print(f"Total number of trajectories: {len(rew_ret)}")

    plt.scatter(cost_ret, rew_ret)
    plt.xlabel("Costs")
    plt.ylabel("Rewards")
    output_path = os.path.join(dir_path, args.output)
    plt.savefig(output_path)
    plt.clf()
