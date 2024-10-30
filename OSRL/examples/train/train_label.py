import torch
from torch import nn
import h5py
import numpy as np
from osrl.algorithms.casual_model import CausalDecomposition, TransitionDataset, CUDA
from torch.utils.data import DataLoader
import gymnasium as gym  # noqa
import dsrl
from tqdm.auto import trange  # noqa
import pyrallis
from dataclasses import dataclass
from fsrl.utils import BaseLogger, WandbLogger
from dataclasses import asdict
from osrl.common.exp_util import auto_name
import os
import matplotlib.pyplot as plt
import seaborn
from torch.nn import functional as F

def load_data(filename):
    hfiles = []
    for file in filename:
        hfiles.append(h5py.File(file, 'r'))

    keys = ['observations', 'next_observations', 'actions', 'rewards', 'costs', 'terminals', 'timeouts']

    print("*" * 10, "concatenating dataset from", "*" * 10)
    for file in filename:
        print("*" * 10, file, "*" * 10)
    dataset_dict = {}
    for k in keys:
        d = [hfile[k] for hfile in hfiles]
        combined = np.concatenate(d, axis=0)
        print(k, combined.shape)
        dataset_dict[k] = combined
    print("*" * 10, "dataset concatenation finished", "*" * 10)

    return dataset_dict

@dataclass
class TrainCfg:
    # general task params
    task: str = "OfflineDroneRun-v0"
    device: str = "cuda:0"
    epoch: int = 301
    step_per_epoch: int = 500
    batch_size: int = 32
    num_workers: int = 1
    lr: float = 1e-4

    learning_mode = "cost"
    
    # task:
    velocity_constraint: float = 1.5

    # causal model
    sparse_start_ratio: float = 0.
    state_dim: int = 8 # 76
    sub_cost_num: int = 2
    hidden_dim: int = 128
    causal_dim: int = 10
    num_hidden: int = 10
    mse_weight: float = 20
    sparse_weight: float = 0.1 # [0.5, 0.1, 0.02]
    sim_weight: float = 0. # [0.5, 0.1, 0.02]
    norm_sim_weight: float = 0.0 #  [0.5, 0.1, 0.02]
    sparse_norm: float = 0. # [0.5, 0.25, 0.1, 0.05]
    log_suffix: str = "1028"

    # logger 
    verbose: bool = True
    project: str = "OASIS-" + task + "-" + learning_mode + "-camera-ready"
    suffix: str = ""
    prefix: str = ""
    name = None
    group = None
    logdir = "label_log"
    

@pyrallis.wrap()
def train(args: TrainCfg):
    env = gym.make(args.task)

    cfg = asdict(args)
    # default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = auto_name(cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = "Model-" + args.task + "-" + args.learning_mode
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir) # BaseLogger() # 
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    data = env.get_dataset()

    data = env.pre_process_data(data,
                                density=1.0,
                                removed_c_min=0,
                                removed_c_max=30,
                                removed_ratio=0.9)

    trainloader = DataLoader(
        TransitionDataset(data),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        
        # pin_memory_device = args.device
    )
    trainloader_iter = iter(trainloader)

    decomposition = CausalDecomposition(
        input_dim=env.observation_space.shape[0] * 2, 
        output_dim=args.sub_cost_num,
        hidden_dim=args.hidden_dim,
        causal_dim=args.causal_dim,
        num_hidden=args.num_hidden,
        sparse_weight=args.sparse_weight,
        sim_weight=args.sim_weight,
        norm_sim_weight=args.norm_sim_weight,
        sparse_norm=args.sparse_norm,
        mse_weight = args.mse_weight,
        device=args.device,
        learning_mode=args.learning_mode
    )

    causal_optim = torch.optim.Adam(decomposition.parameters(), lr=args.lr) # , weight_decay=self.weight_decay_q

    save_epoch_int = 20
    sparse_start_ratio = args.sparse_start_ratio
    figure_index = 0

    for step in trange(args.epoch, desc="Training"):
        loss_mean = 0
        logger.write_without_reset(step)

        if step >= args.epoch * sparse_start_ratio:
            decomposition.sparse_flag = True
        else:
            decomposition.sparse_flag = False

        for _ in range(args.step_per_epoch):
            batch = next(trainloader_iter)
            observations, next_observations, _, rewards, costs, _ = [b.to(args.device) for b in batch]

            if args.learning_mode == "reward":
                true_val = rewards * 0.1
            elif args.learning_mode == "cost":
                true_val = costs
            
            # print(batch)
            input_val = torch.cat((observations, next_observations), axis=-1)
            pred_val = decomposition(input_val)
            true_val = torch.as_tensor(true_val, dtype=torch.float32)

            loss, info = decomposition.loss_function(true_val, pred_val)
            causal_optim.zero_grad()
            loss.backward()
            causal_optim.step()
            loss_mean += info["train/causal_mse"]

        loss_mean = loss_mean / args.step_per_epoch
        logger.store(**info)
        logger.store(**{"train/lossmean": loss_mean})

        logger.write(step, display=args.verbose)
        

        # store mask figrue
        if step % save_epoch_int == 0 and step >= args.epoch * sparse_start_ratio:
            mask_prob, mask = decomposition.get_mask()
            mask = mask.detach().cpu().numpy()
            mask_prob = mask_prob.detach().cpu().numpy()

            plt.style.use('seaborn-v0_8')
            plt.figure(figure_index) # , figsize=(2, 5)
            seaborn.heatmap(mask_prob)
            name = "epoch_"+str(step) + "_prob"
            plt.title(name)

            mask_logdir = "label_results/"+args.task+"/log"+ args.learning_mode + args.log_suffix 
            if not os.path.exists(mask_logdir):
                os.makedirs(mask_logdir)

            
            plt.savefig(mask_logdir+"/"+name+".png", dpi=400)

            figure_index += 1

    # model evaluation:
    mse_total = 0
    evaluation_num = 10
    for _ in range(evaluation_num):
        batch = next(trainloader_iter)
        observations, next_observations, _, _, costs, _ = [b.to(args.device) for b in batch]
        input_val = torch.cat((observations, next_observations), axis=-1)
        cost_predict = decomposition(input_val)
        # cost_predict = decomposition(observations)
        costs = torch.as_tensor(costs, dtype=torch.float32)
        velocity_cost = velocity_violation(state=observations, velocity_constraint=args.velocity_constraint)
        costs += velocity_cost
        mse = F.mse_loss(costs, cost_predict) 
        mse_total += mse.item()
    mse_total = mse_total / evaluation_num
    print("mse evaluation:", mse_total)
    
    # save model
    path = "label_results/"+args.task+"/log"+ args.learning_mode + args.log_suffix+"/"+args.learning_mode + "-epoch_"+str(args.epoch)+".pt"
    decomposition.save_model(path=path)

    return

def velocity_violation(state, velocity_constraint):
    velocity = state[:, 2:4].cpu().numpy()
    # print(velocity.shape)
    v_norm = np.sum(np.abs(velocity)**2,axis=-1)**(1./2)
    cost = torch.as_tensor([1 if vi >= velocity_constraint else 0 for vi in v_norm ], dtype=torch.float32)
    # print(cost.shape)
    return CUDA(cost)



if __name__ == "__main__":
    train()