import os
import uuid
import types
from dataclasses import asdict

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym  # noqa
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import wrap_env, NormalizationEnvWrapper  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from osrl.configs.oasis_configs import OASISTrainConfig, DD_DEFAULT_CONFIG

from osrl.algorithms import OASIS, OASISTrainer
from osrl.common import SequenceDataset
from osrl.common.exp_util import auto_name, seed_all


@pyrallis.wrap()
def train(args: OASISTrainConfig):
    # update config
    # cfg, old_cfg = asdict(args), asdict(OASISTrainConfig())
    # args = types.SimpleNamespace(**cfg)
    
    cfg, old_cfg = asdict(args), asdict(OASISTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(DD_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(DD_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix) + '-camera-ready-1026' # + '-condition-ablation'
    # args.name += '-test_ret-' + str(args.test_ret) + "-w-" + str(args.condition_guidance_w)
    if args.group is None:
        args.group = "OASIS-" + args.task + '-camera-ready-1026'
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    visualization_logdir = os.path.join(args.visualization_log, args.group, args.name)
    if not os.path.exists(visualization_logdir):
        os.makedirs(visualization_logdir)

    visualization_logdir += "/"

    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # initialize environment
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
                                removed_ratio=args.removed_ratio,
                                filter_len=args.seq_len
                                )

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = NormalizationEnvWrapper(env)

    print("max_action:", env.action_space.high[0])

    # model & optimizer & scheduler setup
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
        stochastic=args.stochastic,
        n_timesteps = args.n_timesteps,
        returns_condition = args.returns_condition,
        condition_guidance_w = args.condition_guidance_w
    ).to(args.device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}
    
    if args.resume:
        model_state = torch.load(args.resume_path) #  model.load_state_dict
        model.load_state_dict(model_state['model_state'])


    logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = OASISTrainer(model,
                         env,
                         logger=logger,
                         learning_rate=args.learning_rate,
                         weight_decay=args.weight_decay,
                         betas=args.betas,
                         clip_grad=args.clip_grad,
                         lr_warmup_steps=args.lr_warmup_steps,
                         reward_scale=args.reward_scale,
                         cost_scale=args.cost_scale,
                         loss_cost_weight=args.loss_cost_weight,
                         loss_state_weight=args.loss_state_weight,
                         cost_reverse=args.cost_reverse,
                         no_entropy=args.no_entropy,
                         device=args.device,
                         visualization_log=visualization_logdir)

    ct = lambda x: 70 - x if args.linear else 1 / (x + 10)

    dataset = SequenceDataset(
        data,
        seq_len=args.seq_len,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
        deg=args.deg,
        pf_sample=args.pf_sample,
        max_rew_decrease=args.max_rew_decrease,
        beta=args.beta,
        augment_percent=args.augment_percent,
        cost_reverse=args.cost_reverse,
        max_reward=args.max_reward,
        min_reward=args.min_reward,
        pf_only=args.pf_only,
        rmin=args.rmin,
        cost_bins=args.cost_bins,
        npb=args.npb,
        cost_sample=args.cost_sample,
        cost_transform=ct,
        start_sampling=args.start_sampling,
        prob=args.prob,
        random_aug=args.random_aug,
        aug_rmin=args.aug_rmin,
        aug_rmax=args.aug_rmax,
        aug_cmin=args.aug_cmin,
        aug_cmax=args.aug_cmax,
        cgap=args.cgap,
        rstd=args.rstd,
        cstd=args.cstd,
    )

    env.update_mean_std(
        obs_mean = dataset.obs_mean,
        obs_std = dataset.obs_std,
        action_mean = dataset.action_mean,
        action_std = dataset.action_std
    )

    max_reward_return, max_cost_return = dataset.max_reward_return, dataset.max_cost_return
    print("max_reward_return", max_reward_return)
    print("max_cost_return", max_cost_return)
    # max_cost_return = 75

    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    max_returns = 0

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        
        states, actions, returns, costs_return, time_steps, mask, episode_reward, episode_cost, costs = [
            b.to(args.device) for b in batch
        ]

        returns = returns / max_reward_return
        costs_return = costs_return / max_cost_return

        episode_reward = episode_reward / max_reward_return
        episode_cost = episode_cost / max_cost_return

        max_returns = max(max_returns, returns.detach().cpu().max().item())

        visualization_flag = False

        if not args.resume:
            trainer.train_one_step(states, actions, episode_reward, costs_return, time_steps, mask,
                               episode_cost, costs, visualization_flag, step = step)
        
        # print("========Train one step==========")
        # evaluation
        # print("dataset max reward return:", max_returns)

        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1 or args.resume: # True or 
            # save the current weight
            logger.save_checkpoint(suffix = "-{}".format(step + 1))
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)
        
        # diffuser evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1 or args.resume: # True or 
            average_reward, average_cost = [], []
            log_cost, log_reward, log_len = {}, {}, {}
 
            ret, cost, length = trainer.evaluate(
                args.eval_episodes,
                args.test_condition)
            average_cost.append(cost)
            average_reward.append(ret)

            name = "eval_rollouts"
            log_cost.update({name: cost})
            log_reward.update({name: ret})
            log_len.update({name: length})
            

            logger.store(tab="cost", **log_cost)
            logger.store(tab="reward", **log_reward)
            logger.store(tab="length", **log_len)

            # save the current weight
            logger.save_checkpoint(suffix = "-{}".format(step + 1))
            # save the best weight
            mean_ret = np.mean(average_reward)
            mean_cost = np.mean(average_cost)

            if args.resume:
                print("mean_ret:", mean_ret)
                print("mean_cost:", mean_cost)
            # if mean_cost < best_cost or (mean_cost == best_cost
            #                              and mean_ret > best_reward):
            #     best_cost = mean_cost
            #     best_reward = mean_ret
            #     best_idx = step
            #     logger.save_checkpoint(suffix="best")

            # logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)

if __name__ == "__main__":
    train()
