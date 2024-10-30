from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import numpy as np

@dataclass
class OASISTrainConfig:
    # wandb params
    project: str = "OASIS-(Camera Ready)-1027"
    task: str = "OfflineDroneRun-v0" 
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "OASIS"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float], ...] = None
    epsilon: float = None
    density: float = 1
    # model params
    embedding_dim: int = 128 # 32 # 128
    num_layers: int = 3
    num_heads: int = 8
    action_head_layers: int = 1
    seq_len: int = 32 # 10 # Horizon length
    episode_len: int = 300 # 300 for carcircle
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    time_emb: bool = True
    # training params
    
    dataset: str = None
    learning_rate: float = 3e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 256
    update_steps: int = 200_000
    lr_warmup_steps: int = 500
    reward_scale: float = 0.1
    cost_scale: float = 1
    num_workers: int = 6

    resume: bool = False
    resume_path: str = None 
    # general params
    seed: int = 20
    device: str = "cuda:1"

    # additional dataset operation
    removed_r_min: float = -np.inf
    removed_r_max: float = np.inf 
    removed_c_min: float = 0.
    removed_c_max: float = 30.
    removed_ratio: float = 0.9

    # test condition
    visualization_log: str = "visualization"
    test_condition: Tuple[float, float] = (0.2, 0.7) # cost, reward
    
    condition_guidance_w: float = 2. # TODO weight
    saving_interval = 10000
    test_ret = 0.9 # TODO
    returns_condition: bool = True
    n_timesteps: int = 20 # denoising timestep

    # evaluation params
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((450.0, 10), (500.0, 20), (550.0, 50))  # reward, cost
    cost_limit: int = 20
    eval_episodes: int = 5
    eval_every: int = 20000
    
    threads: int = 6
    # augmentation param
    deg: int = 4
    pf_sample: bool = False
    beta: float = 1.0
    augment_percent: float = 0.2

    max_reward: float = 1000.0
    # minimum reward above the PF curve
    min_reward: float = 1.0
    # the max drecrease of ret between the associated traj
    # w.r.t the nearest pf traj
    max_rew_decrease: float = 100.0
    # model mode params
    use_rew: bool = True
    use_cost: bool = True
    cost_transform: bool = True
    cost_prefix: bool = False
    add_cost_feat: bool = False
    mul_cost_feat: bool = False
    cat_cost_feat: bool = False
    loss_cost_weight: float = 0.02
    loss_state_weight: float = 0
    cost_reverse: bool = False
    # pf only mode param
    pf_only: bool = False
    rmin: float = 300
    cost_bins: int = 60
    npb: int = 5
    cost_sample: bool = True
    linear: bool = True  # linear or inverse
    start_sampling: bool = False
    prob: float = 0.2
    stochastic: bool = True
    init_temperature: float = 0.1
    no_entropy: bool = False
    # random augmentation
    random_aug: float = 0
    aug_rmin: float = 400
    aug_rmax: float = 700
    aug_cmin: float = -2
    aug_cmax: float = 25
    cgap: float = 5
    rstd: float = 1
    cstd: float = 0.2


@dataclass
class DDCarCircleConfig(OASISTrainConfig):
    pass

@dataclass
class DDAntRunConfig(OASISTrainConfig):
    # model params
    seq_len: int = 32
    episode_len: int = 200
    # training params
    task: str = "OfflineAntRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((700.0, 10), (750.0, 20), (800.0, 40))
    # augmentation param
    deg: int = 3
    max_reward: float = 1000.0
    max_rew_decrease: float = 150
     


@dataclass
class DDDroneRunConfig(OASISTrainConfig):
    # model params
    seq_len: int = 32
    episode_len: int = 200
    embedding_dim: int = 128 # 128
    update_steps: int = 600_000
    
    # training params
    task: str = "OfflineDroneRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((400.0, 10), (500.0, 20), (600.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 700.0
    max_rew_decrease: float = 100
    min_reward: float = 1
     


@dataclass
class DDDroneCircleConfig(OASISTrainConfig):
    # model params
    seq_len: int = 32
    embedding_dim: int = 128 # 128
    episode_len: int = 300
    update_steps: int = 600_000
    # training params
    task: str = "OfflineDroneCircle-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((700.0, 10), (750.0, 20), (800.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 1000.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    #  


@dataclass
class DDCarRunConfig(OASISTrainConfig):
    # model params
    seq_len: int = 32
    embedding_dim: int = 64
    episode_len: int = 200
    # training params
    task: str = "OfflineCarRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((575.0, 10), (575.0, 20), (575.0, 40))
    # augmentation param
    deg: int = 0
    max_reward: float = 600.0
    max_rew_decrease: float = 100
    min_reward: float = 1
     


@dataclass
class DDAntCircleConfig(OASISTrainConfig):
    # model params
    seq_len: int = 32
    episode_len: int = 500
    # training params
    task: str = "OfflineAntCircle-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
     


@dataclass
class DDBallRunConfig(OASISTrainConfig):
    # model params
    # seq_len: int = 32
    episode_len: int = 100
    embedding_dim: int = 64
    # training params
    task: str = "OfflineBallRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((500.0, 10), (500.0, 20), (700.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 1400.0
    max_rew_decrease: float = 200
    min_reward: float = 1

@dataclass
class DDBallCircleConfig(OASISTrainConfig):
    pass

DD_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineBallRun-v0": DDBallRunConfig,
    "OfflineBallCircle-v0": DDBallCircleConfig,
    "OfflineCarRun-v0": DDCarRunConfig,
    "OfflineCarCircle-v0": DDCarCircleConfig,
    "OfflineDroneRun-v0": DDDroneRunConfig,
    "OfflineDroneCircle-v0": DDDroneCircleConfig,
}
