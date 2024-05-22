from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field

import numpy as np

@dataclass
class BCQLTrainConfig:
    project: str = "OASIS" 
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "OASIS-BCQL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1
    # training params
    task: str = "OfflineBallCircle-v0"
    dataset: str = None
    seed: int = 22
    device: str = "cuda:3"
    new_data_path: Optional[str] = None

    update_data: bool = True
    replace: bool = True

    removed_r_min: float = -np.inf
    removed_r_max: float = np.inf 
    removed_c_min: float = 0.
    removed_c_max: float = 30.
    removed_ratio: float = 0.9
    removed_all_ratio: float = 0.2

    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    vae_lr: float = 0.001
    phi: float = 0.05
    lmbda: float = 0.75
    beta: float = 0.5
    cost_limit: int = 20
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    num_q: int = 2
    num_qc: int = 2
    PID: List[float] = field(default=[0.1, 0.003, 0.001], is_mutable=True)
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class BCQLCarCircleConfig(BCQLTrainConfig):
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineCarCircle-v0-condition-[[0.15, 0.7], [0.3, 0.85]]-num-630000learned_label.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineCarCircle-v0-n-40-condition-[[0.1, 0.5], [0.1, 0.4]]-num-192000_tempting_dataset_clip-17.hdf5"
    
    new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineCarCircle-v0-num-189000CVAE_tempting_dataset-batch_size-3000.hdf5"

@dataclass
class BCQLBallCircleConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineBallCircle-v0-condition-[[0.1, 0.7], [0.1, 0.65]]-num-310000learned_label-0504.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineBallCircle-v0-condition-[[0.25, 0.85], [0.3, 0.9]]-num-160000medium_dataset_clip-17.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineBallCircle-v0-condition-[[0.2, 0.8], [0.15, 0.8]]-num-160000full_dataset_clip-17.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineBallCircle-v0-condition-[[0.35, 0.85], [0.4, 0.9]]-num-160000conservative_dataset_clip-17.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineBallCircle-v0-condition-[[0.5, 0.8], [0.45, 0.8]]-num-160000conservative_dataset_clip-17.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineBallCircle-v0-condition-[[0.5, 0.85], [0.45, 0.8]]-num-160000conservative_dataset_clip-17.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineBallCircle-v0-n-20-condition-[[0.2, 0.8]]-num-1600_tempting_dataset_clip-17-batch_size-100.hdf5"

    new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineBallCircle-v0-num-189000CVAE_tempting_dataset-batch_size-3000.hdf5"

@dataclass
class BCQLAntRunConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class BCQLDroneRunConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineDroneRun-v0-condition-[[0.2, 0.5], [0.2, 0.6]]-num-630000learned_label-0504.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineDroneRun-v0-condition-[[0.1, 0.5], [0.1, 0.6]]-num-630000learned_label-0504.hdf5"
    
    new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineDroneRun-v0-num-63000CVAE_tempting_dataset-batch_size-1000.hdf5"
    
    update_data: bool = True
    replace: bool = True

@dataclass
class BCQLDroneCircleConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineDroneCircle-v0-condition-[[0.15, 0.95], [0.1, 0.95]]-num-630000learned_label.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineDroneCircle-v0-condition-[[0.15, 0.8], [0.1, 0.8]]-num-630000learned_label.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineDroneCircle-v0-condition-[[0.1, 0.75], [0.1, 0.8]]-num-630000learned_label.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineDroneCircle-v0-condition-[[0.1, 0.75], [0.1, 0.8]]-num-310000learned_label-0504.hdf5"
    # new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineDroneCircle-v0-condition-[[0.15, 0.85], [0.1, 0.8]]-num-310000learned_label-0504.hdf5"
    new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineDroneCircle-v0-num-189000CVAE_tempting_dataset-batch_size-3000.hdf5"
    update_data: bool = True
    replace: bool = True

@dataclass
class BCQLCarRunConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200
    new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineCarRun-v0-num-189000CVAE_tempting_dataset-batch_size-3000.hdf5"
    update_data: bool = True
    replace: bool = True

@dataclass
class BCQLAntCircleConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class BCQLBallRunConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100
    new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineBallRun-v0-num-189000CVAE_tempting_dataset-batch_size-3000.hdf5"
    update_data: bool = True
    replace: bool = True


@dataclass
class BCQLCarButton1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarButton2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarCircle1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQLCarCircle2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQLCarGoal1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarGoal2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarPush1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLCarPush2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointButton1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointButton2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointCircle1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500
    new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflinePointCircle1Gymnasium-v0-condition-[[0.05, 0.625], [0.05, 0.65], [0.05, 0.575], [0.05, 0.6]]-num-310000learned_label-0504.hdf5"
    update_data: bool = True
    replace: bool = True

@dataclass
class BCQLPointCircle2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQLPointGoal1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointGoal2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointPush1Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLPointPush2Config(BCQLTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQLAntVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000
    
    new_data_path: Optional[str] = "/home/yihang/.dsrl/datasets/generated/OfflineAntVelocityGymnasium-v1-condition-[[0.05, 0.625], [0.05, 0.65], [0.05, 0.575], [0.05, 0.6]]-num-630000learned_label-0504.hdf5"
    update_data: bool = True
    replace: bool = True

@dataclass
class BCQLHalfCheetahVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQLHopperVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQLSwimmerVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQLWalker2dVelocityConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQLEasySparseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLEasyMeanConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLEasyDenseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLMediumSparseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLMediumMeanConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLMediumDenseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLHardSparseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLHardMeanConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQLHardDenseConfig(BCQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


BCQL_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": BCQLCarCircleConfig,
    "OfflineAntRun-v0": BCQLAntRunConfig,
    "OfflineDroneRun-v0": BCQLDroneRunConfig,
    "OfflineDroneCircle-v0": BCQLDroneCircleConfig,
    "OfflineCarRun-v0": BCQLCarRunConfig,
    "OfflineAntCircle-v0": BCQLAntCircleConfig,
    "OfflineBallCircle-v0": BCQLBallCircleConfig,
    "OfflineBallRun-v0": BCQLBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": BCQLCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": BCQLCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": BCQLCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": BCQLCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": BCQLCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": BCQLCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": BCQLCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": BCQLCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": BCQLPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": BCQLPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": BCQLPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": BCQLPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": BCQLPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": BCQLPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": BCQLPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": BCQLPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": BCQLAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": BCQLHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": BCQLHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": BCQLSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": BCQLWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": BCQLEasySparseConfig,
    "OfflineMetadrive-easymean-v0": BCQLEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": BCQLEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": BCQLMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": BCQLMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": BCQLMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": BCQLHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": BCQLHardMeanConfig,
    "OfflineMetadrive-harddense-v0": BCQLHardDenseConfig
}