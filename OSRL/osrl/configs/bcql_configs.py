from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field

import numpy as np

@dataclass
class BCQLTrainConfig:
    project: str = "OASIS-RL_agent" 
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
    seed: int = 33
    device: str = "cuda:0"
    new_data_path: Optional[str] = "../../../dataset/from_tempting/OfflineBallCircle-v0-from_tempting.hdf5"

    # True means to use curated dataset
    update_data: bool = True
    replace: bool = True

    removed_r_min: float = -np.inf
    removed_r_max: float = np.inf 
    removed_c_min: float = 0.
    removed_c_max: float = 30.
    removed_ratio: float = 0.9
    removed_all_ratio: float = 0.9

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
    update_steps: int = 25_000
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
class BCQLBallCircleConfig(BCQLTrainConfig):
    # training params
    episode_len: int = 200
    new_data_path: Optional[str] = "../../../dataset/from_tempting/OfflineBallCircle-v0-from_tempting.hdf5"


@dataclass
class BCQLBallRunConfig(BCQLTrainConfig):
    # dataset info
    episode_len: int = 100
    new_data_path: Optional[str] = "../../../dataset/from_tempting/OfflineBallRun-v0-from_tempting.hdf5"


@dataclass
class BCQLCarCircleConfig(BCQLTrainConfig):
    # dataset info
    new_data_path: Optional[str] = "../../../dataset/from_tempting/OfflineCarCircle-v0-from_tempting.hdf5"
    
@dataclass
class BCQLCarRunConfig(BCQLTrainConfig):
    # dataset info
    episode_len: int = 200
    new_data_path: Optional[str] = "../../../dataset/from_tempting/OfflineCarRun-v0-from_tempting.hdf5"

@dataclass
class BCQLDroneRunConfig(BCQLTrainConfig):
    # dataset info
    episode_len: int = 200    
    new_data_path: Optional[str] = "../../../dataset/from_tempting/OfflineDroneRun-v0-from_tempting.hdf5"
    update_steps: int = 50_000

@dataclass
class BCQLDroneCircleConfig(BCQLTrainConfig):
    # dataset info
    episode_len: int = 300
    new_data_path: Optional[str] = "../../../dataset/from_tempting/OfflineDroneCircle-v0-from_tempting.hdf5"
    update_steps: int = 50_000


BCQL_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": BCQLCarCircleConfig,
    "OfflineDroneRun-v0": BCQLDroneRunConfig,
    "OfflineDroneCircle-v0": BCQLDroneCircleConfig,
    "OfflineCarRun-v0": BCQLCarRunConfig,
    "OfflineBallCircle-v0": BCQLBallCircleConfig,
    "OfflineBallRun-v0": BCQLBallRunConfig,
}