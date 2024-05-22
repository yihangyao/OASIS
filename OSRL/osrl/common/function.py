import numpy as np
import torch 
import os
import h5py
from tianshou.data.utils.converter import to_hdf5
from tianshou.data import Batch

class BaseModel():
    """
        
    """
    def __init__(self):
        return

    def forward(self, s, a, s_next):
        raise NotImplementedError

class RewardTruthModel(BaseModel):
    def __init__(self, radius=None, normalizer = 50.0):
        super().__init__()
        assert radius is not None
        self.radius = radius
        self.normalizer = normalizer
    
    def forward(self, s, a, s_next):
        """
            s: [seq_len, s_dim]
            a: [seq_len, a_dim]
            s_next: [seq_len, s_dim]
        """

        if len(s_next.shape) == 1:
            s = s.reshape(1, -1)
            a = a.reshape(1, -1)
            s_next = s_next.reshape(1, -1)

        pos = s_next[:, :2] * 10.0
        vel = s_next[:, 2:4] * 5.0
        dist = np.sqrt(np.sum(pos**2, axis=1))
        # position vector and optimal velocity are orthogonal to each other:
        # optimal reward when position vector and orthogonal velocity
        # point into same direction
        vel_orthogonal = np.array([-vel[:, 1], vel[:, 0]]).transpose()
        r = 0.1*np.sum(pos*vel_orthogonal, axis=-1)/(1+np.abs(dist-self.radius))
        r = r / self.normalizer
        return r
    

class CostTruthModel(BaseModel):
    def __init__(self, 
                 xlim=None,
                 grad_clip = 10000,
                 dataset = None,
                 device = "cuda:1"
                 ):
        super().__init__()
        assert xlim is not None
        self.xlim = xlim
        self.grad_clip = grad_clip
        if dataset is not None:
            self.obs_mean = torch.tensor(dataset.obs_mean, device=device)
            self.obs_std = torch.tensor(dataset.obs_std, device=device)
            self.action_mean = torch.tensor(dataset.action_mean, device=device)
            self.action_std = torch.tensor(dataset.action_std, device=device)
    
    def forward(self, s, a, s_next):
        if len(s.shape) == 1:
            s = s.reshape(1, -1)
            a = a.reshape(1, -1)
            s_next = s_next.reshape(1, -1)

        x = s_next[:, 0] * 10.0
        cost = np.zeros(s.shape[0]) # batch size
        index_cost = np.abs(x) > np.ones(x.shape) * self.xlim
        cost[index_cost] = 1.

        return cost

    def c_gradient(self, s, need_denormalize = False):
        # s should be a tensor
        
        if len(s.shape) == 1:
            s = s.reshape(1, -1)
        if need_denormalize:
            s = self.denormalize_obs(s)
        x = s[:, 0] * 10.0
        grad = torch.zeros(s.shape, device=s.device) # batch size
        index_cost = torch.abs(x) > torch.ones(x.shape, device=s.device) * self.xlim

        # for index in index_cost:
        grad_i = (torch.abs(s[index_cost, 0] * 10.0) - self.xlim) * s[index_cost, 0] / torch.abs(s[index_cost, 0])
        grad_i = torch.clip(grad_i, -1 * self.grad_clip, self.grad_clip)
        grad[index_cost, 0] = grad_i

        return grad
    
    def normalize_obs(self, obs):
        self.obs_mean = self.obs_mean.to(obs.device)
        self.obs_std = self.obs_std.to(obs.device)
        return (obs-self.obs_mean) / self.obs_std

    def denormalize_obs(self, obs):
        self.obs_mean = self.obs_mean.to(obs.device)
        self.obs_std = self.obs_std.to(obs.device)
        return obs * self.obs_std + self.obs_mean
    
    def normalize_action(self, action):
        self.action_mean.to(action.device)
        self.action_std.to(action.device)
        return (action-self.action_mean) / self.action_std

    def denormalize_action(self, action):
        self.action_mean.to(action.device)
        self.action_std.to(action.device)
        return action * self.action_std + self.action_mean


def save(log_dir: str, dataset_name: str = "dataset.hdf5", data = None) -> None:
    """Saves the entire buffer to disk as an HDF5 file.

    :param log_dir: Directory to save the dataset in.
    :type log_dir: str
    :param dataset_name: Name of the dataset file to save.
    :type dataset_name: str, optional (default="dataset.hdf5")
    """
    print("Saving dataset...")
    if not os.path.exists(log_dir):
        print(f"Creating saving dir {log_dir}")
        os.makedirs(log_dir)
    dataset_path = os.path.join(log_dir, dataset_name)
    all_data = data
    with h5py.File(dataset_path, "w") as f:
        to_hdf5(all_data, f, compression='gzip')
    print(f"Finish saving dataset to {dataset_path}!")


        

