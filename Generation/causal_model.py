import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import IterableDataset

# This script is built based on the reference: 
# Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation
# https://arxiv.org/abs/2307.07907

def CUDA(var, device = 'cuda:2'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    var = var.to(device)
    return var


def temp_sigmoid(x, temp=1.0):
    return torch.sigmoid(x/temp) 

class TransitionDataset_Subcosts(IterableDataset):
    """
    """

    def __init__(self,
                 dataset: dict,
                 reward_scale: float = 1.0,
                 cost_scale: float = 1.0,
                 state_init: bool = False):
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.sample_prob = None
        self.state_init = state_init
        self.dataset_size = self.dataset["observations"].shape[0]

        self.dataset["done"] = np.logical_or(self.dataset["terminals"],
                                             self.dataset["timeouts"]).astype(np.float32)
        if self.state_init:
            self.dataset["is_init"] = self.dataset["done"].copy()
            self.dataset["is_init"][1:] = self.dataset["is_init"][:-1]
            self.dataset["is_init"][0] = 1.0

    def get_dataset_states(self):
        """
        Returns the proportion of initial states in the dataset, 
        as well as the standard deviations of the observation and action spaces.
        """
        init_state_propotion = self.dataset["is_init"].mean()
        obs_std = self.dataset["observations"].std(0, keepdims=True)
        act_std = self.dataset["actions"].std(0, keepdims=True)
        return init_state_propotion, obs_std, act_std

    def __prepare_sample(self, idx):
        observations = self.dataset["observations"][idx, :]
        next_observations = self.dataset["next_observations"][idx, :]
        actions = self.dataset["actions"][idx, :]
        rewards = self.dataset["rewards"][idx] * self.reward_scale
        costs = self.dataset["costs"][idx] * self.cost_scale
        sub_cost_0 = self.dataset["sub_cost_0"][idx] * self.cost_scale
        sub_cost_1 = self.dataset["sub_cost_1"][idx] * self.cost_scale
        sub_cost_2 = self.dataset["sub_cost_2"][idx] * self.cost_scale
        done = self.dataset["done"][idx]
        if self.state_init:
            is_init = self.dataset["is_init"][idx]
            return observations, next_observations, actions, rewards, costs, done, is_init, sub_cost_0, sub_cost_1, sub_cost_2
        return observations, next_observations, actions, rewards, costs, done, sub_cost_0, sub_cost_1, sub_cost_2

    def __iter__(self):
        while True:
            idx = np.random.choice(self.dataset_size, p=self.sample_prob)
            yield self.__prepare_sample(idx)

class TransitionDataset(IterableDataset):
    """


    """

    def __init__(self,
                 dataset: dict,
                 reward_scale: float = 1.0,
                 cost_scale: float = 1.0,
                 state_init: bool = False):
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.sample_prob = None
        self.state_init = state_init
        self.dataset_size = self.dataset["observations"].shape[0]

        self.dataset["done"] = np.logical_or(self.dataset["terminals"],
                                             self.dataset["timeouts"]).astype(np.float32)
        if self.state_init:
            self.dataset["is_init"] = self.dataset["done"].copy()
            self.dataset["is_init"][1:] = self.dataset["is_init"][:-1]
            self.dataset["is_init"][0] = 1.0

    def get_dataset_states(self):
        """
        Returns the proportion of initial states in the dataset, 
        as well as the standard deviations of the observation and action spaces.
        """
        init_state_propotion = self.dataset["is_init"].mean()
        obs_std = self.dataset["observations"].std(0, keepdims=True)
        act_std = self.dataset["actions"].std(0, keepdims=True)
        return init_state_propotion, obs_std, act_std

    def __prepare_sample(self, idx):
        observations = self.dataset["observations"][idx, :]
        next_observations = self.dataset["next_observations"][idx, :]
        actions = self.dataset["actions"][idx, :]
        rewards = self.dataset["rewards"][idx] * self.reward_scale
        costs = self.dataset["costs"][idx] * self.cost_scale
        
        done = self.dataset["done"][idx]
        if self.state_init:
            is_init = self.dataset["is_init"][idx]
            return observations, next_observations, actions, rewards, costs, done, is_init #, sub_cost_0, sub_cost_1, sub_cost_2
        return observations, next_observations, actions, rewards, costs, done

    def __iter__(self):
        while True:
            idx = np.random.choice(self.dataset_size, p=self.sample_prob)
            yield self.__prepare_sample(idx)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_hidden=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = nn.ReLU()

        self.fc_list = nn.ModuleList()
        self.fc_list.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden):
            self.fc_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        assert x.shape[-1] == self.input_dim
        for i in range(len(self.fc_list)):
            x = self.activation(self.fc_list[i](x))
        
        # no activation for the last layer
        return self.output_fc(x)


class CausalDecomposition(nn.Module):
    def __init__(self, 
                input_dim, 
                output_dim, 
                hidden_dim=128, 
                causal_dim=3, 
                num_hidden=8, 
                mse_weight = 20,
                sparse_weight=1, 
                sim_weight = 0.2, 
                norm_sim_weight = 0.3, 
                F_weight = 0.1,
                sparse_norm=0.1, 
                use_full=False, 
                learning_mode = "reward",
                device = "cuda:0"):
        super(CausalDecomposition, self).__init__()
        self.input_dim = input_dim         # S
        self.output_dim = output_dim       # A
        self.causal_dim = causal_dim       # C
        self.emb_dim = 20                  # E
        self.device = device
        self.sim_weight = sim_weight
        self.norm_sim_weight = norm_sim_weight
        self.mse_weight = mse_weight
        self.F_weight = F_weight
        self.learning_mode = learning_mode

        """
            S: state dim
            F: user-defined dimension
            C: sub-cost dimension
            B: batch size

            sparse_norm --> 0
        """

        self.sparse_flag = False

        # sharable encoder
        self.encoder = CUDA(MLP(self.emb_dim+1, causal_dim, hidden_dim, num_hidden), device=self.device) # $
        self.encoder_idx_emb = CUDA(nn.Embedding(self.input_dim, self.emb_dim), device=self.device)

        # sharable decoder
        self.decoder = CUDA(MLP(causal_dim+self.emb_dim, 1, hidden_dim, num_hidden), device=self.device) # $
        self.decoder_idx_emb = CUDA(nn.Embedding(self.output_dim, self.emb_dim), device=self.device)

        self.use_full = use_full
        self.sparse_weight = sparse_weight
        self.sparse_norm = sparse_norm
        self.tau = 1
        self.mask_prob = nn.Parameter(torch.ones(input_dim, output_dim, device=self.device, requires_grad=True)) # $
        self.mask = CUDA(torch.ones_like(self.mask_prob), device=self.device)

        self.evaluation_mode = False

    def forward(self, inputs, threshold=None, sub_cost_flag = False):
        
        assert len(inputs.shape) == 2
        inputs = inputs.unsqueeze(-1) # [B, S, 1]

        # obtain state-action idx embedding
        # [B, S, 1] -> [B, S, E+1]
        encoder_idx = self.encoder_idx_emb(CUDA(torch.arange(0, self.input_dim).long(), device=self.device))
        batch_encoder_idx = encoder_idx.repeat(inputs.shape[0], 1, 1).detach()
        inputs_feature = torch.cat([inputs, batch_encoder_idx], dim=-1)

        # encoder: [B, S+A, E+1] -> [B, S+A, C]
        latent_feature = self.encoder(inputs_feature) # [B, S, C]

        # prepare mask
        if not self.use_full:
            _, self.mask = self.get_mask(threshold)

        # feature mask: 
        masked_feature = torch.einsum('bnc, ns -> bsc', latent_feature,  self.mask)

        # obtain state idx embedding
        # 
        decoder_idx = self.decoder_idx_emb(CUDA(torch.arange(0, self.output_dim).long(), device=self.device))
        batch_decoder_idx = decoder_idx.repeat(inputs.shape[0], 1, 1).detach()
        masked_feature = torch.cat([masked_feature, batch_decoder_idx], dim=-1) # [B, C, C+E]

        # decoder: 
        # For 0-1 cost (sparse cases)
        if self.learning_mode == "cost":
            cost_predict = torch.sigmoid(self.decoder(masked_feature).squeeze(-1)) # self.decoder(masked_feature).squeeze(-1)
        else:
            cost_predict = torch.sigmoid(self.decoder(masked_feature).squeeze(-1))

        if sub_cost_flag:
            return cost_predict
        
        cost_predict = cost_predict.sum(dim=-1)
        if self.evaluation_mode:
            # cost_predict = cost_predict - min(cost_predict)
            mid_point = torch.ones(cost_predict.shape) * 0.15
            mid_point = torch.as_tensor(mid_point, device=self.device, dtype=torch.float32)
            # print(mid_point)
            cost_predict = cost_predict > mid_point
            cost_predict = torch.as_tensor(cost_predict, device=self.device, dtype=torch.float32)
            # print(cost_predict)

        
        # print(cost_predict)
        return cost_predict
    
    def evaluation(self, inputs, threshold=None):
        
        assert len(inputs.shape) == 2
        inputs = inputs.unsqueeze(-1) # [B, S, 1]

        # obtain state-action idx embedding
        # [B, S, 1] -> [B, S, E+1]
        encoder_idx = self.encoder_idx_emb(CUDA(torch.arange(0, self.input_dim).long(), device=self.device))
        batch_encoder_idx = encoder_idx.repeat(inputs.shape[0], 1, 1).detach()
        inputs_feature = torch.cat([inputs, batch_encoder_idx], dim=-1)

        # encoder: [B, S+A, E+1] -> [B, S+A, C]
        latent_feature = self.encoder(inputs_feature) # [B, S, C]

        # prepare mask
        if not self.use_full:
            _, self.mask = self.get_mask(threshold)

        # feature mask: 
        masked_feature = torch.einsum('bnc, ns -> bsc', latent_feature,  self.mask)

        # obtain state idx embedding
        # 
        decoder_idx = self.decoder_idx_emb(CUDA(torch.arange(0, self.output_dim).long(), device=self.device))
        batch_decoder_idx = decoder_idx.repeat(inputs.shape[0], 1, 1).detach()
        masked_feature = torch.cat([masked_feature, batch_decoder_idx], dim=-1) # [B, C, C+E]

        # decoder: 
        cost_predict = self.decoder(masked_feature).squeeze(-1)
        cost_predict_sum = cost_predict.sum(dim=-1)
        return cost_predict_sum, cost_predict

    def loss_function(self, cost, cost_predict):
        mse = F.mse_loss(cost, cost_predict) * self.mse_weight
        sparse = self.sparse_weight * torch.mean(torch.sigmoid(self.mask_prob)**self.sparse_norm)
        
        vector = torch.sigmoid(self.mask_prob)
        v1 = vector[:, 0]
        v2 = vector[:, 1]

        sim_loss =  (v1*v2).mean() # 2 dim case
        
        sim_loss = self.sim_weight * sim_loss

        norm_sim_loss = torch.abs(torch.linalg.vector_norm(v1)**3-torch.linalg.vector_norm(v2)**3)

        if vector.shape[-1] > 2:
            v3 = vector[:, 2]
            norm_sim_loss += torch.abs(torch.linalg.vector_norm(v1)**3-torch.linalg.vector_norm(v3)**3)
            norm_sim_loss += torch.abs(torch.linalg.vector_norm(v2)**3-torch.linalg.vector_norm(v3)**3)

        norm_sim_loss = self.norm_sim_weight * norm_sim_loss

        Matrix = torch.matmul(torch.transpose(vector, 0, 1), vector) 
        Matrix = F.normalize(Matrix)
        I_m = torch.eye(Matrix.shape[0])
        I_m = torch.as_tensor(I_m, device=self.device, dtype=torch.float32)
        diff = Matrix-I_m
        F_loss = torch.norm(diff) * self.F_weight
        # print(F_loss)
        
        info = {'train/causal_mse': mse.item(), 
                'train/causal_sparsity': sparse.item(), 
                'train/sim': sim_loss.item(),
                'train/sim_norm': norm_sim_loss.item(),
                'train/F_loss': F_loss.item()
                }
        
        loss = mse + sparse + sim_loss + norm_sim_loss + F_loss if self.sparse_flag else mse + sparse + F_loss
        # loss = mse + sparse
        
        return loss, info
    
    def subcost_loss_function(self, sub_cost_0, sub_cost_1, sub_cost_2, cost_predict):
        cost = torch.concatenate((sub_cost_0, sub_cost_1, sub_cost_2)).reshape(-1, 3)

        mse = F.mse_loss(cost, cost_predict) * self.mse_weight
        sparse = self.sparse_weight * torch.mean(torch.sigmoid(self.mask_prob)**self.sparse_norm)

        vector = torch.sigmoid(self.mask_prob)
        v1 = vector[:, 0]
        v2 = vector[:, 1]

        sim_loss =  (v1*v2).mean() # 2 dim case
        
        sim_loss = self.sim_weight * sim_loss

        norm_sim_loss = torch.abs(torch.linalg.vector_norm(v1)**3-torch.linalg.vector_norm(v2)**3)

        if vector.shape[-1] > 2:
            v3 = vector[:, 2]
            norm_sim_loss += torch.abs(torch.linalg.vector_norm(v1)**3-torch.linalg.vector_norm(v3)**3)
            norm_sim_loss += torch.abs(torch.linalg.vector_norm(v2)**3-torch.linalg.vector_norm(v3)**3)

        norm_sim_loss = self.norm_sim_weight * norm_sim_loss

        Matrix = torch.matmul(torch.transpose(vector, 0, 1), vector) 
        Matrix = F.normalize(Matrix)
        I_m = torch.eye(Matrix.shape[0])
        I_m = torch.as_tensor(I_m, device=self.device, dtype=torch.float32)
        diff = Matrix-I_m
        F_loss = torch.norm(diff) * self.F_weight
        # print(F_loss)
        
        info = {'train/causal_mse': mse.item(), 
                'train/causal_sparsity': sparse.item(), 
                'train/sim': sim_loss.item(),
                'train/sim_norm': norm_sim_loss.item(),
                'train/F_loss': F_loss.item()
                }
        
        loss = mse + sparse + sim_loss + norm_sim_loss + F_loss if self.sparse_flag else mse + sparse + F_loss
        # loss = mse + sparse
        
        return loss, info


    def get_mask(self, threshold=None, suffix = "", logdir = "log"):
        # prepare mask
        mask_prob = temp_sigmoid(self.mask_prob, temp=1.0) 

        if threshold is not None:
            # use threshold to sample a mask
            mask = (mask_prob > threshold).float()
        else:
            # use gumbel softmax to sample a mask
            # [S+A, S] -> [S+A, S, 2]
            # build a bernoulli distribution with p=1-mask_prob and q=mask_prob
            mask_bernoulli = torch.cat([(1-mask_prob).unsqueeze(-1), mask_prob.unsqueeze(-1)], dim=-1).log()

            # (S+A) x S x 2 -> (S+A) x S x 2 - (S+A) x S
            mask = F.gumbel_softmax(mask_bernoulli, tau=self.tau, hard=True, dim=-1) 
            mask = mask[:, :, 1] # just keep the second channel since the bernoulli distribution is [0, 1]
        
        return mask_prob, mask

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))


