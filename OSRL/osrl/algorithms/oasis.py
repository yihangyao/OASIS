from typing import Optional, Tuple

import copy
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from torch.distributions.beta import Beta
from torch.nn import functional as F  # noqa
from tqdm.auto import trange  # noqa
from osrl.common.exp_util import visualization

from osrl.common.net import cosine_beta_schedule, extract, apply_conditioning, Losses, TemporalUnet, to_torch

class OASIS(nn.Module):
    """
    OASIS model.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        seq_len: int = 10,
        episode_len: int = 1000,
        n_timesteps: int = 1000,
        embedding_dim: int = 128,
        cost_transform: bool = False,
        add_cost_feat: bool = False,
        mul_cost_feat: bool = False,
        cat_cost_feat: bool = False,
        action_head_layers: int = 1,
        cost_prefix: bool = False,
        stochastic: bool = False,
        init_temperature=0.1,
        target_entropy=None,
        returns_condition=False, # Why no returns_condition?
        predict_epsilon=True, # Why True
        ar_inv=False,
        train_only_inv=False,
        clip_denoised=True,
        condition_guidance_w=0.1,
        test_ret = 0.9
    ):
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        self.predict_epsilon = predict_epsilon
        
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transition_dim = state_dim + action_dim
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.clip_denoised = clip_denoised
        
        self.episode_len = episode_len
        self.n_timesteps = n_timesteps
        self.max_action = max_action
        if cost_transform:
            self.cost_transform = lambda x: 50 - x
        else:
            self.cost_transform = None
        self.add_cost_feat = add_cost_feat
        self.mul_cost_feat = mul_cost_feat
        self.cat_cost_feat = cat_cost_feat
        self.stochastic = stochastic

        self.test_ret = test_ret

        super().__init__()
        self.inv_model = nn.Sequential(
                nn.Linear(2 * self.state_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, self.action_dim),
            )
        self.model = TemporalUnet(horizon=seq_len, 
                                  transition_dim=state_dim, 
                                  cond_dim=state_dim,
                                  dim_mults=(1, 2),
                                  returns_condition = self.returns_condition,
                                  seq_len=self.seq_len)

        betas = cosine_beta_schedule(n_timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        
        loss_weights = self.get_loss_weights(discount=1.0)
        self.loss_fn = Losses['state_l2'](loss_weights)
        
        self.apply(self._init_weights)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.state_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.seq_len, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None, cost_returns = None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False, cost_returns=cost_returns)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True, cost_returns=cost_returns)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-10., 10.) # .clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None, cost_returns = None, cost_model = None):
        # assert t <= self.n_timesteps, "t should <= self.n_timesteps"
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns,cost_returns = cost_returns)
        noise = 0.5*torch.randn_like(x) # why 0.5
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        grad = None
        # gradient calculation
        if cost_model is not None:
            with torch.enable_grad():
                x.requires_grad_() 
                grad = cost_model.c_gradient(s=x, need_denormalize = True)

        if grad is None:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise + \
            nonzero_mask * noise * grad
        # add a gradient * noise * nonzero_mask

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False, cost_returns=None,
                      cost_model = None):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device) # why 0.5?
        # ablation
        x = apply_conditioning(x, cond, 0)

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns, cost_returns)
            # ablation
            x = apply_conditioning(x, cond, 0)

            if return_diffusion: diffusion.append(x)
        
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, return_diffusion=False, cost_returns = None,
                           cost_model = None,
                            *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
            forward inference
            utilized for evaluation
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.seq_len # self.episode_len
        shape = (batch_size, horizon, self.state_dim)

        if returns is None:
            returns = torch.ones(batch_size, 1).to(device) * self.test_ret # [batch_size, 1]
        elif isinstance(returns, float):
            returns = torch.ones(batch_size, 1).to(device) * returns
        
        if cost_returns is None:
            cost_returns = torch.ones(batch_size, 1).to(device) * self.test_ret # [batch_size, 1]
        elif isinstance(cost_returns, float):
            cost_returns = torch.ones(batch_size, 1).to(device) * cost_returns

        return self.p_sample_loop(shape, cond, returns, return_diffusion=return_diffusion,
                                  cost_returns=cost_returns, cost_model = cost_model,
                                   *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None, cost_returns = None):
        noise = torch.randn_like(x_start) # x_start is [B, T, state_dim]

        # if self.predict_epsilon:
        #     # Cause we condition on obs at t=0
        #     noise[:, 0, self.action_dim:] = 0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) # forward sampling
        
        # ablation
        x_noisy = apply_conditioning(x_noisy, cond, 0)
        
        x_recon = self.model(x_noisy, cond, t, returns=returns, cost_returns=cost_returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape 
        # print(x_noisy.shape, x_recon.shape, noise.shape)
        if self.predict_epsilon: # True, 
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, state, action, cond, returns=None, cost_returns = None):
        if self.train_only_inv:
            # Calculating inv loss
            x_t = state[:, :-1]
            a_t = action[:, :-1]
            x_t_1 = state[:, 1:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.state_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss = self.inv_model.calc_loss(x_comb_t, a_t)
                info = {'a0_loss':loss.item()}
            else:
                pred_a_t = self.inv_model(x_comb_t)
                loss = F.mse_loss(pred_a_t, a_t)
                info = {'a0_loss': loss.item()}
        else:
            batch_size = len(state)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=state.device).long()
            # cond[0] shape: [2048, 8] 
            diffuse_loss, _ = self.p_losses(state, cond, t, returns, cost_returns = cost_returns)
            # diffuse_loss, info = 0, {}
            # Calculating inv loss
            x_t = state[:, :-1]
            a_t = action[:, :-1]
            x_t_1 = state[:, 1:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.state_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
            else:
                pred_a_t = self.inv_model(x_comb_t)
                inv_loss = F.mse_loss(pred_a_t, a_t)

            # inv_loss['a0_loss'] = inv_loss.item()
            
            # info["inv_loss"] = inv_loss.item()

            info = {
                "diffuse_loss": diffuse_loss.item(),
                "inv_loss": inv_loss.item()
            }

            loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

    def temperature(self):
        if self.stochastic:
            return self.log_temperature.exp()
        else:
            return None

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class OASISTrainer:
    """
    OASIS data generator Trainer
    """

    def __init__(
            self,
            model: OASIS,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            betas: Tuple[float, ...] = (0.9, 0.999),
            clip_grad: float = 0.25,
            lr_warmup_steps: int = 10000,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            loss_cost_weight: float = 0.0,
            loss_state_weight: float = 0.0,
            cost_reverse: bool = False,
            no_entropy: bool = False,
            update_ema_every=10,
            ema_decay=0.995,  
            step_start_ema=2000,        
            device="cpu",
            test_ret = 0.9, # reward return
            visualization_log = None
            ) -> None:
        self.model = model
        self.logger = logger
        self.env = env
        self.clip_grad = clip_grad
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.cost_weight = loss_cost_weight
        self.state_weight = loss_state_weight
        self.cost_reverse = cost_reverse
        self.no_entropy = no_entropy
        self.update_ema_every = update_ema_every
        self.ema = EMA(ema_decay)
        self.ema_model = deepcopy(self.model)
        self.step_start_ema = step_start_ema
        self.visualization_logdir = visualization_log
        
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lambda steps: min((steps + 1) / lr_warmup_steps, 1),
        )
        self.stochastic = self.model.stochastic
        self.max_action = self.model.max_action
        self.beta_dist = Beta(torch.tensor(2, dtype=torch.float, device=self.device),
                              torch.tensor(5, dtype=torch.float, device=self.device))
        self.step = 0
        self.test_ret = test_ret
        
    def train_one_step(self, states, actions, returns, costs_return, time_steps, mask,
                       episode_cost, costs, visualization_flag=False, step = 0):
        # True value indicates that the corresponding key value will be ignored
        # conditional_mask = ~mask.to(torch.bool)
        condition = {0: states[:, 0]}
        # print(states.shape)
        loss, info = self.model.loss(states, actions, condition, returns=returns, cost_returns=episode_cost)
        if visualization_flag:
            with torch.no_grad():
                visualization(obs_seq=states[0, :], 
                            path=self.visualization_logdir,
                            name="debug_2",
                            model=self.model,
                            returns = returns,
                            step = step)
        
        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()
        
        self.scheduler.step()
        self.logger.store(
            tab="train",
            all_loss=loss.item(),
            train_lr=self.scheduler.get_last_lr()[0],
        )


        self.logger.store(
            tab="train",
            **info
        )
        
        if self.step % self.update_ema_every == 0:

            self.step_ema()
        
        self.step += 1

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
        
    def evaluate(self, num_rollouts, test_condition = [0.4, 0.7]):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.ema_model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(num_rollouts, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(self.ema_model, self.env, test_condition)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.ema_model.train()
        print(np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale)
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens)

    @torch.no_grad()
    def rollout(
        self,
        model: OASIS,
        env: gym.Env,
        test_condition
    ) -> Tuple[float, float]:
        """
        Evaluates the performance of the model on a single episode.
        """
        
        states = torch.zeros(1,
                             model.episode_len + 1,
                             model.state_dim,
                             dtype=torch.float,
                             device=self.device)
        actions = torch.zeros(1,
                              model.episode_len,
                              model.action_dim,
                              dtype=torch.float,
                              device=self.device)

        time_steps = torch.arange(model.episode_len,
                                  dtype=torch.long,
                                  device=self.device)
        time_steps = time_steps.view(1, -1)

        obs, info = env.reset()
        states[:, 0] = torch.as_tensor(obs, device=self.device)

        # cannot step higher than model episode len, as timestep embeddings will crash
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for step in range(model.episode_len):
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important) # fix this noqa!!!
            obs = obs.reshape(1, -1)
            conditions = {0: to_torch(obs, device=self.device)}
            samples, diffusion = model.conditional_sample(conditions,
                                                          return_diffusion=True,
                                                          returns = test_condition[1], 
                                                          cost_returns = test_condition[0]) # Why return is None?
                                                           # Return can be skipped, 
                                                           # when return is None, set to default

            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*model.state_dim)
            acts = model.inv_model(obs_comb)
            
            acts = acts.clamp(-self.max_action, self.max_action)
            act = acts[0].cpu().numpy()
            # act = self.get_ensemble_action(1, model, s, a, r, c, t, epi_cost)

            obs_next, reward, terminated, truncated, info = env.step(act)
            if self.cost_reverse:
                cost = (1.0 - info["cost"]) * self.cost_scale
            else:
                cost = info["cost"] * self.cost_scale
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = torch.as_tensor(act)
            states[:, step + 1] = torch.as_tensor(obs_next)
            obs = obs_next

            episode_ret += reward
            episode_len += 1
            episode_cost += cost

            if terminated or truncated:
                break

        return episode_ret, episode_len, episode_cost

    def get_ensemble_action(self, size: int, model, s, a, r, c, t, epi_cost):
        # [size, seq_len, state_dim]
        s = torch.repeat_interleave(s, size, 0)
        # [size, seq_len, act_dim]
        a = torch.repeat_interleave(a, size, 0)
        # [size, seq_len]
        r = torch.repeat_interleave(r, size, 0)
        c = torch.repeat_interleave(c, size, 0)
        t = torch.repeat_interleave(t, size, 0)
        epi_cost = torch.repeat_interleave(epi_cost, size, 0)
        
        acts, _, _ = model(s, a, r, c, t, None, epi_cost)
        if self.stochastic:
            acts = acts.mean

        # [size, seq_len, act_dim]
        acts = torch.mean(acts, dim=0, keepdim=True)
        acts = acts.clamp(-self.max_action, self.max_action)
        act = acts[0, -1].cpu().numpy()
        return act

    def collect_random_rollouts(self, num_rollouts):
        episode_rets = []
        for _ in range(num_rollouts):
            obs, info = self.env.reset()
            episode_ret = 0.0
            for step in range(self.model.episode_len):
                act = self.env.action_space.sample()
                obs_next, reward, terminated, truncated, info = self.env.step(act)
                obs = obs_next
                episode_ret += reward
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
        return np.mean(episode_rets)
