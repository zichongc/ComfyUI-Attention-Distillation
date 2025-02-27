from typing import Literal, Union
from dataclasses import dataclass
import torch
import numpy as np
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


class SimpleBuffer:
    def __init__(self):
        self.Q = {}
        self.K = {}
        self.V = {}
        self.Attn = {}
        
    def clear(self):
        self.Q = {}
        self.K = {}
        self.V = {}
        self.Attn = {}

    def add(self, Q, K, V, Attn, index):
        self.Q[index] = Q
        self.K[index] = K
        self.V[index] = V
        self.Attn[index] = Attn

    def get(self):
        return self.Q.copy(), self.K.copy(), self.V.copy(), self.Attn.copy()


@dataclass
class StyleLossConfig:
    loss_weight: float
    feat_at_layers: Union[list, tuple]

@dataclass
class ContentLossConfig:
    loss_weight: float
    feat_at_layers: Union[list, tuple]


@dataclass
class TimestepSamplerConfig:
    num_timesteps: int = 100
    mode: Literal['linear', 'exp', 'fix', 'auto']='linear'
    start_t: int = 999
    end_t: int = 1
    scale: float = 2
        

class SimpleHandler:
    def __init__(self, 
        optimizer_config: dict = None,
        content_feat_at_layers=(10, 11, 12, 13, 14, 15),
        style_feat_at_layers=(10, 11, 12, 13, 14, 15),
        content_loss_weight=0.,
        style_loss_weight=0.,
        batch_size=1,
        iters_per_step=3,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'fp16',
        init_type: Literal['content', 'random'] = 'content',
        height=512, 
        width=512,
        loss_fn=torch.nn.functional.l1_loss,
        add_noise_to_reference: bool = False,
        timestep_sampler: TimestepSamplerConfig = None,       
        invert_style: bool = False 
    ):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.init_type = init_type
        self.weight_type = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'no': torch.float32}[mixed_precision]
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config if optimizer_config is not None else {'lr': 0.05}
        self.add_noise_to_reference = add_noise_to_reference
        self.timestep_sampler = timestep_sampler
        self.iters_per_step = iters_per_step
        self.invert_style = invert_style
        
        self.style_loss_config = StyleLossConfig(
            loss_weight=style_loss_weight,
            feat_at_layers=style_feat_at_layers,
        )
        self.content_loss_config = ContentLossConfig(
            loss_weight=content_loss_weight,
            feat_at_layers=content_feat_at_layers,
        )
        self.buffer = SimpleBuffer()
    
    def set_mixed_precision(self, precision):
        self.mixed_precision = precision
        self.weight_type = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'no': torch.float32}[precision]

    def set_timestep_sampler(self, timestep_sampler):
        self.timestep_sampler = timestep_sampler
    
    def set_style_loss_config(self, config):
        self.style_loss_config = config

    def set_content_loss_config(self, config):
        self.content_loss_config = config


def sample_timesteps(num_timesteps=100, mode: Literal['linear', 'exp', 'fix', 'auto']='linear', start_t=999, end_t=1, scale=2, device='cpu', scheduler=None):
    assert 0 <= start_t <= 999, start_t
    assert 0 <= end_t <= 999, end_t
    assert start_t >= end_t
    if mode == 'linear':
        timesteps = torch.from_numpy(
            np.linspace(start_t, end_t, num_timesteps, endpoint=True).round().astype(np.int64)
        ).to(device)
    elif mode == 'fix':
        # use `start_t` as the fixed t
        timesteps = torch.tensor([start_t]*num_timesteps).long().to(device)
    elif mode == 'exp':
        linear = torch.from_numpy(
            (np.linspace(start_t, end_t, num_timesteps, endpoint=True)).round().astype(np.int64)
        ).to(device)
        exp = torch.exp(linear/linear.max()*scale)
        timesteps = (end_t+(exp-exp.min())/(exp.max()-exp.min())*(start_t-end_t)).round().long()
    elif mode == 'auto':
        timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps=num_timesteps, device=device)
    else:
        raise ValueError(f"{mode} is not supported. Please choose one of the `linear`, `exp`, and `fix`.")

    return timesteps
        