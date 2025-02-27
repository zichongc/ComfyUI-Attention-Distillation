import torch
from PIL import Image
from typing import Union, List
from dataclasses import dataclass


sd_self_attn_indexes = {
    'down_blocks.0.attentions.0.transformer_blocks.0.attn1': 0,
    'down_blocks.0.attentions.1.transformer_blocks.0.attn1': 1,
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1': 2,
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1': 3,
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1': 4,
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1': 5,
    'mid_block.attentions.0.transformer_blocks.0.attn1': 6,
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1': 7 ,
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1': 8,
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1': 9,
    'up_blocks.2.attentions.0.transformer_blocks.0.attn1': 10,
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1': 11,
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1': 12,
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1': 13,
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1': 14,
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1': 15,
}

sdxl_self_attn_indexes = {
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1': 0,
    'down_blocks.1.attentions.0.transformer_blocks.1.attn1': 1,
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1': 2,
    'down_blocks.1.attentions.1.transformer_blocks.1.attn1': 3,
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1': 4,
    'down_blocks.2.attentions.0.transformer_blocks.1.attn1': 5,
    'down_blocks.2.attentions.0.transformer_blocks.2.attn1': 6,
    'down_blocks.2.attentions.0.transformer_blocks.3.attn1': 7,
    'down_blocks.2.attentions.0.transformer_blocks.4.attn1': 8,
    'down_blocks.2.attentions.0.transformer_blocks.5.attn1': 9,
    'down_blocks.2.attentions.0.transformer_blocks.6.attn1': 10,
    'down_blocks.2.attentions.0.transformer_blocks.7.attn1': 11,
    'down_blocks.2.attentions.0.transformer_blocks.8.attn1': 12,
    'down_blocks.2.attentions.0.transformer_blocks.9.attn1': 13,
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1': 14,
    'down_blocks.2.attentions.1.transformer_blocks.1.attn1': 15,
    'down_blocks.2.attentions.1.transformer_blocks.2.attn1': 16,
    'down_blocks.2.attentions.1.transformer_blocks.3.attn1': 17,
    'down_blocks.2.attentions.1.transformer_blocks.4.attn1': 18,
    'down_blocks.2.attentions.1.transformer_blocks.5.attn1': 19,
    'down_blocks.2.attentions.1.transformer_blocks.6.attn1': 20,
    'down_blocks.2.attentions.1.transformer_blocks.7.attn1': 21,
    'down_blocks.2.attentions.1.transformer_blocks.8.attn1': 22,
    'down_blocks.2.attentions.1.transformer_blocks.9.attn1': 23,
    'mid_block.attentions.0.transformer_blocks.0.attn1': 24,
    'mid_block.attentions.0.transformer_blocks.1.attn1': 25,
    'mid_block.attentions.0.transformer_blocks.2.attn1': 26,
    'mid_block.attentions.0.transformer_blocks.3.attn1': 27,
    'mid_block.attentions.0.transformer_blocks.4.attn1': 28,
    'mid_block.attentions.0.transformer_blocks.5.attn1': 29,
    'mid_block.attentions.0.transformer_blocks.6.attn1': 30,
    'mid_block.attentions.0.transformer_blocks.7.attn1': 31,
    'mid_block.attentions.0.transformer_blocks.8.attn1': 32,
    'mid_block.attentions.0.transformer_blocks.9.attn1': 33,
    'up_blocks.0.attentions.0.transformer_blocks.0.attn1': 34,
    'up_blocks.0.attentions.0.transformer_blocks.1.attn1': 35,
    'up_blocks.0.attentions.0.transformer_blocks.2.attn1': 36,
    'up_blocks.0.attentions.0.transformer_blocks.3.attn1': 37,
    'up_blocks.0.attentions.0.transformer_blocks.4.attn1': 38,
    'up_blocks.0.attentions.0.transformer_blocks.5.attn1': 39,
    'up_blocks.0.attentions.0.transformer_blocks.6.attn1': 40,
    'up_blocks.0.attentions.0.transformer_blocks.7.attn1': 41,
    'up_blocks.0.attentions.0.transformer_blocks.8.attn1': 42,
    'up_blocks.0.attentions.0.transformer_blocks.9.attn1': 43,
    'up_blocks.0.attentions.1.transformer_blocks.0.attn1': 44,
    'up_blocks.0.attentions.1.transformer_blocks.1.attn1': 45,
    'up_blocks.0.attentions.1.transformer_blocks.2.attn1': 46,
    'up_blocks.0.attentions.1.transformer_blocks.3.attn1': 47,
    'up_blocks.0.attentions.1.transformer_blocks.4.attn1': 48,
    'up_blocks.0.attentions.1.transformer_blocks.5.attn1': 49,
    'up_blocks.0.attentions.1.transformer_blocks.6.attn1': 50,
    'up_blocks.0.attentions.1.transformer_blocks.7.attn1': 51,
    'up_blocks.0.attentions.1.transformer_blocks.8.attn1': 52,
    'up_blocks.0.attentions.1.transformer_blocks.9.attn1': 53,
    'up_blocks.0.attentions.2.transformer_blocks.0.attn1': 54,
    'up_blocks.0.attentions.2.transformer_blocks.1.attn1': 55,
    'up_blocks.0.attentions.2.transformer_blocks.2.attn1': 56,
    'up_blocks.0.attentions.2.transformer_blocks.3.attn1': 57,
    'up_blocks.0.attentions.2.transformer_blocks.4.attn1': 58,
    'up_blocks.0.attentions.2.transformer_blocks.5.attn1': 59,
    'up_blocks.0.attentions.2.transformer_blocks.6.attn1': 60,
    'up_blocks.0.attentions.2.transformer_blocks.7.attn1': 61,
    'up_blocks.0.attentions.2.transformer_blocks.8.attn1': 62,
    'up_blocks.0.attentions.2.transformer_blocks.9.attn1': 63,
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1': 64,
    'up_blocks.1.attentions.0.transformer_blocks.1.attn1': 65,
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1': 66,
    'up_blocks.1.attentions.1.transformer_blocks.1.attn1': 67,
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1': 68,
    'up_blocks.1.attentions.2.transformer_blocks.1.attn1': 69,
}


def export_results(latents, vae, output_dir, name='demo.png'):
    latents = 1 / vae.config.scaling_factor * latents.detach()
    image = vae.decode(latents.to(vae.dtype))['sample']
    image = (image / 2 + .5).clamp(0, 1)
    from torchvision.utils import save_image
    import os
    save_image(image, os.path.join(output_dir, name))


def adain(source, target, eps=1e-6):
    source_mean, source_std = torch.mean(source, dim=(2, 3), keepdim=True), torch.std(
        source, dim=(2, 3), keepdim=True
    )
    target_mean, target_std = torch.mean(
        target, dim=(0, 2, 3), keepdim=True
    ), torch.std(target, dim=(0, 2, 3), keepdim=True)
    normalized_source = (source - source_mean) / (source_std + eps)
    transferred_source = normalized_source * target_std + target_mean

    return transferred_source


@dataclass
class LatentsOutput:
    latents: dict
    
    
@dataclass
class ADOptimizationOutput:
    latents: torch.Tensor
    images: Union[torch.Tensor, List[Image.Image]]


@dataclass
class ADSamplingOutput:
    latents: torch.Tensor
    images: Union[torch.Tensor, List[Image.Image]]
