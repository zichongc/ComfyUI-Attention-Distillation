import os
import torch
from PIL import Image
from diffusers import DDIMScheduler

from comfy.comfy_types import IO
import comfy.model_management as mm
import node_helpers
import folder_paths

from .attention_distillation import AttentionDistillation
from .handlers import TimestepSamplerConfig, SimpleHandler


class PureText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "get_prompt"
    CATEGORY = "AttentionDistillationWrapper"

    def get_prompt(self, text):
        return (text,)


class LoadPILImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "AttentionDistillationWrapper"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path).convert('RGB')
        return (img,)


class LoadDistiller:
    RETURN_TYPES = ("DISTILLER",)
    RETURN_NAMES = ("distiller",)
    FUNCTION = "load_model"
    CATEGORY = "AttentionDistillationWrapper"

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                "model": (['stable-diffusion-v1-5', 'stable-diffusion-2-1', ''], {"default": "stable-diffusion-v1-5"}),
                "precision": (['fp16', 'fp32'], {"default": 'fp16'}),
            },  
        }

    @torch.inference_mode(False)
    def load_model(self, model, precision):
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        if precision == 'fp32':
            precision = 'no'
        device = mm.get_torch_device()
        
        model_name = os.path.join(folder_paths.models_dir, 'diffusers', model)
        if not os.path.exists(model_name):
            print(f"Please download target model to : {model_name}")
        
        scheduler = DDIMScheduler.from_pretrained(model_name, subfolder='scheduler')
        distiller = AttentionDistillation.from_pretrained(
            model_name, scheduler=scheduler, safety_checker=None, torch_dtype=weight_dtype
        ).to(device)

        return ({"distiller": distiller, "precision": precision, 'weight_dtype': weight_dtype},)


class ADHandler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_weight": ("FLOAT", {"default": 1., "min": 0., "max": 10., "step": 0.001}),
                "content_weight": ("FLOAT", {"default": 0.2, "min": 0., "max": 10., "step": 0.001}),
                "init_type": (["content", "random"], {"default": "content"}),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 8}),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 8}),
                "lr": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5, "step": 0.001}),
            }
        }
    RETURN_TYPES = ("HANDLER",)
    RETURN_NAMES = ("handler",)
    FUNCTION = "load_handler"
    CATEGORY = "AttentionDistillationWrapper"

    def load_handler(self, style_weight, content_weight, init_type, height, width, lr):
        handler = SimpleHandler(
            optimizer_config={'lr': lr},
            content_feat_at_layers=(10, 11, 12, 13, 14, 15),
            style_feat_at_layers=(10, 11, 12, 13, 14, 15),
            style_loss_weight=style_weight,
            content_loss_weight=content_weight,
            init_type=init_type,
            height=height,
            width=width
        )
        return (handler,)
        

class ADOptimizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "distiller": ("DISTILLER",),
                "handler": ("HANDLER",),
                "content": ("IMAGE",),
                "style": ("IMAGE",),
                "steps": ("INT", {"default": 200, "min": 1, "max": 500, "step": 1}),
                "seed": ("INT", {"default": 2025, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "AttentionDistillationWrapper"

    @torch.inference_mode(False)
    def process(self, distiller, handler, content, style, steps, seed):
        precision = distiller['precision']
        attn_distiller = distiller['distiller']
        handler.set_mixed_precision(precision)
        handler.set_timestep_sampler(TimestepSamplerConfig(num_timesteps=steps, mode='auto'))

        images = attn_distiller.optimize(
            style_image=style, contents=content, handler=handler, image_return_type='pt', seed=seed
        ).images
        images = images.permute(0, 2, 3, 1)
        return (images,)
    

class ADSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "distiller": ("DISTILLER",),
                "handler": ("HANDLER",),
                "style": ("IMAGE",),
                "positive": (IO.CONDITIONING,),
                "negative": (IO.CONDITIONING,),
                "seed": ("INT", {"default": 2025, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "iters": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "cfg": ("FLOAT", {"default": 7., "min": 1., "max": 20., "step": 0.01}),
                "eta": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": 0.01}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "AttentionDistillationWrapper"
    
    @torch.inference_mode(False)
    def process(self, distiller, handler, style, positive, negative, seed, steps, iters, cfg, eta):
        precision = distiller['precision']
        attn_distiller = distiller['distiller']
        handler.set_mixed_precision(precision)
        handler.iters_per_step = iters
        
        images = attn_distiller.sample(
            prompt=positive, style_image=style, handler=handler, num_inference_steps=steps, negative_prompt=negative,
            guidance_scale=cfg, seed=seed, eta=eta, image_return_type='pt',
        ).images
        images = images.permute(0, 2, 3, 1)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "LoadDistiller": LoadDistiller,
    "ADHandler": ADHandler,
    "ADOptimizer": ADOptimizer,
    "ADSampler": ADSampler,
    "LoadPILImage": LoadPILImage,
    "PureText": PureText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDistiller": "Load Distiller",
    "ADHandler": "Handler for Attention Distillation",
    "ADOptimizer": "Optimization-Based Style Transfer",
    "ADSampler": "Sampler for Style-Specific Text-to-Image",
    "LoadPILImage": "Load PIL Image",
    "PureText": "Text Prompt",
}
