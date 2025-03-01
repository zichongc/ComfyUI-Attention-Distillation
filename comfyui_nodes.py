import os
import torch
from PIL import Image
from diffusers import DDIMScheduler

from comfy.comfy_types import IO
import comfy.model_management as mm
import node_helpers
import folder_paths

from torchvision.transforms.functional import resize, to_tensor
from accelerate.utils import set_seed
from .pipeline_sd import ADPipeline
from .pipeline_sdxl import ADPipeline as ADXLPipeline
from .utils import Controller


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
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path).convert('RGB')
        return (img,)


class ResizeImage:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_image"

    CATEGORY = "AttentionDistillationWrapper"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 8}),
            },
        }

    def resize_image(self, image, resolution):
        image = resize(image, size=resolution)
        return (image,
        )


class LoadDistiller:
    RETURN_TYPES = ("DISTILLER",)
    RETURN_NAMES = ("distiller",)
    FUNCTION = "load_model"
    CATEGORY = "AttentionDistillationWrapper"

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                "model": (['stable-diffusion-v1-5'], {"default": "stable-diffusion-v1-5"}),
                "precision": (['bf16', 'fp32'], {"default": 'bf16'}),
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
        distiller = ADPipeline.from_pretrained(
            model_name, scheduler=scheduler, safety_checker=None, torch_dtype=weight_dtype
        ).to(device)
        distiller.classifier = distiller.unet

        return ({"distiller": distiller, "precision": precision, 'weight_dtype': weight_dtype},)


class ADOptimizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "distiller": ("DISTILLER",),
                "content": ("IMAGE",),
                "style": ("IMAGE",),
                "steps": ("INT", {"default": 200, "min": 1, "max": 500, "step": 1}),
                "content_weight": ("FLOAT", {"default": 0.25, "min": 0., "max": 10., "step": 0.001}),
                "lr": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5, "step": 0.001}),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 8}),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 2025, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "AttentionDistillationWrapper"

    @torch.inference_mode(False)
    def process(self, distiller, content, style, steps, content_weight, lr, height, width, seed):
        precision = distiller['precision']
        attn_distiller = distiller['distiller']

        style = to_tensor(resize(style, (512, 512))).unsqueeze(0)
        content = to_tensor(content).unsqueeze(0)

        controller = Controller(self_layers=(10, 16))
        set_seed(seed)

        images = attn_distiller.optimize(
            lr=lr,
            batch_size=1,
            iters=1,
            width=width,
            height=height,
            weight=content_weight,
            controller=controller,
            style_image=style,
            content_image=content,
            mixed_precision=precision,
            num_inference_steps=steps,
            enable_gradient_checkpoint=False,
        )
        images = images.permute(0, 2, 3, 1).float()
        return (images,)
    

class ADSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "distiller": ("DISTILLER",),
                "style": ("IMAGE",),
                "positive": (IO.CONDITIONING,),
                "negative": (IO.CONDITIONING,),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "lr": ("FLOAT", {"default": 0.015, "min": 0.001, "max": 1., "step": 0.001}),
                "iters": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 1., "max": 20., "step": 0.01}),
                "eta": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": 0.01}),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "seed": ("INT", {"default": 2025, "min": 0, "max": 0xffffffffffffffff}),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 8}),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 8}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "AttentionDistillationWrapper"
    
    @torch.inference_mode(False)
    def process(self, distiller, style, positive, negative, steps, lr, iters, cfg, eta, num_images_per_prompt, seed, height, width):
        precision = distiller['precision']
        attn_distiller = distiller['distiller']
        controller = Controller(self_layers=(10, 16))
        
        style = to_tensor(resize(style, (512, 512))).unsqueeze(0)

        set_seed(seed)
        images = attn_distiller.sample(
            controller=controller,
            iters=iters,
            lr=lr,
            adain=True,
            height=height,
            width=width,
            mixed_precision=precision,
            style_image=style,
            prompt=positive,
            negative_prompt=negative,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=num_images_per_prompt,
            enable_gradient_checkpoint=False
        )
        images = images.permute(0, 2, 3, 1).float()
        return (images,)


NODE_CLASS_MAPPINGS = {
    "LoadDistiller": LoadDistiller,
    "ADOptimizer": ADOptimizer,
    "ADSampler": ADSampler,
    "LoadPILImage": LoadPILImage,
    "PureText": PureText,
    "ResizeImage": ResizeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDistiller": "Load Distiller",
    "ADHandler": "Handler for Attention Distillation",
    "ADOptimizer": "Optimization-Based Style Transfer",
    "ADSampler": "Sampler for Style-Specific Text-to-Image",
    "LoadPILImage": "Load PIL Image",
    "PureText": "Text Prompt",
    "ResizeImage": "Resize Image",
}
