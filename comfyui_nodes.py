import os
import torch
from PIL import Image
from diffusers import DDIMScheduler

from comfy.comfy_types import IO
import comfy.model_management as mm
import node_helpers
import folder_paths
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from torchvision.transforms.functional import resize, to_tensor
from accelerate.utils import set_seed
from .pipeline_sd import ADPipeline
from .pipeline_sdxl import ADPipeline as ADXLPipeline
from .pipeline_flux import ADPipeline as ADFluxPipeline
from .utils import Controller
from .utils import sd15_file_names, sdxl_file_names, flux_file_names


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
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4
            if (image.shape[1] != 3 and image.shape[-1] == 3):
                image = image.permute(0, 3, 1, 2)
        image = resize(image, size=resolution)
        return (image,)


class LoadDistiller:
    RETURN_TYPES = ("DISTILLER",)
    RETURN_NAMES = ("distiller",)
    FUNCTION = "load_model"
    CATEGORY = "AttentionDistillationWrapper"

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                "model": (['stable-diffusion-v1-5', 'stable-diffusion-xl-base-1.0', 'FLUX.1-dev'], {"default": "stable-diffusion-v1-5"}),
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
        model_class = {
            "stable-diffusion-v1-5": ADPipeline,
            "stable-diffusion-xl-base-1.0": ADXLPipeline,
            "FLUX.1-dev": ADFluxPipeline,
        }[model]

        if not os.path.exists(model_name):
            print(f"Please download target model to : {model_name}")
        
        try:
            if model == "FLUX.1-dev":
                distiller = model_class.from_pretrained(
                    model_name, safety_checker=None, torch_dtype=weight_dtype
                ).to(device)
            else:
                scheduler = DDIMScheduler.from_pretrained(model_name, subfolder='scheduler')
                distiller = model_class.from_pretrained(
                    model_name, scheduler=scheduler, safety_checker=None, torch_dtype=weight_dtype
                ).to(device)
        except:
            print('Download models...')
                
            repo_name = {
                "stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
                "FLUX.1-dev": "black-forest-labs/FLUX.1-dev",
            }[model]

            file_names = {
                "stable-diffusion-v1-5": sd15_file_names,
                "stable-diffusion-xl-base-1.0": sdxl_file_names,
                "FLUX.1-dev": flux_file_names,
            }[model]

            pbar = tqdm(file_names)
            for file_name in pbar:
                pbar.set_description(f'Downloading {file_name}')
                if not os.path.exists(os.path.join(model_name, file_name)):
                    hf_hub_download(repo_id=repo_name, filename=file_name, local_dir=model_name)
                pbar.update()


            if model == "FLUX.1-dev":
                distiller = model_class.from_pretrained(
                    model_name, safety_checker=None, torch_dtype=weight_dtype
                ).to(device)
            else:
                scheduler = DDIMScheduler.from_pretrained(model_name, subfolder='scheduler')
                distiller = model_class.from_pretrained(
                    model_name, scheduler=scheduler, safety_checker=None, torch_dtype=weight_dtype
                ).to(device)

        if hasattr(distiller, 'unet'):
            distiller.classifier = distiller.unet
        elif hasattr(distiller, 'transformer'):
            distiller.classifier = distiller.transformer
        else:
            raise ValueError("Failed to initialize the classifier.")

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

        assert isinstance(attn_distiller, ADPipeline), "Only support SD1.5 for style transfer."
        assert isinstance(style, Image.Image) and isinstance(content, Image.Image), "Please use the image loader in `AttentionDistillationWrapper->Load PIL Image` for loading image."

        if isinstance(style, torch.Tensor) and style.ndim == 3:
            style = resize(style.unsqueeze(0), (512, 512))
        elif isinstance(style, Image.Image):
            style = to_tensor(resize(style, (512, 512))).unsqueeze(0)
                
        if isinstance(content, torch.Tensor) and content.ndim == 3:
            content = content.unsqueeze(0)
        elif isinstance(content, Image.Image):
            content = to_tensor(content).unsqueeze(0)
        
        assert isinstance(style, torch.Tensor) and style.ndim == 4
        assert isinstance(content, torch.Tensor) and content.ndim == 4

        if (style.shape[1] != 3 and style.shape[-1] == 3):
            style = style.permute(0, 3, 1, 2)
        if (content.shape[1] != 3 and content.shape[-1] == 3):
            content = content.permute(0, 3, 1, 2)

        print(content.shape)
        controller = Controller(self_layers=(10, 16))
        set_seed(seed)

        print('style', style.min(), style.max())
        print('content', content.min(), content.max())

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
    
    DEFAULT_CONFIGS = {
        ADPipeline: {'self_layers': (10, 16), 'resolution': (512, 512), 'enable_gradient_checkpoint': False},
        ADXLPipeline: {'self_layers': (64, 70), 'resolution': (1024, 1024), 'enable_gradient_checkpoint': True},
        ADFluxPipeline: {'self_layers': (50, 57), 'resolution': (512, 512), 'enable_gradient_checkpoint': True},
    }

    @torch.inference_mode(False)
    def process(self, distiller, style, positive, negative, steps, lr, iters, cfg, num_images_per_prompt, seed, height, width):
        precision = distiller['precision']
        attn_distiller = distiller['distiller']
        
        assert isinstance(style, Image.Image), "Please use the image loader in `AttentionDistillationWrapper->Load PIL Image` for loading image."

        default_config = self.DEFAULT_CONFIGS[type(attn_distiller)]
        print(default_config)

        controller = Controller(self_layers=default_config['self_layers'])
        
        if isinstance(style, torch.Tensor) and style.ndim == 3:
            style = resize(style.unsqueeze(0), default_config['resolution'])
        elif isinstance(style, Image.Image):
            style = to_tensor(resize(style, default_config['resolution'])).unsqueeze(0)

        assert isinstance(style, torch.Tensor) and style.ndim == 4

        if (style.shape[1] != 3 and style.shape[-1] == 3):
            style = style.permute(0, 3, 1, 2)

        print('style', style.min(), style.max(), style.mean())
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
            enable_gradient_checkpoint=default_config['enable_gradient_checkpoint']
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
