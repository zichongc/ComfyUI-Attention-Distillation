import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm
from dataclasses import asdict
from typing import Union, Literal, List, Optional, Dict
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from .handlers import SimpleHandler, SimpleBuffer, sample_timesteps
from .utils import sd_self_attn_indexes, ADOptimizationOutput, ADSamplingOutput, LatentsOutput


def custom_self_attention(block_class, layer_index, buffer: SimpleBuffer, op: Literal['store', 'none']='none'):
    """
    customize for attention distillation
    """
    class Attention(block_class):
        _layer_index = layer_index
        
        def forward(
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            
            attn_output = hidden_states.clone()
                
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            # -------------------------------------- store features ----------------------------------------
            if op == 'store' and buffer is not None:
                buffer.add(query, key, value, attn_output, index=attn._layer_index)
            # ----------------------------------------------------------------------------------------------  

            return hidden_states

    return Attention


class AttentionDistillation(StableDiffusionPipeline):
    def register_denoising_network(self, content_feat_at_layers=[], style_feat_at_layers=[], buffer: SimpleBuffer = None, op: Literal['store']='store'):
        denoiser = self.unet
        indexes = sd_self_attn_indexes
        
        registered_layers = []
        for name, module in denoiser.named_modules():
            if name.endswith('attn1'):
                if indexes[name] in content_feat_at_layers or indexes[name] in style_feat_at_layers:
                    registered_layers.append(indexes[name])
                    module.__class__ = custom_self_attention(module.__class__, layer_index=name, buffer=buffer, op=op)

    def reset_denoising_network(self):
        denoiser = self.unet
        for name, module in denoiser.named_modules():
            if name.endswith('attn1'):
                module.__class__ = custom_self_attention(module.__class__, layer_index=name, buffer=None, op='none')

    def cast(self, device='cuda', weight_dtype=torch.float32):
        self.unet.to(device, weight_dtype)
        self.vae.to(device, weight_dtype)
        self.text_encoder.to(device, weight_dtype)
    
    def fix_weights(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
    @torch.no_grad()
    def image2latents(self, image: Union[Image.Image, List[Image.Image]]):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=.5, std=.5)
        ])
        
        if isinstance(image, Image.Image):
            image = [image]
        
        latents = []
        for img in image:
            img = transform(img).unsqueeze(0).to(device=self._execution_device, dtype=self.vae.dtype)
            latent = self.vae.encode(img)['latent_dist'].mean
            latent = latent * self.vae.config.scaling_factor
            latents.append(latent)
            
        latents = torch.cat(latents, dim=0)
        return latents

    @torch.no_grad()
    def latents2images(self, latents: torch.Tensor, return_type: Literal['pil', 'pt'] = 'pil'):
        dtype = self.vae.dtype
        device = self.vae.device
        
        images = []
        for latent in latents:
            latent = latent.unsqueeze(0).to(device=device, dtype=dtype)
            latent = latent / self.vae.config.scaling_factor
            image = self.vae.decode(latent)[0]
            image = (image * 0.5 + 0.5).clamp(0, 1)
            images.append(image)
        
        if return_type == 'pt':
            images = torch.cat(images)
        elif return_type == 'pil':
            images = [to_pil_image(img.squeeze(0)) for img in images]
        return images
    
    def extract_features(self, latents, t, prompt_embeds, handler: SimpleHandler):
        add_noise = handler.add_noise_to_reference
        handler.buffer.clear()
        if add_noise:
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)
        
        self.unet(latents, t, prompt_embeds)
        return handler.buffer.get()
    
    def optimize(
        self, 
        style_image: Image.Image, 
        contents: Union[Image.Image, List[Image.Image]], 
        handler: SimpleHandler, 
        image_return_type: Literal['pil', 'pt']='pil',
        seed: int = None,
    ) -> ADOptimizationOutput:
        self.fix_weights()
        accelerator = Accelerator(mixed_precision=handler.mixed_precision)
        
        ori_device, ori_dtype = self.vae.device, self.vae.dtype
        self.cast(device=accelerator.device, weight_dtype=handler.weight_type)
        self.reset_denoising_network()
        self.register_denoising_network(
            content_feat_at_layers=handler.content_loss_config.feat_at_layers, 
            style_feat_at_layers=handler.style_loss_config.feat_at_layers, 
            buffer=handler.buffer,
            op='store'
        )

        height, width = handler.height // self.vae_scale_factor, handler.width // self.vae_scale_factor
        
        style_latents = self.image2latents(style_image.resize((handler.height, handler.width)))
        content_latents = None
        if contents is not None:
            if isinstance(contents, Image.Image):
                contents = [contents]
            assert handler.batch_size == len(contents)
            content_latents = self.image2latents(contents)
        
        generator = torch.Generator(accelerator.device).manual_seed(seed) if seed is not None else None
        if handler.init_type == 'content' and content_latents is not None:
            latents = content_latents.clone().detach().float()
        else:
            latents = torch.randn([handler.batch_size, 4, height, width], generator=generator).to(accelerator.device)

        latents.requires_grad = True
        
        prompt_embeds = self.encode_prompt("", accelerator.device, 1, False)[0]
        style_prompt_embeds = prompt_embeds.repeat(style_latents.shape[0], 1, 1)
        target_prompt_embeds = prompt_embeds.repeat(latents.shape[0], 1, 1)      
        
        timesteps = sample_timesteps(device=accelerator.device, scheduler=self.scheduler, **asdict(handler.timestep_sampler))

        optimizer = torch.optim.Adam([latents], **handler.optimizer_config)
        self.unet, latents, optimizer = accelerator.prepare(self.unet, latents, optimizer)
        
        compute_content_loss = content_latents is not None and handler.content_loss_config.loss_weight > 0.
        pbar = tqdm(timesteps, desc='Optimize')
        for t in pbar:
            # style features
            style_querys, style_keys, style_values, style_attns = self.extract_features(
                style_latents, t, style_prompt_embeds, handler=handler)
            # content features
            if compute_content_loss:
                content_querys, content_keys, content_values, content_attns = self.extract_features(
                    content_latents, t, target_prompt_embeds, handler=handler)
            # target features
            querys, keys, values, attns = self.extract_features(
                latents, t, target_prompt_embeds, handler=handler)
            
            optimizer.zero_grad()

            style_ref_attns = {}
            target_attns = {}
            for key in style_keys.keys():
                if sd_self_attn_indexes[key] in handler.style_loss_config.feat_at_layers:
                    ref_query, style_key, style_value = querys[key], style_keys[key], style_values[key]
                    target_attns[key] = attns[key]

                    batch_size = ref_query.shape[0]
                    attn = F.scaled_dot_product_attention(
                        ref_query, style_key.repeat(batch_size, 1, 1, 1), style_value.repeat(batch_size, 1, 1, 1)
                    )
                    B, H, S, D = attn.shape
                    style_ref_attns[key] = attn.transpose(1, 2).reshape(B, S, H*D)
            
            style_loss = sum([
                handler.loss_fn(target_attns[key].float(), style_ref_attns[key].float().detach())
                for key in style_ref_attns.keys()
            ])
            
            # content loss
            content_loss = 0.
            if compute_content_loss:
                content_loss = sum([
                    handler.loss_fn(querys[key].float(), content_querys[key].float().detach())
                    for key in content_querys.keys() if sd_self_attn_indexes[key] in handler.style_loss_config.feat_at_layers
                ])

            loss = style_loss * handler.style_loss_config.loss_weight + content_loss * handler.content_loss_config.loss_weight
            pbar.set_postfix({'t': int(t.cpu()), 'style': float(style_loss), 'content': float(content_loss)})
            
            accelerator.backward(loss)
            optimizer.step()

        self.cast(device=ori_device, weight_dtype=ori_dtype)
        self.reset_denoising_network()
        handler.buffer.clear()

        images = self.latents2images(latents, return_type=image_return_type)
        return ADOptimizationOutput(latents=latents, images=images)
    
    @torch.no_grad()
    def ddim_inversion(self, images: Union[List[Image.Image], Image.Image], prompts: Union[List[str], str]="", num_inversion_steps=50) -> LatentsOutput:
        if isinstance(images, list):
            batch = len(images)
            if isinstance(prompts, str):
                prompts = [prompts]*batch
            assert len(prompts) == batch
        else:
            batch = 1
                  
        z0 = self.image2latents(images)
        prompt_embeds = self.encode_prompt(prompts, device=self._execution_device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]

        inv_latents = {}
        timesteps, num_inversion_steps = retrieve_timesteps(self.scheduler, num_inference_steps=num_inversion_steps, device=self._execution_device)
        timesteps = reversed(timesteps)

        latents = z0
        for i, t in enumerate(tqdm(timesteps, desc='DDIM Inversion')):
            model_input = latents
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i-1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )
            
            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1-alpha_prod_t) ** 0.5
            sigma_prev = (1-alpha_prod_t_prev) ** 0.5
            
            pred_noise = self.unet(model_input, t, encoder_hidden_states=prompt_embeds).sample
            pred_x0 = (latents - sigma_prev * pred_noise) / mu_prev
            latents = mu * pred_x0 + sigma * pred_noise
            
            inv_latents[int(t.cpu())] = latents.clone()
            
        return LatentsOutput(latents=inv_latents)
    
    def sample(
        self,
        prompt: str,
        style_image: Image.Image, 
        handler: SimpleHandler,
        num_inference_steps: int = 50,
        negative_prompt=None,
        latents: torch.Tensor = None,
        image_return_type: Literal['pil', 'pt']='pil',
        guidance_scale: float = 7.5,
        eta: float = 0.,
        cross_attention_kwargs=None,
        seed: int = None,
    ):
        self._guidance_scale = guidance_scale
        self._cross_attention_kwargs = cross_attention_kwargs
        dtype = self.vae.dtype

        self.fix_weights()
        self.reset_denoising_network()
        accelerator = Accelerator(mixed_precision=handler.mixed_precision)
        ori_device, ori_dtype = self.vae.device, self.vae.dtype
        self.cast(device=accelerator.device, weight_dtype=handler.weight_type)
        
        generator = torch.Generator(accelerator.device,).manual_seed(seed) if seed is not None else None

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, accelerator.device, None, None
        )
        
        style_latents = self.image2latents(style_image.resize((handler.height, handler.width)))
        if handler.invert_style:
            style_latents = self.ddim_inversion(style_image, "", num_inference_steps).latents

        height, width = handler.height // self.vae_scale_factor, handler.width // self.vae_scale_factor
        if latents is None:
            latents = torch.randn([handler.batch_size, 4, height, width], generator=generator, device=accelerator.device)
        
        latents = latents.to(dtype)
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        do_classifier_free_guidance = guidance_scale > 1.
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, accelerator.device, num_images_per_prompt=handler.batch_size,  negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance, lora_scale=lora_scale
        )
        
        distill_prompt_embeds = self.encode_prompt("", accelerator.device, 1, False)[0]
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if handler.iters_per_step > 0:
                    latents = self.distill(
                        input_latents=latents.clone(), t=t, style_latents=style_latents, 
                        prompt_embeds=distill_prompt_embeds, handler=handler, accelerator=accelerator
                    )
                progress_bar.update()
              
                latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
        self.cast(device=ori_device, weight_dtype=ori_dtype)  
        handler.buffer.clear()
        images = self.latents2images(latents, return_type=image_return_type)
        return ADSamplingOutput(latents=latents, images=images)

    def distill(
        self, 
        input_latents: torch.Tensor, 
        t: torch.Tensor, 
        style_latents: Union[torch.Tensor, Dict[int, torch.Tensor]], 
        prompt_embeds: torch.Tensor, 
        handler: SimpleHandler, 
        accelerator: Accelerator,
    ):
        dtype = input_latents.dtype
        self.reset_denoising_network()
        self.register_denoising_network(
            # handler.content_loss_config.feat_at_layers, 
            handler.style_loss_config.feat_at_layers, 
            buffer=handler.buffer
        )
        if isinstance(style_latents, torch.Tensor):
            noisy_style_latents = self.scheduler.add_noise(style_latents, torch.randn_like(style_latents), t)
        elif isinstance(style_latents, dict):
            noisy_style_latents = style_latents[int(t.cpu())]
        else:
            raise ValueError
        
        style_prompt_embeds = prompt_embeds.repeat(noisy_style_latents.shape[0], 1, 1)
        target_prompt_embeds = prompt_embeds.repeat(input_latents.shape[0], 1, 1)

        latents = input_latents.detach().float()
        latents.requires_grad = True

        optimizer = torch.optim.Adam([latents], **handler.optimizer_config)
        self.unet, latents, optimizer = accelerator.prepare(self.unet, latents, optimizer)
        
        # style features
        style_querys, style_keys, style_values, style_attns = self.extract_features(
            noisy_style_latents, t, style_prompt_embeds, handler=handler)
        content_querys = None

        iterations = handler.iters_per_step
        for _ in range(iterations):
            optimizer.zero_grad()

            querys, keys, values, attns = self.extract_features(
                latents, t, target_prompt_embeds, handler=handler)
            if content_querys is None:
                content_querys = querys.copy()

            style_loss = 0.
            if handler.style_loss_config.loss_weight > 0.:
                style_ref_attns = {}
                target_attns = {}
                for key in style_keys.keys():
                    if sd_self_attn_indexes[key] not in handler.style_loss_config.feat_at_layers:
                        continue
                    ref_query, style_key, style_value = querys[key].clone(), style_keys[key], style_values[key]
                    target_attns[key] = attns[key]
                    
                    batch_size = ref_query.shape[0]
                    attn = F.scaled_dot_product_attention(
                        ref_query, style_key.repeat(batch_size, 1, 1, 1), style_value.repeat(batch_size, 1, 1, 1),
                    )
                    B, H, S, D = attn.shape
                    style_ref_attns[key] = attn.transpose(1, 2).reshape(B, S, H*D)
                
                style_loss = sum([
                    handler.loss_fn(target_attns[key].float(), style_ref_attns[key].float().detach())
                    for key in style_ref_attns.keys()
                ])

            content_loss = 0.
            compute_content_loss = handler.content_loss_config.loss_weight > 0.
            if compute_content_loss:
                content_loss = sum([
                    handler.loss_fn(querys[key].float(), content_querys[key].float().detach())
                    for key in content_querys.keys() if sd_self_attn_indexes[key] in handler.style_loss_config.feat_at_layers
                ])
                   
            loss = style_loss * handler.style_loss_config.loss_weight + content_loss * handler.content_loss_config.loss_weight
            accelerator.backward(loss)
            optimizer.step()

        self.reset_denoising_network()
        return latents.detach().to(dtype)
