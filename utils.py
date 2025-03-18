import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
# import matplotlib.pyplot as plt
import math


sd15_file_names = [
    'feature_extractor/preprocessor_config.json', 
    'scheduler/scheduler_config.json', 
    'text_encoder/config.json', 
    'text_encoder/model.safetensors',
    'tokenizer/merges.txt', 
    'tokenizer/special_tokens_map.json', 
    'tokenizer/tokenizer_config.json', 
    'tokenizer/vocab.json', 
    'unet/config.json', 
    'unet/diffusion_pytorch_model.safetensors', 
    'vae/config.json', 
    'vae/diffusion_pytorch_model.safetensors', 
    'model_index.json'
]

sdxl_file_names = [
    'model_index.json',
    'vae/config.json', 
    'vae/diffusion_pytorch_model.safetensors', 
    'unet/config.json', 
    'unet/diffusion_pytorch_model.safetensors', 
    'tokenizer/merges.txt', 
    'tokenizer/special_tokens_map.json', 
    'tokenizer/tokenizer_config.json', 
    'tokenizer/vocab.json', 
    'tokenizer_2/merges.txt', 
    'tokenizer_2/special_tokens_map.json', 
    'tokenizer_2/tokenizer_config.json', 
    'tokenizer_2/vocab.json', 
    'text_encoder/config.json', 
    'text_encoder/model.safetensors',
    'text_encoder_2/config.json', 
    'text_encoder_2/model.safetensors',
    'scheduler/scheduler_config.json', 
]

flux_file_names = [
    'model_index.json',
    'vae/config.json', 
    'vae/diffusion_pytorch_model.safetensors', 
    'transformer/config.json',
    'transformer/diffusion_pytorch_model-00001-of-00003.safetensors',
    'transformer/diffusion_pytorch_model-00002-of-00003.safetensors',
    'transformer/diffusion_pytorch_model-00003-of-00003.safetensors',
    'transformer/diffusion_pytorch_model.safetensors.index.json',
    'tokenizer/merges.txt', 
    'tokenizer/special_tokens_map.json', 
    'tokenizer/tokenizer_config.json', 
    'tokenizer/vocab.json', 
    'tokenizer_2/spiece.model', 
    'tokenizer_2/special_tokens_map.json', 
    'tokenizer_2/tokenizer_config.json', 
    'tokenizer_2/tokenizer.json', 
    'text_encoder/config.json', 
    'text_encoder/model.safetensors',
    'text_encoder_2/config.json',
    'text_encoder_2/model-00001-of-00002.safetensors',
    'text_encoder_2/model-00002-of-00002.safetensors',
    'text_encoder_2/model.safetensors.index.json',
    'scheduler/scheduler_config.json', 
]


def register_attn_control(unet, controller, cache=None):
    def attn_forward(self):
        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
        ):
            residual = hidden_states
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            q = self.to_q(hidden_states)
            is_self = encoder_hidden_states is None

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            inner_dim = k.shape[-1]
            head_dim = inner_dim // self.heads

            q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            if is_self and controller.cur_self_layer in controller.self_layers:
                cache.add(q, k, v, hidden_states)

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            )
            hidden_states = hidden_states.to(q.dtype)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            if is_self:
                controller.cur_self_layer += 1

            return hidden_states

        return forward

    def modify_forward(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":  # spatial Transformer layer
                net.forward = attn_forward(net)
                return count + 1
            elif hasattr(net, "children"):
                count = modify_forward(subnet, count)
        return count

    cross_att_count = 0
    for net_name, net in unet.named_children():
        cross_att_count += modify_forward(net, 0)
    controller.num_self_layers = cross_att_count // 2


def register_attn_control_flux(unet, controller, cache=None):
    def attn_forward(self):

        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
            *args,
            **kwargs,
        ):
            batch_size, _, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            # `sample` projections.
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
            if encoder_hidden_states is not None:
                # `context` projections.
                encoder_hidden_states_query_proj = self.add_q_proj(
                    encoder_hidden_states
                )
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(
                    encoder_hidden_states
                )

                encoder_hidden_states_query_proj = (
                    encoder_hidden_states_query_proj.view(
                        batch_size, -1, self.heads, head_dim
                    ).transpose(1, 2)
                )
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = (
                    encoder_hidden_states_value_proj.view(
                        batch_size, -1, self.heads, head_dim
                    ).transpose(1, 2)
                )

                if self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.norm_added_q(
                        encoder_hidden_states_query_proj
                    )
                if self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.norm_added_k(
                        encoder_hidden_states_key_proj
                    )

                # attention
                query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb

                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            if controller.cur_self_layer in controller.self_layers:
                # print("cache added")
                cache.add(query, key, value, hidden_states)
                # if encoder_hidden_states is None:
            controller.cur_self_layer += 1

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            )

            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1] :],
                )

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)
                encoder_hidden_states = self.to_add_out(encoder_hidden_states)

                return hidden_states, encoder_hidden_states
            else:
                return hidden_states

        return forward

    def modify_forward(net, count):
        # print(net.named_children())
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":  # spatial Transformer layer
                net.forward = attn_forward(net)
                return count + 1
            elif hasattr(net, "children"):
                count = modify_forward(subnet, count)
        return count

    cross_att_count = 0
    cross_att_count += modify_forward(unet, 0)
    controller.num_self_layers += cross_att_count


def load_image(image_path, size=None, mode="RGB"):
    img = Image.open(image_path).convert(mode)
    if size is None:
        width, height = img.size
        new_width = (width // 64) * 64
        new_height = (height // 64) * 64
        size = (new_width, new_height)
    img = img.resize(size, Image.BICUBIC)
    return ToTensor()(img).unsqueeze(0)


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


def adain_flux(source, target, eps=1e-6):
    source_mean, source_std = torch.mean(source, dim=1, keepdim=True), torch.std(
        source, dim=1, keepdim=True
    )
    target_mean, target_std = torch.mean(
        target, dim=(0, 1), keepdim=True
    ), torch.std(target, dim=(0, 1), keepdim=True)
    normalized_source = (source - source_mean) / (source_std + eps)
    transferred_source = normalized_source * target_std + target_mean

    return transferred_source


class Controller:
    def step(self):
        self.cur_self_layer = 0

    def __init__(self, self_layers=(0, 16)):
        self.num_self_layers = -1
        self.cur_self_layer = 0
        self.self_layers = list(range(*self_layers))


class DataCache:
    def __init__(self):
        self.q = []
        self.k = []
        self.v = []
        self.out = []

    def clear(self):
        self.q.clear()
        self.k.clear()
        self.v.clear()
        self.out.clear()

    def add(self, q, k, v, out):
        self.q.append(q)
        self.k.append(k)
        self.v.append(v)
        self.out.append(out)

    def get(self):
        return self.q.copy(), self.k.copy(), self.v.copy(), self.out.copy()
    


# def show_image(path, title, display_height=3, title_fontsize=12):
#     img = Image.open(path)
#     img_width, img_height = img.size

#     aspect_ratio = img_width / img_height
#     display_width = display_height * aspect_ratio

#     plt.figure(figsize=(display_width, display_height))
#     plt.imshow(img)
#     plt.title(title, 
#              fontsize=title_fontsize, 
#              fontweight='bold', 
#              pad=20) 
#     plt.axis('off')    
#     plt.tight_layout() 
#     plt.show()
