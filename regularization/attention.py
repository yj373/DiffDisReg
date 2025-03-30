from einops import rearrange
from typing import Optional, Dict, Tuple, Any
from torch.nn import BCELoss

from diffusers.models.attention import _chunked_feed_forward
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import (
    deprecate,
)
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
)
from diffusers.training_utils import (
    compute_snr
)

import torch
import math
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


def attn_call (
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
):
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim
    # print(hidden_states.shape)
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    ##
    if hasattr(self, "store_attn_map"):
        height = int(math.sqrt(attention_probs.shape[-2]))
        self.attn_map = rearrange(attention_probs, 'b (h w) d -> b d h w', h=height) # (10,9216,77) -> (10,77,96,96)
    ##
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def attn_call2_0(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
):
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim
    print(hidden_states.shape)
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
    ##
    if hasattr(self, "store_attn_map"):
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        self.attn_map = rearrange(attention_probs, 'b (h w) d -> b d h w', h=height) # (10,9216,77) -> (10,77,96,96)
        self.timestep = int(timestep.item())
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    ##
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def lora_attn_call(self, attn: Attention, hidden_states, height, width, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor()
    ##
    attn.processor.__call__ = attn_call.__get__(attn.processor, AttnProcessor)
    ##

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, height, width, *args, **kwargs)


def lora_attn_call2_0(self, attn: Attention, hidden_states, height, width, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor2_0()
    ##
    attn.processor.__call__ = attn_call.__get__(attn.processor, AttnProcessor2_0)
    ##

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, height, width, *args, **kwargs)


def cross_attn_init():
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call


def store_attn_maps(unet, self_attn=False):
    target_prefix = 'attn2' if not self_attn else 'attn'
    for name, module in unet.named_modules():
        if not name.split('.')[-1].startswith(target_prefix):
            continue
        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True

    return unet


def aggregate_attention(unet, self_attn=False, res=8, bsz=1, with_prior_preservation=False):
    attn_target = []
    attn_prior = []
    target = 'attn1' if self_attn else 'attn2'
    for name, module in unet.named_modules():
        if not name.split('.')[-1].startswith(target):
            continue
        if hasattr(module.processor, "attn_map") and module.processor.attn_map.shape[-2] == res and module.processor.attn_map.shape[-1] == res:
            # item = module.processor.attn_map if not with_prior_preservation else torch.chunk(module.processor.attn_map, 2)[0]
            # item_prior = None if not with_prior_preservation else torch.chunk(module.processor.attn_map, 2)[0]
            print(">>> module.processor.attn_map.shape", module.processor.attn_map.shape) # (8 * num_images, 77, res, res)
            if not with_prior_preservation:
                item, item_prior = module.processor.attn_map, None
            else:
                item, item_prior = torch.split(module.processor.attn_map, [bsz * 8, len(module.processor.attn_map) - bsz * 8])
                # print("item.shape", item.shape)
                # print("item_prior.shape", item_prior.shape)
            if bsz == 1:
                item = torch.mean(item, axis=0, keepdim=True)
                if item_prior is not None:
                    item_prior = torch.mean(item_prior, axis=0, keepdim=True)
            else:
                # print("item.shape", item.shape)
                item_ = torch.chunk(item, int(len(item) / 8))
                item_ = [torch.mean(c, axis=0) for c in item_]
                item = torch.stack(item_, dim=0)
                # print("post item.shape", item.shape)
                if item_prior is not None:
                    item_prior_ = torch.chunk(item_prior, int(len(item_prior) / 8))
                    item_prior_ = [torch.mean(c, axis=0) for c in item_prior_]
                    item_prior = torch.stack(item_prior_, dim=0)

            attn_target.append(item)
            print("res", res, item.shape)
            if item_prior is not None:
                attn_prior.append(item_prior)
                print("res", res, item_prior.shape)
    attn_target_ = torch.mean(torch.stack(attn_target), dim=0)
    if len(attn_prior) > 0:
        attn_prior_ = torch.mean(torch.stack(attn_target), dim=0)
    else:
        attn_prior_ = None
    return [attn_target_, attn_prior_]


def generate_score_map(
    select,
    prior_select,
    attn_8,
    attn_16,
    attn_32,
    attn_64,
    attn_64_self=None,
    weight=[0.3, 0.5, 0.1, 0.1],
    height=780, 
    width=780, 
    bsz=1,
    alpha=8, 
    beta=0.4,
    with_prior_preservation=False, 
):
    images = []
    if not with_prior_preservation:
        imgs_ = []
        for idx, att in enumerate([attn_8, attn_16, attn_32, attn_64]):
            if (len(att) == 1):
                att_ = att[0].unsqueeze(0)
            else:
                att_ = torch.stack(att, dim=0) # [n, bsz, 77, 64, 64]

            att_ = torch.mean(att_, dim=0, keepdim=True) # [1, bsz, 77, 64, 64]
            att_ = att_.squeeze(0) # [bsz, 77, 64, 64]

            if idx < 3:
                att_ = F.interpolate(
                    att_,
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=False
                )

            imgs_.append(att_ * weight[idx])
        images.append(imgs_)
    else:
        imgs_target = []
        imgs_prior = []
        for idx, att in enumerate([attn_8, attn_16, attn_32, attn_64]):
            # if (len(att) == 1):
            #     att_ = att[0].unsqueeze(0)
            # else:
            #     att_ = torch.stack(att, dim=0) # [n, bsz, 77, 64, 64]
            # print(len(att))
            print(idx, len(att))
            att_target, att_prior = att
            # att_target, att_prior = torch.chunk(att_, 2)
            # att_target = torch.mean(att_target, dim=0, keepdim=True) # [1, bsz, 77, 64, 64]
            # att_target = att_target.squeeze(0) # [bsz, 77, 64, 64]
            # print("att_target.shape", att_target.shape)
            # att_prior = torch.mean(att_prior, dim=0, keepdim=True) # [1, bsz, 77, 64, 64]
            # att_prior = att_prior.squeeze(0) # [bsz, 77, 64, 64]
            # print("att_prior.shape", att_prior.shape)

            if idx < 3:
                att_target = F.interpolate(
                    att_target,
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=False
                )
                att_prior = F.interpolate(
                    att_prior,
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=False
                )
            imgs_target.append(att_target * weight[idx])
            imgs_prior.append(att_prior * weight[idx])
        images.append(imgs_target)
        images.append(imgs_prior)

    self_att_maps = []
    if attn_64_self is not None:
        if not with_prior_preservation:
            attn_64_self = torch.stack(attn_64_self) # [n, bsz, 4096, 64, 64]
            attn_64_self = torch.mean(attn_64_self, dim=0, keepdim=True).squeeze(0) # [bsz, 4096, 64, 64]
            self_att_map_ = attn_64_self.view(bsz, 64*64, 64*64) # [bsz, 4096, 4096]
            self_att_map_ = torch.stack([slice_ / slice_.max() for slice_ in self_att_map_])
            # self_att_map_ = self_att_map_.to(cross_att_map.dtype)
            self_att_maps.append(self_att_map_)
        else:
            # attn_64_self = torch.stack(attn_64_self)
            # attn_64_self_target = torch.stack(attn_64_self[:bsz])
            # attn_64_self_target = torch.mean(attn_64_self_target, dim=0, keepdim=True).squeeze(0)
            # attn_64_self_prior = torch.stack(attn_64_self[bsz:])
            # attn_64_self_prior = torch.mean(attn_64_self_prior, dim=0, keepdim=True).squeeze(0)
            attn_64_self_target, attn_64_self_prior = attn_64_self
            self_att_map_target = attn_64_self_target.view(bsz, 64*64, 64*64)
            self_att_map_target = torch.stack([slice_ / slice_.max() for slice_ in self_att_map_target])
            self_att_map_prior = attn_64_self_prior.view(len(attn_64_self_prior), 64*64, 64*64)
            self_att_map_prior = torch.stack([slice_ / slice_.max() for slice_ in self_att_map_prior])
            self_att_maps.append(self_att_map_target)
            self_att_maps.append(self_att_map_prior)

    score_maps = []
    for i, imgs in enumerate(images):
        cross_att_map = torch.stack(imgs) # [4, bsz, 77, 64, 64]
        cross_att_map = torch.sum(cross_att_map, dim=0, keepdim=True).squeeze(0) # [bsz, 77, 64, 64]
        select_ = select if i == 0 else prior_select
        cross_att_map_ = cross_att_map[:, select_[0], :, :]
        for s in select_[1:]:
            cross_att_map_ += cross_att_map[:, s, :, :]
        if len(select_) > 1:
            cross_att_map_ /= len(select_) # [bsz, 64, 64]
        cross_att_map = cross_att_map_.view(bsz, -1, 1) # [bsz, 4096, 1]
        if i < len(self_att_maps):
            print(self_att_maps[i].shape, cross_att_map.shape)
            cross_att_map = torch.matmul(self_att_maps[i], cross_att_map) # [bsz, 4096, 1]

        score_map = cross_att_map.view(bsz, 64, 64)
        score_map = F.interpolate(score_map.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False)
        score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min())
        score_map = F.sigmoid(alpha * (score_map - beta))
        score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min())
        score_maps.append(score_map)

    return score_maps


def compute_attn_reg(
    reg, 
    timesteps, 
    noise_scheduler, 
    reg_snr_gamma, 
    with_attention_reg_snr, 
    with_attention_reg_sigmoid, 
    reg_sigmoid_delta, 
    reg_sigmoid_kappa,
):
    assert not (with_attention_reg_snr and with_attention_reg_sigmoid), "Only one weight method can be used."
    if not with_attention_reg_snr and not with_attention_reg_sigmoid:
        return reg.mean()
    if with_attention_reg_snr:
        snr = compute_snr(noise_scheduler, timesteps)
        base_weight = (
            torch.stack([snr, reg_snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        )
        if noise_scheduler.config.prediction_type == "v_prediction":
            # Velocity objective needs to be floored to an SNR weight of one.
            reg_loss_weights = base_weight + 1
        else:
            # Epsilon and sample both use the same loss weights.
            # SD 1.4 prediction type: epsilon
            reg_loss_weights = base_weight
        reg = reg.mean(dim=list(range(1, len(reg.shape))))
        reg = (reg * reg_loss_weights).mean()
        return reg
    
    reg_loss_weights = 1/(1 + torch.exp((timesteps - reg_sigmoid_delta) / reg_sigmoid_kappa))
    reg = reg.mean()
    reg = (reg * reg_loss_weights).mean()
    return reg


def attention_regularization(
    unet,
    seg_values,
    select,
    prior_select,
    weight=[0.3, 0.5, 0.1, 0.1],
    height=780, 
    width=780, 
    bsz=1,
    save=False, 
    img=None, 
    step=0, 
    with_prior_preservation=False, 
    self_attn=False, 
    alpha=8, 
    beta=0.4,
    with_attention_reg_snr=False,
    timesteps=None,
    reg_snr_gamma=5.0,
    noise_scheduler=None,
    with_attention_reg_sigmoid=False,
    reg_sigmoid_delta = 200,
    reg_sigmoid_kappa = 10,
    disable_input_reg=False,
):
    assert with_prior_preservation or not(disable_input_reg), "Either input loss or prior BCE loss must be computed."
    attn_8 = aggregate_attention(unet, self_attn=False, res=8, bsz=bsz, with_prior_preservation=with_prior_preservation)
    attn_16 = aggregate_attention(unet, self_attn=False, res=16, bsz=bsz, with_prior_preservation=with_prior_preservation)
    attn_32 = aggregate_attention(unet, self_attn=False, res=32, bsz=bsz, with_prior_preservation=with_prior_preservation)
    attn_64 = aggregate_attention(unet, self_attn=False, res=64, bsz=bsz, with_prior_preservation=with_prior_preservation)
    if self_attn:
        attn_64_self = aggregate_attention(unet, self_attn=True, res=64, bsz=bsz, with_prior_preservation=with_prior_preservation)
        # print(len(attn_64_self))
        # for item in attn_64_self:
        #     print(item.shape)
    else:
        attn_64_self = None

    score_maps = generate_score_map(
        select,
        prior_select,
        attn_8,
        attn_16,
        attn_32,
        attn_64,
        attn_64_self=attn_64_self,
        weight=weight,
        height=height, 
        width=width, 
        bsz=bsz,
        alpha=alpha, 
        beta=beta,
        with_prior_preservation=with_prior_preservation,
    )
    print(len(score_maps))
    loss = BCELoss(reduction='none')
    if not with_prior_preservation:
        score_map = score_maps[0]
        seg_values = seg_values.to(score_map.dtype)
        reg = loss(score_map, seg_values)
        reg_prior = None
    else:
        seg_values = seg_values.to(score_maps[0].dtype)
        print("seg_values.shape", seg_values.shape)
        seg_valuse_target, seg_valuse_prior = torch.split(seg_values, [bsz, len(seg_values) - bsz])
        print("score_maps[0].shape", score_maps[0].shape)
        print("seg_valuse_target.shape", seg_valuse_target.shape)
        reg = loss(score_maps[0], seg_valuse_target)
        print("reg.shape", reg.shape) 
        reg_prior = loss(score_maps[1], seg_valuse_prior)
        print("reg_prior.shape", reg_prior.shape)

    if disable_input_reg:
        return compute_attn_reg(reg_prior, timesteps, noise_scheduler, reg_snr_gamma, with_attention_reg_snr, with_attention_reg_sigmoid, reg_sigmoid_delta, reg_sigmoid_kappa)
    reg = compute_attn_reg(reg, timesteps, noise_scheduler, reg_snr_gamma, with_attention_reg_snr, with_attention_reg_sigmoid, reg_sigmoid_delta, reg_sigmoid_kappa)
    if  reg_prior is not None:
        reg +=  compute_attn_reg(reg_prior, timesteps, noise_scheduler, reg_snr_gamma, with_attention_reg_snr, with_attention_reg_sigmoid, reg_sigmoid_delta, reg_sigmoid_kappa)
        reg /= 2.0
    return reg


######## debugging and case study ########
import numpy as np
import cv2
import os
import argparse
# import shutil
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from typing import Optional, Union, Tuple, List, Callable, Dict
from torchvision import transforms
from sklearn.metrics import f1_score, roc_curve, auc

from diffusers import (
    DiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    CLIPTextModel,
)


def show_cam_on_image(img, mask):
    # print(mask.shape, img.size)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img.size[1],img.size[0]), mode='bilinear', align_corners=False).squeeze().squeeze()
    mask = mask.to(torch.float32)
    # mask = np.interp()
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, palette, num_rows=1, offset_ratio=0.02, save_dir="./", name='output.pdf'):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    palette_image = pil_img.convert('P')
    palette_image.putpalette(palette)
    # display(palette_image)
    palette_image.save(os.path.join(save_dir, name))


def show_cross_attention(prompts, tokenizer, unet, bsz, with_prior_preservation, palette, res: int, self_attn: bool, select: int = 0, save_dir="./", cls_name=''):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    # attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    attention_maps = aggregate_attention(unet, self_attn=self_attn, res=res, bsz=bsz, with_prior_preservation=with_prior_preservation)
    if (len(attention_maps) == 1):
        att_ = attention_maps[0].unsqueeze(0)
    else:
        att_ = torch.stack(attention_maps, dim=0) # [n, bsz, 77, 64, 64]

    attention_maps = torch.mean(att_, dim=0, keepdim=True) # [1, bsz, 77, 64, 64]
    attention_maps = attention_maps.squeeze(0).squeeze(0).cpu() # [bsz, 77, 64, 64]
    images = []
    j = 0
    for i in range(len(tokens)):
        image = attention_maps[i, :, :]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.float().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if decoder(int(tokens[j])) == "++":
            j += 1
        image = text_under_image(image, decoder(int(tokens[j])))
        images.append(image)
        j += 1
        if j >= len(tokens):
            break
    view_images(np.stack(images, axis=0), palette, save_dir=save_dir, name='output_{}_{}.pdf'.format(res, cls_name))


def create_palette(cmap_name, num_colors=256):
    cmap = plt.get_cmap(cmap_name)
    palette = []
    for i in range(num_colors):
        r, g, b, _ = cmap(i / num_colors)
        palette.extend((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def encode_imgs(imgs, vae):
    # imgs: [B, 3, H, W]
    imgs = 2 * imgs - 1
    # print(imgs.dtype)
    posterior = vae.encode(imgs).latent_dist.mean
    latents = posterior * 0.18215
    return latents


def attention_test(
    pretrained_model_name_or_path,
    image_path,
    seg_path,
    prompt,
    cls_name,
    t=100,
    self_attn=True,
    lora=True,
    lora_weights_root=None,
    lora_weights_subfolder='checkpoint-1000',
    scratch_weights_root=None,
    scratch_weights_subfolder='checkpoint-1000',
    map_weight=[0.3, 0.5, 0.1, 0.1],
    map_alpha=16.0,
    map_beta=4.0,
    cur_best=0.0,
    pos_indices=[1, 2],
):  
    image_file_ = "{}_{:.2f}_{}_{}_{}".format(map_alpha, map_beta, self_attn, len(pos_indices)==1, cls_name)
    seg = Image.open(seg_path).convert('L')
    seg_arr = np.asarray(seg)
    seg_arr = np.where(seg_arr > 200, 1, 0)

    device = torch.device("cuda:0")
    if lora:
        pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, dtype=torch.bfloat16, safety_checker=None,
        ).to(device, dtype=torch.bfloat16)
        vae = pipeline.vae.to(device, dtype=torch.bfloat16)
        scheduler = pipeline.scheduler

        if lora_weights_root:
            print(">>>>> load LoRA weights")
            pipeline.load_lora_weights(lora_weights_root, subfolder=lora_weights_subfolder, weight_name="pytorch_lora_weights.safetensors")
    
        unet = pipeline.unet
        unet = unet.to(device, dtype=torch.bfloat16)
        cross_attn_init()
        unet = store_attn_maps(unet, self_attn=self_attn)
    
    else:
        if scratch_weights_root:
            unet = UNet2DConditionModel.from_pretrained(scratch_weights_root, subfolder=os.path.join(scratch_weights_subfolder, "unet"))
            text_encoder = CLIPTextModel.from_pretrained(scratch_weights_root, subfolder="text_encoder")
            pipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, unet=unet, text_encoder=text_encoder, dtype=torch.bfloat16, safety_checker=None
            ).to(device, dtype=torch.bfloat16)
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, dtype=torch.bfloat16, safety_checker=None,
            ).to(device, dtype=torch.bfloat16)
        unet = pipeline.unet
        unet = unet.to(device, dtype=torch.bfloat16)
        cross_attn_init()
        unet = store_attn_maps(unet, self_attn=self_attn)
        vae = pipeline.vae.to(device, dtype=torch.bfloat16)
        scheduler = pipeline.scheduler

    timesteps = torch.tensor(t, device=device)

    with torch.no_grad():
        input_img = Image.open(image_path).convert("RGB")
        if prompt is None:
            processor = BlipProcessor.from_pretrained("/dev/shm/alexJiang/source/BLIP")
            model = BlipForConditionalGeneration.from_pretrained("/dev/shm/alexJiang/source/BLIP").to(device)
            object_name = cls_name.split('_')[-1]
            text = f"a photograph of sks {object_name}"
            inputs = processor(input_img, text, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            prompt = processor.decode(out[0], skip_special_tokens=True)
            print(">>> generated prompt:", prompt)

        trans = []
        trans.append(transforms.ToTensor())

        trans = transforms.Compose(trans)

        img_tensor = (trans(input_img).unsqueeze(0)).to(device)
        rgb_512 = F.interpolate(img_tensor, (512, 512), mode='bilinear', align_corners=False).bfloat16()

        input_latent = encode_imgs(rgb_512, vae)
        noise = torch.randn_like(input_latent).to(device)
        bsz, channels, height, width = input_latent.shape
        latents_noisy = scheduler.add_noise(input_latent, noise, timesteps)

        text_input = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_input.input_ids
        prompt_embeds = pipeline.text_encoder(text_input_ids.to(device), attention_mask=None)
        text_embeddings = prompt_embeds[0]

        model_pred = unet(
            latents_noisy,
            timesteps,
            text_embeddings,
            class_labels=None,
            return_dict=False,
        )[0]

        attn_8 = aggregate_attention(unet, self_attn=False, res=8, bsz=bsz, with_prior_preservation=False)
        attn_16 = aggregate_attention(unet, self_attn=False, res=16, bsz=bsz, with_prior_preservation=False)
        attn_32 = aggregate_attention(unet, self_attn=False, res=32, bsz=bsz, with_prior_preservation=False)
        attn_64 = aggregate_attention(unet, self_attn=False, res=64, bsz=bsz, with_prior_preservation=False)

        if self_attn:
            attn_64_self = aggregate_attention(unet, self_attn=True, res=64, bsz=bsz, with_prior_preservation=False)

        imgs = []
        for idx, att in enumerate([attn_8, attn_16, attn_32, attn_64]):
            if (len(att) == 1):
                att_ = att[0].unsqueeze(0)
            else:
                att_ = torch.stack(att, dim=0) # [n, bsz, 77, 64, 64]

            att_ = torch.mean(att_, dim=0, keepdim=True) # [1, bsz, 77, 64, 64]
            att_ = att_.squeeze(0) # [bsz, 77, 64, 64]

            if idx < 3:
                att_ = F.interpolate(
                    att_,
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=False
                )

            imgs.append(att_ * map_weight[idx])

        cross_att_map = torch.stack(imgs) # [4, bsz, 77, 64, 64]
        cross_att_map = torch.sum(cross_att_map, dim=0, keepdim=True).squeeze(0) # [bsz, 77, 64, 64]
        cross_att_map_ = cross_att_map[:, pos_indices[0], :, :]
        for s in pos_indices[1:]:
            # print(cross_att_map.shape, s)
            cross_att_map_ += cross_att_map[:, s, :, :]
        if len(pos_indices) > 1:
            cross_att_map_ /= len(pos_indices) # [bsz, 64, 64]
        cross_att_map = cross_att_map_.view(bsz, -1, 1) # [bsz, 4096, 1]

        if self_attn:
            attn_64_self = torch.stack(attn_64_self) # [n, bsz, 4096, 64, 64]
            attn_64_self = torch.mean(attn_64_self, dim=0, keepdim=True).squeeze(0) # [bsz, 4096, 64, 64]
            self_att_map = attn_64_self.view(bsz, 64*64, 64*64) # [bsz, 4096, 4096]
            self_att_map = self_att_map / self_att_map.max()
            self_att_map = self_att_map.to(cross_att_map.dtype)
            cross_att_map = torch.matmul(self_att_map, cross_att_map) # [bsz, 4096, 1]

        score_map = cross_att_map.view(bsz, 64, 64)
        # torch.set_printoptions(threshold=10_000)
        # print(score_map)
        score_map = F.interpolate(score_map.unsqueeze(1), size=(input_img.size[1], input_img.size[0]), mode='bilinear', align_corners=False)
        score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min())
        score_map = F.sigmoid(map_alpha * (score_map - map_beta))
        score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min())

        loss = BCELoss(reduction='mean')
        seg_values = torch.tensor(seg_arr).float()
        score_map = score_map.cpu().float().squeeze(0).squeeze(0)
        # print(score_map.shape, seg_values.shape)
        l = loss(score_map, seg_values).item()

        f1 = []
        thres_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        score_map = score_map.numpy()
        for thres in thres_list:
            mask_binary = np.where(score_map > thres, 1, 0)
            f1_ = f1_score(seg_arr.flatten(), mask_binary.flatten())
            f1.append(f1_)
            # print(">>> thres: {}, f1: {:.4f}".format(thres, f1_))
        f1 = np.asarray(f1)
        best_f1 = f1.max()
        dice_auc = auc(np.asarray(thres_list), f1)
        print('Dice AUC: {:.4f}, best Dice: {:.4f}, BCE loss: {:.4f}'.format(dice_auc, best_f1, l))

        

    if dice_auc > cur_best:
        for file in os.listdir(save_dir):
            if file.endswith(".pdf"):
                os.remove(os.path.join(save_dir, file))
        print(">>>>> update current best")
        palette = create_palette('viridis')
        print("8x8 cross att map")
        show_cross_attention([prompt], pipeline.tokenizer, unet, bsz, False, palette, res=8, self_attn=False, select=0, save_dir=save_dir, cls_name=image_file_)
        print("16x16 cross att map")
        show_cross_attention([prompt], pipeline.tokenizer, unet, bsz, False, palette, res=16, self_attn=False, select=0, save_dir=save_dir, cls_name=image_file_)
        print("32x32 cross att map")
        show_cross_attention([prompt], pipeline.tokenizer, unet, bsz, False, palette, res=32, self_attn=False, select=0, save_dir=save_dir, cls_name=image_file_)
        print("64x64 cross att map")
        show_cross_attention([prompt], pipeline.tokenizer, unet, bsz, False, palette, res=64, self_attn=False, select=0, save_dir=save_dir, cls_name=image_file_)

        print("cam visualization")

        cam = show_cam_on_image(input_img, torch.tensor(score_map))
        pil_img = Image.fromarray(cam[:,:,::-1])
        pil_img.save(os.path.join(save_dir, 'cam_{}.pdf'.format(image_file_)))

        del unet
        del pipeline
        del vae
        torch.cuda.empty_cache()
        return dice_auc

    del unet
    del pipeline
    del vae
    torch.cuda.empty_cache()
    return cur_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos_index1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--pos_index2",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/dev/shm/alexJiang/source/SD1_4",
    )
    parser.add_argument(
        "--lora",
        action="store_true"
    )
    parser.add_argument(
        "--lora_weight_root",
        type=str,
        default="output/LoRA-DreamBooth-dog6-v1-4",
    )
    parser.add_argument(
        "--scratch_weight_root",
        type=str,
        default="output/DreamBooth-dog6-v1-4",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint-1000",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="Dataset/DreamBooth/dog6/00.jpg",
    )
    parser.add_argument(
        "--seg_path",
        type=str,
        default="Dataset/DreamBooth/dog6_seg/00.jpg",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="output/attention_test/dog6",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--map_alpha_min",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--map_alpha_max",
        type=int,
        default=110,
    )
    parser.add_argument(
        "--map_alpha_step",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--map_beta_min",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--map_beta_max",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--map_beta_step",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--without_baseline",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    pos_indices = [args.pos_index1]
    if args.pos_index2 > 0:
        pos_indices.append(args.pos_index2)

    # pretrained_model_name_or_path = "/dev/shm/alexJiang/source/SD1_4"
    # image_path = "Dataset/DreamBooth/dog6/00.jpg"
    # seg_path = "Dataset/DreamBooth/dog6_seg/00.jpg"
    # prompt = "a sks dog"
    # prompt = None
    class_name_prefix = "long_prompt" if args.prompt is None else "short_prompt"
    # cls_name = "long_prompt_original_dog"
    # self_attn = True
    # map_alpha=100.0
    # map_beta=0.15
    # cls_name = "{}_{}_{}_long_prompt_LoRA_DreamBooth_dog".format(self_attn, map_alpha, map_beta)
    # lora_weights_root = "output/LoRA-DreamBooth-backpack-v1-4"
    # t = 100
    # single_token = True
    # attention_test(
    #     pretrained_model_name_or_path,
    #     image_path,
    #     seg_path,
    #     prompt,
    #     cls_name,
    #     t=t,
    #     self_attn=self_attn,
    #     lora_weights_root=lora_weights_root,
    #     lora_weights_subfolder='checkpoint-1000',
    #     map_alpha=map_alpha,
    #     map_beta=map_beta,
    # )

    if args.without_baseline:
        weight_list = [args.lora_weight_root] if args.lora else [args.scratch_weight_root]
    else:
        weight_list = [args.lora_weight_root, None] if args.lora else [args.scratch_weight_root, None]

    for weight_root in weight_list:
        if weight_root is None:
            save_dir = os.path.join(args.save_root, "LoRA_Original") if args.lora else os.path.join(args.save_root, "Original")
        else:
            save_dir = os.path.join(args.save_root, "LoRA_DreamBooth") if args.lora else os.path.join(args.save_root, "DreamBooth")
        for self_attn in [True, False]:
            cur_best = 0.0
            save_dir = os.path.join(save_dir, "self_attn") if self_attn else os.path.join(save_dir, "wo_self_attn")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            for map_alpha in range(args.map_alpha_min, args.map_alpha_max, args.map_alpha_step):
                for map_beta in np.arange(args.map_beta_min, args.map_beta_max, args.map_beta_step):
                    if weight_root is None:
                        cls_name = "{}_original_dog".format(class_name_prefix)
                    else:
                        cls_name = "{}_LoRA_DreamBooth_dog".format(class_name_prefix) if args.lora else "{}_DreamBooth_dog".format(class_name_prefix)
                    print(" ")
                    print(">>> map_alpha: {:.2f}, map_beta: {:.2f}, cur_best: {:.2f}".format(map_alpha, map_beta, cur_best))
                    cur_best = attention_test(
                        args.pretrained_model_name_or_path,
                        args.image_path,
                        args.seg_path,
                        args.prompt,
                        cls_name,
                        t=args.t,
                        self_attn=self_attn,
                        lora=args.lora,
                        lora_weights_root=weight_root,
                        lora_weights_subfolder=args.ckpt,
                        scratch_weights_root=weight_root,
                        scratch_weights_subfolder=args.ckpt,
                        map_alpha=map_alpha,
                        map_beta=map_beta,
                        cur_best=cur_best,
                        pos_indices=pos_indices,
                    )

            print(">>>>>>>>>> weight_root: {}, self_attn: {}, best Dice AUC: {:.4f}\n\n".format(weight_root, self_attn, cur_best))

