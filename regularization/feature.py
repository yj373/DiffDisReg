import torch
import torch.nn.functional as F

from typing import Optional, Dict, Tuple, Any
from diffusers.utils.torch_utils import apply_freeu, is_torch_version
from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D, UpBlock2D

def store_res_features(unet):
    for upsample_block in unet.up_blocks:
        upsample_block.store_res_features = True
    return unet


def CrossAttnUpBlock2DForward(
    self,
    hidden_states: torch.Tensor,
    res_hidden_states_tuple: Tuple[torch.Tensor, ...],
    temb: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    upsample_size: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
):
    # if cross_attention_kwargs is not None:
    #     if cross_attention_kwargs.get("scale", None) is not None:
    #         logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    is_freeu_enabled = (
        getattr(self, "s1", None)
        and getattr(self, "s2", None)
        and getattr(self, "b1", None)
        and getattr(self, "b2", None)
    )
    for j, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        # FreeU: Only operate on the first two stages
        if is_freeu_enabled:
            hidden_states, res_hidden_states = apply_freeu(
                self.resolution_idx,
                hidden_states,
                res_hidden_states,
                s1=self.s1,
                s2=self.s2,
                b1=self.b1,
                b2=self.b2,
            )

        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        ## save the features like ODISE
        if j == len(self.resnets) - 1 and self.store_res_features:
            self.res_features = hidden_states.contiguous()
        ##

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward
            
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states


def UpBlock2DForward(
    self,
    hidden_states: torch.Tensor,
    res_hidden_states_tuple: Tuple[torch.Tensor, ...],
    temb: Optional[torch.Tensor] = None,
    upsample_size: Optional[int] = None,
    *args,
    **kwargs,
):
    is_freeu_enabled = (
        getattr(self, "s1", None)
        and getattr(self, "s2", None)
        and getattr(self, "b1", None)
        and getattr(self, "b2", None)
    )
    for j, resnet in enumerate(self.resnets):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        # FreeU: Only operate on the first two stages
        if is_freeu_enabled:
            hidden_states, res_hidden_states = apply_freeu(
                self.resolution_idx,
                hidden_states,
                res_hidden_states,
                s1=self.s1,
                s2=self.s2,
                b1=self.b1,
                b2=self.b2,
            )

        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        ## save the features like ODISE
        if j == len(self.resnets) - 1 and self.store_res_features:
            self.res_features = hidden_states.contiguous()
        ##

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                )
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
        else:
            hidden_states = resnet(hidden_states, temb)

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)
    
    return hidden_states


def feature_set_unet_forward_function(unet):
    for block in unet.up_blocks:
        if block.__class__.__name__ == "CrossAttnUpBlock2D":
            block.forward = CrossAttnUpBlock2DForward.__get__(block, CrossAttnUpBlock2D)
        elif block.__class__.__name__ == "UpBlock2D":
            block.forward = UpBlock2DForward.__get__(block, UpBlock2D)

    return unet


def feature_regularization(unet, decoder, segmentation_head, seg_values):
    features = []
    try:
        blocks = unet.up_blocks
    except AttributeError:
        blocks = unet.module.up_blocks
    for block in blocks:
        if hasattr(block, "res_features"):
            features.append(torch.chunk(block.res_features, 2, dim=0)[0])

    features = features[::-1]
    # torch.Size([1, 640, 64, 64])
    # torch.Size([1, 960, 32, 32])
    # torch.Size([1, 1920, 16, 16])
    # torch.Size([1, 2560, 8, 8])
    decoder_output = decoder(*features)
    mask = segmentation_head(decoder_output)
    loss = F.binary_cross_entropy_with_logits(mask, seg_values, reduction='none')
    return loss