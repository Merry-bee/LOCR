"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import math
import os
import warnings
import inspect
from typing import List, Optional, Tuple, Union, Callable
from collections import defaultdict
from pathlib import Path
import copy
import json
import fitz
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import cv2
import timm
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.distributed as dist
from locr.visualization import visual_box,interact_with_human

from timm.models.swin_transformer import SwinTransformer
from torchvision.transforms.functional import resize, rotate
from transformers import (
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    MBartConfig,
    MBartForCausalLM
)
from transformers.models.mbart.modeling_mbart import (
    MBartDecoderLayer,
    MBartAttention,
    MBartLearnedPositionalEmbedding,
    _make_causal_mask,
    _expand_mask
)
from transformers.activations import ACT2FN
from transformers.utils import replace_return_docstrings
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from locr.postprocessing import postprocess
from locr.transforms import train_transform, test_transform

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.logits_process import LogitsProcessorList
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging
logger = logging.get_logger(__name__)

import logging
from dataclasses import dataclass

from locr.prompt_encoder import PromptEncoder
from locr.position_decoder import PositionDecoder
from locr.cal_loss import cal_loss

class PromptAttention(nn.Module):
    '''
    modification of MBartAttention
    Q and K/V with different dim (1024/256)
    '''
    def __init__(
        self,
        embed_dim: int,
        q_input_dim:int,
        kv_input_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_input_dim = q_input_dim
        self.kv_input_dim = kv_input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(kv_input_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(kv_input_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(q_input_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
            # use_cache
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else :
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
      
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)    # tgt_len: seq_len(Q); src_len:588(KV)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

@dataclass
class CausalLMOutput(ModelOutput):
    '''Modification of CausalLMOutputWithCrossAttentions'''
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: torch.FloatTensor = None
    prompt_pred: Tuple = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class GreedySearchEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    prompt_pred: Tuple = None
    torch.FloatTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

 
class SwinEncoder(nn.Module):
    r"""
    Encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations

    Args:
        input_size: Input image size (width, height)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """

    def __init__(
        self,
        input_size: List[int],
        align_long_axis: bool,
        window_size: int,
        encoder_layer: List[int],
        patch_size: int,
        embed_dim: int,
        num_heads: List[int],
        name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size  # 7
        self.encoder_layer = encoder_layer
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.model = SwinTransformer(   
            img_size=self.input_size,  
            depths=self.encoder_layer, 
            window_size=self.window_size,   
            patch_size=self.patch_size, 
            embed_dim=self.embed_dim,   
            num_heads=self.num_heads,   
            num_classes=0,
        )

        # weight init with swin
        if not name_or_path:
            swin_state_dict = timm.create_model(
                "swin_base_patch4_window12_384", pretrained=True
            ).state_dict()
            new_swin_state_dict = self.model.state_dict()
            for x in new_swin_state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(math.sqrt(len(pos_bias)))
                    new_len = int(2 * window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(
                        0, 3, 1, 2
                    )
                    pos_bias = F.interpolate(
                        pos_bias,
                        size=(new_len, new_len),
                        mode="bicubic",
                        align_corners=False,
                    )
                    new_swin_state_dict[x] = (
                        pos_bias.permute(0, 2, 3, 1)
                        .reshape(1, new_len**2, -1)
                        .squeeze(0)
                    )
                else:
                    new_swin_state_dict[x] = swin_state_dict[x]
            self.model.load_state_dict(new_swin_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        x = self.model.patch_embed(x)  
        x = self.model.pos_drop(x)     
        x = self.model.layers(x)        
        
        return x

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    @property
    def to_tensor(self):
        if self.training:
            return train_transform
        else:
            return test_transform

    def prepare_input_padding(
        self, img: Image.Image, random_padding: bool = False
    ) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            return
        # crop margins
        try:
            img = self.crop_margin(img.convert("RGB"))  
        except OSError:
            # might throw an error for broken files
            return
        if img.height == 0 or img.width == 0:
            return
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))   
        
    def prepare_input(
        self, img: Image.Image, random_padding: bool = False
    ) -> torch.Tensor:
        """
        Convert PIL Image to tensor by forcing resize, image scale may be changed
        """
        if img is None:
            return
        try:
            img = img.resize([self.input_size[1],self.input_size[0]])
        except OSError:
            return
        
        return self.to_tensor(img)

class PromptBartConfig(PretrainedConfig):
    model_type = "promptbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=None,
        forced_eos_token_id=2,
        prompt_embed_dim=1024,  # 256
        image_embedding_size: List[int] = [28, 21],
        input_size: List[int] = [896, 672],
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.prompt_embed_dim = prompt_embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_size = input_size
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

class PromptBartDecoderLayer(nn.Module):
    '''
    Modification of MBartDecoderLayer: 加入adapter层, 进行prompt和image的cross_attention
    '''
    def __init__(self, config: PromptBartConfig):
        super().__init__()
        self.embed_dim = config.d_model # 1024
        self.prompt_embed_dim = config.prompt_embed_dim
        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.image_size = config.input_size
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.encoder_attn = PromptAttention(
            embed_dim=self.embed_dim,
            q_input_dim = self.embed_dim,
            kv_input_dim = self.prompt_embed_dim,
            num_heads = config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_positional_encoding: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        prompt_hidden_states: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention: 修改hidden_states
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,                # [bs,len(input_ids),1024]
            past_key_value=self_attn_past_key_value,    # ([bs,16,cur_len,64],[bs,16,cur_len,64])
            attention_mask=attention_mask,              # [1,1,len(input_ids),len(input_ids)]: dim=-1,0:[0,0,...0] -> -1:[0,0,0,-inf,-inf,...]
            layer_head_mask=layer_head_mask,            # None
            output_attentions=output_attentions,        # True
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block：修改hidden_states
        cross_attn_present_key_value = None
        cross_attn_weights = None
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
        # add 2D positional encoding
        hidden_states = hidden_states + prompt_hidden_states.sum(dim=2)   
        encoder_hidden_states = encoder_hidden_states + encoder_positional_encoding.sum(dim=2)    

        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,      
            key_value_states=encoder_hidden_states, 
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,     
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  
        hidden_states = residual + hidden_states
        # prompt_hidden_states = hidden_states
        

        # add cross-attn to positions 3,4 of present_key_value tuple
        if use_cache:
            present_key_value = present_key_value + cross_attn_present_key_value 

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)  

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights) 

        if use_cache:
            outputs += (present_key_value,)       

        return outputs

class PromptBartPreTrainedModel(PreTrainedModel):
    config_class = PromptBartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PromptBartDecoderLayer", "MBartAttention"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PromptBartDecoder, PromptBartDecoder)):
            module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

class PromptBartDecoder(PromptBartPreTrainedModel):

    def __init__(self, config: PromptBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([PromptBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None: 
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_positional_encoding: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        prompt_hidden_states: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_tensors: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_positional_encoding=encoder_positional_encoding,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                        ),
                    prompt_hidden_states= prompt_hidden_states,
                    prompt_attention_mask= prompt_attention_mask,
                    prompt_attn_layer_head_mask = prompt_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
 
class PromptBartDecoderWrapper(PromptBartPreTrainedModel):
    """
    Between PromptBartForCausalLM and PromptDecoder
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = PromptBartDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

class PromptBartForCausalLM(PromptBartPreTrainedModel):   
    '''
    Modifacation of MBartForCausalLM 
    '''
    
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config,tokenizer_file):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = PromptBartDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_embedding_size = config.image_embedding_size #[28,21]
        self.embed_ratio = config.input_size[0]/config.image_embedding_size[0]  # 896/28=32
        self.alpha = nn.Parameter(torch.zeros(1))

        self.prompt_encoder = PromptEncoder(
            embed_dim=config.prompt_embed_dim,
            image_embedding_size=config.image_embedding_size,
            input_image_size=config.input_size, # [896,672]
            mask_in_chans=16,
        )
        self.position_decoder = PositionDecoder(
            decoder_attention_heads=config.decoder_attention_heads,decoder_layers=config.decoder_layers,input_dim=config.image_embedding_size[0]*config.image_embedding_size[1], hidden_dim=256, output_dim=5, num_layers=3,image_size=config.input_size
        )
        
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        self.pad_idx = self.tokenizer.pad_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.decoder = decoder

    def get_decoder(self):
        return self.decoder

    def decode_position(self,cross_attn_weights,attention_mask, full_prompt_in = None, image_tensors = None):
        prompt_pred=torch.zeros([attention_mask.shape[0],attention_mask.shape[1],2,2]).to(cross_attn_weights.device) # [bs,len,2,2]      
        coords,hm=self.position_decoder(cross_attn_weights,attention_mask, full_prompt_in, image_tensors) # [4,bs,16,len(input_ids),588]->coords:[bs,len,2,2]
        prompt_pred[:,:,0,0],prompt_pred[:,:,0,1],prompt_pred[:,:,1,0],prompt_pred[:,:,1,1]=coords
        return (prompt_pred,hm)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        prompt_in: Optional[torch.Tensor] = None,
        prompt_true: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        full_input_ids:Optional[torch.tensor] = None,
        full_prompt_in:Optional[torch.tensor] = None,
        pdf = None,
        validation = False,
        add_noise = False,
        omit_ratio = 0,
        current_step = None,
        image_tensors = None,
    ) -> Union[Tuple, CausalLMOutput]:
   

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
         
        prompt_hidden_states = self.prompt_encoder(points=None,boxes=prompt_in,add_noise=add_noise,omit_ratio=omit_ratio) if prompt_in is not None else None    # [bs,len,2,2]->[bs,len,2,d]
        if current_step is not None:
            prompt_hidden_states *= min(0.001*current_step,1)
            
        bs = encoder_hidden_states.shape[0]
        img_coord = torch.empty((bs,self.image_embedding_size[0],self.image_embedding_size[1],2,2),dtype=torch.float32)    # [bs,28,21,2,2]
        y1,x1=torch.meshgrid(torch.arange(self.image_embedding_size[0]),torch.arange(self.image_embedding_size[1]))
        y2,x2=torch.meshgrid(torch.arange(1,self.image_embedding_size[0]+1),torch.arange(1,self.image_embedding_size[1]+1))
        img_coord[:,:,:,0,0],img_coord[:,:,:,0,1],img_coord[:,:,:,1,0],img_coord[:,:,:,1,1]=x1,y1,x2,y2
        img_coord = img_coord*self.embed_ratio
        encoder_positional_encoding = self.prompt_encoder(points=None,boxes=img_coord.reshape(bs,-1,2,2))  # [bs,588,2,2]->[bs,588,2,d]
        if current_step is not None:
            encoder_positional_encoding *= min(0.001*current_step,1)
            
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # self.model.decoder: PromptBartDecoder

        if self.model.training:     # 使用cross_attention作为prompt box        
            outputs = self.model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,    # [bs,588,1024]
                encoder_positional_encoding = encoder_positional_encoding,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                prompt_hidden_states = prompt_hidden_states,    # [bs,len,2,prompt_embed_dim]
                prompt_attention_mask = prompt_attention_mask,
                prompt_attn_layer_head_mask = prompt_attn_layer_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,    # train的过程中只调用一次decoder，不需要缓存kv
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            
            logits = self.lm_head(outputs[0])   # [bs,max_length,50000]
            # train with prompt，取对应位置的logits
            if prompt_in is not None:
                if prompt_in.shape[1] != logits.shape[1]:  # 非全文prompt 
                    bs = logits.shape[0]    
                    prompt_label_length = labels.shape[-1] 
                    pred = torch.zeros(bs,prompt_label_length,logits.shape[-1])
                    pred[:,:,self.pad_idx] = 100
                    for b in range(bs):
                        start_idx = max(torch.where(attention_mask[b,:]==1)[0]) # target token start, 但input_ids还拼接了labels
                        if labels is not None and self.pad_idx in labels[b]:    
                        
                            label_len = min(torch.where(labels[b]==self.pad_idx)[0])-2     # 除去<s>和</s>的长度
                        else:
                            
                            label_len = prompt_label_length-2
                    
                        pred[b,:label_len,:] = logits[b,start_idx-label_len:start_idx,:]                                    # [bs,prompt_label_len,50000]
                        labels[b] = torch.cat((labels[b,1:label_len+1],torch.full([prompt_label_length-label_len],self.pad_idx).to(labels.device)))      # [bs,prompt_label_len]，将[<s>,x,x,x,</s>,<pad>]改为[x,x,x,<pad>,<pad><,pad>]
                    logits = pred.to(logits.device) # [bs,prompt_label_len,50000]

        else:   # 推理
            outputs = self.model.decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,    # [bs,588,1024]
                    encoder_positional_encoding = encoder_positional_encoding,
                    encoder_attention_mask=encoder_attention_mask,
                    head_mask=head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    prompt_hidden_states = prompt_hidden_states,
                    prompt_attention_mask = prompt_attention_mask,
                    prompt_attn_layer_head_mask = prompt_attn_layer_head_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    image_tensors = image_tensors
                )
            logits = self.lm_head(outputs[0])   # [bs,len,50000]
        
        cross_attn_weights = torch.stack(outputs['cross_attentions']) # [4, bs,16,len(input_ids),588]
        mask = attention_mask[:,-1:] if input_ids.shape[1] == 1 else attention_mask # inference with cache->attention_mask=1位
        prompt_pred = self.decode_position(cross_attn_weights,mask, full_prompt_in = full_prompt_in, image_tensors = image_tensors)
     
        loss_txt,loss_math,loss_table,loss_start,diou,iou,fl = None,None,None,None,None,None,None
        if labels is not None :      # train/validation
            labels = labels.to(logits.device)   # [bs,length], 用padding补齐
    
            loss_txt,loss_math,loss_table,loss_start,diou,iou,fl = cal_loss(bs=logits.shape[0],
                                                          logits=logits.view(-1, self.config.vocab_size), labels=labels.view(-1),
                                                          prompt_pred=prompt_pred,prompt_true=prompt_true)  
           
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=(loss_txt,loss_math,loss_table,loss_start,diou,iou,fl),
            logits=logits,
            prompt_pred = prompt_pred,  # box,prob
            past_key_values=outputs.past_key_values,    
            hidden_states=outputs.hidden_states,        
            attentions=outputs.attentions,             
            cross_attentions=outputs.cross_attentions,  
        )

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        # attention_mask: Optional[LogitsProcessorList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        pdf = None,
        prompt_in = None,
        validation = False,
        current_step = None,
        image_tensors = None,
        **model_kwargs, # attention_masks, encoder_outputs, use_cache
    ) -> GreedySearchEncoderDecoderOutput:
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,f
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only

        batch_size = input_ids.shape[0]
        # encoder_attention_mask = torch.ones([batch_size,28,21]).to(input_ids.device)    # 0: mask; 1: not
        # glob_cross_attn = torch.zeros([batch_size,28,21]).to(input_ids.device)           # 每个位置被注意到过的attn总和
        cur_line = 0
       
        while True:
            back_flag = False   # 是否需要回退一步
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids,prompt_in, **model_kwargs)                
            
            # forward pass to get next token
            outputs = self.forward(     # self: PromptBartForCausalLM
                **model_inputs, # input_ids, attention_mask, past_key_values, encoder_hidden_states,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                full_input_ids=input_ids,    # 只用于输出current tokens
                full_prompt_in=prompt_in,
                image_tensors = image_tensors,
                pdf=pdf,
                add_noise=False,
                omit_ratio=0,
            )
        
            soft_logits = torch.softmax(outputs.logits,dim=2)
            p_token = soft_logits[:,-1,:].max(dim=-1).values   # # hm=[bs,1,28,21]
            bs,seq_len,output_h,output_w = outputs.prompt_pred[1].shape 
            p_position = outputs.prompt_pred[1][:,-1,:].view(bs,-1).softmax(dim=-1).max(dim=-1).values   # [bs,1,28,21]->[bs,588]->[bs]
            p_token_thres = 0.2
            p_position_thres = 0.5

            if not validation: # human-interactive mode
                for b in range(p_token.shape[0]):
                    # 可视化预测结果
                    save_path = f'data/case/visual_png/{Path(pdf[0]).stem}_{pdf[1]}_infer.png'
                    origin_path = f'data/case/visual_png/{Path(pdf[0]).stem}_{pdf[1]}_origin.png'
                    pd = fitz.open(pdf[0])
                    page = pd[pdf[1]]
                    if not os.path.exists(save_path):   # 保存原图
                        with open(save_path, "wb") as f:
                            f.write(page.get_pixmap(dpi=300).pil_tobytes(format="PNG")) 
                    if not os.path.exists(origin_path): # 保存原图
                        with open(origin_path, "wb") as f:
                            f.write(page.get_pixmap(dpi=300).pil_tobytes(format="PNG")) 
                    image_size = [672,896]
                    ori_img = Image.open(origin_path).resize(image_size)
                    prompt_pred = outputs.prompt_pred[0].squeeze(0)[-1].clone() # [1,1,2,2]->[2,2]
                    hm_indices = outputs.prompt_pred[1][b,-1,:].view(-1).max(dim=0).indices    # [28,21]
                    h,w = hm_indices//output_w, hm_indices % output_w
                    if p_token[b] < p_token_thres and p_position[b] < p_position_thres:   # token置信度低
                        visual_box(png_path=save_path,boxes=prompt_pred.clone(),save_path=save_path,color='red',image_size = [672,896])     
                        flask_png_path=f'flask-image-annotator/images/{Path(pdf[0]).stem}_{pdf[1]}.png'
                        if os.listdir('flask-image-annotator/images'):
                            os.system('rm flask-image-annotator/images/*')
                        os.system(f'cp {save_path} {flask_png_path}')
                        print(f'pdf:{pdf[0]}\npretexts:{repr(self.tokenizer.decode(input_ids[b]))}\nthis_token:{self.tokenizer.decode(torch.argmax(soft_logits[b,-1]))}\np_token={p_token[b]};p_position={p_position[b].item()}')
                        prompt_user,token_user = interact_with_human(prompt_pred.clone(),flask_png_path,save_path,color='red',image_size=[672,896])   # 人工参与
                        if token_user=='q':  # 回退一个token
                            back_flag = True
                        elif token_user:
                            outputs.logits[b,-1,self.tokenizer(token_user)['input_ids'][1]]=outputs.logits[b,-1,:].max()+1
                        if prompt_user: # 拉框，不回退
                            outputs.prompt_pred[0][:,-1:,:,:] = torch.tensor(prompt_user,dtype=torch.float32).unsqueeze(0).unsqueeze(0)   
                    else:   # token和position都对：直接继续
                        visual_box(png_path=save_path,boxes=prompt_pred.clone(),save_path=save_path,color='orange',image_size = [672,896],fill=False)     
                      
            else:   # validation:non-human-interactive mode
                pass


            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    if 'decoder_attentions' in dir(outputs):
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                    elif 'attentions' in dir(outputs):
                        decoder_attentions += (     # outputs.attentions:(num_layer,[batch_size,16,1,cur_len])
                            (outputs.attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,) # # outputs.cross_attentions:(num_layer,[batch_size,16,1,588])

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # next_token_scores:(batch_size, 50000); next_tokens: (batch_size)
            next_tokens = torch.argmax(next_token_scores, dim=-1)  
    

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            
            if not back_flag:
                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)    # (batchsize,cur_len)
                # update generated prompt
                next_prompt = outputs.prompt_pred[0][:,-1:,:,:] #[bs,seq_len,2,2]
                prompt_in = torch.cat([prompt_in.to(next_prompt.device),next_prompt],dim=1)
            else:   # 回退一位
                # input_ids: 不变
                # prompt:删一个，加一个
                next_prompt = outputs.prompt_pred[0][:,-1:,:,:] #[bs,seq_len,2,2]
                prompt_in = torch.cat([prompt_in[:,:-1,:,:].to(next_prompt.device),next_prompt],dim=1)
                # pask_key_values: 删除一位
                past_kv = tuple()
                for i_layer in range(len(outputs['past_key_values'])):
                    layer_kv = tuple()
                    for idx in range(4):
                        if idx<2:
                            layer_kv += (outputs['past_key_values'][i_layer][idx][:,:,:-1,:],)
                        else:
                            layer_kv += (outputs['past_key_values'][i_layer][idx],)
                    past_kv += (layer_kv,)
                outputs['past_key_values'] = past_kv
            
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break
            


        if return_dict_in_generate:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                prompt_pred = outputs['prompt_pred'],
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
            
        else:
            return input_ids

    
    @torch.no_grad()
    # ref: transformers/generation/utils.py: generate()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,  # None
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        pdf = None,
        prompt_in = None,
        validation = False,
        current_step = None,
        image_tensors = None,
        **kwargs,   # input_ids, attention_masks,encoder_outputs,use_cache
    ) -> GreedySearchEncoderDecoderOutput:
     
       

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            # model_kwargs['encoder_outputs']['last_hidden_state'].shape=[3,588,1024]
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")


        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)


        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        # 11. run greedy search
        return self.greedy_search(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            pdf=pdf,
            prompt_in=prompt_in[:,:input_ids.shape[1],:,:],
            validation=validation,
            current_step=current_step,
            image_tensors=image_tensors,
            **model_kwargs,     # attention_mask, encoder_output, use_cache
        )

       

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            input_ids = input_ids[:, -1:]
           
            
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }


    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
    

    

class PromptDecoder(nn.Module):
    """
    Modification of BARTDecoder
    """

    def __init__(
        self,
        decoder_layer: int,
        max_position_embeddings: int,
        input_size:List[int],
        image_embedding_size:List[int],
        hidden_dimension: int = 1024,
        name_or_path: Union[str, bytes, os.PathLike] = None,
        
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings
        if not name_or_path:
            tokenizer_file = Path(__file__).parent / "dataset" / "tokenizer.json"
        else:
            tokenizer_file = Path(name_or_path) / "tokenizer.json"
        if not tokenizer_file.exists():
            raise ValueError("Could not find tokenizer file")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.unk_token = "<unk>"

        self.model = PromptBartForCausalLM(
            config=PromptBartConfig(
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=len(self.tokenizer),
                scale_embedding=True,
                add_final_layer_norm=True,
                d_model=hidden_dimension,
                prompt_embed_dim=hidden_dimension,#1024,256,
                decoder_start_token_id=0,
                input_size = input_size,
                image_embedding_size = image_embedding_size,
            ),
            tokenizer_file='checkpoints/tokenizer.json'
        )
        self.model.config.is_encoder_decoder = True  # to get cross-attention
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference

        if not name_or_path:
            bart_state_dict = PromptBartForCausalLM.from_pretrained(
                "facebook/mbart-large-50"
            ).state_dict()
            new_bart_state_dict = self.model.state_dict()
            for x in new_bart_state_dict:
                if (
                    x.endswith("embed_positions.weight")
                    and self.max_position_embeddings != 1024
                ):
                    new_bart_state_dict[x] = torch.nn.Parameter(
                        self.resize_bart_abs_pos_emb(
                            bart_state_dict[x],
                            self.max_position_embeddings
                            + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                        )
                    )
                elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                    new_bart_state_dict[x] = bart_state_dict[x][
                        : len(self.tokenizer), :
                    ]
                else:
                    new_bart_state_dict[x] = bart_state_dict[x]
            self.model.load_state_dict(new_bart_state_dict, strict=False)

    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(list_of_tokens))}
        )
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_for_inference(
        self,
        input_ids: torch.Tensor,
        prompt_in: torch.Tensor,
        encoder_outputs: torch.Tensor,
        past=None,
        past_key_values=None,
        use_cache: bool = None,
        attention_mask: torch.Tensor = None,
    ):
        """
        Args:
            input_ids: (batch_size, sequence_lenth)

        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        past = past or past_key_values
        if past is not None:    # (n_layers,2+2,[bs,16,cur_len or 588,1024])
            input_ids = input_ids[:, -1:]
            prompt_in = prompt_in[:,-1:,:,:]
        output = {
            "input_ids": input_ids,
            "prompt_in": prompt_in,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return output

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_in: Optional[torch.Tensor] = None,
        prompt_true: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = True,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
        current_step = None,
        full_prompt_in = None,
    ):
        return self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            prompt_in = prompt_in,
            prompt_true = prompt_true,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            add_noise=True,
            omit_ratio=0.0,
            current_step=current_step,
            full_prompt_in = full_prompt_in,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of MBart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight

class BARTDecoder(nn.Module):
    """
    Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a decoder

    Args:
        decoder_layer:
            Number of layers of BARTDecoder
        max_position_embeddings:
            The maximum sequence length to be trained
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local,
            otherwise, `facebook/mbart-large-50` will be set (using `transformers`)
    """

    def __init__(
        self,
        decoder_layer: int,
        max_position_embeddings: int,
        hidden_dimension: int = 1024,
        name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings
        if not name_or_path:
            tokenizer_file = Path(__file__).parent / "dataset" / "tokenizer.json"
        else:
            tokenizer_file = Path(name_or_path) / "tokenizer.json"
        if not tokenizer_file.exists():
            raise ValueError("Could not find tokenizer file")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.unk_token = "<unk>"

        self.model = MBartForCausalLM(
            config=MBartConfig(
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=len(self.tokenizer),
                scale_embedding=True,
                add_final_layer_norm=True,
                d_model=hidden_dimension,
            )
        )
        self.model.config.is_encoder_decoder = True  # to get cross-attention
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference

        if not name_or_path:
            bart_state_dict = MBartForCausalLM.from_pretrained(
                "facebook/mbart-large-50"
            ).state_dict()
            new_bart_state_dict = self.model.state_dict()
            for x in new_bart_state_dict:
                if (
                    x.endswith("embed_positions.weight")
                    and self.max_position_embeddings != 1024
                ):
                    new_bart_state_dict[x] = torch.nn.Parameter(
                        self.resize_bart_abs_pos_emb(
                            bart_state_dict[x],
                            self.max_position_embeddings
                            + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                        )
                    )
                elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                    new_bart_state_dict[x] = bart_state_dict[x][
                        : len(self.tokenizer), :
                    ]
                else:
                    new_bart_state_dict[x] = bart_state_dict[x]
            self.model.load_state_dict(new_bart_state_dict, strict=False)

    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(list_of_tokens))}
        )
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_for_inference(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        past=None,
        past_key_values=None,
        use_cache: bool = None,
        attention_mask: torch.Tensor = None,
    ):
        """
        Args:
            input_ids: (batch_size, sequence_lenth)

        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        past = past or past_key_values
        if past is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return output

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
    ):
        return self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of MBart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight

class LOCRConfig(PretrainedConfig):
    
    model_type = "locr"

    def __init__(
        self,
        input_size: List[int] = [896, 672],
        align_long_axis: bool = False,
        window_size: int = 7,
        encoder_layer: List[int] = [2, 2, 14, 2],
        decoder_layer: int = 10,
        max_position_embeddings: int = None,
        max_length: int = 4096,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        patch_size: int = 4,
        embed_dim: int = 128,
        num_heads: List[int] = [4, 8, 16, 32],
        hidden_dimension: int = 1024,
        omit_ratio: float = 0,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = (
            max_length if max_position_embeddings is None else max_position_embeddings
        )
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dimension = hidden_dimension
 


class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0


def batch(l, b=15):
    subs = []
    for i in range(len(l) - b):
        subs.append(l[i : i + b])
    return subs


def subdiv(l, b=10):
    subs = []
    for i in range(len(l) - b):
        subs.append(l[: i + b])
    return subs


class LOCRModel(PreTrainedModel):
  
    config_class = LOCRConfig
    base_model_prefix = "locr"

    def __init__(self, config: LOCRConfig):
        super().__init__(config)
        self.config = config
        self.encoder = SwinEncoder(
            input_size=self.config.input_size,
            align_long_axis=self.config.align_long_axis,
            window_size=self.config.window_size,
            encoder_layer=self.config.encoder_layer,
            name_or_path=self.config.name_or_path,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,    
            num_heads=self.config.num_heads,
        )
       

        self.decoder = PromptDecoder(
            max_position_embeddings=self.config.max_position_embeddings,
            decoder_layer=self.config.decoder_layer,
            name_or_path=self.config.name_or_path,
            hidden_dimension=self.config.hidden_dimension,
            input_size = self.config.input_size,
            image_embedding_size = [int(size/self.config.patch_size/2**(len(self.config.encoder_layer)-1)) for size in self.config.input_size]  # [896/4/2^3,672/4/2^3]=[28,21]
        )
        self.pad_id = self.decoder.tokenizer.pad_token_id

    def forward(
        self,
        image_tensors: torch.Tensor,
        pre_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        label_id: torch.Tensor = None,
        prompt_in: Optional[torch.Tensor] = None,
        prompt_true: Optional[torch.Tensor] = None,
        current_step = None,
        full_prompt_in = None,
    ):
        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            pre_input_ids: (batch_size, sequence_length, embedding_dim)
        """
        encoder_outputs = self.encoder(image_tensors)
      
        input_ids = pre_input_ids   #[bs,max_len]
        labels = label_id   # [bs,max_len/label_len]
       
        decoder_outputs = self.decoder(
            input_ids=input_ids.contiguous(),   # [bs,len(input_ids)-1]
            encoder_hidden_states=encoder_outputs,
            attention_mask=attention_mask,      # input_ids是否为padding_token
            labels=labels.contiguous(),      
            prompt_in = prompt_in,
            prompt_true = prompt_true,
            current_step=current_step,
            full_prompt_in = full_prompt_in,
        )
        return decoder_outputs

    def _init_weights(self, *args, **kwargs):
        return

    def inference(
        self,
        image: Image.Image = None,
        image_tensors: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = True,
        pdf = None,
        prompt = None,
        validation = False,
        use_cache = True,
        current_step = None,
    ):
        """
        Generate a token sequence in an auto-regressive manner.

        Args:
            image: input document image (PIL.Image)
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
        """
        output = {
            "predictions": list(),
            "sequences": list(),
            "repeats": list(),
            "repetitions": list(),
            'logits': list(),
            "prompt_pred":list(),
        }
        if image is None and image_tensors is None:
            logging.warn("Image not found")
            return output

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)
        image_tensors = image_tensors.to(torch.float32)
        if self.device.type == "cuda":  # half is not compatible in cpu implementation.
            image_tensors = image_tensors.to(self.device)

        last_hidden_state = self.encoder(image_tensors)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        last_hidden_state = self.encoder(image_tensors).to(torch.float32)
        encoder_outputs = ModelOutput(
            last_hidden_state=last_hidden_state, attentions=None
        )

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = (
                encoder_outputs.last_hidden_state.unsqueeze(0)
            )

        # get decoder output: generate texts
        
        decoder_output = self.decoder.model.generate(
            encoder_outputs=encoder_outputs,
            prompt_in=prompt,
            # decoder_input_ids = input_ids,  # kwargs
            attention_mask=attention_mask,   # kwargs
            min_length=1,
            max_length=self.config.max_length,
            pad_token_id=self.pad_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            bad_words_ids=[
                [self.decoder.tokenizer.unk_token_id],
            ],
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=return_attentions,
            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores(threshold=0.005)]),
            do_sample=False,
            pdf=pdf,
            validation=validation,
            use_cache = use_cache,
            current_step=current_step,
            image_tensors=image_tensors,
        )

        output["repetitions"] = decoder_output.sequences.clone()    # sequences: token_ids[batch_size,seq_len]
        output["sequences"] = decoder_output.sequences.clone()      # dtype=torch.int64, from input_ids
        batch_size = len(decoder_output.sequences)  

        output["logits"] = decoder_output.scores                        # scores: token_scores(seq_len,[batch_size,50000])
        logits = torch.stack(decoder_output.scores, 1).cpu().max(-1)    # stack(): convert to [batch_size, seq_len, 50000]
        values = logits.values      # [batch_size,seq_len]
        indices = logits.indices    # [batch_size,seq_len]

        output["prompt_pred"] = decoder_output.prompt_pred

        for b in range(batch_size):
            mask = indices[b] != self.pad_id
            N = mask.sum().item()   # seq_len
            var = np.array(
                [np.var(s) / len(s) for s in batch(values[b, mask].float().numpy())]
            )
            if len(var) < 10:  
                output["repeats"].append(None)
                continue
            varvar = np.array([np.var(v) for v in subdiv(var[::-1])][::-1]) # formula in paper
            minlen = 120
            if (
                indices[b] == self.decoder.tokenizer.eos_token_id
            ).any() and N + 1 < indices.shape[1]:
                # there is an end to the generation, likely no repetitions
                output["repeats"].append(None)
                continue
            # small_var = np.where(varvar < 0.045)[0] # the indices of repetition
            small_var = np.where(varvar < 0.03)[0] # the indices of repetition
            if len(small_var) > 1:
                if np.all(np.diff(small_var) < 2):  
                    idx = int(min(max(small_var[0], 1) * 1.08 + minlen, 4095))
                    # if idx / N > 0.9:  # at most last bit
                    if idx / N > 0.85:  # at most last bit
                        output["repeats"].append(None)
                        continue
                
                    output["repeats"].append(idx) 
                    output["sequences"][b, idx:] = self.pad_id
                    output["repetitions"][b, :idx] = self.pad_id
                    
                    
                else:
                    output["repeats"].append(None)
            else:
                output["repeats"].append(None)
        output["repetitions"] = self.decoder.tokenizer.batch_decode(
            output["repetitions"], skip_special_tokens=True
        )
        for b in range(len(output["repetitions"])):
            if output["repeats"][b] and output["repetitions"][b]:
                print(f'Rep {b}_{output["repeats"][b]}: {output["repetitions"][b]}')
       
        output["predictions"] = postprocess(                   
            self.decoder.tokenizer.batch_decode(
                output["sequences"], skip_special_tokens=True
            ),
            markdown_fix=True,
        )

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,   # (seq_len,num_layer,[batchsize,16,1,cur_len])
                "cross_attentions": decoder_output.cross_attentions,    # (seq_len,num_layer,[batchsize,16,1,588])
            }
           
            

        return output

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, bytes, os.PathLike],
        *model_args,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained model from a pre-trained model configuration

        Args:
            model_path:
                Name of a pretrained model name either registered in huggingface.co. or saved in local.
        """
        model = super(LOCRModel, cls).from_pretrained(
            model_path, *model_args, **kwargs
        )

        # truncate or interpolate position embeddings of decoder
        max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        if (
            max_length != model.config.max_position_embeddings
        ):  # if max_length of trained model differs max_length you want to train
            model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                model.decoder.resize_bart_abs_pos_emb(
                    model.decoder.model.model.decoder.embed_positions.weight,
                    max_length
                    + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                )
            )
            model.config.max_position_embeddings = max_length

        return model
