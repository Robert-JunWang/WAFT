import numpy as np
from loguru import logger    
from typing import Callable, List, Any, Tuple, Dict

import torch
from torch import nn, Tensor

from .linear_attention import Attention, crop_feature, pad_feature
from .position_encoding import RoPEPositionEncodingSine
from .utils import get_autocast_params

from einops.einops import rearrange

class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size0=4,
                 agg_size1=4,
                 no_flash=True,
                 rope=False,
                 npe=None,
                 fp32=False,
                 ):
        super(AG_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False, groups=d_model) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature
        query, source = self.norm1(self.aggregate(x).permute(0,2,3,1)), self.norm1(self.max_pool(source).permute(0,2,3,1)) # [N, H, W, C]
        if x_mask is not None:
            x_mask, source_mask = map(lambda x: self.max_pool(x.float()).bool(), [x_mask, source_mask])
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        # m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        m = self.attention.export(query, key, value)
        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m



class AG_SAEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size=4,
                 no_flash=True,
                 rope=False,
                 npe=None,
                 fp32=False,
                 ):
        super(AG_SAEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size = agg_size
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=agg_size, padding=0, stride=agg_size, bias=False, groups=d_model) if self.agg_size != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size, stride=self.agg_size) if self.agg_size != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, x_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()

        # Aggragate feature
        query, source = self.norm1(self.aggregate(x).permute(0,2,3,1)), self.norm1(self.max_pool(x).permute(0,2,3,1)) # [N, H, W, C]
        if x_mask is not None:
            x_mask = self.max_pool(x_mask.float()).bool()
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        # m = self.attention(query, key, value, q_mask=x_mask, kv_mask=x_mask)
        m = self.attention.export(query, key, value)
        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size, w=W0 // self.agg_size) # [N, C, H0, W0]
        if self.agg_size != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m

class SelfAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 no_flash=True,
                 rope=False,
                 npe=None,
                 fp32=False,
                 up_scale=None
                 ):
        super(SelfAttentionLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.up_scale = up_scale

        self.rope = rope

        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, x_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
        """
        bs, C, H0, W0 = x.size()

        # Aggragate feature
        query = self.norm1(x.permute(0,2,3,1))

        query, key, value = self.q_proj(query), self.k_proj(query), self.v_proj(query)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        # m = self.attention(query, key, value, q_mask=x_mask, kv_mask=x_mask)
        m = self.attention.export(query, key, value)
        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0, w=W0) # [N, C, H0, W0]
        if self.up_scale:
            x = torch.nn.functional.interpolate(m, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
            m = torch.nn.functional.interpolate(m, scale_factor=self.up_scale, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m
    

class AG_RoPE_EncoderLayer2(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                #  agg_size0=4,
                #  agg_size1=4,
                 no_flash=True,
                 rope=False,
                 attention_only=True,
                 npe=None,
                 fp16=False
                 ):
        super(AG_RoPE_EncoderLayer2, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.rope = rope
        self.attention_only = attention_only

        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=fp16)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        
        self.attention = Attention(no_flash, self.nhead, self.dim, not fp16)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, H0, W0, C = x.size()
        query, source = self.norm1(x), self.norm1(source) # [N, H, W, C]
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        # m = self.attention.export(query, key, value)
        m = self.attention(query, key, value)
        # m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        m = self.merge(m.reshape(bs, H0, W0, self.nhead*self.dim)) # [N, L, C]

        if not self.attention_only:
            # feed-forward network
            m = self.mlp(torch.cat([x, m], dim=3)) # [N, H0, W0, C]
            m = self.norm2(m)

        return x + m

class FeatureEnhance(nn.Module):
    def __init__(self, dim, temperature = 0.2, amp_dtype = torch.float16):
        super().__init__()

        self.dim = dim
        num_heads = 8
        self.amp = True
        self.amp_dtype = amp_dtype
        fp16 = amp_dtype == torch.float16

        self.self_attn = AG_RoPE_EncoderLayer2(dim, num_heads,
                npe = None,
                rope = False, 
                no_flash=True, fp16=fp16)
        self.cross_attn = AG_RoPE_EncoderLayer2(dim, num_heads,
                npe = None,
                rope = False, 
                no_flash=True, fp16=fp16)
        


    def forward(self, x, y, **kwargs):
        x = x.permute(0,2,3,1)
        y = y.permute(0,2,3,1)

        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            x = self.self_attn(x, x)
            y = self.self_attn(y, y)

            out = self.cross_attn(y, x)

            # return x.permute(0, 3, 1, 2), out.permute(0, 3, 1, 2) # [N, C, H0, W0]
            return out.permute(0, 3, 1, 2) # [N, C, H0, W0]