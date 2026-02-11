import torch
import torch.nn as nn
import timm
import numpy as np
import torchvision
import torch.nn.functional as F

import sys
import os
from model.backbone.head import DPTHead

REPO_DIR = "thirdparty/dinov3"

'''
Please request the weights from https://ai.meta.com/dinov3/ and fill in the links below.
'''

WEIGHTS_URLS = {
    'vits': 'weights/dino3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    'vitb': 'weights/dino3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    'vitl': 'weights/dino3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
}

class DinoV3Feature(nn.Module):
    def __init__(self, model_name='vits', lvl=-3):
        super().__init__()
        self.dpt_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], "dim": 1024},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], "dim": 768},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], "dim": 384},
        }
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23]
        }
        self.model_name = model_name
        self.encoder = self.freeze_(torch.hub.load(REPO_DIR, f"dinov3_{model_name}16", source='local', weights=WEIGHTS_URLS[model_name]))
        self.embed_dim = self.dpt_configs[model_name]['dim']
        self.output_dim = self.dpt_configs[model_name]['features']
        self.out_channels = self.dpt_configs[model_name]['out_channels']
        self.dpt_head = DPTHead(self.embed_dim, features=self.output_dim, out_channels=self.out_channels, lvl=lvl)

    def freeze_(self, model):
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        return model

    def forward(self, x):
        """
        @x: (B,C,H,W)
        """
        h, w = x.shape[-2], x.shape[-1]
        features = self.encoder.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.model_name], return_class_token=True)
        patch_size = self.encoder.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        outs = self.dpt_head.forward(features, patch_h, patch_w)
        final = F.interpolate(outs[0], size=(h//2, w//2), mode='bilinear', align_corners=True)
        return final


class DinoV3Feature2(nn.Module):
    def __init__(self, model_name='vits', output_dim=256, lvl=-3):
        super().__init__()
        self.dpt_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], "dim": 1024},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], "dim": 768},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], "dim": 384},
        }
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23]
        }
        self.model_name = model_name
        self.encoder = self.freeze_(torch.hub.load(REPO_DIR, f"dinov3_{model_name}16", source='local', weights=WEIGHTS_URLS[model_name]))
        self.embed_dim = self.dpt_configs[model_name]['dim']
        self.output_dim = output_dim
        self.out_channels = self.dpt_configs[model_name]['out_channels']
        # self.dpt_head = DPTHead(self.embed_dim, features=self.output_dim, out_channels=self.out_channels, lvl=lvl)
        self.proj = nn.Conv2d(self.embed_dim, self.output_dim, 1, 1)
        # self.encoder.rope_embed = None

    def freeze_(self, model):
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        return model

    def forward(self, x):
        """
        @x: (B,C,H,W)
        """
        # h, w = x.shape[-2], x.shape[-1]
        b, c, h, w = x.shape
        # out = self.encoder.forward_features(x)
        # out = out['x_norm_patchtokens']
        # out = out.permute(0,2,1).view(b, -1, h//16, w//16)
        # print('dino3', out.shape)
        features = self.encoder.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.model_name], return_class_token=True)
        out = features[-1][0]
        # print('input', x.shape, 'dino3', len(features), out.shape)
        # print('input', x.shape, 'dino3', len(features), [o.shape for o in out])
        out = out.permute(0,2,1).view(b, -1, h//16, w//16)

        return self.proj(out)

        # features = self.encoder.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.model_name], return_class_token=True)
        # patch_size = self.encoder.patch_size
        # # print(patch_size, 'features', [f.shape for f in features] )
        # # return features[-1]
        # # patch_h, patch_w = h // patch_size, w // patch_size
        # # outs = self.dpt_head.forward(features, patch_h, patch_w)
        # print('dino3', outs[0].shape)
        # return outs[0]
        # final = F.interpolate(outs[0], size=(h//2, w//2), mode='bilinear', align_corners=True)
        # return final

if __name__ == '__main__':
    model = DinoV3Feature(model_name='vits', lvl=-3)
    input = torch.randn(1, 3, 512, 768)
    output = model(input)
    print("Output shape: ", output.shape)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        with_flops=True) as prof:
            output = model(input)
    
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    events = prof.events()
    forward_MACs = sum([int(evt.flops) for evt in events])
    print("forward MACs: ", forward_MACs / 2 / 1e9, "G")