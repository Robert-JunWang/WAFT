import torch
from torch import nn, Tensor
from .feature_enhance import FeatureEnhance
from .transformer import TransformerDecoder, Block, MemEffAttention
from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DPTHead2
# from .dpt import DPTHead
import torch.nn.functional as F

class CoarseMatcher(nn.Module):
    def __init__(self,
                embed_dim=256,
                depth=3,
                dpt_dim=64,
                out_dim=2,
                upsample=None,
                dpt_dims= [48, 96, 192, 384]
                ):
        super().__init__()

        self.fe = FeatureEnhance(embed_dim)
        self.embed_dim = embed_dim

        # print('in_dim',in_dim)

        # self.proj = nn.Sequential(nn.Conv2d(in_dim, embed_dim, 1, 1), nn.BatchNorm2d(embed_dim))

        self.att_proj = nn.Conv2d(2*embed_dim, embed_dim, 1, 1)

        self.attentions = nn.Sequential(*[Block(embed_dim, 8, attn_class=MemEffAttention) for _ in range(depth)])
        
        self.dpt_head = DPTHead2(self.embed_dim, dpt_dim, out_channels=dpt_dims)
        self.upsample = upsample

        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(dpt_dim//2, dpt_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dpt_dim, out_dim, 1, padding=0, bias=True)
        )

    # def forward(self, f1c, f2c):
    def forward(self, feats):
        # x = self.proj(feats)

        f1c, f2c = feats.chunk(2)
        # print('f1c', f1c.shape)

        gp_posterior = self.fe(f1c, f2c)
        x = torch.cat((gp_posterior, f1c), dim = 1)
        x = self.att_proj(x)
        B,C,H,W = x.shape
        tokens = x.reshape(B,C,H*W).permute(0,2,1)

        # out = self.attentions(tokens)
        out = []
        for block in self.attentions:
            tokens = block(tokens)
            out.append(tokens.permute(0,2,1).reshape(B,C,H,W))

        x = [f1c, out[0], out[1], out[-1]]
        # print( [i.shape for i in x])


        out = self.dpt_head(x)
        if self.upsample is not None:
            out = F.interpolate(out,  scale_factor=(self.upsample, self.upsample), mode="bilinear", align_corners=True)

        out = self.flow_head(out)

        return out

if __name__ == '__main__':
    device = torch.device('cpu')
    resolution = 448
    x = torch.randn(2, 384, resolution//16, resolution//16).to(device)

    model = CoarseMatcher(384).eval()
         
    out = model(x)
    print('out', out.shape)