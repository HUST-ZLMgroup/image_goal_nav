"""Optional feature-conditioning and fusion layers."""

from typing import Type
import torch
import torch.nn as nn
import os
from src.residual_networks import ResNet, BasicBlock
from einops import reduce
from habitat import logger
from src.context_backbone import CompactContextBackbone
from src.context_relation import (
    ContextStyleEncoder,
    PairRelationClassifier,
    load_matching_weights,
)
from src.common.relation_runtime import prepare_relation_run
import torch.nn.functional as F
class FeatureModulation(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def __init__(self, channels=[32,64,128,256], res=[32,16,8,4], reduction="none"):
        super().__init__()
        self.reduction = reduction
        if self.reduction == "none":
            self.f = nn.ModuleList([
                nn.Conv2d(channels[i], channels[i], kernel_size=1, stride=1)
                for i in range(len(channels))
            ])
            self.h = nn.ModuleList([
                nn.Conv2d(channels[i], channels[i], kernel_size=1, stride=1)
                for i in range(len(channels))
            ])
        else:
            self.f = nn.ModuleList([
                nn.Linear(channels[i], channels[i])
                for i in range(len(channels))
            ])
            self.h = nn.ModuleList([
                nn.Linear(channels[i], channels[i])
                for i in range(len(channels))
            ])
        
    def forward(self, x, conds, i):
        if self.reduction == "none":
            gammas = self.f[i](conds).view_as(x)
            betas = self.h[i](conds).view_as(x)
        elif self.reduction == "global":
            gammas = self.f[i](reduce(conds, "b c h w -> b c", reduction="mean"))[:,:,None,None].expand_as(x)
            betas = self.h[i](reduce(conds, "b c h w -> b c", reduction="mean"))[:,:,None,None].expand_as(x)
        else:
            raise TypeError("not implemented")
        return (gammas * x) + betas


class ConditioningBackbone(ResNet):
    def forward(self, x):
        x = self.stem(x)

        interm_o = []
        for l in self.layers:
            x = l(x)
            interm_o.append(x)

        return x, interm_o
    
class ContextMapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        args = prepare_relation_run(arg_mode='test')
        self.encoder = CompactContextBackbone(args)
        # self.encoder = load_model(self.encoder, os.path.join(args.save_path, 'max_acc.pth'))
        self.conv = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=2, stride=1, padding=0)
    def forward(self,x):
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(x)
        x = self.conv(x)
        return x

class ContextPairEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = ContextStyleEncoder()
        load_matching_weights('/data/lpn/room-expert/checkpoints/feature_encoder.pkl', self.feature_encoder, strip='module.')
        # self.feature_encoder.load_state_dict(torch.load('/data/lpn/room-expert/checkpoints/feature_encoder.pkl',))
        self.relation_network = PairRelationClassifier()
        load_matching_weights('/data/lpn/room-expert/checkpoints/relation_network.pkl', self.relation_network, strip='module.')
        # self.relation_network.load_state_dict(torch.load('/data/lpn/room-expert/checkpoints/relation_network.pkl'))
    def forward(self,x_o, x_g):
        with torch.no_grad():
            feature1 = self.feature_encoder(x_o)
            feature2 = self.feature_encoder(x_g)
            relation_pairs = torch.cat((feature1,feature2),dim=1)
            relations = self.relation_network(relation_pairs)
            relations = torch.softmax(relations, dim=1)
        return relations


class ModulatedBackbone(ResNet):
    def __init__(self, reduction, film_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.film = FeatureModulation(reduction=reduction)
        self.film_layers = film_layers
        logger.info(f"filmed resnet encoder with layers: {film_layers}, reduction: {reduction}")
    
    def forward(self, x, x_cond):
        x = self.stem(x)

        for i,l in enumerate(self.layers):
            x = l(x)
            if i in self.film_layers:
                x = self.film(x, x_cond[i], i)

        return x


class ConditionalFusionEncoder(nn.Module):
    def __init__(
        self,
        reduction,
        film_layers,
        *args, **kwargs
    ):
        super().__init__()
        self.stem_o = ModulatedBackbone(reduction, film_layers, *args, **kwargs)
        self.stem_g = ConditioningBackbone(*args, **kwargs)
        self.room_encoder = ContextMapEncoder()
        self.conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        # Pairwise context conditioning can be substituted here if required.
        self.film_layers = film_layers
        self.final_spatial_compress = self.stem_o.final_spatial_compress
        # self.final_channels = self.stem_o.final_channels * 2
        self.final_channels = self.stem_o.final_channels
        
    def forward(self, x):
        b,c,h,w = x.shape
        x_o = x[:,:3,...]
        x_g = x[:,3:,...]
        r_o = self.room_encoder(x_o)
        r_g = self.room_encoder(x_g)
        r = torch.cat((r_o,r_g),dim=1)
        # r = self.room_encoder(x_o,x_g)
        x_g, x_cond = self.stem_g(x_g)
        x_o = self.stem_o(x_o, x_cond)
        x_o = self.conv(torch.cat((x_o,r),dim=1))

        # return torch.cat((x_o, x_g), dim=1)
        return x_o
        # return x


def fusion_resnet9(in_channels, base_planes, ngroups, film_reduction, film_layers):
    return ConditionalFusionEncoder(film_reduction, film_layers, 3, base_planes, ngroups, BasicBlock, [1, 1, 1, 1])


if __name__ == "__main__":
    mid_fusion_resnet = ConditionalFusionEncoder(
        3, 32, 16, BasicBlock, [1, 1, 1, 1]
    )
    dummy = torch.rand([2,6,128,128])
    out = mid_fusion_resnet(dummy)
    print(out.shape)
