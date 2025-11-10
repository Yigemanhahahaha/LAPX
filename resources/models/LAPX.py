import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


class DW_Bottleneck(nn.Module):
    expansion = 2  
    def __init__(self, in_channels, neck_channels, stride=1, downsample=None, GroupConv=False, SoftGate=False):
        super().__init__()
        # channel shuffle group number
        self.groups = 2
        self.group_conv = GroupConv
        #
        self.downsample = downsample
        # ELU activation function
        self.elu = nn.ELU(inplace=True)
        #
        self.bn1 = nn.BatchNorm2d(in_channels)
        #
        if not GroupConv:
            self.pw_conv1 = nn.Conv2d(in_channels, neck_channels, kernel_size=1, bias=False)  
        else:
            self.pw_conv1 = nn.Conv2d(in_channels, neck_channels, kernel_size=1, groups=self.groups, bias=False)  
        #
        self.bn2 = nn.BatchNorm2d(neck_channels)
        self.dw_conv = nn.Conv2d(neck_channels, neck_channels, kernel_size=3, stride=stride,
                                  padding=1, groups=neck_channels, bias=False)  
        self.bn3 = nn.BatchNorm2d(neck_channels)
        #
        if not GroupConv:
            self.pw_conv2 = nn.Conv2d(neck_channels, neck_channels*2, kernel_size=1, bias=True)  
        else:
            self.pw_conv2 = nn.Conv2d(neck_channels, neck_channels*2, kernel_size=1, groups=self.groups, bias=True)  
        #
        # soft gate residual
        self.use_soft_gate = SoftGate
        if SoftGate:
            self.alpha = nn.Parameter(torch.zeros((1, neck_channels*2, 1, 1)))

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.shape
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.elu(out)
        out = self.pw_conv1(out)
        if self.group_conv:
            out = self.channel_shuffle(out, self.groups)  

        out = self.bn2(out)
        out = self.elu(out)  
        out = self.dw_conv(out)

        out = self.bn3(out)
        out = self.elu(out)  
        out = self.pw_conv2(out)
        if self.group_conv:
            out = self.channel_shuffle(out, self.groups)   

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.use_soft_gate:
            out = out + self.alpha*residual
        else:
            out = out + residual

        return out



class FC_Block(nn.Module):
    def __init__(self, in_channels, out_channels, GroupConv=False, act=False):
        super().__init__()
        # channel shuffle group number
        self.groups = 2
        # 
        self.act = act
        #
        self.use_group_conv = GroupConv
        #
        if act:
            #
            if not GroupConv:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=self.groups, bias=False)  
            #
            self.bn = nn.BatchNorm2d(out_channels)
            self.elu = nn.ELU(inplace=True)
        else:
            #
            if not GroupConv:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)  
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=self.groups, bias=True)  
            #

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.shape
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

    def forward(self, x):
        out = self.conv(x)
        if self.act:
            out = self.bn(out)
            out = self.elu(out)
        if self.use_group_conv:
            out = self.channel_shuffle(out, self.groups)

        return out


class ECA(nn.Module):
    def __init__(self, channels, k_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)  # [B, C, 1, 1]
        max_out = self.max_pool(x)  # [B, C, 1, 1]

        y = avg_out + max_out       
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)

        return x * y
        

class ECA_CBAM(nn.Module):
    def __init__(self, channels, spatial_kernel_size=7, eca_kernel_size=7):
        super().__init__()
        self.channel_attention = ECA(channels, k_size=eca_kernel_size)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=spatial_kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention (using ECA)
        x = self.channel_attention(x)
        # Spatial attention
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_avg_out = torch.cat([max_out, avg_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(max_avg_out))

        out = x * spatial_att
        
        return out



class NonLocal(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.theta = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.phi = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.g = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.out_proj = nn.Conv2d(channels // 8, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
        #
        self.enabled = False

    def forward(self, x):
        
        if not self.enabled:
            return x
            
        B, C, H, W = x.size()
        N = H * W
        C_ = C // 8

        theta_x = self.theta(x).view(B, C_, N).permute(0, 2, 1)     # [B, N, C']
        phi_x = self.phi(x).view(B, C_, N)                          # [B, C', N]
        g_x = self.g(x).view(B, C_, N)                              # [B, C', N]

        attention = torch.bmm(theta_x, phi_x)                # [B, N, N]
        attention = self.softmax(attention)

        y = torch.bmm(attention, g_x.permute(0, 2, 1))             # [B, N, C']
        y = y.permute(0, 2, 1).contiguous().view(B, C_, H, W)      # [B, C', H, W]

        out = self.out_proj(y)                                     # [B, C, H, W]
        out = self.gamma * out + x

        return out

        

class ECA_NonLocal(nn.Module):
    def __init__(self, channels, eca_kernel_size=7):
        super().__init__()
        self.channel_attention = ECA(channels, k_size=eca_kernel_size)
        self.position_attention = NonLocal(channels)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.position_attention(x)

        return x



class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.spatial_kernel_size = spatial_kernel_size

        # Channel Attention Module
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention Module
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=spatial_kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        max_out = max_out.view(max_out.size(0), -1)
        avg_out = avg_out.view(avg_out.size(0), -1)
        out = self.fc(max_out) + self.fc(avg_out)
        out = out.view(x.size(0), self.channels, 1, 1)
        channel_att = x * out.expand_as(x)

        # Spatial attention
        max_out = torch.max(channel_att, 1, keepdim=True)[0]
        avg_out = torch.mean(channel_att, 1, keepdim=True)
        out = torch.cat([max_out, avg_out], dim=1)
        out = self.conv(out)
        spatial_att = self.sigmoid(out)
        final = channel_att * spatial_att.expand_as(channel_att)

        return final
        






class Hourglass(nn.Module):
    def __init__(self, block, blocks_number, neck_channels, depth, 
                 use_soft_gate=False, use_group_conv = False,
                 attention_block=None, self_attention_block=None, 
                 use_self_attention=False): # depth=4
        super().__init__()
        self.depth = depth
        self.block = block
        # a hourglass with depth 4
        self.hg = self._make_hour_glass(block, blocks_number, neck_channels, depth, group_conv=use_group_conv, soft_gate=use_soft_gate)
        # attention module at the lowest reoslution
        if self_attention_block and use_self_attention:
               self.attention = self_attention_block(neck_channels*block.expansion)
        elif attention_block is not None:
            self.attention = attention_block(neck_channels*block.expansion)
        else:
            self.attention = None
            
    def _make_residual(self, block, blocks_number, neck_channels, group_conv, soft_gate):
        layers = []
        for i in range(0, blocks_number):
            layers.append(block(neck_channels*block.expansion, neck_channels, GroupConv=group_conv, SoftGate=soft_gate)) 
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, blocks_number, neck_channels, depth, group_conv, soft_gate):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, blocks_number, neck_channels, group_conv, soft_gate))
            if i == 0:
                res.append(self._make_residual(block, blocks_number, neck_channels, group_conv, soft_gate))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):  # n=4
        up1 = self.hg[n-1][0](x)

        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1) 
        else:
            low2 = self.hg[n-1][3](low1)
            if self.attention is not None:
                low2 = self.attention(low2)

        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)  

        out = up1 + up2
        
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, fc_block, use_stem_GroupConv=False, use_hg_GroupConv=False, 
                       stem_SoftGate=False, hg_SoftGate=False,
                       hourglass_channel_width=256, hourglass_depth=4, num_stacks=2, num_blocks=1, 
                       stem_attention=None,
                       backbone_attention=None,
                       hg_neck_attention=None,
                       hg_neck_self_attention=None, self_attention_stages=[],
                       num_classes=16):
        super().__init__()
        #
        self.in_channels = hourglass_channel_width//4
        self.num_feats = hourglass_channel_width//2
        self.num_stacks = num_stacks
        
        # Stem
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.elu = nn.ELU(inplace=True)
        self.layer1 = self._make_residual(block, neck_channels=self.in_channels, use_group_conv=use_stem_GroupConv, soft_gate=stem_SoftGate)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer2 = self._make_residual(block, neck_channels=self.in_channels, use_group_conv=use_stem_GroupConv, soft_gate=stem_SoftGate) # after layer2 self.in_channels=256
        self.layer3 = self._make_residual(block, neck_channels=self.num_feats, use_group_conv=use_stem_GroupConv, soft_gate=stem_SoftGate)
        
        # a stem attention module
        if stem_attention is not None:
            self.stem_attention = stem_attention(self.num_feats * block.expansion)   # a attention module refines features from the stem
        else:
            self.stem_attention = None
        
        # backbone attention modules after every hourglass
        if backbone_attention is not None:
            self.backbone_attention = nn.ModuleList([backbone_attention(self.num_feats * block.expansion) for _ in range(num_stacks)])
        else:
            self.backbone_attention = None
        
        # Hourglasses
        ch = self.num_feats*block.expansion # 256
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            self_attention_flag = (i in self_attention_stages)
            hg.append(Hourglass(block, num_blocks, self.num_feats, hourglass_depth, 
                                use_soft_gate=hg_SoftGate, use_group_conv=use_hg_GroupConv,
                                attention_block=hg_neck_attention, self_attention_block=hg_neck_self_attention, 
                                use_self_attention=self_attention_flag))
            res.append(self._make_residual(block, self.num_feats, num_blocks, use_group_conv=use_hg_GroupConv, soft_gate=stem_SoftGate))  # a 256-128-256 bottleneck
            fc.append(fc_block(ch, ch, GroupConv=use_hg_GroupConv, act=True)) 
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:                                                  # 在第一阶段，将预测的结果返回输入第二个glass
                fc_.append(fc_block(ch, ch, GroupConv=use_hg_GroupConv, act=False))
                score_.append(nn.Conv2d(num_classes, ch,                          # 从num_classes到256
                              kernel_size=1, bias=True))
                
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, neck_channels, blocks_num=1, stride=1, use_group_conv=False, soft_gate=False):
        downsample = None
        if stride != 1 or self.in_channels != neck_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, neck_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.in_channels, neck_channels, stride, downsample, GroupConv=use_group_conv, SoftGate=soft_gate))
        self.in_channels = neck_channels * block.expansion
        for i in range(1, blocks_num):
            layers.append(block(self.in_channels, neck_channels, GroupConv=use_group_conv, SoftGate=soft_gate))

        return nn.Sequential(*layers)

    def _make_fc(self, in_channels, out_channels):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        return nn.Sequential(
            conv,
            bn,
            self.elu,
        )

    def forward(self, x):
        out = [] 
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # a stem attention module
        if self.stem_attention is not None:
           x = self.stem_attention(x)       
        # Hourglasses Backbone
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            # backbone attention modules
            if self.backbone_attention is not None:
                y = self.backbone_attention[i](y)
            #
            score = self.score[i](y)
            out.append(score) 
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out


def pose_net():
    return HourglassNet(DW_Bottleneck, FC_Block, use_stem_GroupConv=False, use_hg_GroupConv=False, 
                        stem_SoftGate=False, hg_SoftGate=True,
                        hourglass_channel_width=208, hourglass_depth=4, num_stacks=3, num_blocks=1, 
                        stem_attention=ECA_CBAM,
                        backbone_attention=ECA_CBAM,
                        hg_neck_attention=ECA,
                        hg_neck_self_attention=ECA_NonLocal, self_attention_stages=[0, 2],
                        num_classes=16)
