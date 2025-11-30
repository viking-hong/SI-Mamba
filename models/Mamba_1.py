import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from mamba.mamba_ssm.modules.mamba_simple import Mamba
from geoseg.modules.vim import VisionMamba
from geoseg.modules.vim_vi import VisionMamba1
import math
import numpy as np
import os
import matplotlib.pyplot as plt



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_channels = max(1, in_planes // ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_planes, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)
        max_out = self.mlp(self.max_pool(x))  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)
        out = avg_out + max_out
        return self.sigmoid(out)
 
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1)     # 在通道上拼接两个矩阵:(B,2,H,W)
        x = self.conv1(x)                                   # 通过卷积层得到注意力权重:(B,2,H,W)-->(B,1,H,W)
        return self.sigmoid(x)
 
 
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
 
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
 
    def forward(self, x):
        out = x * self.ca(x)                # 通过通道注意力机制得到的特征图,x:(B,C,H,W),ca(x):(B,C,1,1),out:(B,C,H,W)
        result = out * self.sa(out)           # 通过空间注意力机制得到的特征图,out:(B,C,H,W),sa(out):(B,1,H,W),result:(B,C,H,W)
        return result


class MambaLayer(nn.Module):
    def __init__(self, img_size=16, dim=4, d_state=2, d_conv=2, expand=1):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.mamba = VisionMamba(
            channels=dim,  # Model dimension d_model
            embed_dim=dim,
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,
            img_size=img_size
        ).cuda()
        self.conv1 = ConvBNReLU(dim, dim, kernel_size=3)

    def forward(self, ah, x):
        #save_feature_map(x, 'data/Sentinel-2/view/test/1.jpg')
        x_mamba, extra_feature = self.mamba(ah, x)
        x = self.conv1(x_mamba)
        ah = extra_feature + ah
        return x, ah


class MambaBlock(nn.Module):
    def __init__(self, img_size=16, in_channels=32):
        super().__init__()  
        self.local1 = ConvBN(in_channels, in_channels, kernel_size=1)
        self.local2 = ConvBN(in_channels, in_channels, kernel_size=3)
        self.local3 = ConvBN(in_channels, in_channels, kernel_size=3)
        self.conv = ConvBNReLU(in_channels, in_channels, kernel_size=3)
        self.mamba = MambaLayer(img_size=img_size, dim=in_channels)
        self.cbam = CBAM(in_planes=in_channels)

    def forward(self, ah, x):
        h, w = x.size()[-2:]
        ah = nn.functional.interpolate(ah, size=(h, w), mode='bilinear', align_corners=False)

        local1 = self.local1(x)
        local2 = self.local2(local1)
        local = self.local3(self.cbam(local1 + local2))
        x, ah = self.mamba(ah,x)
        x = self.conv(x) + self.conv(local1 + local2 + local)
        
        #save_feature_map(x, 'data/Sentinel-2/view/test/2.jpg')
        return x, ah

class MambaLayer1(nn.Module):
    def __init__(self, img_size=16, dim=4, d_state=2, d_conv=2, expand=1):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.mamba = VisionMamba1(
            channels=dim,  # Model dimension d_model
            embed_dim=dim,
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,
            img_size=img_size
        ).cuda()
        self.conv1 = ConvBNReLU(dim, dim, kernel_size=3)

    def forward(self, x):
        x_mamba = self.mamba(x)
        x = self.conv1(x_mamba)
        return x

class MambaBlock1(nn.Module):
    def __init__(self, img_size=16, in_channels=32):
        super().__init__()  
        self.local1 = ConvBN(in_channels, in_channels, kernel_size=1)
        self.local2 = ConvBN(in_channels, in_channels, kernel_size=3)
        self.local3 = ConvBN(in_channels, in_channels, kernel_size=3)
        self.conv = ConvBNReLU(in_channels, in_channels, kernel_size=3)
        self.mamba = MambaLayer1(img_size=img_size, dim=in_channels)
        self.cbam = CBAM(in_planes=in_channels)

    def forward(self, x):
        local1 = self.local1(x)
        local2 = self.local2(local1)
        local = self.local3(self.cbam(local1 + local2))
        x = self.conv(self.mamba(x)) + self.conv(local1 + local2 + local)
        #save_feature_map(x, 'data/Sentinel-2/view/test/2.jpg')
        return x

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # 多尺度深度可分离卷积
        # 改进1：使用自适应感受野
        self.conv1 = nn.Conv2d(in_features, in_features, 3, padding=1, groups=8)
        self.conv2 = nn.Conv2d(in_features, in_features, 3, padding=2, dilation=2, groups=8)
        self.conv3 = nn.Conv2d(in_features, in_features, 3, padding=3, dilation=3, groups=8)
        
        # 改进2：自适应特征融合
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features*3, in_features//2, 1),
            nn.ReLU(),
            nn.Conv2d(in_features//2, 3, 1),
            nn.Softmax(dim=1)
        )
        
        # 改进3：增强通道交互
        self.projector = nn.Sequential(
            nn.Conv2d(in_features, in_features*2, 1),
            nn.GELU(),
            nn.Conv2d(in_features*2, in_features, 1)
        )
        
    def forward(self, x):
        c1 = F.gelu(self.conv1(x))
        c2 = F.gelu(self.conv2(x))
        c3 = F.gelu(self.conv3(x))
        
        # 自适应融合
        attn_weights = self.attn(torch.cat([c1, c2, c3], dim=1))
        fused = attn_weights[:,0:1]*c1 + attn_weights[:,1:2]*c2 + attn_weights[:,2:3]*c3
        
        # 增强通道交互
        return x + self.projector(fused)


class Mona(nn.Module):
    def __init__(self, in_dim, factor=2):
        super().__init__()
        self.in_dim = in_dim
        self.factor = factor
        self.inner_dim = in_dim // factor 
        
        self.project1 = nn.Linear(in_dim, self.inner_dim)  # 降维投影
        self.project2 = nn.Linear(self.inner_dim, in_dim) 
        
        self.nonlinear = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(in_dim)
        
        self.adapter_conv = MonaOp(self.inner_dim)
        
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)  # 归一化权重
        self.gammax = nn.Parameter(torch.ones(in_dim))        # 原始输入权重
    
    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)

        x_seq = self.norm(x_seq) * self.gamma + x_seq * self.gammax
        project1 = self.project1(x_seq)
        project1_spatial = project1.reshape(B, H, W, self.inner_dim).permute(0, 3, 1, 2)

        conv_out = self.adapter_conv(project1_spatial)
        conv_seq = conv_out.permute(0, 2, 3, 1).reshape(B, H*W, self.inner_dim)

        nonlinear = self.nonlinear(conv_seq)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        output = project2.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return identity + output

class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        
    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        res = self.pre_conv(res)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()
        self.mona = Mona(in_dim=decode_channels).cuda()

    def forward(self, x):
        #save_feature_map(x, 'data/Sentinel-2/view/test/3.jpg')
        #x = self.mona(x)
        #save_feature_map(x, 'data/Sentinel-2/view/test/4.jpg')
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)
        #save_feature_map(x, 'data/Sentinel-2/view/test/5.jpg')
        return x


class rgb_indices(nn.Module):
    def __init__(self, eps=1e-8):
        super(rgb_indices, self).__init__()
        self.conv1 =ConvBN(in_channels=3, out_channels=64, kernel_size=3)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = ConvBN(in_channels=64, out_channels=3, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(in_features=3, out_features=3, bias=True)
        self.eps = eps

    def forward(self, x):
        weights = self.conv1(x)
        weights = self.avg(weights)
        weights = self.conv2(weights)
        weights = self.sig(weights)
        weights = 0.8 + 0.4 * weights

        B, C, H, W = weights.shape
        weights = weights.view(B, C)       
        weights = self.fc(weights)     
        weights = torch.clamp(weights, 0.8, 1.2)   
        weights = weights.view(B, C, H, W)

        x_w = x * weights

        r = x_w[:, [0], :, :]  # [B, 1, H, W]
        g = x_w[:, [1], :, :]
        b = x_w[:, [2], :, :]
        r = r/255; g = g/255; b = b/255

        # exg = 2*g - r - b
        # ndwi = (g - b) / (g + b)#归一化水体指数（RGB）
        # exb =  1.3*b - g #超蓝指数
        si = (r - b) / (r + b)#雪盖指数
        return si



class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.p3 = WF(encoder_channels[-2], decode_channels)
        self.p2 = WF(encoder_channels[-3], decode_channels)
        self.p1 = WF(encoder_channels[-4], decode_channels)

        self.b1 = MambaBlock(img_size=64, in_channels=decode_channels)
        self.b2 = MambaBlock(img_size=32, in_channels=decode_channels)
        self.b3 = MambaBlock(img_size=16, in_channels=decode_channels)

        # self.b0 = MambaBlock(img_size=128, in_channels=decode_channels)

        self.head = FeatureRefinementHead(in_channels=decode_channels, decode_channels=decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, ah, res1, res2, res3, res4, h, w):
        x, ah3 = self.b3(ah, self.pre_conv(res4))
        x = self.p3(x, res3)
        x, ah2 = self.b2(ah, x)

        x = self.p2(x, res2)
        x, ah1 = self.b1(ah, x)

        x = self.p1(x, res1)
        # x = self.b0(ah, x)
        x = self.head(x)

        ah1 = F.interpolate(ah1, size=(h, w), mode='bilinear', align_corners=False)
        ah2 = F.interpolate(ah2, size=(h, w), mode='bilinear', align_corners=False)
        ah3 = F.interpolate(ah3, size=(h, w), mode='bilinear', align_corners=False)
        ah = ah1 + ah2 + ah3

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = self.segmentation_head(x)

        return x, ah

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Decoder1(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 num_classes=6):
        super(Decoder1, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.p3 = WF(encoder_channels[-2], decode_channels)
        self.p2 = WF(encoder_channels[-3], decode_channels)
        self.p1 = WF(encoder_channels[-4], decode_channels)

        self.b1 = MambaBlock1(img_size=64, in_channels=decode_channels)
        self.b2 = MambaBlock1(img_size=32, in_channels=decode_channels)
        self.b3 = MambaBlock1(img_size=16, in_channels=decode_channels)

        # self.b0 = MambaBlock(img_size=128, in_channels=decode_channels)

        self.head = FeatureRefinementHead(in_channels=decode_channels, decode_channels=decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b3(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b2(x)

        x = self.p2(x, res2)
        x = self.b1(x)

        x = self.p1(x, res1)
        # x = self.b0(ah, x)
        x = self.head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = self.segmentation_head(x)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat
    
def save_feature_map(feature, save_path):
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().cpu().numpy()
    if feature.ndim == 4:
        feature = feature[0]
    fmap = np.mean(feature, axis=0)
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

    plt.imshow(fmap, cmap='jet')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

class SIMamba(nn.Module):
    def __init__(self,
                 decode_channels=32,
                 dropout=0.1,
                 backbone_name='resnet18.fb_swsl_ig1b_ft_in1k',#'resnet18.fb_swsl_ig1b_ft_in1k',#'swsl_resnet18',
                 pretrained=True,
                 num_classes=6,
                 index=True
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.idx = rgb_indices()
        self.index = index
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, num_classes)
        self.decoder1 = Decoder1(encoder_channels, decode_channels, dropout, num_classes)

        if self.training:
            self.aux_head = AuxHead(1, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.index:
            if self.training:
                ah = self.idx(x)
                x1, ah1 = self.decoder(ah, res1, res2, res3, res4, h, w)
                ah2 = self.aux_head(ah1)
                return x1, ah2
            else:
                ah = self.idx(x)
                x1, _ = self.decoder(ah, res1, res2, res3, res4, h, w)
                return x1
        else:
            x1 = self.decoder1(res1, res2, res3, res4, h, w)
            return x1
