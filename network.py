import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.attention import TransformerEncoderBlock,DualTransformerEncoderBlock
from models.common import MIM,SEM
from models.SEWeight import SEWeightModule as SE
from mamba_module import VSSLayer, VSSLayer_cross
from functools import partial
from tools.wavelet import create_wavelet_filter, wavelet_transform, inverse_wavelet_transform
import os
import matplotlib.pyplot as plt

# AMSA: Adaptive Multi-Scale Attention (自适应多尺度注意力机制)
class AMSA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, num_scales=2):
        super(AMSA, self).__init__()
        
        # 改进的通道注意力 - 多池化策略
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的多层感知机
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.GELU(),  # 使用 GELU 替代 ReLU
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 多尺度融合
        kernel_sizes = [3, 5]  # 多尺度卷积核
        self.spatial_convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.spatial_convs.append(
                nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=ks, padding=ks//2, bias=False),
                    nn.BatchNorm2d(1)
                )
            )
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(len(kernel_sizes), 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # 通道 - 空间交互
        self.gamma = nn.Parameter(torch.tensor(0.5))  # 可学习的权重
    
    def forward(self, x):
        # 通道注意力 - 融合平均和最大池化
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        channel_out = self.shared_mlp(avg_out + max_out)
        x_channel = channel_out * x
        
        # 空间注意力 - 多尺度特征提取
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_pool, max_pool], dim=1)
        
        # 多尺度空间特征
        spatial_features = []
        for spatial_conv in self.spatial_convs:
            spatial_features.append(spatial_conv(spatial_in))
        
        # 融合多尺度特征
        spatial_out = self.spatial_fusion(torch.cat(spatial_features, dim=1))
        
        # 自适应融合通道和空间注意力
        out = (1 - self.gamma) * x_channel + self.gamma * (spatial_out * x_channel)
        
        return out


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)


class WTAMSA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db4'):
        super(WTAMSA, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)

        self.base_conv = AMSA(in_channels)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        # 创建用于处理小波分解后子带特征的 AMSA 注意力机制列表
        # 每个 AMSA 处理一个小波分解层级，每个层级有 4 个子带 (LL、LH、HL、HH)，因此通道数扩展 4 倍
        self.wavelet_convs = nn.ModuleList(
            [AMSA(in_channels*4) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:,:,0,:,:]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


def d_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class Decoder_2(nn.Module):
    def __init__(self, dim):
        super(Decoder_2, self).__init__()

        self.dconv_up3 = d_conv(256 + 512, 256)
        self.dconv_up2 = d_conv(128 + 256, 128)
        self.dconv_up1 = d_conv(128 + 64, 64)

        self.dconv_last = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x, add_fea, H, W, encoder_fea):
        x1 = F.interpolate(x,scale_factor=2, mode='bilinear', align_corners=False)
        x1 = torch.cat([x1, encoder_fea[3]], dim=1)

        x2 = self.dconv_up3(x1)
        x2 = x2 + add_fea[1]
        x2 = F.interpolate(x2,scale_factor=2, mode='bilinear', align_corners=False)
        x2 = torch.cat([x2, encoder_fea[2]], dim=1)

        x3 = self.dconv_up2(x2)
        x3 = x3 + add_fea[0]
        x3 = F.interpolate(x3,scale_factor=2, mode='bilinear', align_corners=False)
        x3 = torch.cat([x3, encoder_fea[1]], dim=1)

        x4 = self.dconv_up1(x3)
        x4 = F.interpolate(x4,scale_factor=2, mode='bilinear', align_corners=False)
        x4 = torch.cat([x4, encoder_fea[0]], dim=1)

        x5 = self.dconv_last(x4)#x5:C=64
        x5 = F.interpolate(x5,scale_factor=2, mode='bilinear', align_corners=True)

        x_final = nn.Tanh()(self.conv_out(x5)) / 2 + 0.5

        return x_final


class FDSNet(nn.Module):

    def __init__(self):
        super(FDSNet, self).__init__()
        self.num_resnet_layers = 34
        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)

        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)

        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)

        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)

        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)

        self.encoder_thermal_conv1 = resnet_raw_model1.conv1
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool

        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool

        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4
        
        # 添加 Mamba 层用于增强特征提取
        self.mamba_thermal_layer1 = VSSLayer(dim=64, d_state=16)
        self.mamba_thermal_layer2 = VSSLayer(dim=128, d_state=16)
        self.mamba_thermal_layer3 = VSSLayer(dim=256, d_state=16)
        self.mamba_thermal_layer4 = VSSLayer(dim=512, d_state=16)
        
        self.mamba_rgb_layer1 = VSSLayer(dim=64, d_state=16)
        self.mamba_rgb_layer2 = VSSLayer(dim=128, d_state=16)
        self.mamba_rgb_layer3 = VSSLayer(dim=256, d_state=16)
        self.mamba_rgb_layer4 = VSSLayer(dim=512, d_state=16)
        
        # 添加交叉流 Mamba 层用于特征融合
        self.mamba_cross_layer2 = VSSLayer_cross(dim=256, d_state=16)
        self.mamba_cross_layer3 = VSSLayer_cross(dim=512, d_state=16)
        self.mamba_cross_layer4 = VSSLayer_cross(dim=1024, d_state=16)

        self.dim_in_channel_list=[64,128,256,512]
        self.seg_channel = 256
        self.sft_in_channel = [64,128,256,256]
        self.sft_out_channel = [64,128,256,512]
        self.high_fuse2 = SDFM_dual(self.dim_in_channel_list[3], self.seg_channel, self.sft_in_channel[3], self.sft_out_channel[3])
        self.high_fuse1 = SDFM_dual(self.dim_in_channel_list[2], self.seg_channel, self.sft_in_channel[2], self.sft_out_channel[2])
        self.low_fuse = SDFM_one(self.dim_in_channel_list[1], self.seg_channel, self.sft_in_channel[1], self.sft_out_channel[1])

        self.decoder1 = Decoder_2(self.dim_in_channel_list)
        self.con3x3_64 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ) for i in range(2)
        ])
        channel=[128,256,512]
        self.con3x3_down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=channel[i]*2, out_channels=channel[i], kernel_size=3,padding=1),
                nn.BatchNorm2d(channel[i]),
                nn.ReLU()
            )for i in range(2)])

        self.con3x3_add =nn.Sequential(
            nn.Conv2d(in_channels=channel[2], out_channels=channel[2], kernel_size=3,padding=1),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU()
            )
        
        # 添加小波变换卷积层用于特征增强
        self.wtconv_rgb1 = WTAMSA(64, 64, wt_levels=1)
        self.wtconv_thermal1 = WTAMSA(64, 64, wt_levels=1)
        self.wtconv_rgb2 = WTAMSA(64, 64, wt_levels=1)
        self.wtconv_thermal2 = WTAMSA(64, 64, wt_levels=1)
        self.wtconv_rgb3 = WTAMSA(128, 128, wt_levels=1)
        self.wtconv_thermal3 = WTAMSA(128, 128, wt_levels=1)
        self.wtconv_rgb4 = WTAMSA(256, 256, wt_levels=1)
        self.wtconv_thermal4 = WTAMSA(256, 256, wt_levels=1)
        self.wtconv_rgb5 = WTAMSA(512, 512, wt_levels=1)
        self.wtconv_thermal5 = WTAMSA(512, 512, wt_levels=1)
   

    def forward(self, rgb, depth, vis_sig, ir_seg,H,W,C):

        _,_,H_seg,W_seg = vis_sig.size()

        rgb = rgb
        thermal = depth[:, :1, ...]
        thermal = torch.cat([thermal, thermal, thermal], dim=1) 


        decoder_add_fea=[]
        encoder_fea_list=[]

        rgb = self.encoder_rgb_conv1(rgb)  #
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)
        rgb = self.wtconv_rgb1(rgb)  # 应用小波变换增强特征
        thermal = self.encoder_thermal_conv1(thermal)  #
        thermal = self.encoder_thermal_bn1(thermal)
        thermal = self.encoder_thermal_relu(thermal)
        thermal = self.wtconv_thermal1(thermal)  # 应用小波变换增强特征

        encoder_fea = torch.cat([rgb,thermal],dim = 1)
        encoder_fea = self.con3x3_64[0](encoder_fea)
        encoder_fea_list.append(encoder_fea)

        #maxpooling
        rgb = self.encoder_rgb_maxpool(rgb)
        thermal = self.encoder_thermal_maxpool(thermal)

        rgb1 = self.encoder_rgb_layer1(rgb)
        rgb1 = self.mamba_rgb_layer1(rgb1)  # 应用 Mamba 增强特征
        rgb1 = self.wtconv_rgb2(rgb1)  # 应用小波变换增强特征
        thermal1 = self.encoder_thermal_layer1(thermal)
        thermal1 = self.mamba_thermal_layer1(thermal1)  # 应用 Mamba 增强特征
        thermal1 = self.wtconv_thermal2(thermal1)  # 应用小波变换增强特征
        encoder_fea = torch.cat([rgb1,thermal1],dim = 1)
        encoder_fea = self.con3x3_64[1](encoder_fea)
        encoder_fea_list.append(encoder_fea)

        rgb2 = self.encoder_rgb_layer2(rgb1)
        rgb2 = self.mamba_rgb_layer2(rgb2)  # 应用 Mamba 增强特征
        rgb2 = self.wtconv_rgb3(rgb2)  # 应用小波变换增强特征
        thermal2 = self.encoder_thermal_layer2(thermal1)
        thermal2 = self.mamba_thermal_layer2(thermal2)  # 应用 Mamba 增强特征
        thermal2 = self.wtconv_thermal3(thermal2)  # 应用小波变换增强特征
        encoder_fea = torch.cat([rgb2,thermal2],dim = 1)
        encoder_fea = self.con3x3_down[0](encoder_fea)
        encoder_fea_list.append(encoder_fea)

        rgb2_sem, thermal2_sem, add_fea,vis_cross,ir_cross,vis_ploss1,ir_ploss1  = self.low_fuse(rgb2,thermal2,vis_sig,ir_seg,C)
        rgb2 = rgb2 + rgb2_sem
        thermal2 = thermal2 + thermal2_sem
        decoder_add_fea.append(add_fea)

        rgb3 = self.encoder_rgb_layer3(rgb2) 
        rgb3 = self.mamba_rgb_layer3(rgb3)  # 应用 Mamba 增强特征
        rgb3 = self.wtconv_rgb4(rgb3)  # 应用小波变换增强特征
        thermal3 = self.encoder_thermal_layer3(thermal2) 
        thermal3 = self.mamba_thermal_layer3(thermal3)  # 应用 Mamba 增强特征
        thermal3 = self.wtconv_thermal4(thermal3)  # 应用小波变换增强特征
        encoder_fea = torch.cat([rgb3,thermal3],dim = 1)
        encoder_fea = self.con3x3_down[1](encoder_fea)
        encoder_fea_list.append(encoder_fea)

        rgb3_sem, thermal3_sem, add_fea,vis_cross,ir_cross,vis_ploss2,ir_ploss2  = self.high_fuse1(rgb3,thermal3,vis_sig,ir_seg,C)
        rgb3 = rgb3 + rgb3_sem
        thermal3 = thermal3 + thermal3_sem
        decoder_add_fea.append(add_fea)
        
        rgb4= self.encoder_rgb_layer4(rgb3)
        rgb4 = self.mamba_rgb_layer4(rgb4)  # 应用 Mamba 增强特征
        rgb4 = self.wtconv_rgb5(rgb4)  # 应用小波变换增强特征
        thermal4 = self.encoder_thermal_layer4(thermal3) 
        thermal4 = self.mamba_thermal_layer4(thermal4)  # 应用 Mamba 增强特征
        thermal4 = self.wtconv_thermal5(thermal4)  # 应用小波变换增强特征

        rgb4_sem, thermal4_sem, add_fea,vis_cross,ir_cross,vis_ploss3,ir_ploss3  = self.high_fuse2(rgb4,thermal4,vis_sig,ir_seg,C)
        rgb4 = rgb4 + rgb4_sem
        thermal4 = thermal4 + thermal4_sem

        # 分离基础特征和细节特征
        # 基础特征：使用较低层级的特征（layer2输出）
        feature_V_B = rgb2
        feature_I_B = thermal2
        # 细节特征：使用较高层级的特征（layer4输出）
        feature_V_D = rgb4
        feature_I_D = thermal4

        fuse_fea = rgb4 + thermal4
        fuse_fea = self.con3x3_add(fuse_fea)

        decoder_fea = self.decoder1(fuse_fea,decoder_add_fea,H,W,encoder_fea_list)

        vis_ploss_all = vis_ploss1+vis_ploss2+vis_ploss3
        ir_ploss_all = ir_ploss1+ir_ploss2+ir_ploss3

        return  decoder_fea,vis_ploss_all,ir_ploss_all, feature_V_B, feature_I_B, feature_V_D, feature_I_D

class SDFM_one(nn.Module):
    def __init__(self, dim_in=32, dim_out=256, feature_channel=32,out_channel=32, nhead=8):
        super(SDFM_one,self).__init__()

        self.encoder_block_one_1 = TransformerEncoderBlock(dim_in,dim_out, nhead)
        self.encoder_block_one_2 = TransformerEncoderBlock(dim_in, dim_out, nhead)

        self.MIM_1=MIM(feature_channel=feature_channel,out_channel=out_channel)
        self.MIM_2=MIM(feature_channel=feature_channel,out_channel=out_channel)
        self.conv = nn.Conv2d(in_channels=feature_channel * 2, out_channels=out_channel , kernel_size=3, padding=1)

        self.se = SE(out_channel)
        self.conv1x1_in = nn.Conv2d(dim_in, dim_out, kernel_size=1)


    def forward(self, rgb, ir, vis_seg, ir_seg,C):
        _,c,h,w = rgb.size()
        _,_,h_seg,w_seg = vis_seg.size()
        if h > h_seg :
            n = h // h_seg
            n = float(n)
            vis_seg = F.interpolate(vis_seg, scale_factor=n)
            ir_seg = F.interpolate(ir_seg, scale_factor=n)
        elif h < h_seg :
            n = h_seg // h
            vis_seg=F.avg_pool2d(vis_seg,kernel_size=n,stride=n)
            ir_seg=F.avg_pool2d(ir_seg,kernel_size=n,stride=n)

        vis_cross = self.encoder_block_one_1(rgb, vis_seg,C)
        ir_cross = self.encoder_block_one_2(ir, ir_seg,C)

        if c != C:
            vis_prior = self.conv1x1_in(vis_cross)
            ir_prior = self.conv1x1_in(ir_cross)
        vis_ploss = F.mse_loss(vis_prior,vis_seg)
        ir_ploss = F.mse_loss(ir_prior,ir_seg)

        vis_out = self.MIM_1(vis_cross, ir_cross)
        ir_out = self.MIM_2(ir_cross, vis_cross)

        add_fea = torch.cat([vis_cross,ir_cross],dim=1)
        add_fea = self.conv(add_fea)
        add_fea_w = self.se(add_fea) 
        add_fea = add_fea * add_fea_w

        return vis_out,ir_out,add_fea,vis_cross,ir_cross,vis_ploss,ir_ploss

class SDFM_dual(nn.Module):
    def __init__(self, dim_in=32, dim_out=256, feature_channel=32,out_channel=32, nhead=8):
        super(SDFM_dual,self).__init__()

        self.encoder_block_3 = DualTransformerEncoderBlock(dim_in,dim_out, nhead)
        self.encoder_block_4 = DualTransformerEncoderBlock(dim_in,dim_out, nhead)

        self.SEM_1 = SEM(feature_channel=feature_channel, out_channel=out_channel)
        self.SEM_2 = SEM(feature_channel=feature_channel, out_channel=out_channel)

        # self.conv1 = nn.Conv2d(in_channels=feature_channel * 2, out_channels=out_channel , kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel , kernel_size=3, padding=1)

        # self.se1 = SE(out_channel)
        self.se2 = SE(out_channel)
        self.conv1x1_in = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, rgb, ir, vis_seg, ir_seg,C):
        _,c,h,w = rgb.size()
        _,_,h_seg,w_seg = vis_seg.size()
        if h > h_seg :
            n = h // h_seg
            vis_seg = F.interpolate(vis_seg, scale_factor=float(n))
            ir_seg = F.interpolate(ir_seg, scale_factor=float(n))

        elif h < h_seg :
            n = h_seg // h
            vis_seg=F.avg_pool2d(vis_seg,kernel_size=(n,n),stride=(n,n))
            ir_seg=F.avg_pool2d(ir_seg,kernel_size=(n,n),stride=(n,n))

        vis_cross = self.encoder_block_3(rgb, vis_seg,C)
        ir_cross = self.encoder_block_4(ir, ir_seg,C)

        if c != C:
            vis_prior = self.conv1x1_in(vis_cross)
            ir_prior = self.conv1x1_in(ir_cross)
        else:
            vis_prior = vis_cross
            ir_prior = ir_cross
        vis_ploss = F.mse_loss(vis_prior,vis_seg)
        ir_ploss = F.mse_loss(ir_prior,ir_seg)

        vis_out = self.SEM_1(vis_cross, vis_seg)
        ir_out = self.SEM_2(ir_cross, ir_seg)

        add_fea = torch.cat([vis_cross, ir_cross], dim=1)
        add_fea = self.conv2(add_fea)
        add_fea_w = self.se2(add_fea) 
        add_fea = add_fea * add_fea_w

        return vis_out,ir_out,add_fea,vis_cross,ir_cross,vis_ploss,ir_ploss
