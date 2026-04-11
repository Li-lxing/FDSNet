
import torch.nn as nn

shape = (256, 256)

import torch
import torch.nn.functional as F

from math import exp

# 相关系数计算函数
def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

# 特征分解约束损失函数
def decomp_loss(feature_V_B, feature_I_B, feature_V_D, feature_I_D):
    """特征分解约束损失，用于控制基础特征和细节特征之间的关系"""
    cc_loss_B = cc(feature_V_B, feature_I_B)
    cc_loss_D = cc(feature_V_D, feature_I_D)
    loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
    return loss_decomp, cc_loss_B, cc_loss_D

class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]

        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)

def corr_loss(image_ir, img_vis, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss


class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res



def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

def final_mse1(img_ir, img_vis, img_fuse, mask=None):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map_ir=torch.where(map1+mask>0, one, zero)
    map_vi= 1 - map_ir

    res = map_ir * mse_ir + map_vi * mse_vi
    return res.mean()


def Fusion_loss(vi,ir,  fu,vis_ploss,ir_ploss, mask, weights, decomp_loss_val=0.0, cc_loss_B=0.0, cc_loss_D=0.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sobelconv = Sobelxy(device)

    vi_grad_x, vi_grad_y = sobelconv(vi)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fu)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)

    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)
    loss_intensity = final_mse1(ir, vi, fu, mask) + 0 * F.l1_loss(fu, torch.max(ir, vi))
    loss_prior = vis_ploss + ir_ploss
    
    # 添加特征分解约束损失权重，默认为0.0
    decomp_weight = weights[3] if len(weights) > 3 else 0.0
    loss_total = weights[0]*loss_prior + weights[1] * loss_grad + weights[2] * loss_intensity + decomp_weight * decomp_loss_val
    return loss_total, loss_prior, loss_grad, loss_intensity, decomp_loss_val, cc_loss_B, cc_loss_D

