import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Optional, Callable
from einops import rearrange, repeat

# 导入选择性扫描函数
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# 备用实现
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2D_cross(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj0 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d0 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj0 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj1 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight0 = nn.Parameter(torch.stack([t.weight for t in self.x_proj0], dim=0))  # (K=4, N, inner)
        self.x_proj_weight1 = nn.Parameter(torch.stack([t.weight for t in self.x_proj1], dim=0))
        del self.x_proj0
        del self.x_proj1

        self.dt_projs0 = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs1 = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight0 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs0], dim=0))  # (K=4, inner, rank)
        self.dt_projs_weight1 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs1], dim=0))  # (K=4, inner, rank)
        
        self.dt_projs_bias0 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs0], dim=0))  # (K=4, inner)
        self.dt_projs_bias1 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs1], dim=0)) 
        
        del self.dt_projs0
        del self.dt_projs1
        
        self.A_logs0 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.A_logs1 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        
        self.Ds0 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds1 = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0

        self.out_norm0 = nn.LayerNorm(self.d_inner)
        self.out_proj0 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_norm1 = nn.LayerNorm(self.d_inner)
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x0, x1):
        self.selective_scan0 = selective_scan_fn
        self.selective_scan1 = selective_scan_fn
        
        B, C, H, W = x0.shape
        L = H * W
        K = 4

        x_hwwh0 = torch.stack([x0.view(B, -1, L), torch.transpose(x0, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs0 = torch.cat([x_hwwh0, torch.flip(x_hwwh0, dims=[-1])], dim=1)  # (b, k, d, l)
        
        x_hwwh1 = torch.stack([x1.view(B, -1, L), torch.transpose(x1, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs1 = torch.cat([x_hwwh1, torch.flip(x_hwwh1, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl0 = torch.einsum("b k d l, k c d -> b k c l", xs0.view(B, K, -1, L), self.x_proj_weight0)
        x_dbl1 = torch.einsum("b k d l, k c d -> b k c l", xs1.view(B, K, -1, L), self.x_proj_weight1)
        
        dts0, Bs0, Cs0 = torch.split(x_dbl0, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts0 = torch.einsum("b k r l, k d r -> b k d l", dts0.view(B, K, -1, L), self.dt_projs_weight0)
        
        dts1, Bs1, Cs1 = torch.split(x_dbl1, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts1 = torch.einsum("b k r l, k d r -> b k d l", dts1.view(B, K, -1, L), self.dt_projs_weight1)

        xs0 = xs0.float().view(B, -1, L)  # (b, k * d, l)
        dts0 = dts0.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs0 = Bs0.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs0 = Cs0.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds0 = self.Ds0.float().view(-1)  # (k * d)
        As0 = -torch.exp(self.A_logs0.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias0 = self.dt_projs_bias0.float().view(-1)  # (k * d)
        
        xs1 = xs1.float().view(B, -1, L)  # (b, k * d, l)
        dts1 = dts1.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs1 = Bs1.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs1 = Cs1.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds1 = self.Ds1.float().view(-1)  # (k * d)
        As1 = -torch.exp(self.A_logs1.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias1 = self.dt_projs_bias1.float().view(-1)  # (k * d)

        out_y0 = self.selective_scan0(
            xs0, dts1, 
            As1, Bs1, Cs1, Ds1, z=None,
            delta_bias=dt_projs_bias1,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y0.dtype == torch.float
        
        out_y1 = self.selective_scan1(
            xs1, dts0, 
            As0, Bs0, Cs0, Ds0, z=None,
            delta_bias=dt_projs_bias0,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y1.dtype == torch.float

        inv_y0 = torch.flip(out_y0[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y0 = torch.transpose(out_y0[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y0 = torch.transpose(inv_y0[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        inv_y1 = torch.flip(out_y1[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y1 = torch.transpose(out_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y1 = torch.transpose(inv_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y0[:, 0], inv_y0[:, 0], wh_y0, invwh_y0, out_y1[:, 0], inv_y1[:, 0], wh_y1, invwh_y1

    def forward(self, x0, x1):
        B, H, W, C = x0.shape

        xz0 = self.in_proj0(x0)
        xz1 = self.in_proj1(x1)
               
        x0, z0 = xz0.chunk(2, dim=-1)  # (b, h, w, d)
        x1, z1 = xz1.chunk(2, dim=-1)

        x0 = x0.permute(0, 3, 1, 2).contiguous()
        x0 = self.act(self.conv2d0(x0))  # (b, d, h, w)
        
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.act(self.conv2d1(x1))
        
        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x0, x1)
        
        assert y1.dtype == torch.float32
        
        yir = y1 + y2 + y3 + y4
        yvi = y5 + y6 + y7 + y8
        
        yir = torch.transpose(yir, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        yir = self.out_norm0(yir)
        
        yvi = torch.transpose(yvi, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        yvi = self.out_norm1(yvi)
        
        yir = yir * F.silu(z0)
        out0 = self.out_proj0(yir)
        
        yvi = yvi * F.silu(z1)
        out1 = self.out_proj1(yvi)
        
        return out0, out1


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.conv_vss1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.relu_vss1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input: torch.Tensor):
        x_vss = self.drop_path(self.self_attention(self.ln_1(input.permute(0, 2, 3, 1))))
        x_conv = self.relu_vss1(self.conv_vss1(input))
        x = x_conv + x_vss.permute(0, 3, 1, 2)
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage. """
    def __init__(
        self, 
        dim, 
        depth=1, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        return x


class VSSBlock_cross(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.self_attention = SS2D_cross(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.conv_vss1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.relu_vss1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu_vss2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_vss2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        
        self.conv_vss3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.relu_vss3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu_vss4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_vss4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x0, x1):
        x_vss0, x_vss1 = self.drop_path(self.self_attention(self.ln_1(x0.permute(0, 2, 3, 1)), self.ln_2(x1.permute(0, 2, 3, 1))))
        x0 = x_vss0.permute(0, 3, 1, 2)
        x1 = x_vss1.permute(0, 3, 1, 2)
        return x0, x1


class VSSLayer_cross(nn.Module):
    """ A basic Swin Transformer layer for one stage. """
    def __init__(
        self, 
        dim, 
        depth=1, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock_cross(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, xir, xvi):
        for blk in self.blocks:
            xir, xvi = blk(xir, xvi)

        return xir, xvi
