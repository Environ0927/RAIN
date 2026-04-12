# trust_sign.py
# 作用：从服务器可信模型（全局模型 net）提取可信方向 s_trust，并提供把符号方向写回模型/按符号方向走一步的工具函数。

from typing import Iterator, Tuple
import torch
import torch.nn as nn

@torch.no_grad()
def flatten_params(net: nn.Module, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    把模型参数拼成一个一维向量（按 state_dict / parameters 顺序）。
    """
    flats = []
    for p in net.parameters():
        flats.append(p.detach().reshape(-1).to(dtype=dtype))
    out = torch.cat(flats, dim=0)
    if device is not None:
        out = out.to(device)
    return out  # [D]

@torch.no_grad()
def sign_from_model(net: nn.Module, device=None) -> torch.Tensor:
    """
    从服务器可信模型（net）提取符号方向 s_trust ∈ {−1,+1}^D，返回 int8（节省内存/带宽）。
    约定：0 按 +1 处理，避免死区。
    """
    w = flatten_params(net, device=device, dtype=torch.float32)   # [D], float
    s = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    return s.to(torch.int8)  # [-1,+1] cast int8

@torch.no_grad()
def apply_signed_direction_step(net: nn.Module, s_dir: torch.Tensor, step_size: float) -> None:
    """
    用符号方向做一步更新：W ← W − η · sign_dir
    - s_dir: {−1,+1}^D, int8/float 都行
    - step_size: 步长（你可以用 tssc 里聚合出的 scale_bar）
    """
    if s_dir.dtype != torch.float32 and s_dir.dtype != torch.float16:
        s_dir = s_dir.to(torch.float32)
    off = 0
    for p in net.parameters():
        n = p.numel()
        d = s_dir[off:off+n].view_as(p)
        p.add_(d, alpha=-float(step_size))
        off += n

@torch.no_grad()
def overwrite_model_sign_to(net: nn.Module, s_dir: torch.Tensor) -> None:
    """
    可选：把模型的符号强制对齐到 s_dir（仅改符号不改幅度），用于“可信方向回写同步”。
    """
    if s_dir.dtype != torch.float32 and s_dir.dtype != torch.float16:
        s_dir = s_dir.to(torch.float32)
    off = 0
    for p in net.parameters():
        n = p.numel()
        tgt = s_dir[off:off+n].view_as(p)
        mag = p.data.abs()
        p.data = torch.sign(tgt) * mag
        off += n
