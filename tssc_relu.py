# tssc_relu.py
import numpy as np
import torch
from typing import List, Dict, Optional
from camel_hd_rpc import pack_row_kplus1, s_trust_bits16_from_pm1, call_camel_hd
import os

@torch.no_grad()
def tssc_relu_aggregate(
    msgs: List[dict],
    net: torch.nn.Module,
    s_trust: torch.Tensor,          # shape [d], values in {-1,+1} (dtype int8)
    device: torch.device,
    lambda_mad: float = 2.2,
    tau_override: Optional[float] = None,
    w_max: Optional[float] = None,
    use_scale: bool = False,
    global_step_size: float = 1.0,
) -> torch.Tensor:
    """
    Thresholded Sparse Sign Consistency + ReLU weighting (TSSC-ReLU).
    1) 对每个客户端：仅在其上传的 k 个索引上，计算与 s_trust 的汉明距离 hd_i
    2) tau = median(hd) + 1.4826 * lambda_mad * MAD(hd)  （或使用 tau_override）
    3) 权重 w_i = max(0, tau - hd_i)，可设 w_max 截断；归一化得到 alpha_i
    4) 稀疏加权符号聚合：对每个观测到的坐标 j，S[j] += alpha_i * (+1/-1)
       s_trust_next[j] = sign(S[j])；未观测坐标沿用旧符号
    5) scale_bar：若 use_scale=True，从 layer_scale 估计一个（加权）中位数幅度
    6) ΔW = (global_step_size * scale_bar) * s_trust_next，原地更新模型
    返回：更新后的 s_trust_next
    """
    # ----- 1) 汉明距离（仅在稀疏坐标上） -----
    # ref_bits = (s_trust > 0).to(torch.uint8).cpu().numpy()  # {0,1}
    # dists = []
    # for m in msgs:
    #     hd = 0
    #     for (j, b) in m["pairs"]:
    #         hd += (b ^ int(ref_bits[j]))
    #     dists.append(hd)

    # k = max(1, len(msgs[0]["pairs"]))
    # d = torch.tensor(dists, dtype=torch.float32) / float(k)
    
    # ----- 1) 汉明距离（使用 Camel-Go HammingPlain） -----
    camel_bin = os.path.expanduser("~/517/safefl/Secure-Shuffling/camel_hd")
    D = len(s_trust)
    k = len(msgs[0]["pairs"])
    tau_frac = None  # 占位; 这里仍由下方的 MAD 计算产生 tau

    # 打包消息行 (每行 k 比特 + 16B 种子)
    rows = []
    for m in msgs:
        if "seed16" not in m:
            # 如果客户端还没生成种子，用固定伪种子占位（方便先跑通）
            m["seed16"] = bytes([len(rows) % 256]) * 16
        rows.append(pack_row_kplus1(m["pairs"], m["seed16"]))

    # s_trust {-1,+1} → {0,1} → 16B/位
    s_bits16 = s_trust_bits16_from_pm1(s_trust.cpu().numpy())

    # # 调用 camel_hd 可执行文件
    # tau_count = len(msgs[0]["pairs"])  # 仅为接口需要，真实 τ 仍在后面计算
    # # pass_mask, hd_vals = call_camel_hd(camel_bin, D, k, tau_count, rows, s_bits16)
    # pass_mask, hd_vals = call_camel_hd(camel_bin, D, k, len(msgs[0]["pairs"]),rows, s_bits16, mode="mpc")
    # print("MPC Pass mask:", pass_mask)

    # d = torch.tensor(hd_vals, dtype=torch.float32) / float(k)
    _, hd_vals = call_camel_hd(camel_bin, D, k, len(msgs[0]["pairs"]), rows, s_bits16, mode="plain")
    d = torch.tensor(hd_vals, dtype=torch.float32) / float(k)

    # ----- 2) tau -----
    if tau_override is None:
        med = torch.median(d)
        mad = torch.median(torch.abs(d - med)) + 1e-9
        tau = float(med + 1.4826 * lambda_mad * mad)
    else:
        tau = float(tau_override)

    # ----- 3) ReLU 权重 -----
    w = torch.clamp(tau - d, min=0.0)
    if w_max is not None:
        w = torch.clamp(w, max=float(w_max))
    sumw = float(w.sum().item()) + 1e-12
    alpha = (w / sumw).cpu().numpy()  # sum alpha_i = 1

    # ----- 4) 稀疏加权符号聚合 -----
    S: Dict[int, float] = {}
    for i, m in enumerate(msgs):
        ai = float(alpha[i])
        if ai <= 0:
            continue
        for (j, b) in m["pairs"]:
            S[j] = S.get(j, 0.0) + ai * (1.0 if b == 1 else -1.0)

    s_trust_next = s_trust.clone()
    for j, Sj in S.items():
        s_trust_next[j] = 1 if Sj >= 0 else -1

    # ----- 5) scale_bar -----
    if use_scale:
        scales = []
        for i, m in enumerate(msgs):
            ls = m.get("layer_scale")
            if ls is None or len(ls) == 0:
                continue
            s = float(np.median(np.asarray(ls, dtype=np.float32)))  # per-client summary
            scales.append((s, float(alpha[i])))
        if len(scales) > 0:
            vals = np.array([v for v, _ in scales], dtype=np.float32)
            ws   = np.array([w for _, w in scales], dtype=np.float32)
            order = np.argsort(vals)
            vals, ws = vals[order], ws[order]
            cum = np.cumsum(ws) / (ws.sum() + 1e-12)
            pos = np.searchsorted(cum, 0.5)
            scale_bar = float(vals[min(pos, len(vals) - 1)])
        else:
            scale_bar = 1.0
    else:
        scale_bar = 1.0

    # ----- 6) 应用 ΔW -----
    step = float(global_step_size) * float(scale_bar)
    direction = s_trust_next.to(torch.float32).to(device)  # {-1,+1}
    offset = 0
    for p in net.parameters():
        n = p.numel()
        seg = direction[offset:offset + n].view_as(p)
        p.add_(seg, alpha=-step)
        offset += n

    return s_trust_next

@torch.no_grad()
def rainy_tssc_aggregate(grad_list, net, device, s_trust: torch.Tensor,
                         step_size: float = 1.0):
    """
    grad_list: 你这一轮收集到的客户端更新（若你已做客户端压缩/取符号/稀疏采样，此处也可以是 Msg 列表）
    s_trust:   来自服务器可信模型的方向锚点（int8, [-1,+1]^D）
    step_size: 幅度控制（可用你在 TSSC 里聚合出的尺度中位数/截断均值；这里给默认值）
    备注：真正的 TSSC-ReLU（Hamming、τ=med+λ·MAD、ReLU 权重、更新 s_trust_next）请放在你自己的实现里；
          这个入口只演示如何接“可信方向 → 按方向更新全局模型”。
    """
    # 典型流程（伪代码，留给你已有实现）：
    # 1) 用 s_trust 计算每个客户端的稀疏 Hamming 距离 hd_i
    # 2) τ = median(hd) + λ * MAD(hd)
    # 3) w_i = ReLU(τ - hd_i)，归一化得到 alpha_i
    # 4) 在稀疏坐标上做加权投票，得到 s_trust_next
    # 5) 估计一个全局幅度 scale_bar（例如幸存客户端的尺度中位数）
    # 6) 最终方向/步长：DeltaW = scale_bar * s_trust_next

    # 这里演示“按符号方向更新”（先用 s_trust；如果你已经算出 s_trust_next，请替换成它）
    apply_signed_direction_step(net, s_trust, step_size)