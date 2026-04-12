# aggregation_rules_tssc.py
import torch
from typing import List, Dict, Tuple, Optional
from shuffler import Shuffler 
from client_codec import ClientCompressor
from tssc_relu import tssc_relu_aggregate
from trust_sign import apply_signed_direction_step
from camel_hd_rpc import call_camel_shuffle_perm
import os

@torch.no_grad()
def rainy_tssc_step(
    grad_list: List[List[torch.Tensor]],
    net: torch.nn.Module,
    device: torch.device,
    s_trust: Optional[torch.Tensor] = None,
    round_id: int = 0,
    k_bits: int = 8,                 # 将被覆盖为 d（dense）
    dp_clip: float = 1.0,
    dp_sigma: float = 0.0,
    use_scale: bool = False,
    use_EF: bool = False,
    lambda_mad: float = 2.2,
    tau_override: Optional[float] = None,
    w_max: Optional[float] = None,
    global_lr: float = 0.05,
) -> Tuple[torch.Tensor, Dict]:
    """
    Rainy-TSSC（dense 版本）：
      - Client端：clip -> noise -> (EF) -> sign -> dense pairs (idx,bit for all indices)
      - Shuffle：Camel-HD / MPC 洗牌
      - Server端：Hamming -> τ(median+λ·MAD) -> ReLU权重 -> 加权投票 -> 得到 s_trust_next
      - Update：按符号方向步长 global_lr 更新模型
    返回：
      s_trust_next, stats(dict)
    """
    # 总维度 d
    d = 0
    for p in net.parameters():
        d += p.numel()

    if s_trust is None:
        s_trust = torch.ones(d, dtype=torch.int8, device=device)

    # 强制 dense：k_bits = d
    k_bits = d
    compressor = ClientCompressor(
        d=d, C=dp_clip, sigma=dp_sigma, k=k_bits,
        use_scale=use_scale, use_EF=use_EF,
        dense_mode=True, device=device
    )

    # 1) 客户端编码（dense 符号）
    msgs = []
    for cid, grads in enumerate(grad_list):
        msg = compressor.compress(client_id=cid, round_id=round_id, grads=grads, seed_i=None)
        msgs.append(msg)

    # 2) Shuffle（匿名打乱）
    try:
        perm = call_camel_shuffle_perm(
            os.environ.get("CAMEL_SHUFFLE_BIN", ""),
            n_rows=len(msgs),
            seed=round_id
        )
        msgs = [msgs[i] for i in perm]
    except Exception:
        # 若 Camel 未部署，回退为本地顺序（不影响功能，仅少了匿名放大）
        pass

    # 3) TSSC-ReLU 聚合（在 dense 下逐维有符号/权重）
    s_trust_next, tau, hd_median, hd_mad, kept = tssc_relu_aggregate(
        msgs=msgs, net=net, s_trust=s_trust, device=device,
        lambda_mad=lambda_mad, tau_override=tau_override, w_max=w_max
    )

    # 4) 符号方向更新
    apply_signed_direction_step(net, s_trust_next, global_lr)

    # 5) 统计信息
    stats = dict(
        round_id=int(round_id),
        d=int(d),
        k_bits=int(k_bits),           # = d（dense）
        dp_clip=float(dp_clip),
        dp_sigma=float(dp_sigma),
        lambda_mad=float(lambda_mad),
        tau=float(tau),
        hd_median=float(hd_median),
        hd_mad=float(hd_mad),
        kept=int(kept),
        n_clients=len(grad_list),
        mode="dense",
    )
    return s_trust_next, stats
