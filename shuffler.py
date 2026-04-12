# shuffler.py
from typing import List, Dict, Optional
import copy
import numpy as np

class Shuffler:
    """
    明文 Shuffler：去掉可识别字段（如 client_id），并随机重排消息顺序。
    - 这是 Shuffle-DP 流水线的最小化接口；后续你可以把它替换为“多方混洗/匿名通道”。
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def anonymize_and_shuffle(self, msgs: List[dict]) -> List[dict]:
        # 深拷贝，去除潜在的标识字段，然后随机打乱
        stripped: List[dict] = []
        for m in msgs:
            mm = copy.deepcopy(m)
            # 保守起见，移除/覆盖可能的标识字段
            for k in ("client_id", "sender", "ip", "uid"):
                if k in mm:
                    mm.pop(k)
            # seed_i/round_id 仍可保留（如需 MPC 复现 PRG），也可按需置空
            stripped.append(mm)
        self.rng.shuffle(stripped)
        return stripped
