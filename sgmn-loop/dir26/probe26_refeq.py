"""DIR-26 · probe26_refeq.py —— 核 (e) 的框架层发现：§3.5 的「恒定门控 ⇔ λ 变更」在**哪一步**成立

正式轮 (e) 实测（$r_0{=}8$，因子化 §3.3）：$\\cos P(\\alpha{=}.5,\\lambda{=}.05)=0.113$ vs $\\cos P(\\alpha{=}1,\\lambda{=}.10)=0.255$，
差 **0.142**（阈值 0.05）⇒ 等价性 FAIL。

**手算预期（本探针要判的就是它）**：
  · §3.2 **全矩阵参考步**：$W\\leftarrow W-\\eta G$，且 $G(\\alpha{.}5,\\lambda{.}05)=0.5\\,G(\\alpha{=}1,\\lambda{.}10)$ **逐位成立**
    ⇒ $W$ 的轨迹恰好是 0.5 倍 ⇒ $P$ 共线 ⇒ $\\arg\\max$ 与 $\\cos$ **完全相同** ⇒ 等价性 **精确**（差应为 0）。
  · §3.3 **因子化步**：$U\\leftarrow U-\\eta GV,\\ V\\leftarrow V-\\eta G^\\top U$（用旧 $U,V$）⇒ $G$ 减半**不**等于整步减半
    （交叉项 $\\eta^2GG^\\top$ 与 $\\eta^2GVV^\\top$ 不同阶）⇒ 等价性**失效**。
⇒ 若参考步差 ≈0 而因子化步差 0.142，则 (e) 的 FAIL 是**装置实现层的二阶项**，不是框架 §3.5 的错。
"""
import json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
from sgmn_module_hebb_v7 import HebbianCore
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26")
from R1_gating import build_stream, drive, metrics_plas, unit, D_MODEL, N_DIST, ETA, M, S_KEY, N_SLOTS

R0 = 8
THETA = math.radians(25.8419)
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]


def core_ref(d, seed, alpha, lam):
    """§3.2 全矩阵参考步：$W^0=0$、$U=V=0$（空记忆），与 dir24/dir25 的 r0=0 臂同一构造。"""
    g = torch.Generator().manual_seed(seed)
    c = HebbianCore(d, d, alpha, lam, ETA, g, "cpu", factorized=False, n_entries=N_SLOTS, d_model_sig=8)
    c.W.zero_(); c.W0.zero_()
    return c


def run(arm, seed, alpha, lam):
    D = build_stream(seed, D_MODEL, N_DIST, 4, THETA)
    if arm == "ref":
        c = core_ref(D_MODEL, seed, alpha, lam)
    else:
        g = torch.Generator().manual_seed(seed)
        c = HebbianCore(D_MODEL, D_MODEL, alpha, lam, ETA, g, "cpu", factorized=True,
                        n_entries=N_SLOTS, d_model_sig=8)
        U = torch.randn(D_MODEL, D_MODEL, generator=g) / math.sqrt(D_MODEL)
        V = torch.randn(D_MODEL, D_MODEL, generator=g) / math.sqrt(D_MODEL)
        U[:, R0:] = 0.0; V[:, R0:] = 0.0
        c.U.copy_(U); c.V.copy_(V)
    drive(c, D, "A")
    m = metrics_plas(c._W().detach(), c.W0, D["proto"], D["vb"], N_DIST)
    return {"cosP": m["cosP"], "ret1": m["ret1"], "P_norm": m["P_norm"]}


if __name__ == "__main__":
    out = {"rows": []}
    for arm in ("ref", "factor"):
        for s in SEEDS:
            a = run(arm, s, 0.5, 0.05)
            b = run(arm, s, 1.0, 0.10)
            d = abs(a["cosP"] - b["cosP"])
            out["rows"].append({"arm": arm, "seed": s, "cosP_half_lam05": a["cosP"],
                                "cosP_full_lam10": b["cosP"], "abs_diff": d})
            print(f"  {arm:7s} seed={s} cosP(.5,.05)={a['cosP']:.4f} vs (1,.10)={b['cosP']:.4f} "
                  f"差={d:.2e}  ret1={a['ret1']:.3f}/{b['ret1']:.3f}")
    for arm in ("ref", "factor"):
        v = [r["abs_diff"] for r in out["rows"] if r["arm"] == arm]
        print(f"  ⇒ {arm:7s} max|ΔcosP| = {max(v):.3e}  mean = {float(np.mean(v)):.3e}")
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "probe26_refeq.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print("→", p)
