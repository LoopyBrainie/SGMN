r"""DIR-26-N · 机制探针 `probe26n_mech.py` —— **为什么** R 在 m=60 掉基线、却在 m=24 保住免疫？

**定位**：正式轮（`DIR26Nr11_formal.json`）已给出裁决（通道 GATE-ONLY），但卡片要求裁决必须解释**机制**，
不只是读数。本探针**不新增判据、不改任何阈值**，只做**事后诊断**：把每条规则**跳过**与**写下**的机会按
$\cos$ 分层，回答三个问题：

  Q1  τ 的跳过集是**低 $\cos$** 的吗？（设计如此）
  Q2  R 的跳过集是**低 $\cos$** 还是**高 $\cos$**？⇒ 若高 $\cos$，说明 R 是"**修复规则**"（这个条目还准不准）
      而不是"**选择规则**"（这次写入值不值得），这是它在 m=60 掉基线的机制候选。
  Q3  H 的跳过集里有多少是条目的**首次机会**？⇒ H 把"首次就高 $\cos$"的写也跳掉 ⇒ 过度跳写机制。

**纪律**：
  · 规则定义逐字沿用 `probe26n_feas.drive_rule`（唯一实现），不重写谓词；
  · 只跑 τ / H / R 三条（不跑 J：它恒等全写，机制已由 A26N-7 钉死）；
  · 种子取前 6 个（探针级：**不参与裁决**），阈值一个都不改；
  · 输出 `out/probe26n_mech.json`；本探针**不出判决**。
"""
import json
import os
import sys

import numpy as np
import torch

for _p in ("dir26n", "dir26m", "dir26l", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g",
           "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating12 as K
import R1_gating14 as M14
import probe26n_feas as PR
from R1_gating import D_MODEL, make_core
from R1_gating5 import theta_of

N_KV = 60
N_READ = 20
SEEDS = K.SEEDS[:6]
BS = (40, 4)
MODES = ("A", "H", "R")          # A = τ 基线
OUT = "/home/ec2-user/sgmn-loop/dir26n/out"


def _stats(cos_sk, cos_wr):
    def s(x):
        x = np.asarray(x, float)
        if x.size == 0:
            return {"n": 0, "mean": None, "median": None}
        return {"n": int(x.size), "mean": float(np.mean(x)), "median": float(np.median(x))}
    return {"skip": s(cos_sk), "write": s(cos_wr)}


def unit(seed, b, tau):
    D = M14.build_stream_b(b, seed, D_MODEL, int(N_KV), K.R_PRES, theta_of(0.9))
    out = {"seed": seed, "b": int(b), "m": N_READ + int(b), "rules": {}}
    logs = {}
    for mode in MODES:
        core = make_core(D_MODEL, seed, K.R0)
        ki, lg, _diag = PR.drive_rule(core, D, mode, tau=tau)
        logs[mode] = lg
        cos = np.array([e["cos"] for e in lg], float)
        iss = np.array([e["issued"] for e in lg], bool)
        cos_sk, cos_wr = cos[~iss], cos[iss]
        st = _stats(cos_sk, cos_wr)
        st["issued"] = int(ki)
        st["gate_rate"] = float(np.mean(~iss))
        # 「τ 会保留的高 cos 写」被本规则跳掉的比例（R/H 的核心诊断）
        st["frac_skip_above_tau"] = float(np.mean(cos_sk > tau)) if cos_sk.size else None
        # 本规则**写下**的、τ 会跳掉的低价值写（cos ≤ τ）的比例
        st["frac_write_le_tau"] = float(np.mean(cos_wr <= tau)) if cos_wr.size else None
        # H 专项：跳过集中有多少是条目的首次机会（hist_n==0）
        if mode == "H":
            first = np.array([e["hist_n"] == 0 for e in lg], bool)
            st["frac_skip_is_first"] = float(np.mean(first[~iss])) if (~iss).any() else None
        out["rules"][mode] = st
    # 规则间的跳过集重叠（Jaccard）
    def skipmask(mode):
        return np.array([not e["issued"] for e in logs[mode]], bool)
    taus, hs, rs = skipmask("A"), skipmask("H"), skipmask("R")
    for a, b2, nm in ((hs, taus, "H"), (rs, taus, "R")):
        inter = int(np.sum(a & b2))
        union = int(np.sum(a | b2))
        out.setdefault("overlap", {})[f"{nm}_vs_tau"] = {"inter": inter, "union": union,
                                                        "jaccard": (inter / union) if union else None}
    # R 的跳过集里，有多少是 τ 会**保留**的（高价值被丢）
    out["overlap"]["R_skip_but_tau_write"] = int(np.sum(rs & ~taus))
    out["overlap"]["tau_skip_but_R_write"] = int(np.sum(taus & ~rs))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    from R1_gating12 import cal_at
    tau = cal_at(N_KV)["c_tau"]
    print(f"=== 机制探针（种子 {len(SEEDS)} × 档 {BS}，τ={tau:.4f}）===", flush=True)
    rows = []
    for b in BS:
        for s in SEEDS:
            u = unit(s, b, tau)
            rows.append(u)
            r = u["rules"]
            print(f"  b={b:2d} seed={s:3d} | τ skip cos {r['A']['skip']['mean']:.4f} "
                  f"write {r['A']['write']['mean']:.4f} | H skip {r['H']['skip']['mean']:.4f} "
                  f"write {r['H']['write']['mean']:.4f} first {r['H']['frac_skip_is_first']:.3f} | "
                  f"R skip {r['R']['skip']['mean']:.4f} write {r['R']['write']['mean']:.4f} "
                  f"aboveτ {r['R']['frac_skip_above_tau']:.3f} | Jacc(R,τ) {u['overlap']['R_vs_tau']['jaccard']:.3f}",
                  flush=True)
    # 汇总
    summ = {}
    for b in BS:
        sel = [u for u in rows if u["b"] == b]
        summ[str(b)] = {}
        for mode in MODES:
            summ[str(b)][mode] = {
                "gate_rate": float(np.mean([u["rules"][mode]["gate_rate"] for u in sel])),
                "skip_cos_mean": float(np.mean([u["rules"][mode]["skip"]["mean"] for u in sel])),
                "write_cos_mean": float(np.mean([u["rules"][mode]["write"]["mean"] for u in sel])),
                "skip_minus_write_cos": float(np.mean([u["rules"][mode]["skip"]["mean"] - u["rules"][mode]["write"]["mean"]
                                                       for u in sel])),
                "frac_skip_above_tau": float(np.mean([u["rules"][mode]["frac_skip_above_tau"] for u in sel])),
                "frac_write_le_tau": float(np.mean([u["rules"][mode]["frac_write_le_tau"] for u in sel])),
            }
        if b == 40:
            summ[str(b)]["H"]["frac_skip_is_first"] = float(np.mean([u["rules"]["H"]["frac_skip_is_first"] for u in sel]))
        summ[str(b)]["overlap"] = {
            "R_skip_but_tau_write": float(np.mean([u["overlap"]["R_skip_but_tau_write"] for u in sel])),
            "tau_skip_but_R_write": float(np.mean([u["overlap"]["tau_skip_but_R_write"] for u in sel])),
            "R_vs_tau_jaccard": float(np.mean([u["overlap"]["R_vs_tau"]["jaccard"] for u in sel])),
            "H_vs_tau_jaccard": float(np.mean([u["overlap"]["H_vs_tau"]["jaccard"] for u in sel])),
        }
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    json.dump({"tau": tau, "seeds": list(SEEDS), "rows": rows, "summary": summ},
              open(f"{OUT}/probe26n_mech.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  → {OUT}/probe26n_mech.json")


if __name__ == "__main__":
    main()
