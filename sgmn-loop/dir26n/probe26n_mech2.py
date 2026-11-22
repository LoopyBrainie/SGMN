r"""DIR-26-N · 机制探针 **2**（`probe26n_mech2.py`）—— R 在 m=60 掉基线的机制归因

**定位**：正式轮已裁决（通道 GATE-ONLY）。卡片要求裁决解释**机制**。本探针**不新增判据、不改阈值**，
只做**事后诊断**，把"为什么 R 在 m=60 比**无条件写**还差、却在 m=24 保住免疫"落到可测量的量上。

**四个候选机制 + 对应测量**：

| 假设 | 测量 |
|---|---|
| M1 **R 不是选择规则**（它按"残差漂移"选，不按"这次写值不值"选） | R 的跳集 $\cos$ 均值 vs 写集 $\cos$ 均值（若相等 ⇒ 与 $\cos$ 无关）；Jaccard(R 跳集, τ 跳集) |
| M2 **R 跳掉的是"末次强化写"**（残差还没被干扰抬高 ⇒ 跳过） ⇒ 条目在末轮失去收尾强化 ⇒ 读侧命中下降 | 跳集的**轮次分布**（尤其末轮占比 vs 机会占比）；逐条目"末轮是否有写"；逐读条目命中数 |
| M3 **m=24 上 R 跳的是"兄弟已覆盖"的冗余写** ⇒ 省下干扰 ⇒ 读侧命中上升 | 非读条目的**模式组大小**（`pat_of`：b=4 ⇒ 40 个非读条目落进 4 组 × 10）与逐条目写次数的分布 |
| M4 H 是**逐条目棘轮锁**（一旦 $\cos>$ 自身历史中位数就永久跳过 ⇒ 条目被冻结在半训练态） | H 的跳集 $\cos$ 均值（应≈写集的 2–3 倍）；H 的条目级"永久跳过"计数 |

**纪律**：规则谓词仍只有一份（`probe26n_feas.drive_rule`）；逐读条目命中用 `ret_jitter_margin` 的**副本**实现，
并**断言**其 `ret1` 与官方 `measure_hit_bounded` **逐位相等**（否则本探针作废）。
种子取前 3 个（与 `probe26n_feas` 一致；探针级：**不参与裁决**）。输出 `out/probe26n_mech2.json`；**本探针不出判决**。
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
from R1_gating12 import R_PRES, R_HO8, cal_at, read_stream
from rerun_missing import measure_hit_bounded

N_KV = 60
N_READ = 20
SEEDS = K.SEEDS[:3]
BS = (40, 4)
OUT = "/home/ec2-user/sgmn-loop/dir26n/out"


def unit(x):
    return torch.nn.functional.normalize(x, dim=-1, eps=1e-12)


def ret_jitter_margin_peritem(P, Kq, vb, n, r_q=R_HO8):
    r"""`probe26i_margin.ret_jitter_margin` 的**逐读条目命中**版本（副本；ret1 必须逐位相等）。"""
    vbU = unit(vb)
    m_all, hit = [], np.zeros((r_q, n), dtype=bool)
    for rho in range(r_q):
        Y = P @ Kq[rho].T
        S = unit(Y.T) @ vbU.T
        for i in range(n):
            row = S[i]
            j = int(torch.argmax(row).item())
            hit[rho, i] = j == i
            other = torch.cat([row[:i], row[i + 1:]])
            m_all.append(float(row[i] - other.max()))
    return {"mu": float(np.mean(m_all)), "ret1": float(hit.mean()), "hit": hit}


def unit_stats(seed, b, tau):
    D = M14.build_stream_b(b, seed, D_MODEL, int(N_KV), K.R_PRES, theta_of(0.9))
    RS8 = read_stream(seed, R_HO8)
    proto, vb = D["proto"][:N_READ], D["vb"][:N_READ]
    pat = D.get("pat")
    if pat is None:
        pat = M14.pat_of(b, N_KV)
    group = {}
    for i in range(int(N_KV)):
        group[int(pat[i])] = group.get(int(pat[i]), 0) + 1
    out = {"seed": seed, "b": int(b), "m": N_READ + int(b), "rules": {}}
    logs = {}
    for mode, tag in (("A", "tau"), ("H", "H"), ("R", "R")):
        core = make_core(D_MODEL, seed, K.R0)
        ki, lg, _d = PR.drive_rule(core, D, mode, tau=tau)
        logs[mode] = lg
        W = core._W().detach()
        P = W - core.W0
        m8 = measure_hit_bounded(P, RS8["K"], vb, N_READ, R_HO8)
        pm = ret_jitter_margin_peritem(P, RS8["K"], vb, N_READ, R_HO8)
        assert pm["ret1"] == m8["ret1"], f"命中副本与官方口径不符：{pm['ret1']} vs {m8['ret1']}"
        cos = np.array([e["cos"] for e in lg], float)
        iss = np.array([e["issued"] for e in lg], bool)
        rnd = np.array([e["round"] for e in lg], int)
        itm = np.array([e["item"] for e in lg], int)
        # 逐条目写次数 / 末轮是否有写
        wr = np.zeros(int(N_KV), int)
        for i, w in zip(itm, iss):
            wr[i] += int(w)
        last_rnd = int(rnd.max())
        has_last = np.array([bool(iss[(itm == i) & (rnd == last_rnd)].any()) for i in range(int(N_KV))])
        # 跳集的轮次分布（归一化：末轮跳占比 vs 末轮机会占比）
        sk = ~iss
        st = {
            "issued": int(ki), "ret1": float(m8["ret1"]),
            "gate_rate": float(np.mean(sk)),
            "skip_cos_mean": float(np.mean(cos[sk])) if sk.any() else None,
            "write_cos_mean": float(np.mean(cos[~sk])) if (~sk).any() else None,
            "frac_skip_above_tau": float(np.mean(cos[sk] > tau)) if sk.any() else None,
            "frac_write_le_tau": float(np.mean(cos[~sk] <= tau)) if (~sk).any() else None,
            "skip_last_round_frac": float(np.mean(rnd[sk] == last_rnd)) if sk.any() else None,
            "opp_last_round_frac": float(np.mean(rnd == last_rnd)),
            "skip_round_hist": [float(np.mean(rnd[sk] == r)) if sk.any() else None for r in range(last_rnd + 1)],
            "opp_round_hist": [float(np.mean(rnd == r)) for r in range(last_rnd + 1)],
            "writes_per_item_min": int(wr.min()), "writes_per_item_med": float(np.median(wr)),
            "writes_per_item_max": int(wr.max()),
            "frac_items_last_round_written": float(has_last.mean()),
            # 读条目命中（逐条目，8 轮）
            "read_hits": pm["hit"].sum(axis=0).tolist(),
            "read_writes": wr[:N_READ].tolist(),
            "read_group_size": [group[int(pat[i])] for i in range(N_READ)],
        }
        # 非读条目：按模式组大小分桶的写次数（m=24 的冗余核）
        nb = [i for i in range(N_READ, int(N_KV))]
        st["nonread_writes_by_grp"] = {str(g): float(np.mean([wr[i] for i in nb if group[int(pat[i])] == g]))
                                       for g in sorted(set(group[int(pat[i])] for i in nb))}
        out["rules"][tag] = st
    # 跳集重叠（★ DEF-26N-7：直接用本单元已算的 log，**不再重复驱动**）
    def mask(mode):
        return np.array([not e["issued"] for e in logs[mode]], bool)
    t, h, r = mask("A"), mask("H"), mask("R")
    out["overlap"] = {
        "J_R_tau": float(np.sum(r & t) / max(np.sum(r | t), 1)),
        "J_H_tau": float(np.sum(h & t) / max(np.sum(h | t), 1)),
        "R_skip_tau_write": int(np.sum(r & ~t)), "tau_skip_R_write": int(np.sum(t & ~r)),
    }
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    tau = cal_at(N_KV)["c_tau"]
    print(f"=== 机制探针 2（种子 {len(SEEDS)} × 档 {BS}，τ={tau:.4f}）===", flush=True)
    rows = []
    for b in BS:
        for s in SEEDS:
            u = unit_stats(s, b, tau)
            rows.append(u)
            for tag in ("tau", "H", "R"):
                st = u["rules"][tag]
                print(f"  b={b:2d} seed={s:3d} {tag:>3s} | gate {st['gate_rate']:.3f} ret1 {st['ret1']:.4f} "
                      f"| skip cos {st['skip_cos_mean']:.4f} vs write {st['write_cos_mean']:.4f} "
                      f"| 末轮跳占比 {st['skip_last_round_frac']:.3f} (机会 {st['opp_last_round_frac']:.3f}) "
                      f"| 末轮有写条目 {st['frac_items_last_round_written']:.3f} "
                      f"| 写/条目 {st['writes_per_item_min']}-{st['writes_per_item_max']} "
                      f"| 读命中 {int(np.sum(st['read_hits']))}/160", flush=True)
            print(f"        J(R,τ)={u['overlap']['J_R_tau']:.3f} J(H,τ)={u['overlap']['J_H_tau']:.3f} "
                  f"R跳而τ写={u['overlap']['R_skip_tau_write']} τ跳而R写={u['overlap']['tau_skip_R_write']}", flush=True)
    summ = {}
    for b in BS:
        sel = [u for u in rows if u["b"] == b]
        summ[str(b)] = {}
        for tag in ("tau", "H", "R"):
            keys = ("gate_rate", "ret1", "skip_cos_mean", "write_cos_mean", "frac_skip_above_tau",
                    "frac_write_le_tau", "skip_last_round_frac", "opp_last_round_frac",
                    "frac_items_last_round_written", "writes_per_item_min", "writes_per_item_max")
            summ[str(b)][tag] = {k: float(np.mean([u["rules"][tag][k] for u in sel])) for k in keys}
            summ[str(b)][tag]["read_hits_mean"] = float(np.mean([np.sum(u["rules"][tag]["read_hits"]) for u in sel]))
            summ[str(b)][tag]["read_writes_mean"] = float(np.mean([np.sum(u["rules"][tag]["read_writes"]) for u in sel]))
            summ[str(b)][tag]["nonread_writes_by_grp"] = {
                g: float(np.mean([u["rules"][tag]["nonread_writes_by_grp"][g] for u in sel]))
                for g in sel[0]["rules"][tag]["nonread_writes_by_grp"]}
        summ[str(b)]["overlap"] = {k: float(np.mean([u["overlap"][k] for u in sel]))
                                   for k in ("J_R_tau", "J_H_tau", "R_skip_tau_write", "tau_skip_R_write")}
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    json.dump({"tau": tau, "seeds": list(SEEDS), "rows": rows, "summary": summ},
              open(f"{OUT}/probe26n_mech2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  → {OUT}/probe26n_mech2.json")


if __name__ == "__main__":
    main()
