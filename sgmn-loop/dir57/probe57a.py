#!/opt/pytorch/bin/python
r"""DIR-57 · **前提探针 A**：两条**不同键**能否绑定**同一个值**？（跨键绑定 / 自我一致性）

★ 探针纪律：**不设阈值、不写判据、不产出裁决读数**。
★ 装置**逐字复用** DIR-51：`R1_task51.{gen_vec, q95, read_P}` + 写入内层 `R1_task48.step_pair`（α=1、N_REP=16、λ=0.05、η=0.05）。
★ 四臂（同一起跑状态、同一对数）：
  `a_only`：只写 `(k_a, v_P)` ⇒ 读 `k_a`、`k_b`（**溢出基线**：没写 `k_b` 时它读到多少）
  `ab`    ：再写 `(k_b, v_P)` ⇒ 读 `k_a`、`k_b`（★ 核心：两键**都**指向 `v_P` 吗）
  `ba`    ：**换序**（先 b 后 a）⇒ 顺序敏感性（DIR-55 曾见近因 = 1.000）
  `distinct`：写 `(k_a, v_P)` 再写 `(k_b, v_Q)`（`v_Q ⊥ v_P`）⇒ 第二笔是否**擦掉**第一笔
★ 零假设：同一条读出对 32 个**随机目标向量**的余弦 `q95`（`DEF-51-3` 的实例内配对法，独立 generator `DEF-51-9`）。
用法：`/opt/pytorch/bin/python dir57/probe57a.py`
输出：`dir57/out/PROBE57A.json` + `dir57/out/PROBE57A.pt`（供 57c 复用读出向量）
"""
import json
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402
import R1_task51 as T51      # noqa: E402

OUT = os.path.join(ROOT, "dir57", "out")
SEED0 = 20261117
SEEDS = [SEED0 + 58000 + i for i in range(64)]
N_ANCHOR = 32
OFF_K1, OFF_K2, OFF_VAL, OFF_ANCHOR = 57101, 57102, 57103, 57104


def write_pair(U, Vm, k, v):
    return T48.step_pair(U, Vm, k, v, 1.0)


def key_like(g, ref=None, max_cos=0.2):
    r"""键：`KEY_SCALE × 单位向量`；给定 `ref` 时**拒绝采样**要求 `|cos| ≤ max_cos`。"""
    while True:
        k = R.KEY_SCALE * T51.gen_vec(g, M.D_K)
        if ref is None:
            return k
        c = float((k @ ref) / (k.norm() * ref.norm()))
        if abs(c) <= max_cos:
            return k, c


def value_perp(vP, g):
    r"""与 `v_P` 严格正交的单位向量（拒绝采样：投影残差够大才归一化，`DEF-51-1` 的教训）。"""
    while True:
        b = T51.gen_vec(g, M.D_V)
        b = b - (b @ vP) * vP
        if float(b.norm()) > 0.5:
            return b / b.norm()


def null_q95(U, Vm, k, g):
    r"""同一条读出 `W k` 对 32 个随机目标向量的余弦 `q95`。"""
    cs = [T51.read_P(U, Vm, k, T51.gen_vec(g, M.D_V))[0] for _ in range(N_ANCHOR)]
    return T51.q95(cs)


def svals(U, Vm):
    s = torch.linalg.svdvals(M._W(U, Vm).detach())
    return float(s[0]), float(s[1]), float(s[2]) if s.numel() > 2 else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, tensors = [], {}
    for sd in SEEDS:
        P = R.build(sd, "V0")
        U0, V0 = P["U1"].clone(), P["V1"].clone()
        g1 = torch.Generator().manual_seed(sd + OFF_K1)
        g2 = torch.Generator().manual_seed(sd + OFF_K2)
        gv = torch.Generator().manual_seed(sd + OFF_VAL)
        ga = torch.Generator().manual_seed(sd + OFF_ANCHOR)   # ★ 独立 generator（DEF-51-9）
        kA = key_like(g1)
        kB, cos_ab = key_like(g2, ref=kA, max_cos=0.2)
        vP = T51.gen_vec(gv, M.D_V)
        vQ = value_perp(vP, gv)

        # ── 臂 1：只写 a
        U1, V1 = write_pair(U0, V0, kA, vP)
        ca_only = T51.read_P(U1, V1, kA, vP)[0]
        cb_only = T51.read_P(U1, V1, kB, vP)[0]
        # ── 臂 2：再写 b（核心）
        U2, V2 = write_pair(U1, V1, kB, vP)
        ca_end = T51.read_P(U2, V2, kA, vP)[0]
        cb_end = T51.read_P(U2, V2, kB, vP)[0]
        ra = (M._W(U2, V2) @ kA).detach()
        rb = (M._W(U2, V2) @ kB).detach()
        cos_r = float((ra @ rb) / (ra.norm() * rb.norm())) if float(ra.norm() * rb.norm()) > 0 else float("nan")
        s1, s2, s3 = svals(U2, V2)
        null_a = null_q95(U2, V2, kA, ga)
        null_b = null_q95(U2, V2, kB, ga)
        # ── 臂 3：换序（先 b 后 a）
        Ub, Vb = write_pair(U0, V0, kB, vP)
        cb_mid = T51.read_P(Ub, Vb, kB, vP)[0]
        Uba, Vba = write_pair(Ub, Vb, kA, vP)
        ca_ba = T51.read_P(Uba, Vba, kA, vP)[0]
        cb_ba = T51.read_P(Uba, Vba, kB, vP)[0]
        # ── 臂 4：第二笔写**正交值**
        Ud, Vd = write_pair(U0, V0, kA, vP)
        Ud, Vd = write_pair(Ud, Vd, kB, vQ)
        ca_dist = T51.read_P(Ud, Vd, kA, vP)[0]
        cb_distQ = T51.read_P(Ud, Vd, kB, vQ)[0]
        ca_distQ = T51.read_P(Ud, Vd, kA, vQ)[0]

        rows.append({"seed": sd, "cos_k_ab": cos_ab,
                     "ca_only": ca_only, "cb_only": cb_only,
                     "ca_end": ca_end, "cb_end": cb_end,
                     "dca": ca_end - ca_only, "dcb": cb_end - cb_only,
                     "ca_ba": ca_ba, "cb_ba": cb_ba, "cb_mid": cb_mid,
                     "ca_dist": ca_dist, "cb_distQ": cb_distQ, "ca_distQ": ca_distQ,
                     "cos_r_ab": cos_r, "s1": s1, "s2": s2, "s3": s3,
                     "null_a": null_a, "null_b": null_b,
                     "rnorm_a": float(ra.norm()), "rnorm_b": float(rb.norm())})
        tensors[str(sd)] = {"ra": ra, "rb": rb, "vP": vP}

    def med(k):
        return statistics.median([x[k] for x in rows])

    def frac(pred):
        return sum(1 for x in rows if pred(x)) / len(rows)

    summ = {
        "n_instances": len(rows),
        "cos_k_ab": {"median": med("cos_k_ab"), "max_abs": max(abs(x["cos_k_ab"]) for x in rows)},
        "a_only": {"ca_median": med("ca_only"), "cb_median": med("cb_only")},
        "ab": {"ca_median": med("ca_end"), "cb_median": med("cb_end"),
               "dca_median": med("dca"), "dcb_median": med("dcb"),
               "frac_both_above_null": frac(lambda x: x["ca_end"] > x["null_a"] and x["cb_end"] > x["null_b"]),
               "frac_a_above_null": frac(lambda x: x["ca_end"] > x["null_a"]),
               "frac_b_above_null": frac(lambda x: x["cb_end"] > x["null_b"])},
        "order": {"ca_ba_median": med("ca_ba"), "cb_ba_median": med("cb_ba"),
                  "cb_mid_median": med("cb_mid")},
        "distinct_values": {"ca_dist_median": med("ca_dist"), "cb_distQ_median": med("cb_distQ"),
                            "ca_distQ_median": med("ca_distQ"),
                            "frac_a_kept": frac(lambda x: x["ca_dist"] > x["null_a"])},
        "geometry": {"cos_r_ab_median": med("cos_r_ab"),
                     "s1_median": med("s1"), "s2_median": med("s2"), "s3_median": med("s3"),
                     "rnorm_a_median": med("rnorm_a"), "rnorm_b_median": med("rnorm_b")},
        "null": {"a_median": med("null_a"), "b_median": med("null_b")},
    }
    res = {"probe": "57a", "device": "DIR-51 (dir31/dir33/dir46/dir47/dir48, write=dir48.step_pair)",
           "seeds": [SEEDS[0], SEEDS[-1]], "rows": rows, "summary": summ}
    with open(os.path.join(OUT, "PROBE57A.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    torch.save(tensors, os.path.join(OUT, "PROBE57A.pt"))

    s = summ
    print("── ① 键对：|cos(k_a,k_b)| 中位 %.4f（max %.4f）" % (s["cos_k_ab"]["median"], s["cos_k_ab"]["max_abs"]))
    print("── 只写 a：c_a %.4f · c_b %.4f（溢出）" % (s["a_only"]["ca_median"], s["a_only"]["cb_median"]))
    print("── 写 a+b：c_a %.4f · c_b %.4f ⇒ Δc_a %.4f · Δc_b %.4f；两键都超零假设 %d/%d"
          % (s["ab"]["ca_median"], s["ab"]["cb_median"], s["ab"]["dca_median"], s["ab"]["dcb_median"],
             round(s["ab"]["frac_both_above_null"] * s["n_instances"]), s["n_instances"]))
    print("── 换序 b→a：c_a %.4f · c_b %.4f（对照 a→b：%.4f / %.4f）"
          % (s["order"]["ca_ba_median"], s["order"]["cb_ba_median"],
             s["ab"]["ca_median"], s["ab"]["cb_median"]))
    print("── 第二笔写正交值：c_a(保留) %.4f · c_b→v_Q %.4f · c_a→v_Q %.4f"
          % (s["distinct_values"]["ca_dist_median"], s["distinct_values"]["cb_distQ_median"],
             s["distinct_values"]["ca_distQ_median"]))
    print("── 几何：cos(r_a,r_b) %.4f · σ=[%.3f, %.3f, %.3f] · ‖r‖=%.3f/%.3f · 零假设 q95 %.4f/%.4f"
          % (s["geometry"]["cos_r_ab_median"], s["geometry"]["s1_median"], s["geometry"]["s2_median"],
             s["geometry"]["s3_median"], s["geometry"]["rnorm_a_median"], s["geometry"]["rnorm_b_median"],
             s["null"]["a_median"], s["null"]["b_median"]))


if __name__ == "__main__":
    main()