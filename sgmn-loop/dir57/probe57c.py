#!/opt/pytorch/bin/python
r"""DIR-57 · **前提探针 C**：**读出侧**能不能同时服务两条键？（③ 是不是一个独立前提）

★ 探针纪律：**不设阈值、不写判据、不产出裁决读数**。
★ 先讲清代数（这决定了 ③ 在本装置上**是否退化**）：
   DIR-51 / 本装置的读数是 `cos(W k, v) = (Wk·v)/(‖Wk‖‖v‖)` —— **尺度不变**。
   任务层的读出（`Wo1` 等）在本装置里没有对应物；即使有，任何**线性**读出层的可调自由度对
   "方向"而言只剩 **一个正交/伸缩变换**，而 `cos` 已经把伸缩吸收 ⇒ **方向能否读出不取决于读出层**，
   只取决于 `Wk` 本身指不指向 `v`（即 ①②）。
   ⇒ 因此 ③ 在本装置上**只能**测两件非退化的事：
     (i) **符号冲突**：`c_a > 0` 而 `c_b < 0` 时，任何单一线性 A 都不可能把两者同时映到 `+v_P`（余弦口径）；
     (ii) **共线性**：`cos(r_a, r_b) ≈ 1` ⇒ 两条键的读出其实是**同一个方向**（此时"两键都指向 v_P"不是绑定，是**值塌缩**）。
★ 数据：复用 `PROBE57A.pt`（64 实例，每条键的读出向量 `r_a = W k_a`、`r_b = W k_b`、`v_P`）。
   ★ 并显式演示"逐实例拟合 A 会退化"：2 行约束 / 16 个参数 ⇒ 残差 ~1e-30，**不得**当读数（`D-98` 教训 134）。
用法：`/opt/pytorch/bin/python dir57/probe57c.py` → `dir57/out/PROBE57C.json`
"""
import json
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
OUT = "/home/ec2-user/sgmn-loop/dir57/out"


def cos(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def main():
    T = torch.load(os.path.join(OUT, "PROBE57A.pt"))
    rows = []
    for sd, d in T.items():
        ra, rb, vP = d["ra"], d["rb"], d["vP"]
        ca, cb = cos(ra, vP), cos(rb, vP)
        # 单一线性 A（4×4，逐实例）：约束 2 行 / 16 参数 ⇒ **退化演示**
        Y = torch.stack([ra, rb])
        Tg = torch.stack([vP, vP])
        A = torch.linalg.lstsq(Y, Tg).solution
        resid = float((((Y @ A) - Tg) ** 2).mean())
        rows.append({"seed": sd, "ca": ca, "cb": cb, "cos_r": cos(ra, rb),
                     "sign_conflict": bool(ca * cb < 0),
                     "both_positive": bool(ca > 0 and cb > 0),
                     "degenerate_A_resid": resid,
                     "n_constraints": 2, "n_params": 16})

    n = len(rows)
    summ = {
        "n_instances": n,
        "sign_conflict_frac": sum(r["sign_conflict"] for r in rows) / n,
        "both_positive_frac": sum(r["both_positive"] for r in rows) / n,
        "cos_r_median": statistics.median([r["cos_r"] for r in rows]),
        "cos_r_abs_q95": sorted(abs(r["cos_r"]) for r in rows)[int(0.95 * (n - 1))],
        "frac_cos_r_above_0.9": sum(1 for r in rows if abs(r["cos_r"]) > 0.9) / n,
        "ca_median": statistics.median([r["ca"] for r in rows]),
        "cb_median": statistics.median([r["cb"] for r in rows]),
        "degenerate_A_resid_max": max(r["degenerate_A_resid"] for r in rows),
        "degenerate_A_note": "2 行约束 / 16 参数 ⇒ 精确插值，残差 ~0 是**退化**而非证据（D-98 缺陷 134）",
    }
    res = {"probe": "57c", "device": "DIR-51 读数（cos 尺度不变）", "summary": summ, "rows": rows}
    with open(os.path.join(OUT, "PROBE57C.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("── 符号冲突（c_a>0 而 c_b<0）：%d/%d（%.1f%%）；两键同号为正：%d/%d"
          % (round(summ["sign_conflict_frac"] * n), n, 100 * summ["sign_conflict_frac"],
             round(summ["both_positive_frac"] * n), n))
    print("── 共线性：cos(r_a,r_b) 中位 %.4f · |cos| q95 %.4f · |cos|>0.9 的比例 %.1f%%"
          % (summ["cos_r_median"], summ["cos_r_abs_q95"], 100 * summ["frac_cos_r_above_0.9"]))
    print("── c_a 中位 %.4f · c_b 中位 %.4f" % (summ["ca_median"], summ["cb_median"]))
    print("── 退化核对：逐实例 4×4 A 的残差最大 %.2e（%s）"
          % (summ["degenerate_A_resid_max"], "预期 ~0，不作读数"))


if __name__ == "__main__":
    main()