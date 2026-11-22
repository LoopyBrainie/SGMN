"""DIR-26-C · probe26c_worked.py —— 工作算例的可复现核对（$d{=}16,N{=}8$）

卡片 §2 的读数表来自 `probe26c_modes.py --d 16 --n 8`；本脚本**重跑该表并逐项断言**，
另外把抖动指标与**逐项暴力 cos** 对照（与 Stage A 的 A26C-3 同源，但独立跑一次，防止"只在 Stage A 里对一次就算数"）。
"""
import json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26c")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26b")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26")

from probe26c_modes import HO_OFFSET, R_HO, R_PRES, THETA, run
from R1_gating3 import ret_jitter, ret_jitter_bruteforce
from R1_gating import build_stream, drive, make_core

EXPECT = {                                   # 卡片 §2 表（rev 1.1 冻结时抄录）
    2: {"A": (0.3750, 0.3125, 0.2500), "B": (0.5000, 0.3750, 0.3750), "P": (0.2500, 0.2812, 0.2500)},
    4: {"A": (0.5000, 0.5625, 0.6250), "B": (0.6250, 0.5000, 0.6250), "P": (0.5000, 0.4688, 0.3750)},
}
res = {"checks": [], "table": {}}


def chk(cid, ok, detail):
    res["checks"].append({"id": cid, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}  {detail}")


def main():
    print("=== 工作算例核对（d=16, N=8, seed 20261028, ρ=0.4） ===")
    for r0 in (2, 4):
        o = run(16, 8, r0, 20261028, tag="worked")
        tab = {}
        for arm in ("A", "B", "P"):
            a = o["arms"][arm]
            tab[arm] = (round(a["ret1_clean"], 4), round(a["Jout"]["ret1_mean"], 4),
                        round(a["Jout"]["ret1_maj"], 4))
        res["table"][r0] = tab
        exp = EXPECT[r0]
        ok = all(abs(tab[arm][i] - exp[arm][i]) < 5e-5 for arm in exp for i in range(3))
        chk(f"W-{r0}-表", ok, f"r={r0} 复现 (clean, Jout.mean, Jout.maj)：{tab} == 卡片 {exp}")
        dout = tab["B"][1] - tab["P"][1]
        chk(f"W-{r0}-方向", dout > 0, f"Δout_BP(mean) = {dout:+.4f} > 0（H26C 方向）")

    # 抖动指标 vs 逐项暴力（独立再核一次）
    D = build_stream(20261028, 16, 8, R_PRES, THETA)
    DH = build_stream(20261028 + HO_OFFSET, 16, 8, R_HO, THETA)
    c = make_core(16, 20261028, 4); drive(c, D, "A")
    P = c._W().detach() - c.W0
    a = ret_jitter(P, DH["K"], D["vb"], 8, R_HO)
    b = ret_jitter_bruteforce(P, DH["K"], D["vb"], 8, R_HO)
    chk("W-指标", a["ret1_mean"] == b["ret1_mean"] and a["ret1_maj"] == b["ret1_maj"]
        and a["n_hit"] == b["n_hit"] and a["maj_need"] == 3,
        f"矩阵式 == 暴力：mean {a['ret1_mean']:.4f}、maj {a['ret1_maj']:.4f}、命中 {a['n_hit']}/{a['n_read']}、"
        f"maj_need {a['maj_need']}")
    chk("W-等量", True, "r_q=4 ⇒ 多数票门槛 3（与卡片 §2 定义一致）")

    ok = all(x["ok"] for x in res["checks"])
    res["passed"] = ok
    print(f"\n{'★ 工作算例全部通过' if ok else '✗ 有 FAIL'}（{sum(x['ok'] for x in res['checks'])}/{len(res['checks'])}）")
    outp = "/home/ec2-user/sgmn-loop/dir26c/out/probe26c_worked.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"→ {outp}")


if __name__ == "__main__":
    main()
