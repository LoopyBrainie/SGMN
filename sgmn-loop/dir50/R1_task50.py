#!/opt/pytorch/bin/python
r"""DIR-50 · 实现：**模态持续度** —— 两层装置（层 1 / 层 2 = 两个「模态通道」）下的
「单次干扰伤害是否与模态无关」+「上游模态的干扰会不会级联到下游模态」。

卡：`revolve/DIR-50/DIRECTION_rev50_v10.md`（rev 1.0，md5 见卡内，**阈值已冻结**）。
★ **逐字派生**：`dir50/dev50.py`（写动力学 `dir48.step_pair`、读数 `dir48.read_target`、
   内容流 `dir46.content`、两层接线 `dir31.rollout`）—— **一字不改**；
★ **唯一新增**（DIR-49 之后的增量）：**键空间内构造的干扰键** —— 剂量 `cosθ` **精确可控**、
   范数与自然键**逐位匹配** ⇒ ★ **五臂的内容流逐位相同**（只改**写入的键**），
   这比 DIR-49 的「改内容 `a_4`」更干净（内容操纵会连带改变层 2 的 `(k,v)` 流 ⇒ 跨模态对比被污染）。

用法：`R1_task50.py --arm {none,hit1,hit2} [--dose {0.0,1.0}] --n 4000 --seed0 33000 --out PATH [--tag r10]`
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = "/home/ec2-user/sgmn-loop"
sys.path.insert(0, os.path.join(ROOT, "dir50"))
import dev50 as D  # noqa: E402

ARMS = ("none", "hit1", "hit2")
DOSES = ("0.0", "1.0")


def row_of(sd, rr, arm, dose):
    """把一次 rollout 打包成一行（**原始余弦**，供裁决器独立重算 `rel`）。"""
    out = {"seed": sd, "arm": arm, "dose": dose,
           "n_writes": 2 * D.T_STEPS,          # 每层每步一次写 × 2 层
           "c0": [float(rr["c0"][li]) for li in (0, 1)],
           "curve": [[float(x) for x in rr["curve"][li]] for li in (0, 1)],
           "ov_k": rr["ov_k"], "res": rr["res"],
           "med": {}}
    # ★ 中介链：只记 step T_HIT 后的**前 3 步**（含未受影响的第 0 步作为对照）
    for kk, vv in rr["med"].items():
        out["med"][kk] = [[float(z) for z in vec] for vec in vv[:3]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--dose", default=None, choices=DOSES)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed0", type=int, default=33000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="r10")
    a = ap.parse_args()
    if a.arm == "none":
        assert a.dose is None, "none 臂没有剂量"
    else:
        assert a.dose is not None, "hit 臂必须给剂量"
    t0 = time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.n):
            sd = a.seed0 + i
            P = D.build(sd)
            cont = D.content(sd)
            rr = D.rollout(P, cont, a.arm, None if a.dose is None else float(a.dose))
            row = row_of(sd, rr, a.arm, a.dose)
            row["tag"] = a.tag
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"[{a.arm}|{a.dose}] {i+1}/{a.n}  {time.time()-t0:.1f}s", flush=True)
    print(f"DONE {a.arm}|{a.dose} n={a.n} seed0={a.seed0} → {a.out}  墙钟 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
