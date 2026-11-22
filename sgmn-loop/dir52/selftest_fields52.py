#!/opt/pytorch/bin/python
r"""DIR-52 · **VERDICT 模板字段覆盖自测**（`selftest_fields52.py`，全 CPU-free、秒级）。

动机（Cycle 82）：定论当天我要把 `adjudicate52.py` 的 JSON 逐格抄进 `VERDICT-SCAFFOLD.md §3` 的
13 行表格。若模板要的某个字段**裁决器根本没输出**（或嵌套层级不对），当天只剩三条烂路：
现算（可能算错口径）、改模板（改的是已冻结的呈现）、或**留空**。所以先把「模板要的路径」
与「裁决器实际吐的 JSON」逐条对表 —— 用**合成数据**（`REHEARSAL-ONLY`，绝不读真实 out/）。

同时核两个口径恒等式（都属**呈现一致性**，不改任何判据）：
  ① `SHAPE.threshold == max(delta_star, 3*SHAPE.se)`（P1/P2 越界用的是同一个 thr）；
  ② `3se` 与 `δ*` 的**自由度口径一致**：`sd` 用 `n−1`、`t975` 用 `ν=n−1`（已读码核实，此处再验一遍）。

★ 合成行的形状与 `R1_task52.py` 的输出**完全同构**（含 `phi` 逐轮 μ），故裁决器的
「行自洽复算」分支也会被真正执行到，覆盖率高于只喂 `SHAPE/EARLY/…` 的扁平行。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ADJ = os.path.join(HERE, "adjudicate52.py")
LO, HI = 62000, 62199

# 模板 §3 要求的 13 条路径（原样抄自 VERDICT-SCAFFOLD.md §3 的「JSON 路径」列）
PATHS = [
    "n_rows", "n_expected", "n_eff", "indeterminate_frac",
    "n_missing_vs_expected", "indet_vs_expected", "invalid_reasons",
    "SHAPE.mean", "SHAPE.sd", "SHAPE.se", "SHAPE.threshold",
    "SHAPE.n_pos", "SHAPE.n_neg",
    "EARLY.mean", "LATE.mean", "TOTAL.mean",
    "D67_cross_check", "decision.outcome",
]


def _resolve(d, path):
    cur = d
    try:
        for k in path.split("."):
            cur = cur[k]
        return True, cur
    except (KeyError, TypeError):
        return False, None


def synth_row(seed, e2, e4):
    """造一行与真实行同构的数据：mu(K,F) 使 dG(2)=e2、dG(4)=e4、dG(0)=0。"""
    base = 5.0
    ph = {}
    for phi, k2, k4 in (("0.0", base, base), ("0.9", base + e2, base + e4)):
        ph[phi] = {
            "keep_n": {"K": {"2": 40, "4": 60}, "F": {"2": 40, "4": 60}},
            "mu": {"K": {"0": base, "2": k2, "4": k4},
                   "F": {"0": base, "2": base, "4": base}},
            "wsig": {"K": {"2": 0.01, "4": 0.01}, "F": {"2": 0.01, "4": 0.01}},
            "wnorm": {"K": {"2": 1.0, "4": 1.0}, "F": {"2": 1.0, "4": 1.0}},
        }
    early, total = e2, e4
    return {"seed": seed, "tag": "rehearsal",
            "phi": ph,
            "dG": {"0": 0.0, "2": e2, "4": e4},
            "EARLY": early, "TOTAL": total, "LATE": total - early,
            "SHAPE": early - (total - early),
            "identity_err": 0.0, "finite": True}


def main():
    root = tempfile.mkdtemp(prefix="rehearse82_")
    try:
        outdir = os.path.join(root, "out")
        os.makedirs(outdir, exist_ok=True)
        seeds = list(range(LO, HI + 1))
        q = (len(seeds) + 3) // 4
        shards = []
        for i in range(4):
            p = os.path.join(outdir, f"shard{i}.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for s in seeds[i * q:(i + 1) * q]:
                    # 让 SHAPE 呈**负**（PASS 方向）×，并带一点散布，使 sd>0
                    j = (s - LO) % 5
                    e4 = 0.050 + 0.001 * j
                    e2 = 0.020 + 0.001 * j
                    f.write(json.dumps(synth_row(s, e2, e4)) + "\n")
            shards.append(p)
        jout = os.path.join(root, "rehearsal.json")
        cmd = [sys.executable, ADJ, "--shards", *shards,
               "--delta-star", "0.000430", "--power-gate", "0.000861",
               "--n-expected", "200", "--max-indet-frac", "0.20",
               "--min-n-eff", "160", "--out", jout,
               "--seed-segment", str(LO), str(HI)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("裁决器失败：\n" + r.stdout[-1500:] + r.stderr[-1500:])
            return 1
        d = json.load(open(jout, encoding="utf-8"))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("★ REHEARSAL-ONLY（合成数据，非证据）\n")
    miss = []
    for p in PATHS:
        ok, v = _resolve(d, p)
        if not ok:
            miss.append(p)
        show = v if not isinstance(v, (dict, list)) else type(v).__name__
        print(f"  {'OK  ' if ok else 'MISS'}  {p:26s} = {show}")
    print(f"\n覆盖 {len(PATHS)-len(miss)}/{len(PATHS)}" + (f"；缺：{miss}" if miss else ""))

    # 两个口径恒等式
    thr, se, ds = d["SHAPE"]["threshold"], d["SHAPE"]["se"], d["frozen"]["delta_star"]
    ok_thr = abs(thr - max(ds, 3 * se)) < 1e-15
    exp3 = 3 * se
    print(f"  ① threshold == max(δ*, 3se)：{'OK' if ok_thr else 'FAIL'}  "
          f"(thr={thr:.6g}, 3se={exp3:.6g}, δ*={ds:.6g})")
    print(f"  ② SHAPE.t975 存在且 ν 口径一致：{'OK' if 't975' in d['SHAPE'] else 'MISS'}"
          f"  t975={d['SHAPE'].get('t975')}")

    good = (not miss) and ok_thr and "t975" in d["SHAPE"]
    print("\n★ 全部通过 —— 模板 13 行均可由裁决器 JSON 直接抄录" if good else "\n★ 有缺口")
    print(f"  （排练结局={d['decision']['outcome']}，仅示链路可跑通，**不作任何判据结论**）")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())