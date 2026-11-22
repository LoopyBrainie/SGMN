#!/opt/pytorch/bin/python
r"""裁决器**六支自测**：用合成数据把 `adjudicate52.py` 的六个出口逐一打中，确保定论当天不会因逻辑错误白烧。

★ 只用**确定性**合成值（无随机、无 `Date`）：第 `i` 行的 `SHAPE` 由 `i` 的抛物线残差生成，`sd` 可控。
★ 自测**不产生**任何判据读数，**不进**证据表 —— 它只是对裁决器代码的单元测试。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = "/home/ec2-user/sgmn-loop/dir52"
ADJ = f"{HERE}/adjudicate52.py"
N = 200
DELTA_STAR = 0.000430
GATE = 0.000861
MIN_EFF = 160


def rows(shape_fn, n=N, bad=0):
    out = []
    for i in range(n):
        s = 62000 + i
        finite = i >= bad
        sh = shape_fn(i)
        out.append({"seed": s, "SHAPE": sh, "EARLY": sh / 2, "LATE": sh - sh / 2,
                    "TOTAL": sh, "identity_err": 0.0, "finite": finite})
    return out


def spread(amp, mean=0.0):
    """确定性分布：amp·cos(i) 形式，均值≈mean，sd≈amp/√2。"""
    return lambda i: mean + amp * math.cos(i)


def run(tag, rs, extra=()):
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/rows.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        out = f"{d}/adj.json"
        cmd = [sys.executable, ADJ, "--shards", p, "--delta-star", str(DELTA_STAR),
               "--power-gate", str(GATE), "--n-expected", str(N),
               "--max-indet-frac", "0.20", "--min-n-eff", str(MIN_EFF), "--out", out,
               *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise SystemExit(f"[{tag}] 裁决器崩溃")
        j = json.load(open(out, encoding="utf-8"))
        return j["decision"]["outcome"], j


def main():
    cases = []

    # ① PASS：晚段局部化（SHAPE 显著为负，噪声很小）
    cases.append(("PASS", rows(lambda i: -0.001 + 0.00002 * math.cos(i))))

    # ② EARLY-LOCALIZED：SHAPE 显著为正
    cases.append(("EARLY-LOCALIZED", rows(lambda i: +0.001 + 0.00002 * math.cos(i))))

    # ③ NOT-LOCALIZED：均值≈0 且 3se ≤ δ*
    cases.append(("NOT-LOCALIZED", rows(lambda i: 0.00005 * math.cos(i))))

    # ④ INCONCLUSIVE-BELOW-RESOLUTION：均值≈0 但 3se > δ*（sd≈0.0022 ⇒ 3se≈0.00047）
    cases.append(("INCONCLUSIVE-BELOW-RESOLUTION", rows(lambda i: 0.003 * math.cos(i) - 0.0002)))

    # ⑤ INFEASIBLE-POWER：sd 太大 ⇒ 3se > 0.000861
    cases.append(("INFEASIBLE-POWER", rows(lambda i: 0.02 * math.cos(i))))

    # ⑥ INVALID：剔除过多（n_eff = 150 < 160）
    cases.append(("INVALID", rows(lambda i: 0.001 * math.cos(i), bad=50)))

    ok = True
    print("=== adjudicate52 六支自测 ===")
    for want, rs in cases:
        got, j = run(want, rs)
        s = j["SHAPE"]
        flag = "OK " if got == want else "✗  "
        ok = ok and got == want
        print(f"  {flag} 期望 {want:32s} 实得 {got:32s} "
              f"mean={s['mean']:+.6f} 3se={3*s['se']:.6f} n_eff={j['n_eff']}")
    print(f"  总体：{'全部命中' if ok else '有不命中'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
