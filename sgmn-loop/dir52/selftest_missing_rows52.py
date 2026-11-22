#!/opt/pytorch/bin/python
r"""DIR-52 · **缺失行口径自测**（`n_missing_vs_expected` / `indet_vs_expected`）—— 纯报告字段的单元测试。

★ 验证目标：某分片中途退出导致行数 < `--n-expected` 时，
  ① `indet_vs_expected` 如实反映缺失；② **P3 分支不受本字段影响**（冻结口径仍用 `indet`/`n_eff`）；
  ③ 在本卡冻结值（`n_expected=200`、`min_n_eff=160=0.8·200`）下两口径对 P3 **恒等价**。
★ 不产生任何判据读数，不进证据表。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = "/home/ec2-user/sgmn-loop/dir52"
ADJ = f"{HERE}/adjudicate52.py"
DELTA_STAR, GATE, N_EXP = 0.000430, 0.000861, 200


def rows(n_actual, mean=-0.001000, amp=3e-6):
    out = []
    for i in range(n_actual):
        sh = mean + amp * math.cos(i)
        out.append({"seed": 62000 + i, "SHAPE": sh, "EARLY": sh / 2, "LATE": sh - sh / 2,
                    "TOTAL": sh, "identity_err": 0.0, "finite": True})
    return out


def run(n_actual):
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/rows.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows(n_actual):
                f.write(json.dumps(r) + "\n")
        out = f"{d}/adj.json"
        cmd = [sys.executable, ADJ, "--shards", p, "--delta-star", str(DELTA_STAR),
               "--power-gate", str(GATE), "--n-expected", str(N_EXP),
               "--max-indet-frac", "0.20", "--min-n-eff", "160", "--out", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
        return json.load(open(out, encoding="utf-8"))


def run_raw(text, seg=None):
    """把**原始文本**直接喂给裁决器（用于构造截断行/重复行）。"""
    with tempfile.TemporaryDirectory() as d:
        p_ = f"{d}/rows.jsonl"
        open(p_, "w", encoding="utf-8").write(text)
        out = f"{d}/adj.json"
        cmd = [sys.executable, ADJ, "--shards", p_, "--delta-star", str(DELTA_STAR),
               "--power-gate", str(GATE), "--n-expected", str(N_EXP),
               "--max-indet-frac", "0.20", "--min-n-eff", "160", "--out", out]
        if seg:
            cmd += ["--seed-segment", str(seg[0]), str(seg[1])]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
        return json.load(open(out, encoding="utf-8"))


def raw(rs):
    return "".join(json.dumps(r) + "\n" for r in rs)


CASES = [
    (200, "PASS", 0.0, 0.0, 200),
    (190, "PASS", 0.0, 0.05, 190),          # ★ 缺失 10 行：冻结口径 indet=0，期望口径 0.05，分支不变
    (160, "PASS", 0.0, 0.20, 160),          # ★ 恰好压线：两口径都不越界（indet ≤ 0.20 含等号）
    (159, "INVALID", 0.0, 0.205, 159),      # ★ 低于下限：n_eff 口径判 INVALID
    (170, "PASS", 0.0, 0.15, 170),
]

ok = 0
print("=== DIR-52 缺失行口径自测 ===")
for n_actual, want, want_indet, want_indet_exp, want_neff in CASES:
    o = run(n_actual)
    got, neff = o["decision"]["outcome"], o["n_eff"]
    ie, iee = o["indeterminate_frac"], o["indet_vs_expected"]
    miss = o["n_missing_vs_expected"]
    good = (got == want and neff == want_neff
            and abs(ie - want_indet) < 1e-12 and abs(iee - want_indet_exp) < 1e-9
            and miss == N_EXP - n_actual)
    ok += good
    print(f"  {'OK ' if good else 'FAIL'}  行数{n_actual:4d} ⇒ {got:15s} n_eff={neff:3d} "
          f"indet={ie:.3f} indet_vs_expected={iee:.3f} n_missing={miss}")
# ── ★ 第二轮：**完整性加固**自测（截断行 / 重复种子 / 越界种子）──
print("\n=== 完整性加固自测（截断行 / 重复种子 / 越界种子）===")
INT = []


def chk(tag, cond, detail):
    INT.append(cond)
    print(f"  {'OK ' if cond else 'FAIL'}  {tag}：{detail}")


# ① 干净 200 行
o = run_raw(raw(rows(200)))
ig = o["integrity"]
chk("干净 200 行",
    o["decision"]["outcome"] == "PASS" and ig["n_unparsable"] == 0 and ig["n_duplicate_seeds"] == 0
    and o["n_eff"] == 200,
    f"{o['decision']['outcome']} n_eff={o['n_eff']} unparsable={ig['n_unparsable']} dup={ig['n_duplicate_seeds']}")

# ② 末行被截断（模拟进程写一半被杀）⇒ 原实现会崩溃，现应跳过并如实报缺失
txt = raw(rows(200))
cut = txt[:-60]                      # 砍掉末行尾部 ⇒ 末行不成 JSON
o = run_raw(cut)
ig = o["integrity"]
chk("末行截断",
    o["decision"]["outcome"] == "PASS" and ig["n_unparsable"] == 1
    and o["n_eff"] == 199 and o["n_missing_vs_expected"] == 1,
    f"{o['decision']['outcome']} n_eff={o['n_eff']} unparsable={ig['n_unparsable']} "
    f"missing={o['n_missing_vs_expected']}")

# ③ 重复种子（模拟重跑/错 seed0 造成 201 行但只有 200 个种子）
rs = rows(200) + [rows(200)[0]]
o = run_raw(raw(rs))
ig = o["integrity"]
chk("重复种子",
    o["decision"]["outcome"] == "PASS" and ig["n_duplicate_seeds"] == 1
    and ig["n_unique_seeds"] == 200 and o["n_eff"] == 200,
    f"{o['decision']['outcome']} unique={ig['n_unique_seeds']} dup={ig['n_duplicate_seeds']} n_eff={o['n_eff']}")

# ④ 越界种子：判据**不变**（PASS），但必须**大声报**越界数
rs_out = [dict(r, seed=r["seed"] + 1000) for r in rows(200)]
o = run_raw(raw(rs_out), seg=(62000, 62199))
ig = o["integrity"]
chk("越界种子报告",
    o["decision"]["outcome"] == "PASS" and ig["n_outside_segment"] == 200,
    f"{o['decision']['outcome']} outside={ig['n_outside_segment']}（判据不变，仅告警）")

# ⑤ 段内：应为 0
o = run_raw(raw(rows(200)), seg=(62000, 62199))
chk("段内越界数=0", o["integrity"]["n_outside_segment"] == 0,
    f"outside={o['integrity']['n_outside_segment']}")

ok2 = sum(INT)
print(f"总体：{ok2}/{len(INT)} " + ("全部命中" if ok2 == len(INT) else "★ 有失败"))
sys.exit(0 if (ok == len(CASES) and ok2 == len(INT)) else 1)
