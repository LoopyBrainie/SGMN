#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-42 · **判定表的合成逻辑自检**（`merge` vs 卡 §5 的四个手算工作例）

★ **这不是实验**：不导入装置（`dir31`/`dir33`）、不训练、不产生任何实验读数。
  它只把 `R1_task42.py` 里的 `merge`/`paired`/增量落盘函数的**源码本体**抽出来，
  用**合成数**构造 400 行分片，然后检查裁决结局与卡片 §5 **逐例一致**。
⇒ ⇒ 用途：**冻结前**确认「实现里的判定表」与「卡里写的判定表」是同一张表
（这是纪律 97 的延伸：断言要真的读被断言的对象）。

用法：  python3 dir42/verify_merge42.py
退出码：0 = 全部一致；1 = 有不一致。
"""
import ast
import json
import math
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "R1_task42.py")

# ── 与 `R1_task42.py` 顶部的常量**逐字相同**（冻结值）────────────────────────
SEED0, N_INST, N_REP = 20261111, 400, 16
INST_OFF = list(range(22500, 22500 + N_INST))
SIG_K, FRAC_HEADROOM, COS_SIGNATURE = 3.0, 0.20, 0.5
TRAIN_WIN, EVAL_WIN = (1, 8), (9, 15)
STEPS, LR = 250, 0.03
ARMS = ("oracle", "rand", "learned")


def load_funcs():
    """抽取**源码本体**（不是复制品）⇒ 测的就是真正会跑的代码"""
    src = open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    need = {"paired", "merge", "_jsonl_path", "_row_cfg", "_append_jsonl", "_read_jsonl",
            "_md5", "_card_guard", "_fmean_safe"}
    segs = [ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in need]
    if len(segs) != len(need):
        raise SystemExit(f"★ 在 {SRC} 里只找到 {len(segs)}/{len(need)} 个目标函数")
    # ★ `merge` 已挂上**冻结守卫** ⇒ 这里造一份**已冻结**的假卡（md5 对得上）
    #   ⇒ ⇒ 本自检顺带证明「守卫对合法卡片**不误伤**」。
    card = os.path.join(tempfile.mkdtemp(), "CARD.md")
    with open(card, "w", encoding="utf-8") as f:
        f.write("# 合成卡片（自检用）\n")
    import hashlib as _h
    ns = {"os": os, "json": json, "math": math, "statistics": statistics,
          "time": __import__("time"), "OUT": tempfile.mkdtemp(),
          "ARMS": ARMS, "SEED0": SEED0, "N_INST": N_INST, "N_REP": N_REP,
          "INST_OFF": INST_OFF, "SIG_K": SIG_K, "FRAC_HEADROOM": FRAC_HEADROOM,
          "COS_SIGNATURE": COS_SIGNATURE, "TRAIN_WIN": TRAIN_WIN, "EVAL_WIN": EVAL_WIN,
          "STEPS": STEPS, "LR": LR, "ROW_SCHEMA": 1,
          "hashlib": _h, "pathlib": __import__("pathlib"),
          "CARD_PATH": card, "CARD_MD5_EXPECTED": _h.md5(open(card, "rb").read()).hexdigest()}
    exec("\n\n".join(segs), ns)
    return ns


def offsets(target_sd, n=N_INST):
    """零均值、确定性（无随机）的偏移序列，样本 sd 精确等于 `target_sd`"""
    base = [-1.96, -0.98, 0.0, 0.98, 1.96]
    b = [base[i % 5] for i in range(n)]
    s = statistics.stdev(b)
    return [x * target_sd / s for x in b]


def synth(ns, Lr_mean, Ll_mean, Lo_mean, Lrc_mean, sd_delta, sd_C):
    """构造三臂 400 行；均值与 sd 精确匹配卡 §5 的合成数"""
    cfg = ns["_row_cfg"]()
    d_off, c_off = offsets(sd_delta), offsets(sd_C)
    out = {"oracle": [], "rand": [], "learned": []}

    def row(i, L, extra=None, cos=0.0, dW=0.0):
        r = {"idx": i, "seed": SEED0 + INST_OFF[i], "L": L, "L_seen": L, "L_full": L,
             "cos": cos, "align": cos, "q_norm": 1.0, "y_norm": 1.0, "m_abs": 0.0,
             "Wmax": 1.0, "dWq1": dW, "gn_Wq1": 0.1, "cfg": cfg}
        if extra:
            r.update(extra)
        return r

    for i in range(N_INST):
        Ll = Ll_mean
        Lr = Ll + (Lr_mean - Ll_mean) + d_off[i]
        Lrc = Ll + (Lrc_mean - Ll_mean) + c_off[i]
        out["oracle"].append(row(i, Lo_mean, cos=1.0))
        out["rand"].append(row(i, Lr, cos=0.0))
        out["learned"].append(row(i, Ll, extra={"L_rc": Lrc}, cos=0.9, dW=0.5))
    return out


def write_shards(ns, rows):
    OUT = ns["OUT"]
    for arm, rr in rows.items():
        for sh in range(4):
            part = [r for r in rr if r["idx"] % 4 == sh]
            with open(os.path.join(OUT, f"DIR42r10_arm_{arm}_shard{sh}.json"), "w") as f:
                json.dump({"arm": arm, "shard": sh, "rows": part}, f)


def main():
    ns = load_funcs()
    merge = ns["merge"]
    # 卡 §5 给的 se ⇒ sd = se·√n
    sd_d = 0.80 * (N_INST ** 0.5)
    sd_c = 0.90 * (N_INST ** 0.5)

    cases = [
        ("例A · 第6支 PASS_42", dict(Lr_mean=18.00, Ll_mean=4.00, Lo_mean=1.60, Lrc_mean=15.00),
         "PASS"),
        ("例B · 第3支 NO-OUT-OF-SAMPLE-GAIN", dict(Lr_mean=18.00, Ll_mean=17.00, Lo_mean=1.60,
                                                   Lrc_mean=17.05), "NO-OUT-OF-SAMPLE-GAIN"),
        ("例C · 第4支 NOT-CONTENT-DEPENDENT", dict(Lr_mean=18.00, Ll_mean=4.00, Lo_mean=1.60,
                                                    Lrc_mean=4.05), "NOT-CONTENT-DEPENDENT"),
        ("例D · 第3支（P1 不过 ⇒ 第5支不可达）", dict(Lr_mean=18.00, Ll_mean=16.00, Lo_mean=1.60,
                                                     Lrc_mean=4.00), "NO-OUT-OF-SAMPLE-GAIN"),
        ("例D变体 · 第4支（P1 过、P3 不过）", dict(Lr_mean=18.00, Ll_mean=15.40, Lo_mean=1.60,
                                                  Lrc_mean=4.00), "NOT-CONTENT-DEPENDENT"),
        ("★ 第5支 BELOW-EFFECT-FLOOR（P1 过、P3 过、**只有地板不过**）",
         dict(Lr_mean=18.00, Ll_mean=15.00, Lo_mean=1.60, Lrc_mean=26.00),
         "BELOW-EFFECT-FLOOR"),
    ]
    ok = True
    for name, kw, expect in cases:
        rows = synth(ns, sd_delta=sd_d, sd_C=sd_c, **kw)
        write_shards(ns, rows)
        r = merge()
        got = r["outcome"]
        good = got == expect
        ok = ok and good
        print(f"  {'OK ' if good else 'FAIL'} {name}\n"
              f"       outcome={got}（期望 {expect}）  "
              f"Δ={r['delta_out']['mean']:.3f}/3se={r['delta_out']['3se']:.3f}  "
              f"floor={r['floor_0.2R']:.3f}  C={r['content_C']['mean']:.3f}/"
              f"3se={r['content_C']['3se']:.3f}  P1={r['P1_signif']} "
              f"P2={r['P2_floor']} P3={r['P3_content']}")

    # ── 第1支 INCOMPLETE：缺一个分片 ⇒ ⇒ 必须报「缺臂或 n 不符」，**不得**继续裁决
    rows = synth(ns, sd_delta=sd_d, sd_C=sd_c, Lr_mean=18.00, Ll_mean=4.00,
                 Lo_mean=1.60, Lrc_mean=15.00)
    write_shards(ns, rows)
    os.remove(os.path.join(ns["OUT"], "DIR42r10_arm_learned_shard3.json"))
    r = merge()
    good = (r["outcome"] == "INCOMPLETE" and r["counted"] is False)
    ok = ok and good
    print(f"  {'OK ' if good else 'FAIL'} ★ 第1支 INCOMPLETE（缺 1 个分片）"
          f" ⇒ outcome={r['outcome']} counted={r['counted']} missing={r['missing']}")

    # ── 第2支 DIVERGED：某个读数非有限 ⇒ ⇒ 不得被当成「不过」
    rows = synth(ns, sd_delta=sd_d, sd_C=sd_c, Lr_mean=18.00, Ll_mean=4.00,
                 Lo_mean=1.60, Lrc_mean=15.00)
    rows["learned"][7]["L"] = float("nan")
    write_shards(ns, rows)
    r = merge()
    good = (r["outcome"] == "DIVERGED" and r["counted"] is False)
    ok = ok and good
    print(f"  {'OK ' if good else 'FAIL'} ★ 第2支 DIVERGED（读数非有限）"
          f" ⇒ outcome={r['outcome']} counted={r['counted']} finite={r.get('finite')}")

    print("\n★ 判定表与卡 §5 " + ("逐例一致（覆盖**全部六支**）" if ok else "**不一致**"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
