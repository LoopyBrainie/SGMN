"""`test_verdict119.py` —— `DIR-119` 裁决的**独立复核**（只读 JSON + numpy，**无 torch/GPU**，可本地跑）。

★ 为什么要有它：`R1_task119.py --stage merge` 既**汇总**又**裁决**，是同一段代码自证。
  本文件**从 cell JSON 重新独立算一遍**裁决，与 `stage_verdict.json` 对撞。
  两套实现同时错才会一致通过 —— 这是本项目一贯的「交叉校验」做法。

★ 复核四项：
  1. **字段自洽**：`acc_argmin == rank_hist[0]/48`、`acc_argmax == rank_hist[3]/48`、
     `Σrank_hist == 48`、`len(ok) == 48`、`sum(ok)/48 == acc_argmin`；
  2. **统计重算**：9 seed 均值 / 单样本 t（`t(μ0=0.25)` 与 `t_GAP(μ0=0)`）与 `stage_verdict.json` 比；
  3. **判据重判**：用**卡片冻结的阈值**独立重判 `J119-1..4` 与 `verdict`；
  4. **恒等交叉**：`acc_argmax` 必须等于 `rank_hist[3]` 的比例（`mech119` 侧已证，此处再验存档）。
★ `--inject N` 故意破坏第 N 项 ⇒ 证明本复核**会红**。
用法：`/usr/bin/python3 dir119/test_verdict119.py [--inject N]`
"""
import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
OUT = _HERE / "out"
N_ITEM = 48
SEEDS = (20271126, 20271203, 20271210,
         20260501, 20260502, 20260503, 20260504, 20260505, 20260506)
JUDGE = "10000"
#: ★ 卡片冻结阈值（**不得**从 R1_task119 import —— 独立复核对阈值也要独立声明，
#:   这样若有人在实现里偷改阈值，本复核会红）
M2_ACC = 0.90
M1_GAP = 0.20
T_MIN = 2.0
DOSES = (600, 0)

FAILS = []


def chk(tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(tag)


def tstat(xs, mu0):
    xs = [float(x) for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m, sd = float(np.mean(xs)), float(np.std(xs, ddof=1))
    if sd == 0.0:
        # ★ `DEF-119-03`（本 cycle 自查抓到）：退化情形必须**保号**。
        #   原写法返回 `+inf`，于是"均值显著**低于** μ0"（本应 t=−inf、J119-4 通过）
        #   被判成 t=+inf ⇒ J119-4 失败。真数据 sd>0 时不会发作，但仍须正确。
        return 0.0 if m == mu0 else (float("inf") if m > mu0 else float("-inf"))
    return (m - mu0) / (sd / np.sqrt(len(xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", type=int, default=0)
    a = ap.parse_args()
    inj = a.inject

    print("=" * 78)
    print("DIR-119 裁决独立复核")
    print("=" * 78)

    cells, miss = {}, []
    for dose in DOSES:
        for s in SEEDS:
            f = OUT / "cells" / f"COLOCATED_{dose}_{s}.json"
            if not f.exists():
                miss.append(f"{dose}|{s}")
                continue
            cells[f"{dose}|{s}"] = json.loads(f.read_text(encoding="utf-8"))
    chk("① 18 格齐全（0 缺）", not miss, f"缺 = {miss}" if miss else "18/18")

    # ── ② 字段自洽（逐格逐 mult）
    bad = []
    for k, d in cells.items():
        for mg, me in (d.get("mech_by_mult") or {}).items():
            if not me.get("finite"):
                continue
            rh = me["rank_hist"]
            if sum(rh) != N_ITEM:
                bad.append((k, mg, "Σrank_hist", sum(rh)))
            if abs(me["acc_argmin"] - rh[0] / float(N_ITEM)) > 1e-12:
                bad.append((k, mg, "acc_argmin≠rank_hist[0]/48",
                            me["acc_argmin"] - rh[0] / float(N_ITEM)))
            if abs(me["acc_argmax"] - rh[3] / float(N_ITEM)) > 1e-12:
                bad.append((k, mg, "acc_argmax≠rank_hist[3]/48",
                            me["acc_argmax"] - rh[3] / float(N_ITEM)))
            if len(me["ok"]) != N_ITEM:
                bad.append((k, mg, "|ok|", len(me["ok"])))
            elif abs(sum(me["ok"]) / float(N_ITEM) - me["acc_argmin"]) > 1e-12:
                bad.append((k, mg, "sum(ok)/48≠acc", 0))
    chk("② 字段自洽（18 格 × 9 点：Σrank=48、argmin≡rank1、argmax≡rank4、sum(ok)/48==acc）",
        not bad, f"不符 {len(bad)} 处：{bad[:3]}" if bad else "全部相符")

    # ── ③ 未训练参照：dose=0 应回到 ~37.5%（prior=18）
    pri = [cells[f"0|{s}"]["prior_n"] for s in SEEDS if f"0|{s}" in cells]
    chk("③ dose=0 的 prior_n 全为 18（γ=0 先验恒定）",
        bool(pri) and all(p == 18 for p in pri), f"prior_n = {pri}")

    # ── ④ 统计重算 + ⑤ 判据重判
    accmin, accmax, gaps = [], [], []
    for s in SEEDS:
        k = f"600|{s}"
        if k not in cells:
            continue
        me = (cells[k].get("mech_by_mult") or {}).get(JUDGE, {})
        if not me.get("finite"):
            continue
        accmin.append(me["acc_argmin"])
        accmax.append(me["acc_argmax"])
        gaps.append(None if me["err_in_train_frac"] is None
                    else me["err_in_train_frac"] - me["err_in_train_null"])
    gp = [g for g in gaps if g is not None]
    my = {"acc_argmin_mean": float(np.mean(accmin)) if accmin else None,
          "acc_argmax_mean": float(np.mean(accmax)) if accmax else None,
          "GAP_mean": float(np.mean(gp)) if gp else None,
          "t_argmin_vs_025": tstat(accmin, 0.25),
          "t_GAP": tstat(gp, 0.0)}
    j = {"J119-1": bool(my["acc_argmax_mean"] is not None and my["acc_argmax_mean"] >= M2_ACC),
         "J119-2": bool(my["GAP_mean"] is not None and my["GAP_mean"] >= M1_GAP),
         "J119-3": bool(my["t_GAP"] is not None and my["t_GAP"] >= T_MIN),
         "J119-4": bool(my["t_argmin_vs_025"] is not None and my["t_argmin_vs_025"] <= -T_MIN)}
    if len(accmin) < len(SEEDS):
        verd = "CENSORED"
    elif not j["J119-4"]:
        verd = "REPLICATION-FAILURE"
    elif j["J119-1"]:
        verd = "M2"
    elif j["J119-2"] and j["J119-3"]:
        verd = "M1"
    else:
        verd = "M-OTHER"

    if inj == 4:
        my["acc_argmax_mean"] = -1.0
    if inj == 5:
        j["J119-2"] = not j["J119-2"]
    if inj == 6:
        verd = "M-OTHER" if verd != "M-OTHER" else "M2"

    print(f"      重算：acc_argmin_mean={my['acc_argmin_mean']} "
          f"(t vs 0.25 = {my['t_argmin_vs_025']})")
    print(f"      重算：acc_argmax_mean={my['acc_argmax_mean']}")
    print(f"      重算：GAP_mean={my['GAP_mean']} (t = {my['t_GAP']})")
    print(f"      重判：{j} ⇒ ★ verge = {verd}")

    sv = OUT / "stage_verdict.json"
    got = json.loads(sv.read_text(encoding="utf-8")) if sv.exists() else None
    if got is None:
        chk("④ stage_verdict.json 存在", False, "merge 还没跑？")
        print(f"\n断言失败条数 = {len(FAILS)}")
        return 1

    ok4 = all(got.get("judge", {}).get(k2) == v2 for k2, v2 in j.items())
    chk("④ 判据重判与 merge 一致（J119-1..4）", ok4,
        f"merge={got.get('judge')} vs 复核={j}" if not ok4 else "四项一致")

    gs = got.get("stat") or {}
    diffs = []
    for k2, v2 in my.items():
        w = gs.get(k2)
        if v2 is None or w is None:
            if v2 != w:
                diffs.append((k2, v2, w))
        elif abs(float(v2) - float(w)) > 1e-9:
            diffs.append((k2, v2, w))
    chk("⑤ 统计重算与 merge 一致（5 个量）", not diffs, f"不符：{diffs}" if diffs else "一致")

    chk("⑥ 裁决与 merge 一致", got.get("verdict") == verd,
        f"merge={got.get('verdict')} vs 复核={verd}")
    chk("⑦ 裁决属合法枚举", verd in ("M1", "M2", "M-OTHER", "CENSORED", "REPLICATION-FAILURE"),
        f"verdict={verd}")

    # ── ⑧ 未训练参照的直接读数（dose=0 的 acc 应显著高于 dose=600）
    if accmin:
        a0 = [cells[f"0|{s}"]["mech_by_mult"][JUDGE]["acc_argmin"]
              for s in SEEDS if f"0|{s}" in cells]
        chk("⑧ dose=0 的 acc_argmin 高于 dose=600（训练把读出打到随机以下）",
            bool(a0) and float(np.mean(a0)) > float(np.mean(accmin)),
            f"dose=0 均值={float(np.mean(a0)):.4f} vs dose=600 均值={float(np.mean(accmin)):.4f}")

    print("-" * 78)
    print(f"断言失败条数 = {len(FAILS)}")
    print("★ 全部 PASS" if not FAILS else f"★ {len(FAILS)} 项未通过：{FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())