#!/usr/bin/env python3
"""`DIR-125` 开卡前结构核（`S1 … S12`）。

★ 纪律（`precheck` 的本分，已连续五轮立功）：
  本脚本**只**打印/断言三类东西 ——
    ① 结构事实（`N1/N2`、层数、剂量网格、归档路径）
    ② **已发表**常数（`D-154`/`D-155` 的 `dose=600` 读数、`dose=0` 的读数 —— 本卡之前就有）
    ③ 功效与水平（由 `mech125` 机器算）
  ★★ **绝不**打印/读取任何**新剂量**的观测值 —— 那些格子在正式轮跑之前**不存在**。
      `S12` 明文断言本脚本未调用任何「观测读数函数」。

用法：/opt/pytorch/bin/python dir125/precheck125.py
"""
import os
import sys
import json
import hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mech125 as M                                        # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

CHECKS = []
RES = {}


def rec(cid, ok, detail):
    CHECKS.append({"id": cid, "ok": bool(ok), "detail": detail})
    print(("  [%s] %-9s %s" % ("PASS" if ok else "FAIL", cid, detail)))


# ---------------------------------------------------------------- ★ 仪器化
# `S12` 的判据不是"文本扫描"，而是**真的记录** `load_cells` 被用哪个 dose 调用过。
# 这样才排除"我读了新剂量观测却没在文本里露馅"的情形。
_LOADED_DOSES = []
_orig_load = M.load_cells


def _recording_load(dose):
    _LOADED_DOSES.append(int(dose))
    return _orig_load(dose)


M.load_cells = _recording_load

print("=" * 100)
print("DIR-125 开卡前结构核 —— 剂量扫描（超加性是学习的产物还是先验）")
print("=" * 100)

# ---------------------------------------------------------------- S1 冻结常量
print("\n[S1] 冻结常量（卡片 §2）")
want = {"ALPHA": 0.05, "ALPHA_PLACEBO": 0.05, "SIDE": "two",
        "DOSE_ZERO": 0, "DOSE_PILOT": 25, "DOSE_HIGH": 600,
        "POWER_MIN": 0.80, "STRUCT_N1_PER_SEED": 68, "STRUCT_N2_PER_SEED": 99,
        "MC_REPS": 40000, "MC_SEED": 20261250}
bad = []
for k, v in want.items():
    got = getattr(M, k)
    if got != v:
        bad.append("%s: %r != %r" % (k, got, v))
rec("S1", not bad, "冻结常量 11 项" + ("" if not bad else "；不符：" + "；".join(bad)))
RES["frozen"] = want

# ---------------------------------------------------------------- S2 剂量网格
print("\n[S2] 剂量网格与判据剂量")
ok2 = (tuple(M.DOSE_NEW) == (150, 300, 450)
       and M.DOSE_ALL == (0, 25, 150, 300, 450, 600)
       and len(M.SEEDS) == 9)
rec("S2", ok2, "DOSE_NEW=%r；DOSE_ALL=%r；seeds=%d"
    % (M.DOSE_NEW, M.DOSE_ALL, len(M.SEEDS)))

# ---------------------------------------------------------------- S3 结构常量（真算）
print("\n[S3] 结构常量 —— 由 `n_seen` 直方图**真算**（不是抄常量）")
facts = M.M.build_facts()
tr, ev = M.M.partitions_colocated(facts)
tv = {facts[i]["val"] for i in tr}
items = {r: M.M.make_items(facts, ev, r) for r in M.ALL_RNGS}
ns = {r: M.M.n_seen_vec(items[r], tv) for r in M.ALL_RNGS}
per_q = {}
for r in M.ALL_RNGS:
    per_q[r] = (sum(1 for v in ns[r] if v == 1), sum(1 for v in ns[r] if v == 2))
tot1 = sum(a for a, _ in per_q.values())
tot2 = sum(b for _, b in per_q.values())
print("  逐 qset (N1, N2):", {k: per_q[k] for k in per_q})
print("  单 seed 合计: N1=%d N2=%d" % (tot1, tot2))
ok3 = (per_q == {555: (23, 20), 556: (18, 27), 557: (9, 29), 558: (18, 23)}
       and tot1 == M.STRUCT_N1_PER_SEED and tot2 == M.STRUCT_N2_PER_SEED)
rec("S3", ok3, "逐 qset 与卡片 §2-3 逐位相符；单 seed N1=%d N2=%d" % (tot1, tot2))
RES["struct"] = {str(k): list(v) for k, v in per_q.items()}
RES["per_seed"] = [tot1, tot2]

# ---------------------------------------------------------------- S4 工作例（卡片 §3）
print("\n[S4] 工作例（卡片 §3，冻结）")
u35, cl35 = M.solve_u(35, 23, 20, return_clamped=True)
th35 = M.theta_of(35, 23, 20)
u22, cl22 = M.solve_u(22, 23, 20, return_clamped=True)
th22 = M.theta_of(22, 23, 20)
u225 = M.solve_u(22.5, 23, 20)
th225 = M.theta_of(22.5, 23, 20)
print("  (23,20,m=35): u=%.10f  θ=%.10f  clamped=%s" % (u35, th35, cl35))
print("  (23,20,m=22): u=%.10f  θ=%.10f  clamped=%s" % (u22, th22, cl22))
print("  (23,20,m=22.5): u=%.8f θ=%.8f" % (u225, th225))
# 旧版（夹紧）对照
hi_old = 23 * M.P1(1.0) + 20 * M.P2(1.0)
th_old = 23 * M.P1(1.0) / 35.0
print("  旧版: hi_old=%.4f  θ_old=%.10f  Δθ=%+.10f" % (hi_old, th_old, th35 - th_old))
ok4 = (abs(u35 - 0.2065603008) <= 1e-9 and abs(th35 - 0.5956264661) <= 1e-9
       and abs(th_old - 0.4380952381) <= 1e-9
       and abs((th35 - th_old) - 0.1575312280) <= 1e-9
       and abs(u22 - 1.0) <= 1e-9 and abs(u225 - 0.94898202) <= 1e-8
       and cl35 is False and cl22 is False)
rec("S4", ok4, "工作例 6 个读数逐位相符（含 m=22 处 u=1.0、m=22.5 处连续）")
RES["worked"] = {"u35": u35, "th35": th35, "th_old": th_old, "u22": u22, "u225": u225}

# ---------------------------------------------------------------- S5 池化反例
print("\n[S5] 池化反例（卡片 §2-4 的附带事实）")
u68, cl68 = M.solve_u(133, 68, 99, return_clamped=True)
th68 = M.theta_of(133, 68, 99)
hi_old68 = 68 * M.P1(1.0) + 99 * M.P2(1.0)
th_old68 = 68 * M.P1(1.0) / 133.0
print("  (68,99,m=133): u=%.10f θ=%.10f ; hi_old=%.4f θ_old=%.10f"
      % (u68, th68, hi_old68, th_old68))
ok5 = (abs(u68 - 0.1963949625) <= 1e-9 and abs(th68 - 0.4655612531) <= 1e-9
       and abs(th_old68 - 0.3408521303) <= 1e-9 and 133.0 >= hi_old68)
rec("S5", ok5, "池化 133 > hi_old=%.4f ⇒ 旧夹紧会触发（证明「分层」与「放宽域」是两条独立修正）"
    % hi_old68)
RES["pooled_counterexample"] = {"u": u68, "th": th68, "th_old": th_old68, "hi_old": hi_old68}

# ---------------------------------------------------------------- S6 归档可用性
print("\n[S6] 归档（`dose=0`/`dose=600` 冻结只读）")
c0 = M.load_cells(0)
c600 = M.load_cells(600)
print("  dir119 dose=0: %d cells, seeds=%s" % (len(c0), [c["seed"] for c in c0][:3] + ["..."]))
print("  dir119 dose=600: %d cells, c2 keys=%s"
      % (len(c600), sorted((c600[0]["raw"].get("c2") or {}).keys())))
has_c2_0 = "10000" in ((c0[0]["raw"].get("c2")) or {})
ok6 = (len(c0) == 9 and len(c600) == 9 and not has_c2_0
       and "10000" in ((c600[0]["raw"].get("c2")) or {}))
rec("S6", ok6, "两档各 9 cell；dose=0 **无** c2（结构事实，卡片 §2-3）；dose=600 有 c2")

# ---------------------------------------------------------------- S7 旧域不变性
print("\n[S7] ★ 旧域不变性（`DEF-124-01` 的回归守卫）—— 放宽后 `J124-1` 必须逐位不变")
ctx600 = M.build_ctx(600)
st600 = M.strata_of(ctx600)
obs600 = M.stratum_obs(ctx600, st600)
print("  dose=600 逐层 m_s =", [o["m"] for o in obs600])
hi_old_l = 68 * M.P1(1.0) + 99 * M.P2(1.0)
clamped_layers = [o["key"] for o in obs600
                  if o["m"] > 0 and M.solve_u(o["m"], o["N1"], o["N2"], return_clamped=True)[1]]
print("  hi_old(68,99) = %.4f ; 被夹紧的层 = %s" % (hi_old_l, clamped_layers or "无"))
res600 = M.strat_test(obs600)
print("  J124-1 复算: S=%s m=%s E[S]=%.4f p_two=%.9f dir=%s used=%s clamped=%d"
      % (res600["S"], res600["m"], res600["expect"], res600["p_two"],
         res600["direction"], res600["strata_used"], res600["clamped_layers"]))
ok7 = (res600["S"] == 109 and res600["m"] == 133
       and abs(res600["p_two"] - 0.000366782) <= 1e-9
       and res600["direction"] == "high" and res600["clamped_layers"] == 0
       and not clamped_layers)
rec("S7", ok7, "S=109/m=133/p_two=3.66782e-04 逐位不变；0 层被夹紧 ⇒ 放宽域不改 `J124-1`")
RES["J124_1_recheck"] = {k: res600[k] for k in ("S", "m", "expect", "p_two",
                                               "direction", "strata_used",
                                               "clamped_layers", "p_upper")}

# ---------------------------------------------------------------- S8 旧域 vs 放宽域
print("\n[S8] ★ `dose = 0` 的旧域（夹紧）vs 放宽域 —— **只断言机械事实，数值只报告**")
print("     ★ 为什么不把某个 p 值写成期望：那张卡片 §2-4 已**开卡前**冻结「用放宽域」，")
print("       `S8` 若再对放宽域的具体 p 值下断言，就是在看到结果后调期望（`DEF-124-04` 族）。")
print("       所以这里只断言「夹紧可观测」「旧域确实夹紧了几层」这两件机械事实。")
ctx0 = M.build_ctx(0)
st0 = M.strata_of(ctx0)
obs0 = M.stratum_obs(ctx0, st0)
res0_new = M.strat_test(obs0)
hi_old0 = 23 * M.P1(1.0) + 20 * M.P2(1.0)
print("  dose=0 逐层 m_s =", [o["m"] for o in obs0])
print("  hi_old(23,20) = %.4f ⇒ 旧域**会**夹紧的层（m_s > hi_old）= %d/9"
      % (hi_old0, sum(1 for o in obs0 if o["m"] > hi_old0)))
print("  放宽域: S=%s m=%s E[S]=%.4f p_two=%.9f dir=%s clamped=%d"
      % (res0_new["S"], res0_new["m"], res0_new["expect"], res0_new["p_two"],
         res0_new["direction"], res0_new["clamped_layers"]))

# 逐层对照旧域（mech123 的夹紧版）与放宽域的 θ，数出**确实被旧夹紧改变**的层数
import mech123 as M3OLD
n_diff = 0
print("  逐层 θ 对照（旧夹紧 → 放宽）:")
for o in obs0:
    if o["m"] <= 0:
        continue
    th_old = M3OLD.theta_of(o["m"], o["N1"], o["N2"])
    th_new = M.theta_of(o["m"], o["N1"], o["N2"])
    d = (th_new - th_old)
    if abs(d) > 1e-12:
        n_diff += 1
    print("    m=%-3d θ_old=%.6f θ_new=%.6f Δ=%+.6f%s"
          % (o["m"], th_old, th_new, d, "  ← 被夹紧改变" if abs(d) > 1e-12 else ""))
# 旧口径的 p（用旧 θ 重建零分布）—— 只为报告
ms_o, ths_o = [], []
for o in obs0:
    if o["m"] <= 0:
        continue
    ms_o.append(o["m"])
    ths_o.append(M3OLD.theta_of(o["m"], o["N1"], o["N2"]))
pmf_o = M.convolve_null(ms_o, ths_o)
_, _, two_o = M.tails(pmf_o, sum(o["e1"] for o in obs0))
print("  ★ 报告：旧域 p_two=%.9f（= `D-155` `J124-2` 记录但不采用的值）；放宽域 p_two=%.9f"
      % (two_o, res0_new["p_two"]))

ok8 = (res0_new["executable"] and res0_new["clamped_layers"] == 0 and n_diff == 7)
rec("S8", ok8,
    "放宽域可执行且 0 层夹紧；旧域被夹紧改变的层数 = %d/9（机械事实）" % n_diff)
RES["dose0_recheck"] = {k: res0_new[k] for k in ("S", "m", "expect", "p_two",
                                                 "direction", "strata_used",
                                                 "clamped_layers")}
RES["dose0_old_domain"] = {"p_two": two_o, "n_layers_changed": n_diff,
                           "hi_old": hi_old0}

# ---------------------------------------------------------------- S9 功效（27 层）
print("\n[S9] ★ `J125-1` 的功效与水平（27 层 = 3 新剂量 × 9 seed；机器算）")
strata27 = []
for d in M.DOSE_NEW:
    for s in st600:
        strata27.append({"key": "dose=%d,%s" % (d, s["key"]), "N1": s["N1"], "N2": s["N2"]})
print("  27 层结构: 每层 (N1,N2)=(%d,%d) ⇒ ΣN1=%d ΣN2=%d"
      % (strata27[0]["N1"], strata27[0]["N2"],
         sum(s["N1"] for s in strata27), sum(s["N2"] for s in strata27)))
pw = {}
for ratio in (1.0, 1.5, 1.96, 2.5):
    pw[ratio] = M.power_for_strata(strata27, u=9.2, ratio=ratio, reps=20000)
    print("  ratio=%-5s ⇒ 拒绝率 %.4f" % (ratio, pw[ratio]))
lvl = pw[1.0]
print("  水平（ratio=1）= %.4f（须 ≤ 0.05 + 3·MC_SE）" % lvl)
mc_se = (0.05 * 0.95 / 20000) ** 0.5
ok9 = (lvl <= 0.05 + 3 * mc_se and pw[1.96] >= M.POWER_MIN)
rec("S9", ok9, "水平 %.4f ≤ %.4f；功效(ratio=1.96) %.4f ≥ %.2f"
    % (lvl, 0.05 + 3 * mc_se, pw[1.96], M.POWER_MIN))
RES["power27"] = {str(k): v for k, v in pw.items()}

# ---------------------------------------------------------------- S10 功效（单剂量 9 层）
print("\n[S10] 单剂量 9 层的功效（report-only 口径，`R125-1` 用）")
strata9 = [{"key": s["key"], "N1": s["N1"], "N2": s["N2"]} for s in st600]
pw9 = {}
for ratio in (1.0, 1.5, 1.96, 2.5):
    pw9[ratio] = M.power_for_strata(strata9, u=9.2, ratio=ratio, reps=20000,
                                    seed=M.MC_SEED + 1)
    print("  ratio=%-5s ⇒ 拒绝率 %.4f" % (ratio, pw9[ratio]))
ok10 = pw9[1.0] <= 0.05 + 3 * mc_se
rec("S10", ok10, "单剂量水平 %.4f ≤ %.4f；功效(1.96)=%.4f（report-only ⇒ 不作门槛）"
    % (pw9[1.0], 0.05 + 3 * mc_se, pw9[1.96]))
RES["power9"] = {str(k): v for k, v in pw9.items()}

# ---------------------------------------------------------------- S11 求解器单调性
print("\n[S11] 放宽求解器的单调性与边界")
ms = list(range(1, 43))
us = [M.solve_u(m, 23, 20) for m in ms]
mono = all(us[i] > us[i + 1] for i in range(len(us) - 1))
u_edge, cl_edge = M.solve_u(43, 23, 20, return_clamped=True)
u_over, cl_over = M.solve_u(44, 23, 20, return_clamped=True)
print("  m=1 ⇒ u=%.6g ; m=42 ⇒ u=%.6g ; 严格递减=%s" % (us[0], us[-1], mono))
print("  m=n1+n2=43 ⇒ u=%r clamped=%s ; m=44 ⇒ u=%r clamped=%s"
      % (u_edge, cl_edge, u_over, cl_over))
ok11 = (mono and u_edge == 0.0 and cl_edge is True and u_over == 0.0 and cl_over is True)
rec("S11", ok11, "m=1..42 严格递减；m ≥ n1+n2 返回 0 且 clamped=True（夹紧可观测）")

# ---------------------------------------------------------------- S12 仪器化自检
print("\n[S12] ★ 本脚本**没有**读取任何新剂量观测（仪器化断言，不是文本猜测）")
allowed = {0, 600}
extra = sorted(set(_LOADED_DOSES) - allowed)
print("  load_cells 被调用的 dose 集合 =", sorted(set(_LOADED_DOSES)), "（允许", sorted(allowed), "）")
print("  越界调用 =", extra or "无")
have_new = []
for d in (M.DOSE_PILOT,) + tuple(M.DOSE_NEW):
    if os.path.isdir(M.DIR125_CELLS) and any(
            f.startswith("COLOCATED_%d_" % d) for f in os.listdir(M.DIR125_CELLS)):
        have_new.append(d)
print("  dir125/out/cells 里已有的新剂量格 =", have_new or "无（正式轮之前应为空）")
ok12 = (not extra) and (not have_new)
rec("S12", ok12,
    "load_cells 只被 %s 调用；新剂量格%s"
    % (sorted(set(_LOADED_DOSES)), "为空" if not have_new else "居然已存在 %r" % have_new))

# ---------------------------------------------------------------- 汇总
all_ok = all(c["ok"] for c in CHECKS)
RES["checks"] = CHECKS
RES["all_ok"] = all_ok
with open(os.path.join(OUT, "precheck125.json"), "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=1)

print()
print("=" * 100)
print("%d / %d PASS" % (sum(1 for c in CHECKS if c["ok"]), len(CHECKS)))
print("→ out/precheck125.json")
print("=" * 100)
sys.exit(0 if all_ok else 1)