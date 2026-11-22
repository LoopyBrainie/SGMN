"""`DIR-126` 开卡前结构核（**阻塞门**）。

★ 只允许打印三类东西（`D-124` 起确立的纪律）：
  ① 结构事实（题集数、`N₁_s`、闸门、`n_seen` 直方图）；
  ② **已发表**常数的逐位复现（`D-156` 的 6 个剂量读数）—— 它们不是本卡的观测；
  ③ 判据的 **level / 非空转**（逻辑层，不含新剂量观测）。

★★ 严禁打印任何**新剂量**（`350/375/400`）的判据读数 —— 此刻它们还不存在。
  用**仪器化**断言（`S12`）而不是文本扫描来保证这一点（`DEF-125-04` 的修法）。

用法：
    /opt/pytorch/bin/python dir126/precheck126.py
"""
import os
import sys
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import mech126 as M                                   # noqa: E402

OUT = os.path.join(HERE, "out")
RES = []
_LOG = []


def rec(name, ok, detail, blocking=True):
    RES.append({"id": name, "ok": bool(ok), "detail": detail, "blocking": bool(blocking)})
    print("  %-10s %-5s %s" % (name, "PASS" if ok else "✗FAIL", detail))
    return bool(ok)


def close(a, b, tol=1e-4):
    return (a is not None and b is not None and abs(a - b) <= tol)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 104)
    print("DIR-126 开卡前结构核（12 条；只打结构事实 + 已发表常数 + 逻辑层 level）")
    print("=" * 104)

    # ---------------- S1 剂量轴
    print("\n[S1] 剂量轴")
    ok1 = (M.DOSE_ALL9 == (0, 25, 150, 300, 350, 375, 400, 450, 600))
    rec("S1", ok1, "DOSE_ALL9 = %s（9 个；新格 = %s）" % (list(M.DOSE_ALL9), list(M.DOSE_NEW2)))

    # ---------------- S2 结构常量（**真算**，不抄）
    print("\n[S2] 结构常量（每层每 1 个题集的 N₁/N₂ —— 由 n_seen 直方图真算）")
    _, st1, _ = M.strata_and_obs(0)                   # dose=0 只有 555
    _, st4, _ = M.strata_and_obs(600)                 # dose=600 有 4 个题集
    n1_1, n2_1 = st1[0]["N1"], st1[0]["N2"]
    n1_4, n2_4 = st4[0]["N1"], st4[0]["N2"]
    gate1 = (2.0 * n1_1 + n2_1) / 3.0
    gate4 = (2.0 * n1_4 + n2_4) / 3.0
    ok2 = (n1_1 == 23 and n2_1 == 20 and n1_4 == 68 and n2_4 == 99
           and abs(gate1 - 22.0) < 1e-12 and abs(gate4 - 78.3333333) < 1e-6)
    rec("S2", ok2, "1 题集 (N₁,N₂)=(%d,%d) 闸门=%.4f ；4 题集 (%d,%d) 闸门=%.6f"
        % (n1_1, n2_1, gate1, n1_4, n2_4, gate4))

    # ---------------- S3 工作例 W1..W5 逐位
    print("\n[S3] 工作例 W1–W5 逐位（§3-2）")
    W = {}

    def one(e1, e2, n1=68, n2=99):
        rows = [{"key": "w", "N1": n1, "N2": n2, "e1": int(e1), "e2": int(e2),
                 "m": int(e1) + int(e2)}]
        r = M.sign_row(rows)
        return {"frac": r["frac"], "w1": r["w"]["w1"], "w2": r["w"]["w2"],
                "ratio": r["w"]["ratio"], "pred": r["pred"], "obs": r["obs"],
                "agree": r["agree"]}

    W["W1"] = one(46, 35)
    W["W2"] = one(42, 37)
    W["W4"] = one(41, 30)
    W["W5"] = one(49, 38)
    ok3 = (close(W["W1"]["ratio"], 0.955844) and W["W1"]["agree"] is True
           and close(W["W2"]["ratio"], 0.676715) and W["W2"]["agree"] is True
           and W["W4"]["frac"] == 0.0 and close(W["W4"]["ratio"], 0.873148)
           and W["W4"]["agree"] is False
           and W["W5"]["frac"] == 1.0 and close(W["W5"]["ratio"], 1.034972)
           and W["W5"]["agree"] is False)
    rec("S3", ok3,
        "W1 ratio=%.6f 符=%s ；W2 ratio=%.6f 符=%s ；W4 frac=%.1f ratio=%.6f 符=%s(F-a) ；"
        "W5 frac=%.1f ratio=%.6f 符=%s(F-b)"
        % (W["W1"]["ratio"], W["W1"]["agree"], W["W2"]["ratio"], W["W2"]["agree"],
           W["W4"]["frac"], W["W4"]["ratio"], W["W4"]["agree"],
           W["W5"]["frac"], W["W5"]["ratio"], W["W5"]["agree"]))

    # ---------------- S4 crit_rho（机器算 + 逐位验边界）
    print("\n[S4] `J126-2` 的临界 ρ（精确置换，`n=9`，单侧 `α=0.05`）")
    x9 = list(range(9))
    ps = M.perm_spearman(x9, list(reversed(x9)), side="less")
    n_perm = ps["n_perm"]
    crit = ps["crit_rho"]
    # ★ 逐位验边界：p(crit) <= 0.05 且 下一个更靠近 0 的 rho 的 p > 0.05
    import itertools
    rhos = []
    for p in itertools.permutations(range(9)):
        r = M.spearman(x9, [x9[i] for i in p])
        if r is not None:
            rhos.append(r)
    tot = len(rhos)

    def p_at(c):
        return sum(1 for r in rhos if r <= c + 1e-12) / float(tot)

    p_crit = p_at(crit)
    above = sorted(t for t in set(rhos) if crit < t < 0)
    p_nxt = p_at(above[0]) if above else None
    ok4 = (n_perm == 362880 and close(crit, -0.6, 1e-9)
           and p_crit <= 0.05 and (p_nxt is None or p_nxt > 0.05))
    rec("S4", ok4, "n_perm=%d  crit_rho=%.6f  p(crit)=%.6f≤0.05  p(次近0的%.6f)=%.6f>0.05"
        % (n_perm, crit, p_crit, (above[0] if above else float('nan')),
           (p_nxt if p_nxt is not None else float('nan'))))

    # ---------------- S5 非空转（构造性 + 网格）
    print("\n[S5] `J126-4` 非空转（构造性见证 W4/W5 + 网格分歧率）")
    p = M.nonvacuous_probe()
    ok5 = (W["W4"]["agree"] is False and W["W5"]["agree"] is False
           and p["nonvacuous"] is True and p["discordance"] >= M.NONVAC_MIN_DISCORD)
    rec("S5", ok5, "构造性 (F-a)/(F-b) 都可复现 ；网格 discordance=%.6f (bad_a=%.4f, bad_b=%.4f) "
                   "门槛=%.2f ⇒ 非恒等式"
        % (p["discordance"], p["p_bad_a"], p["p_bad_b"], M.NONVAC_MIN_DISCORD))

    # ---------------- S6 已发表常数复现（D-156 的 6 个剂量）
    print("\n[S6] `D-156` 已发表读数逐位复现（**已发表常数**，不是本卡观测）")
    PUB = {0: (0.8589, 0.9353), 25: (0.6422, 0.7262), 150: (0.7376, 0.4596),
           300: (0.3389, 0.7587), 450: (0.0000, 1.4336), 600: (0.0000, 1.9571)}
    ok6 = True
    msgs = []
    for d, (pf, pr) in PUB.items():
        _, _, obs = M.strata_and_obs(d)
        r = M.sign_row(obs)
        good = (abs(r["frac"] - pf) <= 1e-4 and close(r["w"]["ratio"], pr, 1e-3))
        ok6 = ok6 and good
        msgs.append("d%d:frac=%.4f/%.4f ratio=%.4f/%.4f%s"
                    % (d, r["frac"], pf, r["w"]["ratio"], pr, "" if good else "✗"))
    rec("S6", ok6, " ".join(msgs))

    # ---------------- S7 m=0 不进分母
    print("\n[S7] `frac` 的分母：`m_s = 0` 的层不进分母")
    rows7 = [{"key": "z", "N1": 68, "N2": 99, "e1": 0, "e2": 0, "m": 0},
             {"key": "a", "N1": 68, "N2": 99, "e1": 41, "e2": 30, "m": 71}]
    f7 = M.frac_low_u(rows7)
    ok7 = (f7["m_tot"] == 71 and f7["n_valid"] == 1 and f7["frac"] == 0.0)
    rec("S7", ok7, "m_tot=%d(须 71) n_valid=%d(须 1) frac=%.1f" % (f7["m_tot"], f7["n_valid"], f7["frac"]))

    # ---------------- S8 判别力（H126 vs H126-alt）
    print("\n[S8] `H126` 与 `H126-alt` 在三个新剂量上**可区分**（设计有效性）")
    # `H126-alt`：ratio>1 ⟺ dose >= 450；`H126`：用实测 frac。
    # ★ 此刻没有新剂量数据 ⇒ 只能检查「两支的预测函数在剂量轴上不同」这一结构事实。
    alt_pred = {d: (1 if d >= 450 else -1) for d in M.DOSE_ALL9}
    # `frac` 侧的预测在旧剂量上与 alt 完全一致（这是两支拟合 6/6 的原因）；
    # 若某个新剂量的 frac 为 0 ⇒ 两支预测不同 ⇒ 该点有判别力。
    alt_matches_old = True
    for d in M.DOSE_OLD:
        _, _, obs = M.strata_and_obs(d)
        r = M.sign_row(obs)
        if r["pred"] is not None and r["pred"] != alt_pred[d]:
            alt_matches_old = False
    ok8 = bool(alt_matches_old)
    rec("S8", ok8, "两支在 6 个旧剂量上预测完全一致(fit 6/6 的原因)=%s ⇒ 三个新剂量是唯一的判别窗口"
        % alt_matches_old)

    # ---------------- S9 J126-1 的 level（逻辑层，无 α）
    print("\n[S9] `J126-1` 的 level（**逻辑预测，无抽样 α**）")
    rec("S9", True,
        "若 `H126` 精确为真 ⇒ `frac=0 ⇒ ratio>1` 是**逻辑推论** ⇒ 观察不到的失败 ⇒ "
        "level = 0（不是 0.05 的抽样检验）；反之任一 (F-a)/(F-b) 出现即**确定**否证 ⇒ "
        "检出概率 = 1（不是部分功效）。⇒ 本判据唯一需要的是**非空转**（S5 已过）",
        blocking=False)

    # ---------------- S10 J126-2 的 level（精确置换）
    print("\n[S10] `J126-2` 的 level（精确置换 ⇒ 构造上精确）")
    ok10 = (p_crit <= 0.05 and p_nxt is not None and p_nxt > 0.05)
    rec("S10", ok10, "拒绝域 {ρ ≤ -0.600} 在独立性零下的精确概率 = %.6f ≤ α=0.05 ⇒ level 精确"
        % p_crit)

    # ---------------- S11 归档基线 md5
    print("\n[S11] 归档基线（只读；本卡不得改动）")
    import hashlib
    import glob as _g
    def md5s(pat):
        out = {}
        for f in sorted(_g.glob(pat)):
            out[os.path.basename(f)] = hashlib.md5(open(f, "rb").read()).hexdigest()
        return out
    d119 = md5s(os.path.join(M.M.DIR119_CELLS, "COLOCATED_*.json"))
    d125 = md5s(os.path.join(M.M.DIR125_CELLS, "COLOCATED_*.json"))
    ok11 = (len(d119) == 18 and len(d125) == 36)
    rec("S11", ok11, "dir119 = %d 个 cell（须 18）；dir125 = %d 个（须 36）⇒ 基线已记"
        % (len(d119), len(d125)))

    # ---------------- S12 盲态：新剂量在磁盘上没有任何 cell
    print("\n[S12] 盲态断言：本卡的新剂量 `{350,375,400}` 此刻**无任何 cell**（`DEF-125-04` 的修法）")
    # ★ 第一版把断言写成「`load_cells` 不得被请求新剂量」—— 错了：本 check 会故意扫全部 9 个剂量，
    #   `load_cells` 必然被**请求**（然后抛错）。真正要保证的是「**磁盘上不存在**新剂量的读数」
    #   ⇒ precheck 无论怎么写都不可能看到它们（`DEF-126-03`）。
    import glob as _gg
    found = {}
    for d in M.DOSE_NEW2:
        hits = []
        for pth in (M.M.DIR119_CELLS, M.M.DIR125_CELLS, M.DIR126_CELLS):
            hits += _gg.glob(os.path.join(pth, "COLOCATED_%d_*.json" % d))
        found[int(d)] = len(hits)
    bad = {d: n for d, n in found.items() if n > 0}
    ok12 = (len(bad) == 0)
    rec("S12", ok12, "新剂量的既有 cell 数 = %s（须全 0 ⇒ precheck 跑在盲态）" % found)

    # ---------------- 汇总
    n_ok = sum(1 for r in RES if r["ok"])
    blk = [r for r in RES if r["blocking"] and not r["ok"]]
    print("\n" + "=" * 104)
    print("★ 结构核 %d/%d PASS，阻断项 %d" % (n_ok, len(RES), len(blk)))
    print("=" * 104)
    dump = {"id": "DIR-126 precheck", "card": "rev126/v10",
            "n_pass": n_ok, "n_total": len(RES), "n_blocking_fail": len(blk),
            "items": RES,
            "crit_rho": crit, "crit_rho_p": p_crit, "n_perm": n_perm,
            "nonvac": {k: p[k] for k in ("discordance", "p_bad_a", "p_bad_b",
                                         "p_frac0", "p_ratio_hi", "nonvacuous")},
            "worked": W,
            "struct": {"gate_1q": gate1, "gate_4q": gate4,
                       "n1_1q": n1_1, "n2_1q": n2_1, "n1_4q": n1_4, "n2_4q": n2_4},
            "d119_md5_before": d119, "d125_md5_before": d125,
            "new_dose_cells_present": found}
    with open(os.path.join(OUT, "precheck126.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    print("产物：dir126/out/precheck126.json")
    return 0 if not blk else 1


if __name__ == "__main__":
    sys.exit(main())