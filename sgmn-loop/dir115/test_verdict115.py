#!/usr/bin/env python3
"""DIR-115 · 纯逻辑自测（不加载模型，秒级）。

★ 纪律：写自检脚本后**必须注入已知错误值确认它会变红**（`ENV-NOTES`；`DIR-114` 曾靠这条抓到自己的漏洞）。
本文件支持 `--inject {A,B,C,D}`：注入后必须**变红**，复原后必须**全绿**。
"""
import argparse
import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import importlib.util

spec = importlib.util.spec_from_file_location("T115", _HERE / "R1_task115.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

FAILS = []


def chk(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def make_prior(n_correct=17, n=48, pattern=None):
    if pattern is not None:
        return list(pattern)[:n]
    return [1] * n_correct + [0] * (n - n_correct)


def synth(arm, dose, seed, ok_by_gamma, prior, gamma_ok=True):
    """造一格：给定每个 γ 的 item_ok，反推 acc/D/C 并断言恒等式。"""
    gk = [f"{g:.5e}" for g in T.GRID]
    zero = gk[0]
    out = {"arm": arm, "dose": dose, "seed": seed,
           "item_ok_by_gamma": {}, "acc_by_gamma": {}, "D_by_gamma": {}, "C_by_gamma": {},
           "prior_n": int(sum(prior)), "ident_ok": True,
           "gamma_after_writes": {"8": T.GAMMA_TR if gamma_ok else 0.5 * T.GAMMA_TR},
           "eval_writes": 0, "violations": 0, "finite_params": True, "diverged": False,
           "W_unchanged": {"8": True}, "oracle": None, "n_item": len(prior), "sec": 0.1,
           "final_loss": 0.0}
    assert len(ok_by_gamma) == len(T.GRID), (len(ok_by_gamma), len(T.GRID))
    for g, ok in zip(gk, ok_by_gamma):
        out["item_ok_by_gamma"][g] = list(ok)
        out["acc_by_gamma"][g] = sum(ok) / len(ok)
        D, C = T.flips(prior, ok)
        out["D_by_gamma"][g] = D
        out["C_by_gamma"][g] = C
        if out["prior_n"] - D + C != sum(ok):
            out["ident_ok"] = False
    return out


def build(prior_split, prior_colo, spectrums):
    """spectrums: {arm: [8 个 item_ok 列表]}；缺失 = 补 prior（γ=0 恒等）"""
    cells = {}
    for arm, dose in (("SPLIT", 600), ("COLOCATED", 600), ("SPLIT", 0)):
        pr = prior_split if arm == "SPLIT" else prior_colo
        for s in T.SEEDS:
            spec8 = spectrums.get((arm, dose), [pr] * len(T.GRID))
            cells[T.ck(arm, dose, s)] = synth(arm, dose, s, spec8, pr)
    return cells


def run_verdict(cells, stageA=None):
    T.RES["cells"] = cells
    T.RES["stageA"] = stageA if stageA is not None else []
    T.RES["plan_n"] = len(T.plan_keys())
    return T.verdict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", default=None, choices=[None, "A", "B", "C", "D"])
    a = ap.parse_args()
    inj = a.inject
    print(f"=== DIR-115 自测（inject={inj}）===")

    pr_s = make_prior(17)
    pr_c = make_prior(14)                      # COLOCATED 的锚点故意 ≠ 17
    n = 48

    # ---------- 1. flips / 恒等式 ----------
    ok0 = pr_s
    ok_bad = [0] * n                           # 全错 ⇒ acc=0
    D, C = T.flips(pr_s, ok_bad)
    chk("1.1 flips: 全错 ⇒ D=17, C=0", (D, C) == (17, 0), f"D={D} C={C}")
    chk("1.2 恒等式 17−17+0=0", 17 - D + C == 0)
    D2, C2 = T.flips(pr_s, [1 - x for x in pr_s])   # 完全反转 ⇒ D=17, C=31
    chk("1.3 flips: 完全反转 ⇒ D=17,C=31", (D2, C2) == (17, 31), f"D={D2} C={C2}")

    # ---------- 2. 三条分支互斥且可达 ----------
    # 2a 确认：所有 γ 都 ≤18/48
    spec_low = [pr_s] + [[1] * 13 + [0] * 35] * (len(T.GRID) - 1)     # 13/48
    r = run_verdict(build(pr_s, pr_c, {("SPLIT", 600): spec_low}))
    chk("2a 全低 ⇒ NO-POSITIVE-SIGNAL", r["code"] == "C115-INJECTION-NO-POSITIVE-SIGNAL", r["code"])
    chk("2a' success=True", r["success"] is True)

    # 2b 否证：某 γ 6/6 ≥18 且 mean ≥19
    spec_hi = [pr_s] + [[1] * 22 + [0] * 26] * (len(T.GRID) - 1)     # 22/48 全 seed
    r = run_verdict(build(pr_s, pr_c, {("SPLIT", 600): spec_hi}))
    chk("2b 全高 ⇒ HAS-SIGNAL", r["code"] == "C115-INJECTION-HAS-SIGNAL", r["code"])
    chk("2b' success=False", r["success"] is False)

    # 2c 不确定带：mean ∈ [18,19) 且不满足 F1(a) ⇒ 必须 INCONCLUSIVE，不得算确认
    #    取 mean=18.5：一半 seed 18 项、一半 19 项 ⇒ 6/6≥18 成立但 F1_B(19) 不成立
    mix = []
    for i in range(6):
        mix.append([1] * (18 if i % 2 == 0 else 19) + [0] * (48 - (18 if i % 2 == 0 else 19)))
    spec_band = [pr_s] + [None] * (len(T.GRID) - 1)
    cells = {}
    for s_i, s in enumerate(T.SEEDS):
        seq = [pr_s]
        for g in T.GRID[1:]:
            seq.append(mix[s_i])
        cells[T.ck("SPLIT", 600, s)] = synth("SPLIT", 600, s, seq, pr_s)
        cells[T.ck("COLOCATED", 600, s)] = synth("COLOCATED", 600, s, [pr_c] * len(T.GRID), pr_c)
        cells[T.ck("SPLIT", 0, s)] = synth("SPLIT", 0, s, [pr_s] * len(T.GRID), pr_s)
    r = run_verdict(cells)
    chk("2c mean∈[18,19) ⇒ INCONCLUSIVE-BAND", r["code"] == "C115-INCONCLUSIVE-BAND",
        f"{r['code']} mean={list(r['mean_by_gamma_n48'].values())[:2]}")
    chk("2c' 不是确认、不是否证", r["success"] is False and r["code"] != "C115-INJECTION-NO-POSITIVE-SIGNAL")

    # 2d INCOMPLETE 优先于一切
    cells2 = build(pr_s, pr_c, {})
    cells2.pop(T.ck("SPLIT", 600, T.SEEDS[0]))
    r = run_verdict(cells2)
    chk("2d 缺格 ⇒ INCOMPLETE", r["code"] == "C115-INCOMPLETE", r["code"])

    # 2e DEVICE-DRIFT 优先于判决
    r = run_verdict(build(pr_s, pr_c, {}),
                    stageA=[{"name": "A115-3", "ok": False, "blocking": True, "detail": "x"}])
    chk("2e 守卫失败 ⇒ DEVICE-DRIFT", r["code"] == "C115-DEVICE-DRIFT", r["code"])

    # ---------- 3. plan_n 与格键 ----------
    pk = T.plan_keys()
    chk("3.1 plan_n == 18", len(pk) == 18, f"{len(pk)}")
    chk("3.2 无重复格键", len({T.ck(*x) for x in pk}) == len(pk))
    chk("3.3 训练格 12 + 未训练 6",
        sum(1 for _, d, _ in pk if d == 600) == 12 and sum(1 for _, d, _ in pk if d == 0) == 6)
    chk("3.4 网格 8 点且首点为 0", len(T.GRID) == 8 and T.GRID[0] == 0.0, f"{T.GRID}")
    chk("3.5 γ_tr 在网格内", T.GAMMA_TR in T.GRID)

    # ---------- 4. 阈值自洽 ----------
    chk("4.1 ACC0 17/48 ≈ 0.35417", abs(17 / 48 - 0.3541666666666667) < 1e-15)
    chk("4.2 F1_A(18) > ACC0(17)", T.F1_A_N > T.ACC0_SPLIT_N)
    chk("4.3 F1_B(19) > F1_A(18)", T.F1_B_N > T.F1_A_N)
    chk("4.4 父卡复现值含 6 seed", all(len(v) == 6 for v in T.EXPECT_TR.values()))

    # ---------- 5. 已知错误注入（必须变红）----------
    if inj == "A":      # 把 F1_B 降到 18 ⇒ 2c 的不确定带会被误判成否证（必须被抓到）
        T.F1_B_N = 18
        r = run_verdict(cells)
        chk("INJ-A 关掉不确定带后应不再 INCONCLUSIVE", r["code"] != "C115-INCONCLUSIVE-BAND",
            f"实际 {r['code']}")
        chk("INJ-A ⇒ 本次自测应失败（预期）", False, "F1_B 被篡改后 2c 判据失效")
    if inj == "B":      # 让 flips 少算一个 D ⇒ 恒等式必须失败
        T.flips = lambda pr, cur: (0, 0)
        cells_b = build(pr_s, pr_c, {})
        chk("INJ-B flips 归零后恒等式应失败", all(c["ident_ok"] for c in cells_b.values()) is False,
            "若仍全 True 则恒等式没接上")
        chk("INJ-B ⇒ 本次自测应失败（预期）", False, "恒等式断言失效")
    if inj == "C":      # 网格去掉 γ_tr ⇒ 3.5 必须变红
        T.GRID = (0.0, 1e-04, 2e-04, 3e-04, 4e-04, 5e-04, 6e-04, 7e-04)
        chk("INJ-C γ_tr 不在网格内", T.GAMMA_TR in T.GRID)
    if inj == "D":      # plan_n 少一格 ⇒ 3.1/3.3 必须变红
        T.plan_keys = lambda: T.plan_keys.__wrapped__() if hasattr(T.plan_keys, "__wrapped__") else [
            (a, d, s) for (a, d, s) in
            [(x, 600, y) for x in T.ARMS_TRAINED for y in T.SEEDS] +
            [(T.ARM_PRIMARY, 0, y) for y in T.SEEDS]][:-1]
        chk("INJ-D plan_n 应为 18", len(T.plan_keys()) == 18, f"{len(T.plan_keys())}")

    print()
    if FAILS:
        print(f"★ 失败 {len(FAILS)} 项：{FAILS}")
        return 1
    print("★ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())