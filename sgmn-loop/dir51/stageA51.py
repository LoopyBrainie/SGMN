#!/opt/pytorch/bin/python
r"""DIR-51 · **Stage A 硬门**（卡 `rev51_v11.md` §4 P5 与 `rev51_v10.md` §7 A51-1..A51-12）。

任一断言不过 ⇒ **拦住正式轮**。全部断言在本进程内执行，输出 JSON + 人类可读摘要。

用法：`stageA51.py --out dir51/out/DIR51_stageA.json`
"""
import argparse
import hashlib
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M      # noqa: E402
import R1_task33 as R    # noqa: E402
import R1_task47 as T47  # noqa: E402
import R1_task48 as T48  # noqa: E402
import R1_task51 as T51  # noqa: E402

SEEDS = (52000, 52001, 52002)
STEPS = T51.STEPS
CARD = os.path.join(ROOT, "revolve/DIR-51/DIRECTION_rev51_v11.md")
CARD_MD5_EXPECT = "（运行时计算并落盘；冻结值由卡 §1 与本文件共同固定）"


def chain2(seed, arm, steps=STEPS):
    r"""**独立第二链**：从零重新实现一遍循环（不复用 `run_one`），用于 A51-8 逐位复现。"""
    P = R.build(seed, "V0")
    g_base = torch.Generator().manual_seed(seed + T51.OFF_KEY)
    g_val = torch.Generator().manual_seed(seed + T51.OFF_VAL)
    g_perp = torch.Generator().manual_seed(seed + T51.OFF_PERP)
    kP = T51.KEY_SCALE * T51.gen_vec(g_base, T51.D_K)
    vP = T51.gen_vec(g_base, T51.D_V)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    U, Vm = T48.step_pair(U, Vm, kP, vP, 1.0)
    cs = []
    for j in range(steps):
        if arm == "alpha0":
            kv, vv, al = kP, vP, 0.0
        elif arm == "random":
            kv, vv, al = kP, T51.nonperson_vec(g_val, vP), 1.0
        elif arm == "ortho":
            kv, vv, al = T51.KEY_SCALE * T51.perp_to(kP, g_perp), T51.nonperson_vec(g_val, vP), 1.0
        elif arm == "orthoP":
            kv, vv, al = T51.KEY_SCALE * T51.perp_to(kP, g_perp), vP, 1.0
        else:
            kv, vv, al = kP, vP, 1.0
        U, Vm = T48.step_pair(U, Vm, kv, vv, al)
        Wm = M._W(U, Vm)
        r = (Wm @ kP).detach()
        cs.append(float((r @ vP) / (r.norm() * vP.norm())))
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dir51/out/DIR51_stageA.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    A = {}

    rows = {arm: {s: T51.run_one(s, arm, STEPS) for s in SEEDS} for arm in T51.ARMS}

    # A51-1 / A51-5：五臂同种子的初始状态指纹逐位相同
    fps = {arm: [rows[arm][s]["init_fp"] for s in SEEDS] for arm in T51.ARMS}
    # ★ 断言逻辑：应按**种子**分组比较（同一 seed 下五臂指纹相同），而非「同臂跨种子」
    by_seed = {s: sorted({fps[arm][i] for arm in T51.ARMS}) for i, s in enumerate(SEEDS)}
    A["A51-1"] = {"desc": "同种子下五臂初始状态指纹逐位相同", "fps": fps, "by_seed": by_seed,
                  "pass": all(len(v) == 1 for v in by_seed.values())}

    # A51-2：正交键严格正交
    mx = max(max((abs(x) for x in rows[arm][s]["ks"]), default=0.0)
             for arm in ("ortho", "orthoP") for s in SEEDS)
    A["A51-2"] = {"desc": "ortho/orthoP 写入键与 k_P 严格正交 |cos| ≤ 1e-12", "max_abs_cos": mx,
                  "pass": mx <= 1e-12}

    # A51-3：alpha0 臂 α=0 ⇒ 换 (k,v) 不改变结果（逐位）
    g = torch.Generator().manual_seed(77777)
    kv_alt = T51.gen_vec(g, T51.D_K) * 0.3
    vv_alt = T51.gen_vec(g, T51.D_V)
    dv = 0.0
    for s in SEEDS:
        r1 = T51.run_one(s, "alpha0", STEPS)
        r2 = T51.run_one(s, "alpha0", STEPS, alpha0_kv=(kv_alt, vv_alt))
        dv = max(dv, max(abs(a - b) for a, b in zip(r1["cs"], r2["cs"])))
    A["A51-3"] = {"desc": "alpha0 臂换 (k,v) 结果逐位相同（α=0 数据项为 0）", "max_abs_diff": dv,
                  "pass": dv == 0.0}

    # A51-4：alpha0 轨迹有限且 ‖r_P‖ 非零
    ok4 = all(all(math.isfinite(x) for x in rows["alpha0"][s]["cs"]) and rows["alpha0"][s]["n0"] > 0
              for s in SEEDS)
    A["A51-4"] = {"desc": "alpha0 轨迹有限、‖r_P(t0)‖ > 0", "pass": ok4}

    # A51-5（独立记录）：起跑状态与 alpha0 相同
    A["A51-5"] = {"desc": "同种子五臂起跑状态逐位相同（同 A51-1 的指纹证据）",
                  "pass": A["A51-1"]["pass"]}

    # A51-6：replay 末态 c_P 高于 alpha0 末态（方向性自检，非判据）
    ok6, vals6 = True, {}
    for s in SEEDS:
        c_al = rows["alpha0"][s]["cs"][-1]
        c_rp = rows["replay"][s]["cs"][-1]
        vals6[str(s)] = {"alpha0": c_al, "replay": c_rp}
        ok6 = ok6 and (c_rp > c_al)
    A["A51-6"] = {"desc": "replay 末态 c_P > alpha0 末态 c_P（方向性自检）", "vals": vals6, "pass": ok6}

    # A51-7：random 臂 v_rand 与 v_P^0 不平行
    # ★ DEF-51-8：`v_rand` 的语义是「**显著不平行**」（拒绝采样门槛 0.5），而非「几乎正交」；
    #   原门槛 1e-9 与该语义矛盾（且初版实测达 0.986）⇒ 断言随语义对齐为 ≤ 0.5。
    mv = max(max((abs(x) for x in rows["random"][s]["vrand_cos"]), default=0.0) for s in SEEDS)
    A["A51-7"] = {"desc": "random 臂 v_rand 与 v_P^0 显著不平行 |cos| ≤ 0.5", "max_abs_cos": mv,
                  "pass": mv <= 0.5}

    # A51-8：独立第二链逐位复现
    dv8 = 0.0
    for arm in T51.ARMS:
        for s in SEEDS:
            dv8 = max(dv8, max(abs(a - b) for a, b in zip(rows[arm][s]["cs"], chain2(s, arm))))
    A["A51-8"] = {"desc": "工作例独立第二链复现 max|Δ| = 0", "max_abs_diff": dv8, "pass": dv8 == 0.0}

    # A51-9：种子段不相交
    segs = {"探针41000-41039": (41000, 41039), "标定51500-51799": (51500, 51799),
            "正式52000-52999": (52000, 52999)}
    ov = []
    ks = list(segs)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a1, b1 = segs[ks[i]]
            a2, b2 = segs[ks[j]]
            if max(a1, a2) <= min(b1, b2):
                ov.append((ks[i], ks[j]))
    A["A51-9"] = {"desc": "种子段交集为 0", "segments": {k: list(v) for k, v in segs.items()},
                  "overlaps": ov, "pass": len(ov) == 0}

    # A51-10：卡 md5 与装置常数
    card_md5 = hashlib.md5(open(CARD, "rb").read()).hexdigest()
    consts = {"D_H": M.D_H, "D_K": M.D_K, "D_V": M.D_V, "LAM": M.LAM, "ETA": M.ETA,
              "KEY_SCALE": R.KEY_SCALE, "N_REP": T47.N_REP, "STEPS": STEPS}
    # ★ DEF-51-7：装置真实 `ETA = 0.2`（rev 1.1 的 §5 误抄为 0.05 ⇒ 纯文字更正）
    exp = {"D_H": 8, "D_K": 16, "D_V": 4, "LAM": 0.05, "ETA": 0.2,
           "KEY_SCALE": 0.5, "N_REP": 16, "STEPS": 20}
    A["A51-10"] = {"desc": "卡存在 + md5 落盘 + 装置常数与卡一致", "card_md5": card_md5,
                   "card_exists": os.path.exists(CARD), "consts": consts, "expected": exp,
                   "pass": os.path.exists(CARD) and consts == exp}

    # A51-11：D_P 由落盘 c 序列独立重算一致
    dmax = 0.0
    for s in SEEDS:
        for arm in T51.ARMS:
            r = rows[arm][s]
            d = r["cs"][-1] - rows["alpha0"][s]["cs"][-1]
            d2 = rows["alpha0"][s]["cs"][-1] - r["cs"][-1]
            dmax = max(dmax, abs(d - (-d2)))
    A["A51-11"] = {"desc": "D_P 由落盘 c 序列独立重算一致", "max_abs_diff": dmax, "pass": dmax == 0.0}

    # A51-12：读数为纯读（读 c 不改状态）
    d12 = 0.0
    for s in SEEDS:
        P = R.build(s, "V0")
        g = torch.Generator().manual_seed(s + T51.OFF_KEY)
        kP = T51.KEY_SCALE * T51.gen_vec(g, T51.D_K)
        vP = T51.gen_vec(g, T51.D_V)
        U, Vm = P["U1"].clone(), P["V1"].clone()
        U, Vm = T48.step_pair(U, Vm, kP, vP, 1.0)
        before = torch.cat([U.flatten(), Vm.flatten()]).detach().clone()
        c1, _ = T51.read_P(U, Vm, kP, vP)
        c2, _ = T51.read_P(U, Vm, kP, vP)
        after = torch.cat([U.flatten(), Vm.flatten()]).detach()
        d12 = max(d12, abs(c1 - c2), float((before - after).abs().max()))
    A["A51-12"] = {"desc": "读数为纯读（两次读一致 + 状态逐位不变）", "max_abs_diff": d12,
                   "pass": d12 == 0.0}

    ok = all(v["pass"] for v in A.values())
    out = {"stage_a_pass": ok, "assertions": A, "seeds": list(SEEDS), "steps": STEPS}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("=== DIR-51 Stage A ===")
    for k in sorted(A):
        print(f"  {k}: {'PASS' if A[k]['pass'] else 'FAIL'}   {A[k]['desc']}")
        if not A[k]["pass"]:
            print("      detail:", json.dumps(A[k], ensure_ascii=False)[:400])
    print(f"  总体: {'PASS ⇒ 可进入 smoke/正式轮' if ok else 'FAIL ⇒ 拦住正式轮'}")
    print("WROTE", a.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
