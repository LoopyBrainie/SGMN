#!/opt/pytorch/bin/python
r"""DIR-58 · **Stage A 硬门**（卡 §8 的 `A58-1..14`；任一失败 ⇒ 禁止正式轮）。

用法：`stageA58.py` → `dir58/out/DIR58r10_stageA.json`（退出码 0 = 全过）。
★ `A58-14` 在本脚本内**直接跑** smoke（6 种子 × 两臂），因为它必须证明"协议可执行且判据量可算"。
"""
import ast
import hashlib
import inspect
import json
import os
import statistics
import sys
import textwrap

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402
import R1_task58 as T58      # noqa: E402

OUT = os.path.join(ROOT, "dir58", "out")
SEED0_SMOKE, N_SMOKE = 59500, 6
HISTORY = [
    (10000, 10399), (11000, 11549), (12000, 13095), (13500, 14511), (15000, 15899),
    (16000, 16411), (17000, 18911), (20400, 20999), (21000, 21111), (21500, 22063),
    (22100, 22103), (22500, 22899), (23000, 23003), (23990, 24399), (25000, 25019),
    (25200, 26199), (26200, 27199), (27400, 28399), (28800, 28819), (29400, 30419),
    (31000, 31199), (31400, 32419), (33000, 36999), (41000, 41039), (51000, 52999),
    (53000, 53019), (55000, 55023), (56000, 56095), (57000, 57095), (58000, 58463),
    (59000, 59095), (61000, 61007), (62000, 62399),
]
SEG = (59500, 59595)
E_SEED, E_FROZEN, S_FROZEN = 59000, 0.5451143521469336, 0.41518522379015543


def md5(path):
    h = hashlib.md5()
    with open(os.path.join(ROOT, path), "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def sha16(t):
    return hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()[:16]


def indep_alpha0(U, Vm, n):
    U, Vm = U.clone(), Vm.clone()
    for _ in range(n):
        W = M._W(U, Vm)
        G = M.LAM * W
        U = U - M.ETA * (G @ Vm)
        Vm = Vm - M.ETA * (G.t() @ U)
    return U, Vm


def main():
    os.makedirs(OUT, exist_ok=True)
    res, ok = {}, True

    def put(tag, passed, detail):
        nonlocal ok
        res[tag] = {"pass": bool(passed), "detail": detail}
        ok = ok and bool(passed)
        print("%-8s %s  %s" % (tag, "PASS" if passed else "FAIL", detail))

    # A58-1
    bad = {p: md5(p) for p in T58.DEVICE if md5(p) != T58.DEVICE_MD5[p]}
    put("A58-1", not bad, "六件装置 md5 %s" % ("全部一致" if not bad else bad))

    # A58-2
    P1 = R.build(SEED0_SMOKE, "V0")
    P2 = R.build(SEED0_SMOKE, "V0")
    d2 = float((P1["U1"] - P2["U1"]).abs().max()) + float((P1["V1"] - P2["V1"]).abs().max())
    put("A58-2", d2 == 0.0, "build 两次 max|Δ| = %.1e" % d2)

    # A58-3 / A58-4 / A58-5
    mx3 = mx4 = 0.0
    disjoint = True
    for s in range(SEED0_SMOKE, SEED0_SMOKE + N_SMOKE):
        st = T58.seed_state(s)
        ks, ku, kn = st["ks"], st["k_un"], st["k_new"]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                mx3 = max(mx3, abs(float((ks[i] @ ks[j]) / (ks[i].norm() * ks[j].norm()))))
        mx4 = max(mx4, abs(float(st["v_new"] @ st["vP"])))
        for a in ks + ku + [kn]:
            for b in ks + ku + [kn]:
                if a is not b:
                    c = abs(float((a @ b) / (a.norm() * b.norm())))
                    disjoint = disjoint and c < 1 - 1e-12
    put("A58-3", mx3 <= 0.30, "已写键两两 |cos| max = %.4f" % mx3)
    put("A58-4", mx4 <= 0.30, "|v_I·v_P| max = %.4f" % mx4)
    put("A58-5", disjoint and len({T58.OFF_K, T58.OFF_V, T58.OFF_R, T58.OFF_I}) == 4,
        "键集合两两 |cos| < 1−1e-12 = %s；四 generator 偏移互异" % disjoint)

    # A58-6
    src = inspect.getsource(T58.read_metrics)
    _tree = ast.parse(textwrap.dedent(src))
    _fn = _tree.body[0]
    _body = _fn.body[1:] if (_fn.body and isinstance(_fn.body[0], ast.Expr)
                             and isinstance(_fn.body[0].value, ast.Constant)) else _fn.body
    _code = "\n".join(ast.unparse(n) for n in _body)
    put("A58-6", "cos" not in _code, "read_metrics 去文档串后代码 %d 字符，无 'cos'" % len(_code))

    # A58-7
    st = T58.seed_state(SEED0_SMOKE)
    m = T58.read_metrics(st["U0"], st["V0"], st["ks"][0], st["vP"], st["k_un"], st["k_new"], st["v_new"])
    id7 = (m["S_old"] == m["E_null"] - m["E_old"]) and (m["S_new"] == m["E_null"] - m["E_new"]) \
        and (m["E_null"] == statistics.median(m["E_un"]))
    put("A58-7", id7, "S 恒等式逐位 + E_null = 5 值中位：%s" % id7)

    # A58-8
    Pe = R.build(E_SEED, "V0")
    ste = T58.seed_state(E_SEED)
    me = T58.read_metrics(ste["U0"], ste["V0"], ste["ks"][0], ste["vP"],
                          ste["k_un"], ste["k_new"], ste["v_new"])
    Ua, Va = T48.step_pair(ste["U0"].clone(), ste["V0"].clone(), ste["ks"][0], ste["vP"], 0.0)
    Ub, Vb = indep_alpha0(ste["U0"], ste["V0"], T48.N_REP)
    Wa, Wb = M._W(Ua, Va), M._W(Ub, Vb)
    dW = float((Wa - Wb).abs().max())
    dE = abs(float((Wa @ ste["ks"][0] - ste["vP"]).norm() - (Wb @ ste["ks"][0] - ste["vP"]).norm()))
    dU = float((Ua - Ub).abs().max()) + float((Va - Vb).abs().max())
    ok8 = (abs(me["E_old"] - E_FROZEN) < 1e-12) and (abs(me["S_old"] - S_FROZEN) < 1e-12) \
        and dW == 0.0 and dE == 0.0 and dU == 0.0
    put("A58-8", ok8, "seed59000 E=%.16f（冻结 %.16f）S=%.16f；独立重实现 max|ΔW|=%.1e |ΔE|=%.1e"
        % (me["E_old"], E_FROZEN, me["S_old"], dW, dE))

    # A58-9
    st9 = T58.seed_state(SEED0_SMOKE)
    rD, rW = T58.run_cells(st9, "D"), T58.run_cells(st9, "W")
    w_starts = {c["start_sha16"] for c in rW["cells"].values()}
    ok9 = (rD["common_sha16"] == rW["common_sha16"]
           and rD["cells"]["q0"]["start_sha16"] == rD["common_sha16"]
           and all(c["start_sha16"] != rD["common_sha16"] for c in rW["cells"].values())
           and len(w_starts) == 1)
    put("A58-9", ok9, "共同起点 %s；D 不动作；W 各格同一起点且已改（%d 个起点）"
        % (rD["common_sha16"], len(w_starts)))

    # A58-10
    calls = {"a1": 0, "a0": 0}
    orig = T48.step_pair

    def spy(U, Vm, k, v, alpha):
        calls["a1" if float(alpha) == 1.0 else ("a0" if float(alpha) == 0.0 else "x")] += 1
        return orig(U, Vm, k, v, alpha)

    T48.step_pair = spy
    try:
        calls.update({"a1": 0, "a0": 0})
        if "x" in calls:
            calls.pop("x")
        st10 = T58.seed_state(SEED0_SMOKE)
        calls.update({"a1": 0, "a0": 0})
        T58.run_cells(st10, "W")
        tot_w = dict(calls)
        calls.update({"a1": 0, "a0": 0})
        T58.run_cells(st10, "D")
        tot_d = dict(calls)
    finally:
        T48.step_pair = orig
    exp_w = {"a1": len(T58.QS["W"]), "a0": sum(q - 1 for q in T58.QS["W"])}
    exp_d = {"a1": 0, "a0": sum(T58.QS["D"])}
    ok10 = tot_w == exp_w and tot_d == exp_d and 0 not in T58.QS["W"] and T58.QS["W"][0] == 1
    put("A58-10", ok10, "臂 W 总量 %s（期望 %s）· 臂 D 总量 %s（期望 %s）· q=0 不在 W"
        % (tot_w, exp_w, tot_d, exp_d))

    # A58-11
    fin = all(isinstance(x, float) and x == x and abs(x) != float("inf")
              for r in (rD, rW) for c in r["cells"].values() for x in
              (c["E_old"], c["E_new"], c["E_null"], c["S_old"], c["S_new"], c["W_norm"]))
    adj_src = open(os.path.join(ROOT, "dir58", "adjudicate58.py")).read() if \
        os.path.exists(os.path.join(ROOT, "dir58", "adjudicate58.py")) else None
    ok11 = fin and adj_src is not None and 'QSTAR = (1, 2, 4)' in adj_src and '"D"' in adj_src
    put("A58-11", ok11, "读数全有限 %s；裁决器只读臂 D 且 q*∈{1,2,4} %s" % (fin, adj_src is not None))

    # A58-12
    ok12 = all(not (SEG[0] <= lo <= SEG[1] or SEG[0] <= hi <= SEG[1] or (lo < SEG[0] and hi > SEG[1]))
               for lo, hi in HISTORY)
    put("A58-12", ok12, "正式段 %d..%d 与 %d 个历史段不相交" % (SEG[0], SEG[1], len(HISTORY)))

    # A58-13
    s1, s2 = T58.seed_state(SEED0_SMOKE), T58.seed_state(SEED0_SMOKE)
    ok13 = all(sha16(a) == sha16(b) for a, b in zip(s1["ks"] + s1["k_un"] + [s1["k_new"]],
                                                   s2["ks"] + s2["k_un"] + [s2["k_new"]])) \
        and float((s1["vP"] - s2["vP"]).abs().max()) == 0.0 \
        and float((s1["v_new"] - s2["v_new"]).abs().max()) == 0.0
    put("A58-13", ok13, "k_I / v_I / 未写键两次生成逐位相同（配对性）")

    # A58-14：smoke（6 种子 × 两臂），判据量必须可算
    rows = {"D": [T58.run_seed(SEED0_SMOKE + i, "D") for i in range(N_SMOKE)],
            "W": [T58.run_seed(SEED0_SMOKE + i, "W") for i in range(N_SMOKE)]}
    need = {"q", "E_old", "E_new", "E_null", "S_old", "S_new", "W_norm", "Wk_norm",
            "E_un", "start_sha16"}
    keys_ok = all(need <= set(c)
                  for arm in rows for r in rows[arm] for c in r["cells"].values())
    qh = []
    for i in range(N_SMOKE):
        SD = [rows["D"][i]["cells"]["q%d" % q]["S_old"] for q in T58.QS["D"]]
        tgt = SD[0] / 2
        q12 = None
        for j in range(len(SD) - 1):
            if SD[j] >= tgt >= SD[j + 1]:
                q12 = T58.QS["D"][j] + (T58.QS["D"][j + 1] - T58.QS["D"][j]) * \
                    (SD[j] - tgt) / (SD[j] - SD[j + 1])
                break
        qh.append(q12)
    nb = sum(1 for x in qh if x is None)
    smoke = {"n": N_SMOKE, "cells_complete": keys_ok, "q12_computable": N_SMOKE - nb,
             "S_old_D_q0": [rows["D"][i]["cells"]["q0"]["S_old"] for i in range(N_SMOKE)],
             "delta_q1": [rows["W"][i]["cells"]["q1"]["S_new"] - rows["D"][i]["cells"]["q1"]["S_new"]
                          for i in range(N_SMOKE)]}
    put("A58-14", keys_ok and (N_SMOKE - nb) >= N_SMOKE - 1, "smoke %d 种子：字段完整 %s · 半衰期可算 %d/%d"
        % (N_SMOKE, keys_ok, N_SMOKE - nb, N_SMOKE))

    res["smoke"] = smoke
    res["all_pass"] = ok
    with open(os.path.join(OUT, "DIR58r10_stageA.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n★ Stage A %s（%d 条断言）" % ("全过" if ok else "有失败", len([k for k in res if k.startswith("A58")])))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()