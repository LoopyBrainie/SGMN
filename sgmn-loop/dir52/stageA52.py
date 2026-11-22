#!/opt/pytorch/bin/python
r"""DIR-52 · **Stage A 硬门**（卡 §6，A52-1…A52-10）。任一不过 ⇒ 拦住正式轮。

★ 设计：每条断言是一个函数，**结果一算出即落盘**（`DIR52_stageA_partial.json`），
  重跑时 `A52_RESUME=1`（默认）**只补算缺失的断言** ⇒ 不会因为后来某条断言写错而重烧前面的昂贵计算
  （`DEF-52-2/3/4` 三次教训）。

用法：`stageA52.py`（默认 resume）· `A52_RESUME=0 stageA52.py`（全量重算）
"""
import hashlib
import json
import os
import sys

import torch

for _p in ("dir26s", "dir26r", "dir26o", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g",
           "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22", "dir52"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G                      # noqa: E402
import R1_gating12 as K12                  # noqa: E402
import R1_dose_ar1_v3 as V3                # noqa: E402
import R1_task52 as T52                    # noqa: E402
from R1_gating import D_MODEL, M, drive, make_core     # noqa: E402
from R1_gating7 import build_keep, PROFILE             # noqa: E402
from R1_gating12 import (N_READ, R_HO8, R_PRES, cal_at, read_stream,
                         ret_jitter_margin)            # noqa: E402

ROOT = "/home/ec2-user/sgmn-loop"
OUT = f"{ROOT}/dir52/out"
CARD = f"{ROOT}/revolve/DIR-52/DIRECTION_rev52_v10.md"
PART = f"{OUT}/DIR52_stageA_partial.json"
SEEDS_KEY = list(range(62000, 62008))
SEEDS_HEAVY = [62000, 62001]
TAU = None
RS_CACHE = {}


def RS_(seed):
    if seed not in RS_CACHE:
        RS_CACHE[seed] = T52.run_seed(seed)
    return RS_CACHE[seed]


def mu_at(seed, D, keep, vb, RS8):
    c = make_core(D_MODEL, seed, K12.R0)
    drive(c, D, "R", keep=keep)
    P = c._W().detach() - c.W0
    return ret_jitter_margin(P, RS8["K"], vb, N_READ, R_HO8)["mu"]


class A52Dict(dict):
    def __init__(self, path):
        super().__init__()
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        json.dump(dict(self), open(self._path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=str)


# ─────────────────────────────────── 断言
def a1():
    w = json.load(open(f"{OUT}/DIR52_workex.json", encoding="utf-8")) \
        if os.path.exists(f"{OUT}/DIR52_workex.json") else None
    return {"desc": "workex52 三项逐位核验全过", "workex_pass": bool(w and w.get("pass")),
            "workex": (w or {}).get("checks"), "pass": bool(w and w.get("pass"))}


def a2(ctx):
    ok, info = True, {}
    for pk in ("F", "K"):
        keep, _ = build_keep(V3.N_KV, R_PRES, ctx["cnt"], PROFILE[pk])
        lens, prev = [], set()
        for rho in (1, 2, 3, 4):
            kt = T52.trunc_keep(keep, rho)
            lens.append(len(kt))
            ok = ok and prev <= kt
            prev = kt
        ok = ok and all(lens[i] < lens[i + 1] for i in range(3)) and lens[-1] == sum(ctx["cnt"])
        info[pk] = {"lens": lens, "sum_cnt": sum(ctx["cnt"])}
    return {"desc": "keep(1)⊂keep(2)⊂keep(3)⊂keep(4)、长度严格递增、末项=sum(cnt)",
            "detail": info, "pass": ok}


def a3(ctx):
    okk, d3 = True, 0.0
    for s in SEEDS_KEY:
        Da, Db = V3.build_write(s, 0.0, V3.N_KV), V3.build_write(s, 0.9, V3.N_KV)
        okk = okk and bool(torch.equal(Da["K"][0], Db["K"][0])) \
                  and bool(torch.equal(Da["va"], Db["va"])) and bool(torch.equal(Da["vb"], Db["vb"]))
    for s in SEEDS_HEAVY:
        Da, Db = V3.build_write(s, 0.0, V3.N_KV), V3.build_write(s, 0.9, V3.N_KV)
        vbs, RS = Da["vb"][:N_READ], read_stream(s, R_HO8)
        for pk in ("F", "K"):
            keep, _ = build_keep(V3.N_KV, R_PRES, RS_(s)["cnt0"], PROFILE[pk])
            k1 = T52.trunc_keep(keep, 1)
            d3 = max(d3, abs(mu_at(s, Da, k1, vbs, RS) - mu_at(s, Db, k1, vbs, RS)))
    return {"desc": "ΔδG_1 ≡ 0：两档 φ 的第 1 轮读数逐位相同",
            "keys_bit_equal_8seeds": okk, "mu_max_abs_diff_2seeds": d3,
            "pass": okk and d3 == 0.0}


def a4(ctx):
    d4, rows = 0.0, {}
    for s in SEEDS_HEAVY:
        r = RS_(s)
        d4 = max(d4, r["identity_err"])
        rows[str(s)] = {"EARLY": r["EARLY"], "TOTAL": r["TOTAL"],
                        "SHAPE": r["SHAPE"], "identity_err": r["identity_err"]}
    ctx["rows"] = rows
    return {"desc": "Σ_ρ ΔδG_ρ = ΔG(4) 每种子 ≤1e-12（完整性检查，非判据）",
            "max_abs_err": d4, "pass": d4 <= 1e-12}


def a5(ctx):
    ok = True
    for s in SEEDS_HEAVY:
        r = RS_(s)
        ok = ok and (r["phi"]["0.9"]["cnt_source"] == "frozen_cnt0") \
                and (r["phi"]["0.0"]["cnt_source"] == "B@this_phi") \
                and (r["phi"]["0.0"]["keep_n"] == r["phi"]["0.9"]["keep_n"])
    return {"desc": "φ=0.9 用 frozen_cnt0，且两档 keep 数逐位相同（配额被夹住）", "pass": ok}


def a6(ctx):
    s = SEEDS_HEAVY[0]
    Da = V3.build_write(s, 0.0, V3.N_KV)
    RS = read_stream(s, R_HO8)
    vbs = Da["vb"][:N_READ]
    cnt = RS_(s)["cnt0"]
    d6 = 0.0
    for phi in (0.0, 0.9):
        D = V3.build_write(s, phi, V3.N_KV)
        for pk in ("F", "K"):
            keep, _ = build_keep(V3.N_KV, R_PRES, cnt, PROFILE[pk])
            kt = {t for t in keep if t < 2 * V3.N_KV * 2 * M}      # 独立重写：前 2 轮
            d6 = max(d6, abs(mu_at(s, D, kt, vbs, RS) - RS_(s)["phi"][str(phi)]["mu"][pk]["2"]))
    return {"desc": "独立第二链（从零重写截断+drive+读数）max|Δ| = 0", "max_abs_diff": d6,
            "pass": d6 == 0.0}


def a7(ctx):
    s = SEEDS_HEAVY[0]
    D, RS = V3.build_write(s, 0.0, V3.N_KV), read_stream(s, R_HO8)
    vb = D["vb"][:N_READ]
    keep, _ = build_keep(V3.N_KV, R_PRES, RS_(s)["cnt0"], PROFILE["F"])
    c = make_core(D_MODEL, s, K12.R0)
    drive(c, D, "R", keep=T52.trunc_keep(keep, 2))
    before = c._W().detach().clone()
    m1 = ret_jitter_margin(c._W().detach() - c.W0, RS["K"], vb, N_READ, R_HO8)["mu"]
    m2 = ret_jitter_margin(c._W().detach() - c.W0, RS["K"], vb, N_READ, R_HO8)["mu"]
    d7 = max(abs(m1 - m2), float((before - c._W().detach()).abs().max()))
    return {"desc": "读数为纯读（两次读一致 + W 逐位不变）", "max_abs_diff": d7, "pass": d7 == 0.0}


def a8(ctx):
    landed = set()
    p26 = f"{ROOT}/dir26s/out/DIR26S_rows.json"
    if os.path.exists(p26):
        landed |= {int(r["seed"]) for r in json.load(open(p26, encoding="utf-8"))["rows"]}
    d51 = f"{ROOT}/dir51/out"
    if os.path.isdir(d51):
        for f in os.listdir(d51):
            if f.endswith(".jsonl"):
                for line in open(f"{d51}/{f}", encoding="utf-8"):
                    line = line.strip()
                    if line:
                        landed.add(int(json.loads(line)["seed"]))
    formal, probe = set(range(62000, 62400)), set(range(61000, 61008))
    ov = sorted(formal & landed) + sorted(formal & probe)
    return {"desc": "正式段 62000..62399 与既往落盘种子交集为 0",
            "n_landed_seen": len(landed), "overlaps": ov[:20], "pass": len(ov) == 0}


def a9(ctx):
    md5 = hashlib.md5(open(CARD, "rb").read()).hexdigest() if os.path.exists(CARD) else None
    consts = {"N_KV": V3.N_KV, "R_PRES": R_PRES, "R_HO8": R_HO8, "N_READ": N_READ,
              "R0": K12.R0, "M": M, "c_tau": round(float(TAU["c_tau"]), 12)}
    # ★ 只断言卡片 §5 真正冻结的常数（`D_MODEL` 只作 info —— DEF-52-4）
    exp = {"N_KV": 60, "R_PRES": 4, "R_HO8": 8, "N_READ": 20, "R0": 8, "M": 2,
           "c_tau": 0.094970242679}
    ok = (md5 is not None) and all(consts[k] == exp[k] for k in exp)
    return {"desc": "卡存在 + md5 落盘 + 装置常数与卡一致", "card_md5": md5,
            "consts": consts, "expected": exp, "info_D_MODEL": D_MODEL, "pass": ok}


def a10(ctx):
    d10 = 0.0
    for s in SEEDS_HEAVY:
        r = RS_(s)

        def Gf(ph, rho):
            return r["phi"][ph]["mu"]["K"][str(rho)] - r["phi"][ph]["mu"]["F"][str(rho)]
        e = Gf("0.9", 2) - Gf("0.0", 2)
        t_ = Gf("0.9", 4) - Gf("0.0", 4)
        d10 = max(d10, abs((2 * e - t_) - r["SHAPE"]), abs(e - r["EARLY"]), abs(t_ - r["TOTAL"]))
    return {"desc": "SHAPE/EARLY/TOTAL 由落盘 μ 独立重算一致（≤1e-15）",
            "max_abs_diff": d10, "pass": d10 <= 1e-15}


CHECKS = [("A52-1", a1), ("A52-2", a2), ("A52-3", a3), ("A52-4", a4), ("A52-5", a5),
          ("A52-6", a6), ("A52-7", a7), ("A52-8", a8), ("A52-9", a9), ("A52-10", a10)]


def main():
    global TAU
    TAU = cal_at(V3.N_KV)
    T52.TAUS["tau"] = TAU                    # run_seed 依赖模块级 TAUS
    resume = os.environ.get("A52_RESUME", "1") == "1"
    A = A52Dict(PART)
    if resume and os.path.exists(PART):
        for k, v in json.load(open(PART, encoding="utf-8")).items():
            if isinstance(v, dict) and "pass" in v and k.startswith("A52-"):
                A[k] = v
        print(f"[resume] 复用：{sorted(k for k in A if k.startswith('A52-'))}", flush=True)

    ctx = {"cnt": [int(x) for x in V3.run_unit_dose(SEEDS_HEAVY[0], 0.0, TAU, cnt0=None)["counts"]]}
    for name, fn in CHECKS:
        if resume and name in A and isinstance(A[name], dict) and "pass" in A[name]:
            continue
        A[name] = fn(ctx)
        print(f"  {name}: {'PASS' if A[name]['pass'] else 'FAIL'}  {A[name]['desc']}", flush=True)

    asserts = {k: A[k] for k in A if k.startswith("A52-") and isinstance(A[k], dict) and "pass" in A[k]}
    ok = all(v["pass"] for v in asserts.values())
    out = {"stage_a_pass": ok, "assertions": asserts, "rows": ctx.get("rows")}
    json.dump(out, open(f"{OUT}/DIR52_stageA.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("=== DIR-52 Stage A ===")
    for k in sorted(asserts):
        v = asserts[k]
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}   {v['desc']}")
        if not v["pass"]:
            print("      ", json.dumps({kk: vv for kk, vv in v.items() if kk != "pass"},
                                       ensure_ascii=False, default=str)[:400])
    print(f"  总体：{'PASS ⇒ 可进入正式轮' if ok else 'FAIL ⇒ 拦住正式轮'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
