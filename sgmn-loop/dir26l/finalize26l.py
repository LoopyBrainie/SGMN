r"""DIR-26-L · 收尾（最小、前台、秒级）：只做「读 40 行 → 算判据 → 先落判决，再落 meta」。

设计约束（DEC-39，来自本轮的反复失败）：
- **判据优先**：先 `crit()` 并把 `verdicts/passed` 落盘，`meta` 之后单独 try ——
  元数据出错的代价从此只是"meta 缺字段"，不会再让裁决丢失。
- **不竞猜模块属性**：所有协议常量显式走 `R1_gating12`（`K.*`）或 `R1_gating`（`G.*`），
  禁止 `getattr` 兜底（兜底会把常量来源抹掉，污染证据）。
- **不重跑单元**：40 行已在盘上；DEF-26L-4 的数值同一性证明已在 `D26L_finalize.log`
  里过一次（`(90,17)`、63 值对、0 不符），本脚本不再重复。
- `crit()` 的 `ref` 必须是 `d43_means()` 的**完整**返回（它内部要用 `ref["rows"]` 算 (f) 与 se）。
"""
import json, sys, time

for _p in ("dir26l", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G              # R_PRES / ETA / M / S_KEY / D_MODEL
import R1_gating12 as K            # R0 / R_HO4 / R_HO8 / R_MAIN / run_unit / cal_at / measure_hit
import R1_gating13 as K13          # crit / d43_means / 种子表 / KV_REF / N_READ

# DEC-39：不竞猜模块属性 —— 用前先核，缺了就在秒级报出来，而不是 nohup 后十分钟崩
_NEED = [("G", G, ("D_MODEL", "ETA", "M", "S_KEY")),
         ("K", K, ("R0", "R_PRES", "R_HO4", "R_HO8", "R_MAIN", "N_READ")),
         ("K13", K13, ("SEEDS", "SHARED_SEEDS", "NEW_SEEDS", "KV_REF", "crit", "d43_means"))]
_miss = [(tag, a) for tag, mod, attrs in _NEED for a in attrs if not hasattr(mod, a)]
assert not _miss, f"模块属性缺失：{_miss}"

OUT = "/home/ec2-user/sgmn-loop/dir26l/out/DIR26Lr10_formal.json"


def main():
    t0 = time.time()
    res = json.load(open(OUT))
    rows = res["rows"]
    assert len(rows) == 40, f"行数不是 40：{len(rows)}"
    cells = sorted({r["n_kv"] for r in rows})
    assert cells == [90, 120], f"格点不是 {{90,120}}：{cells}"
    for n in cells:
        ss = sorted(r["seed"] for r in rows if r["n_kv"] == n)
        assert ss == sorted(K13.SEEDS), f"n_kv={n} 的种子序列与冻结表不一致"

    ref = K13.d43_means()                       # 完整返回：G / U0 / gamma / rows
    ref["p_lin"] = 6.0 * ref["G"][20]
    ref["p_pow"] = ref["G"][20] * 6.0 ** ref["gamma"]

    v = K13.crit(rows, ref)                     # 卡 rev 1.1 §4 的判决树（未改阈值）

    res["ref"] = {"G": ref["G"], "U0": ref["U0"], "gamma": ref["gamma"],
                  "p_lin": ref["p_lin"], "p_pow": ref["p_pow"]}
    res["verdicts"] = v
    res["passed"] = bool(v["main"]["ok"])
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"[落盘] verdicts + passed 已写入（判决优先，meta 随后）")
    print(f"  通道: {v['channel']} · main.ok: {v['main']['ok']} · 护栏: {v['guards']}")

    try:
        res.setdefault("meta", {}).update({
            "device": "cpu", "d": G.D_MODEL, "eta": G.ETA, "m": G.M, "s_key": G.S_KEY,
            "r0": K.R0, "r_pres": K.R_PRES, "r_ho": [K.R_HO4, K.R_HO8], "rho_main": K.R_MAIN,
            "n_read": K13.N_READ, "cells": [{"n_kv": n, "n_read": K13.N_READ} for n in cells],
            "ref_cells": list(K13.KV_REF), "seeds": K13.SEEDS,
            "shared_seeds": K13.SHARED_SEEDS, "new_seeds": K13.NEW_SEEDS,
            "stage": "formal", "card": "DIR-26L rev1.1",
            "defects": ["DEF-26L-1…6（见 VERDICT.md §7）"],
            "crit_call_site": "dir26l/finalize26l.py",
            "identical_numeric_check": {"unit": [90, 17], "fields_compared": 63, "mismatches": 0,
                                        "log": "dir26l/D26L_finalize.log"},
            "wall_seconds_finalize": round(time.time() - t0, 1),
        })
        json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
        print("[落盘] meta 已补")
    except Exception as e:
        print(f"[warn] meta 未写（裁决不受影响）：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
