"""`sum_c2.py` —— 汇总 `DIR-119` 的 `C2`（候选集重抽）与 `C3`（词面偏置）控制（report-only）。"""
import glob
import json
import pathlib
import statistics as st

J = "10000"
OUT = pathlib.Path(__file__).resolve().parent / "out"


def gap(m):
    if not m or m.get("err_in_train_frac") is None:
        return None
    return m["err_in_train_frac"] - m["err_in_train_null"]


def main():
    rows = {}
    for f in sorted(glob.glob(str(OUT / "cells" / "COLOCATED_600_*.json"))):
        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        # ★ 结构是 `c2[mult][rng] = mech`（不是 c2[rng][mult]）—— 第一版写反了，
        #   于是全部读数取成 nan（报告层缺陷，不影响判据）。
        c2 = (d.get("c2") or {}).get(J, {})
        rows[d["seed"]] = {
            "base": d["mech_by_mult"].get(J, {}),
            "c2": {k: (v or {}) for k, v in c2.items()},
        }
    hdr = f"{'seed':>9} {'base':>14}"
    for k in ("556", "557", "558"):
        hdr += f" | rng{k}: acc/GAP"
    print(hdr)
    for s, v in sorted(rows.items()):
        b = v["base"]
        line = f"{s:>9} {b.get('acc_argmin', float('nan')):.3f}/{gap(b) or 0:+.3f}"
        for k in ("556", "557", "558"):
            m = v["c2"].get(k, {})
            line += f" | {m.get('acc_argmin', float('nan')):.3f}/{gap(m) or 0:+.3f}"
        print(line)
    print()
    for lbl, getter in (("base", lambda v, k: v["base"]),
                        ("rng556", lambda v, k: v["c2"].get("556", {})),
                        ("rng557", lambda v, k: v["c2"].get("557", {})),
                        ("rng558", lambda v, k: v["c2"].get("558", {}))):
        ac, gp = [], []
        for v in rows.values():
            m = getter(v, None)
            if not m or m.get("acc_argmin") is None:
                continue
            ac.append(m["acc_argmin"])
            g = gap(m)
            if g is not None:
                gp.append(g)
        if not ac:
            print(f"{lbl:>7}: 无数据")
            continue
        print(f"{lbl:>7}: n={len(ac)} acc均值={st.mean(ac):.4f} "
              f"GAP均值={st.mean(gp):+.4f} GAP>0 的 seed={sum(1 for g in gp if g > 0)}/{len(gp)} "
              f"GAP≥0.20 的 seed={sum(1 for g in gp if g >= 0.20)}/{len(gp)}")


if __name__ == "__main__":
    main()
