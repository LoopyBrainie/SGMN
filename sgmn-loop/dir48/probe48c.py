#!/opt/pytorch/bin/python
r"""DIR-48 · 开卡前诊断 C（**定窗口**）：`busy` 的发散率与跨半比例随窗口 `W` 的变化。

★ 背景：`probe48a` 报的 `busy` 发散 3% 是**错的**（它把 NaN 与 `min/max` 混在一起）；实测量得 `W=16` 时 **30–35%** 实例发散
   ⇒ 卡 rev 1.0 的 `W_busy=16` 会撞 `N_DIV_MAX=0.20`。本探针在**未用过的种子段** `25800..25999` 上扫 `W∈{6,8,10,12,16}`，
   量出①发散率 ②跨半比例 ③KM 中位，用来把窗口定在"发散率有余量"的最大 `W`。

用法：`probe48c.py --n 200 --out dir48/out/DIR48_probe_c.json`
"""
import argparse, json, math, os, sys
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_task48 as T48      # ★ 用**已修**（固定 NMARK）的实现，避免 DEF-48-4 混入
import probe48b as PB        # 复用 KM 中位数


def run_win(sd, arm, W):
    save = T48.W_WIN[arm]
    try:
        T48.W_WIN[arm] = W
        return T48.trace_one(sd, arm)
    finally:
        T48.W_WIN[arm] = save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=25800)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"n": a.n, "seed0": a.seed0, "sweep": {}}
    for arm, wins in (("busy", [6, 8, 10, 12, 16]), ("decay", [32]), ("quiet", [16])):
        for W in wins:
            Ls, cs, nd, nu, c0s = [], [], 0, 0, []
            for i in range(a.n):
                r = run_win(a.seed0 + i, arm, W)
                if r["undef"]:
                    nd += 1
                    continue
                Ls.append(r["L"]); cs.append(r["censored"]); c0s.append(r["c0"])
            n_use = len(Ls)
            n_cens = sum(1 for x in cs if x)
            med, reached = PB.km_median(Ls, cs, W)
            key = f"{arm}_W{W}"
            out["sweep"][key] = {"arm": arm, "W": W, "n": a.n, "n_div": nd, "div_frac": nd / a.n,
                                 "div_frac_3se": 3 * math.sqrt((nd / a.n) * (1 - nd / a.n) / a.n),
                                 "n_use": n_use,
                                 "p_cross": (n_use - n_cens) / n_use if n_use else None,
                                 "cens_frac": n_cens / n_use if n_use else None,
                                 "km_median": med, "km_reached": reached,
                                 "mean_c0": sum(c0s) / len(c0s) if c0s else None}
            o = out["sweep"][key]
            print("[%-5s W=%2d] n=%d div=%.3f(±%.3f) use=%d cens=%.3f p_cross=%s KM_med=%s%s c0=%.3f" % (
                arm, W, a.n, o["div_frac"], o["div_frac_3se"], n_use, o["cens_frac"] or -1,
                ("%.3f" % o["p_cross"]) if o["p_cross"] is not None else None,
                med, "" if reached else "(未达)", o["mean_c0"] or -1), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
