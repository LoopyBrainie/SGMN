#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-07 · R3 —— 加入信噪比维度：窗口是「窄」还是「空」

R2 的结论：学习阈值 eta_learn ~ 1/mu，稳定天花板 eta_stable ~ 1/mu。
但 R2 的回归目标是【干净的】(v = A k，SNR=1)。真实 LLM 的 (k -> v) 回归信噪比很低
（隐状态中可被键预测的那一小部分）。若学习阈值还正比于 1/SNR，
则当 SNR<1 时 eta_learn 会超过稳定天花板 => 【窗口为空】。
本轮扫 (mu, SNR) 网格，直接判定窗口是否为空，并给出空窗口的边界 SNR*。

判据：
  eta_learn(mu,SNR)  = 使记忆收敛到回归解的最小步长（网格内）
  eta_stable(mu,SNR) = 使因子范数保持有界的最大步长（网格内）
  窗口非空 <=> eta_learn <= eta_stable
  H3 临界信噪比 SNR* 存在 <=> 存在 SNR 使窗口由非空变为空
"""
import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parent
BOUND = 1e3
MU_GRID = [5.0, 50.0, 500.0, 5000.0]
SNR_GRID = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
ETA_GRID = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
T_WRITES = 1500

res = {"dir": "DIR-07", "round": "R3", "stage": "B", "checks": []}


def rec(name, passed, detail, tol=None, value=None):
    def clean(x):
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (list, tuple)):
            return [clean(i) for i in x]
        return x
    res["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                          "tolerance": tol, "value": clean(value)})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def scalar_run(eta, lam, alpha, mu, snr, T=T_WRITES, seed=0, A=1.0):
    """d=1, r=1, W0=0。
    信号分量缩放为 A*sqrt(SNR)，噪声 sigma_n^2 = (1-SNR) A^2 mu
    => Var(信号) = SNR*A^2 mu，Var(噪声) = (1-SNR)*A^2 mu，Var(v) = A^2 mu（与 SNR 无关），
       且「被 k 解释的比例」恰为 SNR。故完美学习者的 R^2 上限 = SNR。"""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.5); V = rng.normal(0, 0.5)
    W0 = U * V
    sig_n = np.sqrt(max(1.0 - snr, 0.0)) * abs(A) * np.sqrt(mu)
    umax = 0.0
    for t in range(T):
        k = rng.normal(0, np.sqrt(mu)); v = A * np.sqrt(snr) * k + sig_n * rng.normal()
        W = U * V
        G = alpha * (W * k - v) * k + lam * W
        Un = U - eta * G * V
        Vn = V - eta * G * U
        U, V = Un, Vn
        umax = max(umax, abs(U))
        if not (np.isfinite(U) and np.isfinite(V)):
            break
    W_T = U * V
    W_star = A * np.sqrt(snr) * snr * mu / (mu + lam)   # 只含可解释成分的回归解
    rg = np.random.default_rng(seed + 99991)
    ks = rg.normal(0, np.sqrt(mu), 4000)
    vs = A * np.sqrt(snr) * ks + sig_n * rg.normal(size=4000)
    pred = W_T * ks
    r2 = float(1.0 - np.mean((pred - vs) ** 2) / np.var(vs))
    stable = bool(np.isfinite(W_T) and umax < BOUND)
    # 「学到」：拟合优度接近可解释成分所能达到的上限
    r2_max = float(snr)                       # 完美拟合可解释成分时的 R^2 上限
    learned = bool(stable and r2 >= 0.9 * r2_max)
    return {"W_T": float(W_T), "r2": r2, "r2_max": r2_max, "stable": stable,
            "learned": learned, "umax": float(umax), "finite": bool(np.isfinite(W_T))}


def main():
    print("=== DIR-07 R3：信噪比维度 —— 窗口是窄还是空 ===")
    grid = []
    for mu in MU_GRID:
        for snr in SNR_GRID:
            cells = []
            for eta in ETA_GRID:
                o = scalar_run(eta, 0.05, 1.0, mu, snr, T=1000, seed=5)
                o.update({"mu": mu, "snr": snr, "eta": eta})
                cells.append(o)
            stable = [c for c in cells if c["stable"]]
            learned = [c for c in cells if c["learned"]]
            win = [c for c in cells if c["stable"] and c["learned"]]
            e_learn = min([c["eta"] for c in learned], default=None)
            e_stable = max([c["eta"] for c in stable], default=None)
            empty = (e_learn is not None and e_stable is not None and e_learn > e_stable) or \
                    (e_learn is None and e_stable is not None)
            grid.append({"mu": mu, "snr": snr, "n_stable": len(stable), "n_learned": len(learned),
                         "n_window": len(win),
                         "eta_learn_min": e_learn, "eta_stable_max": e_stable,
                         "window_empty": bool(empty),
                         "window_etas": [c["eta"] for c in win]})
            print(f"  mu={mu:<7g} SNR={snr:<5g}: 稳定 {len(stable)}  学到 {len(learned)}  "
                  f"窗口 {len(win)}  eta_learn>={e_learn}  eta_stable<={e_stable}  "
                  f"{'【窗口空】' if empty else ''}")
    res["snr_grid"] = grid

    # H3：是否存在「窗口由非空变为空」的临界 SNR
    crit_snr = {}
    for mu in MU_GRID:
        rows = [g for g in grid if g["mu"] == mu]
        empt = [g["snr"] for g in rows if g["window_empty"]]
        nonempty = [g["snr"] for g in rows if not g["window_empty"]]
        crit_snr[mu] = {"empty_at_snr": empt, "nonempty_at_snr": nonempty}
        print(f"    mu={mu:<7g}: 空窗口的 SNR = {empt}；非空的 SNR = {nonempty}")
    res["critical_snr"] = crit_snr

    has_empty = any(v["empty_at_snr"] for v in crit_snr.values())
    rec("H3 存在「窗口由非空变为空」的临界信噪比", has_empty,
        f"逐 mu 的空窗口 SNR 集合：{ {k: v['empty_at_snr'] for k, v in crit_snr.items()} }",
        "至少一个 mu 上存在", crit_snr)

    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (OUT / "R3_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\n=== DIR-07 R3：{npass}/{len(res['checks'])} 通过 ===")


if __name__ == "__main__":
    main()
