#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-96 · 算术前置（卡 rev 1.1 §5 的 `A96-1` / `A96-3` / `A96-4`）。

**不训练、不写、不前向** —— 只用 `make_stack` 造出的**初始** `W0/Wq/Wo` 做纯线性代数，
目的是在花 9 分钟训练**之前**确认 `κ` 的定义与实现是对的。

锚点（解析）：
  `ΔW = s·W0` ⇒ `A − A₀ = s·A₀` ⇒ `ρ_A = s/(1+s)`，`ν = s` ⇒ **`κ_A = 1/(1+s)`**
  ⇒ `κ` 对 `ΔW` 的**整体缩放不变**，只对**方向结构**敏感（这正是本卡能区分
     「结构性不可见」与「单纯小」的全部依据）。
  `ΔW = 0` ⇒ `κ_A = 0`。
  `ΔW ~ 各向同性随机`（`‖ΔW‖_F = ν‖W0‖_F`）⇒ **`E[κ_A] ≈ 1`**（因 `W0` 本身即
     `randn/sqrt(d_v)`，与同尺度高斯扰动**同分布**）⇒ 这是**判据零假设的实测核**。
"""
import json
import math
import pathlib
import sys

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import torch                                             # noqa: E402
import R2_task91 as R91                                  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
S_ANCHOR = 0.036          # 与 D-125 实测 ν 同档（3.6%）
N_RAND = 8                # DEF-96-05：n=8 取中位


def kappa(W0, Wo, Wq, dW, nu):
    """κ = (‖A−A₀‖_F / ‖A‖_F) / ν，A = Wo(W0+dW)Wq，A₀ = Wo W0 Wq。"""
    A0 = Wo @ W0 @ Wq
    A = Wo @ (W0 + dW) @ Wq
    return float((A - A0).norm() / A.norm()) / nu


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)
    st = R91.make_stack(gamma=0.0, no_write=True, seed=20261028)
    G, ok = {}, {}
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(1234)
    for li, m in st.mems.items():
        W0, Wq, Wo = m.W0, m.Wq, m.Wo
        n0 = float(W0.norm())
        d_k = int(W0.shape[1])
        A0 = Wo @ W0 @ Wq
        # ---------------- A96-1：解析锚点 ----------------
        a_k = kappa(W0, Wo, Wq, S_ANCHOR * W0, S_ANCHOR)
        a_exp = 1.0 / (1.0 + S_ANCHOR)
        z_k = kappa(W0, Wo, Wq, torch.zeros_like(W0), 1.0)      # ν 任意（分子为 0）
        # ---------------- A96-4：各向同性随机控制 ----------------
        ks = []
        for _ in range(N_RAND):
            R_ = torch.randn(W0.shape, generator=g, device=W0.device, dtype=W0.dtype)
            R_ = R_ / R_.norm() * (S_ANCHOR * n0)               # ‖ΔW‖_F = ν‖W0‖_F
            ks.append(kappa(W0, Wo, Wq, R_, S_ANCHOR))
        ks_sorted = sorted(ks)
        G[str(li)] = {
            "d_k": d_k, "W0_fro": n0, "sqrt_d_k": math.sqrt(d_k),
            "A96_1_kappa_anchor": a_k, "A96_1_expect": a_exp,
            "A96_1_rel_err": abs(a_k - a_exp) / a_exp,
            "A96_1_kappa_zero": z_k,
            "A96_4_kappa_rand_median": ks_sorted[N_RAND // 2],
            "A96_4_kappa_rand_all": [round(x, 5) for x in ks],
            "fro_A0": float(A0.norm()), "isfinite": bool(torch.isfinite(A0).all()),
        }
    st.remove()

    ok["A96_1_anchor"] = all(v["A96_1_rel_err"] <= 1e-6 for v in G.values())
    ok["A96_1_zero"] = all(v["A96_1_kappa_zero"] == 0.0 for v in G.values())
    ok["A96_3_W0_shape"] = all(abs(v["W0_fro"] - v["sqrt_d_k"]) / v["sqrt_d_k"] <= 0.01
                               for v in G.values())
    ok["A96_4_rand_in_band"] = all(0.5 <= v["A96_4_kappa_rand_median"] <= 2.0
                                   for v in G.values())
    ok["A96_5_finite"] = all(v["isfinite"] and v["fro_A0"] > 0 for v in G.values())

    print("=== 算术前置 ===")
    for li, v in G.items():
        print(f"  L{li}: d_k={v['d_k']} ‖W⁰‖_F={v['W0_fro']:.4f} (√d_k={v['sqrt_d_k']:.4f}) "
              f"‖A₀‖_F={v['fro_A0']:.4f}")
        print(f"        κ_anchor={v['A96_1_kappa_anchor']:.6f} expect={v['A96_1_expect']:.6f} "
              f"rel_err={v['A96_1_rel_err']:.2e}  κ_zero={v['A96_1_kappa_zero']:.1e}")
        print(f"        κ_rand median={v['A96_4_kappa_rand_median']:.4f} "
              f"all={v['A96_4_kappa_rand_all']}")
    print("\n=== 守卫 ===")
    for k, v in ok.items():
        print(f"  {'✓' if v else '✗'} {k} = {v}")
    (OUT / "stage_A.json").write_text(
        json.dumps({"stage": "A_arith", "s_anchor": S_ANCHOR, "readings": G, "guards": ok},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[写出] {OUT/'stage_A.json'}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())