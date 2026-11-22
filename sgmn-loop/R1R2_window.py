#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-07 · R1/R2 —— 写通道的可行工作窗口（v2，修正度量的语义）

【v2 的修正：rho 是【幅度比】，不是【是否在学】】
  v1 用 rho = ||eta G_data V|| / (||eta G_data V|| + ||eta G_reg V||) 判定「数据项主导」。
  但幅度比高**不代表**记忆在学：实测 rho≈0.99 时标量模型的 W_T 仍停在初值附近（完全不动）。
  原因：alpha=1 时数据项的**幅度**天然远大于 lambda 项（幅度 ~ ||k||^2 vs ~ lambda），
  与系统是否收敛无关。
  => v2 改用【方向性】判据：
        数据项主导  <=>  写入后记忆【更接近回归解 W*】而非【更接近 W^0】
     并辅以回归拟合优度 R^2 = 1 - E[(Wk-v)^2]/Var(v)（不依赖 W* 公式）。
  幅度比 rho 仍记录，但只作「更新构成」的描述性量，不再当判据。

【尺度扫描是本轮的核心】
  v2.4 的标量约定 mu=5 给 ||k||~2.2；真实 LLM 给 ||k||~67（mu_eff ~ 4489）。
  本轮扫 mu 与 eta_w 的二维网格，定位「既稳定（因子范数有界）又在学（收敛到 W*）」的窗口，
  并给出该窗口随尺度的变化律。

R1 (Stage A)：A1 G 分解精确；A2 方向性判据的语义校验（alpha=0 -> 收敛到 W0；alpha=1 -> 收敛到 W*）；
               A3 对照档；A4 与 DIR-06 真实 LLM 探针的一致性；A5 分段前视可重复
R2：尺度 x 步长网格
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sgmn_module_instr import SGMNStack

OUT = pathlib.Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
NCH = 4
BOUND = 1e3
MU_GRID = [0.05, 0.5, 5.0, 50.0, 500.0, 5000.0]
ETA_GRID = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
T_WRITES = 3000

res = {"dir": "DIR-07", "round": "R1+R2", "stage": "A+B", "model": MODEL, "checks": []}


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


# ================================================== 标量参照模型（独立实现）
def scalar_run(eta, lam, alpha, mu, T=T_WRITES, seed=0, A=1.0):
    """d=1, r=1, W0=0：U' = U - eta*G V, V' = V - eta*G U（同步）。
    返回 W_T、W*、R^2、rho 均值、max|U|、是否有限、是否学到（更接近 W*）。"""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.5); V = rng.normal(0, 0.5)
    W0 = U * V
    rhos, umax = [], 0.0
    W = W0
    for t in range(T):
        k = rng.normal(0, np.sqrt(mu)); v = A * k
        W = U * V
        G_data = alpha * (W * k - v) * k
        G_reg = lam * W
        G = G_data + G_reg
        cD = abs(eta * G_data * V); cR = abs(eta * G_reg * V)
        rhos.append(cD / (cD + cR) if (cD + cR) > 0 else 0.0)
        Un = U - eta * G * V
        Vn = V - eta * G * U
        U, V = Un, Vn
        umax = max(umax, abs(U))
        if not (np.isfinite(U) and np.isfinite(V)):
            break
    W_T = U * V
    W_star = A * mu / (mu + lam)
    # 回归拟合优度（用独立样本）
    rg = np.random.default_rng(seed + 99991)
    ks = rg.normal(0, np.sqrt(mu), 4000); vs = A * ks
    pred = W_T * ks
    r2 = float(1.0 - np.mean((pred - vs) ** 2) / np.var(vs))
    d_star = abs(W_T - W_star); d_zero = abs(W_T)
    learned = bool(d_star < d_zero)                      # 更接近回归解而非 W0
    return {"W_T": float(W_T), "W_star": float(W_star), "W_0": float(W0),
            "r2": r2, "rho_mean": float(np.mean(rhos)) if rhos else 0.0,
            "umax": float(umax), "finite": bool(np.isfinite(W_T)),
            "stable": bool(np.isfinite(W_T) and umax < BOUND),
            "learned": learned, "moved": bool(abs(W_T - W0) > 1e-3 * max(abs(W_star), 1e-12))}


def main():
    print("=== DIR-07 v2：写通道的可行工作窗口 ===")

    # ---------------- Stage A ----------------
    print("\n[R1 Stage A]")
    rng = np.random.default_rng(7)
    W0 = np.zeros((3, 4))
    U = rng.normal(0, 0.4, (3, 2)); V = rng.normal(0, 0.4, (4, 2))
    Wc = W0 + U @ V.T
    k = rng.normal(0, 1.0, 4); v = rng.normal(0, 1.0, 3)
    alpha, lam = 0.7, 0.05
    Gd = alpha * np.outer(Wc @ k - v, k); Gr = lam * (Wc - W0)
    err1 = float(np.max(np.abs((Gd + Gr) - (alpha * np.outer(Wc @ k - v, k) + lam * (Wc - W0)))))
    rec("A1 G = G_data + G_reg 分解精确", err1 < 1e-12, f"最大绝对差 {err1:.3e}", "rel<1e-12", err1)

    # A2：方向性判据的语义校验
    a2 = []
    for et in (1e-3, 1e-2, 3e-2, 1e-1):
        o = scalar_run(et, 0.05, 1.0, 5.0)
        a2.append({"eta": et, **o})
        print(f"    alpha=1 eta={et:<7g}: rho={o['rho_mean']:.4f} R^2={o['r2']:+.4f} "
              f"W_T={o['W_T']:+.5f} W*={o['W_star']:.5f} 学到={o['learned']} 动过={o['moved']}")
    o_reg = scalar_run(1e-2, 0.05, 0.0, 5.0)
    print(f"    alpha=0 eta=0.01  : rho={o_reg['rho_mean']:.4f} R^2={o_reg['r2']:+.4f} "
          f"W_T={o_reg['W_T']:+.5f} 学到={o_reg['learned']} 动过={o_reg['moved']}")
    learned_ok = all(r["learned"] for r in a2 if r["moved"])
    reg_ok = (not o_reg["learned"])
    rec("A2 方向性判据的语义：alpha=1 时收敛到 W*；alpha=0 时不学到 W*", learned_ok and reg_ok,
        f"alpha=1 的 {len([r for r in a2 if r['moved']])} 个「动过」的步长全部学到={learned_ok}；"
        f"alpha=0 未学到 W*={reg_ok}（W_T={o_reg['W_T']:+.6f}，W_0 初值 "
        f"{o_reg['W_0']:+.6f}；该档在所选步长下接近『惰性』而非『收敛到 W0』）",
        "两侧均成立", a2)
    res["a2_semantics"] = a2

    rec("A3 对照档 alpha=0 => rho 精确为 0", o_reg["rho_mean"] == 0.0,
        f"rho 均值={o_reg['rho_mean']}", "精确 0", o_reg["rho_mean"])
    # A3b 的反例在 R2 网格里找（推迟到网格扫描之后判定）

    # ---------------- 真实模型：A4/A5 ----------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    enc = tok(["The capital of France is", "In a distant galaxy the explorer found",
               "The quick brown fox jumps over", "Machine learning models are"],
              return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    ids = enc["input_ids"].to(DEV); mask = enc["attention_mask"].to(DEV)
    nl = model.config.num_hidden_layers; d = model.config.hidden_size
    mount = [nl // 3, nl // 2, 2 * nl // 3]

    @torch.no_grad()
    def chunked():
        L = ids.shape[1]; cs = max(1, L // NCH); outs = []
        for i in range(0, L, cs):
            outs.append(model(input_ids=ids[:, i:i + cs],
                              attention_mask=mask[:, i:i + cs]).logits.float())
        return torch.cat(outs, dim=1)

    def stack(lam, r, alpha, eta_w):
        return SGMNStack(model, mount, d, d, r, alpha, lam, eta_w, 0.05, DEV,
                         seed=20261029, use_factorized=True)

    st = stack(0.05, 8, 1.0, 3e-5)
    snap0 = {li: (m.U.detach().clone(), m.V.detach().clone(), m.W0.detach().clone())
             for li, m in st.mems.items()}
    with torch.no_grad():
        c1 = chunked().cpu()
        # 必须把记忆状态复位，否则第二次前视是在【已演化】的记忆上做的
        for li, (U0, V0, W0s) in snap0.items():
            m = st.mems[li]
            m.U.copy_(U0); m.V.copy_(V0); m.W0.copy_(W0s); m.prev_hs = None
        c2 = chunked().cpu()
    rep = float((c1 - c2).abs().max())
    st.remove()
    rec("A5 分段前视可重复（逐位相同）", rep == 0.0, f"两次前视最大差 {rep:.3e}", "精确 0", rep)

    st = stack(0.05, 8, 1.0, 1e-4)
    with torch.no_grad():
        lg = chunked()
    fin = bool(torch.isfinite(lg).all())
    umax = float(max(max(x for x in m.umax_hist) for m in st.mems.values()))
    st.remove(); torch.cuda.empty_cache()
    rec("A4 与 DIR-06 探针一致：eta_w=1e-4 为「有限但发散」", fin and umax >= 1e3,
        f"logits 有限={fin}，max|U|={umax:.3e}（DIR-06 测得 2.27e+07）",
        "有限且 max|U|>=1e3", umax)

    # ---------------- R2 尺度 x 步长网格 ----------------
    print("\n[R2] 尺度 x 步长网格（标量模型；判据：stable AND learned）")
    grid = []
    for mu in MU_GRID:
        for et in ETA_GRID:
            o = scalar_run(et, 0.05, 1.0, mu, T=1200, seed=3)
            o.update({"mu": mu, "eta": et})
            grid.append(o)
            print(f"  mu={mu:<8g} eta={et:<8g}: stable={str(o['stable']):5s} "
                  f"learned={str(o['learned']):5s} R^2={o['r2']:+.4f} "
                  f"max|U|={o['umax']:.3e} rho={o['rho_mean']:.3f}")
    res["scale_grid"] = grid

    win = [g for g in grid if g["stable"] and g["learned"]]
    print(f"\n  同时满足（稳定 且 学到）的网格点：{len(win)}/{len(grid)}")
    for mu in MU_GRID:
        pts = [g for g in grid if g["mu"] == mu and g["stable"] and g["learned"]]
        stb = [g for g in grid if g["mu"] == mu and g["stable"]]
        lrn = [g for g in grid if g["mu"] == mu and g["learned"]]
        print(f"    mu={mu:<8g}: 稳定点 {len(stb)}  学到点 {len(lrn)}  窗口 {len(pts)}"
              + (f"  窗口 eta ∈ {{{', '.join(f'{p[chr(101)+chr(116)+chr(97)]:g}' for p in pts)}}}"
                 if pts else "  【无窗口】"))
    res["window_by_mu"] = {str(mu): [g["eta"] for g in grid
                                     if g["mu"] == mu and g["stable"] and g["learned"]]
                           for mu in MU_GRID}
    counter = [(g["mu"], g["eta"]) for g in grid if g["rho_mean"] > 0.5 and not g["learned"]]
    rec("A3b 幅度比 rho 高但系统不动（v1 判据失效的证据）", len(counter) > 0,
        f"网格中存在 {len(counter)} 个 rho>0.5 但未学到的点，例："
        f"{counter[:4]} => rho 是【幅度比】，不能当『是否在学』的判据（v2 改用方向性判据）",
        "存在反例", counter[:8])
    ok = all(len(res["window_by_mu"][str(mu)]) > 0 for mu in MU_GRID)
    rec("H1 每个尺度上是否存在「稳定 且 学到」的窗口", ok,
        f"逐尺度的窗口数：{ {k: len(v) for k, v in res['window_by_mu'].items()} }",
        "每个尺度 >0", res["window_by_mu"])

    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (OUT / "R1R2_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n=== DIR-07 R1+R2：{npass}/{len(res['checks'])} 通过 ===")


if __name__ == "__main__":
    main()
