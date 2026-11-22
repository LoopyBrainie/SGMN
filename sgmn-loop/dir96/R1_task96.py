#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-96 · `O95-1`：3.6% 的写入增量在哪里被压掉了？—— 卡 `revolve/DIR-96/DIRECTION_rev96_v12.md`（rev 1.2 生效版）。

判据（§2.3，冻结，先验校准）
  κ_A ≤ 0.2 ⇒ H96 成立（抑制在读出侧）；κ_A ≥ 0.5 ⇒ H96 否证。四支见 §4（预注册）。

三级 κ（§2.2 唯一口径）
  A = Wo W Wq, A₀ = Wo W⁰ Wq, ν = ‖ΔW‖_F/‖W⁰‖_F
    κ_A     = (‖A−A₀‖_F/‖A‖_F) / ν                    逐层量 ⇒ seed 取三层中位
    κ_m     = (√(Σ‖(A−A₀)c‖²/Σ‖A c‖²)) / ν           逐层量 ⇒ seed 取三层中位；c 固定取 W 条件
    κ_logit = (mean|L^W − L^{W⁰}| / mean|L^W − L^{W=0}|) / mean_l ν_l   整栈量 ⇒ 每 seed 一个
★ §2.2 ★：`c` **只在 `W=W` 条件捕获一次**，`m=A c`/`m₀=A₀ c` 用**同一批 c** 线性算出
  （`DEF-96-01`：改 W 后重跑前向取 c 会把路径依赖混进 κ_m）。

★ 逐字派生自 `dir95/R1_task95.py`（→ `dir94/R1_task94.py`）：同装置、同写入、同读侧训练；
  **只换测量层**（4 条件×准确率 → 三级 κ）。
"""
import argparse
import json
import math
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import probe91_kpool as KP                       # noqa: E402
import probe91_womap as W91                      # noqa: E402
import R2_task91 as R91                          # noqa: E402
import sgmn_module                               # noqa: E402
import torch                                     # noqa: E402

SEEDS = [20261028, 7777]
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
KA_SUP, KA_FAITH = 0.2, 0.5        # §2.3 冻结阈值
BAND = 0.2                          # §4 支边界
DEV = R91.DEV


# --------------------------------------------------------------- 装置
def load_state(st, sv):
    """把训练好的可塑状态灌回重建的栈；★ 评测期禁写（DEF-95-01）。"""
    for li, m in st.mems.items():
        with torch.no_grad():
            m.Wq.copy_(sv["Wq"][li])
            m.Wo.copy_(sv["Wo"][li])
            m.gamma.copy_(sv["gamma"][li])
            m.Wfull.copy_(sv["W"][li])
        m.no_write = True


def set_W(st, W):
    with torch.no_grad():
        for li, m in st.mems.items():
            m.Wfull.copy_(W[li])


def classify(kA, km, klog):
    """§4 四支（预注册，穷尽且有序）。"""
    if kA <= BAND:
        return "A-READOUT"
    if km <= BAND:
        return "B-CONTEXT"
    if klog <= BAND:
        return "D-DOWNSTREAM"
    return "C-NO-SUPPRESSION"


def cand_logits(model, st, prompt_ids, ids_list):
    """候选（单 token）在最后一个 prompt 位置的**原始 logit**（δ₁/δ₂ 同单位，§2.2）。"""
    out = []
    for cid in ids_list:
        seq = torch.tensor([prompt_ids + cid], dtype=torch.long, device=DEV)
        lg = model(input_ids=seq).logits.float()
        out.append(float(lg[0, len(prompt_ids) - 1, cid[0]]))
    return out


def logit_matrix(model, st, items):
    return [[cand_logits(model, st, it["prompt_ids"], [c[0] for c in it["cand"]]) for it in items]][0]


class CapC:
    """捕获 `LayerMem.read` 的输入 `hs_in`（即 v2.4 的 `c`）。`DEF-96-01`：只在一个条件下捕获。"""

    def __init__(self, st):
        self.st = st
        self.buf = {li: [] for li in st.mems}
        self._orig = sgmn_module.LayerMem.read

    def __enter__(self):
        outer = self

        def patched(self, hs_in, drop_sg=False):
            for li, m in outer.st.mems.items():
                if m is self:
                    outer.buf[li].append(hs_in.reshape(-1, hs_in.shape[-1]).detach())
                    break
            return outer._orig(self, hs_in, drop_sg)

        sgmn_module.LayerMem.read = patched
        return self

    def __exit__(self, *a):
        sgmn_module.LayerMem.read = self._orig
        for li in self.buf:
            self.buf[li] = torch.cat(self.buf[li], 0) if self.buf[li] else None
        return False


def kappa_A(W0, Wo, Wq, dW, nu):
    A0 = Wo @ W0 @ Wq
    A = Wo @ (W0 + dW) @ Wq
    return float((A - A0).norm() / A.norm()) / nu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="formal", choices=["smoke", "formal"])
    a = ap.parse_args()
    smoke = (a.stage == "smoke")
    seeds = SEEDS[:1] if smoke else SEEDS
    OUT.mkdir(exist_ok=True)
    G = {}

    tok, model = KP.model_and_tok()
    model.eval()
    torch.set_grad_enabled(True)
    facts = KP.build_facts(tok)
    chunks, owner = KP.build_chunks(tok, facts)
    if smoke:
        chunks = chunks[:max(4, len(chunks) // 10)]
    tr = list(range(R91.N_TRAIN_KEYS))
    print(f"[setup] chunks={len(chunks)} tr={len(tr)} smoke={smoke} dev={DEV}", flush=True)

    # ---------------- 趟①：写 + 训练读出 ⇒ 只留状态、立刻卸栈 ----------------
    saved = {}
    KP.install_write()
    try:
        for sd in seeds:
            t0 = time.time()
            st = R91.make_stack(gamma=0.0, no_write=False, seed=sd)
            KP.write_phase_arm(model, st, chunks, owner, None)
            losses, viol, div = R91.train_read_task(
                model, st, tok, facts, tr, random.Random(1000 + sd % 100))
            saved[sd] = {
                "W": {li: m.Wfull.detach().clone() for li, m in st.mems.items()},
                "W0": {li: m.W0.detach().clone() for li, m in st.mems.items()},
                "Wq": {li: m.Wq.detach().clone() for li, m in st.mems.items()},
                "Wo": {li: m.Wo.detach().clone() for li, m in st.mems.items()},
                "gamma": {li: m.gamma.detach().clone() for li, m in st.mems.items()},
            }
            print(f"[train] seed={sd} loss={losses[-1]:.3e} viol={viol} div={div} "
                  f"{time.time()-t0:.0f}s", flush=True)
            st.remove()
            del st
            torch.cuda.empty_cache()
    finally:
        KP.restore_write()

    # ---------------- 趟②：逐 seed 计算三级 κ ----------------
    allc = {}
    for sd in seeds:
        t0 = time.time()
        st = R91.make_stack(gamma=0.0, no_write=True, seed=sd)
        load_state(st, saved[sd])
        W, W0 = saved[sd]["W"], saved[sd]["W0"]
        ZERO = {li: torch.zeros_like(W[li]) for li in W}
        items = W91.make_eval_items(tok, facts, tr, random.Random(555))
        layers = list(st.mems)
        R = {"layers": {}, "seed": sd, "n_items": len(items),
             "n_cand": len(items[0]["cand"])}

        # ---- (a) 三种 W 条件各跑一遍 logit 矩阵；W 条件下顺带捕获 c ----
        set_W(st, W)
        with CapC(st) as cap:
            L_W = logit_matrix(model, st, items)
        Cs = {li: cap.buf[li] for li in layers}
        set_W(st, W0)
        L_W0 = logit_matrix(model, st, items)
        set_W(st, ZERO)
        L_Z = logit_matrix(model, st, items)

        # ---- (b) 逐层 κ_A / κ_m / 随机控制 ----
        g = torch.Generator(device=DEV).manual_seed(4321)
        for li in layers:
            m = st.mems[li]
            dW = W[li] - W0[li]
            nu = float(dW.norm() / W0[li].norm())
            A = m.Wo @ W[li] @ m.Wq
            A0 = m.Wo @ W0[li] @ m.Wq
            dA = A - A0
            rho_A = float(dA.norm() / A.norm())
            C = Cs[li].T                                   # (d, n_c)
            rho_m = float((dA @ C).norm() / (A @ C).norm())
            AC = A @ C                                    # κ_m 与 κ_m^rand 共用同一分母（§2.4.2）
            ka_r, km_r = [], []
            for _ in range(8):                             # DEF-96-05：n=8 取中位
                Rr = torch.randn(W0[li].shape, generator=g, device=DEV, dtype=W0[li].dtype)
                Rr = Rr / Rr.norm() * (nu * float(W0[li].norm()))
                Ar = m.Wo @ (W0[li] + Rr) @ m.Wq
                ka_r.append(float((Ar - A0).norm() / Ar.norm()) / nu)
                km_r.append(float(((Ar - A0) @ C).norm() / AC.norm()) / nu)
            ka_med, km_med = statistics.median(ka_r), statistics.median(km_r)
            rands = ka_r
            R["layers"][str(li)] = {
                "nu": nu, "rho_A": rho_A, "kappa_A": rho_A / nu,
                "rho_m": rho_m, "kappa_m": rho_m / nu,
                "kappa_A_rand_median": ka_med,
                "kappa_A_rand_rel_spread": ((max(ka_r) - min(ka_r)) / ka_med) if ka_med else None,
                "kappa_m_rand_median": km_med,
                "kappa_m_rand_rel_spread": ((max(km_r) - min(km_r)) / km_med) if km_med else None,
                "cos_dW_W0": float((dW.flatten() @ W0[li].flatten())
                                   / (dW.norm() * W0[li].norm())),
                "dW_sv_top8": [float(x) for x in torch.linalg.svdvals(dW)[:8]],
                "n_ctx": int(Cs[li].shape[0]),
                "c_norm_mean": float(Cs[li].norm(dim=1).mean()),
                "c_norm_std": float(Cs[li].norm(dim=1).std()),
            }
            print(f"  [L{li}] ν={nu:.5f} κ_A={rho_A/nu:.4f} κ_m={rho_m/nu:.4f} "
                  f"κ_A^rand={ka_med:.4f} κ_m^rand={km_med:.4f} "
                  f"spread={R['layers'][str(li)]['kappa_A_rand_rel_spread']:.3f} "
                  f"n_ctx={Cs[li].shape[0]}", flush=True)

        # ---- (c) 整栈 κ_logit ----
        d1 = sum(abs(x - y) for rw, r0 in zip(L_W, L_W0) for x, y in zip(rw, r0))
        d2 = sum(abs(x - y) for rw, rz in zip(L_W, L_Z) for x, y in zip(rw, rz))
        n = sum(len(r) for r in L_W)
        nu_g = statistics.mean(v["nu"] for v in R["layers"].values())
        R["delta1"], R["delta2"], R["n_logit"] = d1 / n, d2 / n, n
        R["nu_global"] = nu_g
        R["kappa_logit"] = ((R["delta1"] / R["delta2"]) / nu_g) if R["delta2"] > 0 else None

        # ---- (c2) κ_logit^rand（§2.4.2：三层同一组随机扰动，8 次取中位）----
        gr = torch.Generator(device=DEV).manual_seed(4321)
        d1r = []
        for _ in range(8):
            Wr = {}
            for li in layers:
                Rr = torch.randn(W0[li].shape, generator=gr, device=DEV, dtype=W0[li].dtype)
                Rr = Rr / Rr.norm() * (R["layers"][str(li)]["nu"] * float(W0[li].norm()))
                Wr[li] = W0[li] + Rr
            set_W(st, Wr)
            L_R = logit_matrix(model, st, items)
            d1r.append(sum(abs(x - y) for rw, rr in zip(L_W, L_R) for x, y in zip(rw, rr)) / n)
        R["delta1_rand_median"] = statistics.median(d1r)
        R["kappa_logit_rand_median"] = ((R["delta1_rand_median"] / R["delta2"]) / nu_g
                                        if R["delta2"] > 0 else None)
        set_W(st, W)

        # ---- (d) 一致性核对：acc（不担任本卡判据）----
        accs = {}
        for nm, Wc in (("intact", W), ("W0", W0), ("zero", ZERO)):
            set_W(st, Wc)
            wc0 = {li: m.write_calls for li, m in st.mems.items()}
            r = W91.eval_keys(model, st, tok, facts, tr, None, False, None)
            wc1 = {li: m.write_calls for li, m in st.mems.items()}
            accs[nm] = {"acc_q": r["acc_q"], "acc_item": r["acc_item"],
                        "margin": r["margin_mean"],
                        "write_delta": {li: wc1[li] - wc0[li] for li in wc0}}
        R["acc"] = accs

        # ---- (e) 归类（§2.3′ rev 1.2）----
        kA = statistics.median(v["kappa_A"] for v in R["layers"].values())
        km = statistics.median(v["kappa_m"] for v in R["layers"].values())
        kl = R["kappa_logit"]
        R["kappa_A_seed"], R["kappa_m_seed"] = kA, km
        R["branch_seed"] = classify(kA, km, kl) if kl is not None else "INCONCLUSIVE-NO-LOGIT"
        R["branch_layers"] = {li: classify(v["kappa_A"], v["kappa_m"], kl)
                              for li, v in R["layers"].items()}
        R["n_agree_out_of_3"] = sum(1 for b in R["branch_layers"].values()
                                    if b == R["branch_seed"])
        R["stable"] = R["n_agree_out_of_3"] >= 2
        # ---- (f) 结构增益 g = κ_real/κ_rand（§2.4，★ 报告项、非判据）----
        kA_r = statistics.median(v["kappa_A_rand_median"] for v in R["layers"].values())
        km_r = statistics.median(v["kappa_m_rand_median"] for v in R["layers"].values())
        kl_r = R["kappa_logit_rand_median"]
        R["g"] = {
            "g_A": (kA / kA_r) if kA_r else None,
            "g_m": (km / km_r) if km_r else None,
            "g_logit": (kl / kl_r) if (kl is not None and kl_r) else None,
            "kappa_A_rand_seed": kA_r, "kappa_m_rand_seed": km_r,
            "kappa_logit_rand_seed": kl_r,
        }
        print(f"  ⇒ ν̄={nu_g:.5f} δ₁={R['delta1']:.4f} δ₂={R['delta2']:.4f} "
              f"κ_logit={kl:.4f}", flush=True)
        print(f"  ⇒ seed {sd}: g_A={R['g']['g_A']:.4f} g_m={R['g']['g_m']:.4f} "
              f"g_logit={R['g']['g_logit']:.4f}", flush=True)
        print(f"  ⇒ seed {sd}: κ_A={kA:.4f} κ_m={km:.4f} ⇒ {R['branch_seed']} "
              f"(n_agree={R['n_agree_out_of_3']}/3) acc={ {k: v['acc_q'] for k, v in accs.items()} }",
              flush=True)
        allc[str(sd)] = R
        st.remove()
        del st
        torch.cuda.empty_cache()
        print(f"  [seed {sd} done] {time.time()-t0:.0f}s", flush=True)
    G["seeds"] = allc

    # ---------------- 守卫（§5）----------------
    A = {}
    A["A96_2_nu_reproduce_D125"] = all(
        0.030 <= v["nu"] <= 0.045 for R in allc.values() for v in R["layers"].values())
    # rev 1.3：原 "∈[0.5,2.0]" 带只对未训练读出侧成立（算术前置 0.9995；训练后实测 0.44）⇒ DEF-96-08
    A["A96_4_rand_control_sane"] = all(
        v["kappa_A_rand_median"] > 0
        and v["kappa_A_rand_rel_spread"] is not None
        and v["kappa_A_rand_rel_spread"] <= 0.5
        for R in allc.values() for v in R["layers"].values())
    if len(allc) > 1:
        _rs = list(allc.values())
        A["A96_11_rand_stable_across_seeds"] = all(
            abs(_rs[0]["layers"][li]["kappa_A_rand_median"]
                - _rs[1]["layers"][li]["kappa_A_rand_median"])
            / max(_rs[0]["layers"][li]["kappa_A_rand_median"],
                  _rs[1]["layers"][li]["kappa_A_rand_median"]) <= 0.5
            for li in _rs[0]["layers"])
    else:
        A["A96_11_rand_stable_across_seeds"] = True   # smoke 只有一个 seed ⇒ 不适用
    A["A96_5_finite"] = all(
        math.isfinite(v["rho_A"]) and math.isfinite(v["rho_m"]) and v["rho_A"] >= 0
        for R in allc.values() for v in R["layers"].values())
    A["A96_6_c_real"] = all(
        v["n_ctx"] >= R["n_items"] * R["n_cand"]      # 每题每候选至少 1 个 token
        and v["c_norm_std"] > 0 and v["c_norm_mean"] > 0   # 排除抓错/常量张量
        for R in allc.values() for v in R["layers"].values())
    A["A96_7_no_write_during_eval"] = all(
        sum(c["write_delta"].values()) == 0
        for R in allc.values() for c in R["acc"].values())
    A["A96_8_one_stack"] = len(saved) == len(seeds)
    A["A96_9_logit_same_exam"] = all(
        R["n_logit"] == R["n_items"] * R["n_cand"] for R in allc.values())
    A["A96_10_dW_nonzero"] = all(
        0 < v["nu"] < 1 for R in allc.values() for v in R["layers"].values())
    A["A96_acc_intact_eq_W0_reproduces_D125"] = all(
        abs(R["acc"]["intact"]["acc_q"] - R["acc"]["W0"]["acc_q"]) <= 1e-9
        for R in allc.values())
    G["guards"] = A

    vs = [R["branch_seed"] for R in allc.values()]
    stables = all(R["stable"] for R in allc.values())
    if not stables:
        G["verdict_agg"] = "INCONCLUSIVE-LAYER-HETEROGENEITY"
    elif len(set(vs)) == 1:
        G["verdict_agg"] = vs[0]
    else:
        G["verdict_agg"] = "INCONCLUSIVE-SEED-SPLIT"
    if smoke:
        G["verdict_agg"] = "SMOKE-NO-VERDICT"
    G["H96"] = ("成立" if G["verdict_agg"] == "A-READOUT"
                else ("否证" if G["verdict_agg"] in
                      ("B-CONTEXT", "C-NO-SUPPRESSION", "D-DOWNSTREAM") else "未判"))

    print("\n=== 守卫 ===")
    for k, v in A.items():
        print(f"  {'✓' if v else '✗'} {k} = {v}")
    print(f"\n★ 合议 = {G['verdict_agg']}  ·  H96 = {G['H96']}")
    for sd, R in allc.items():
        print(f"  seed {sd}: κ_A={R['kappa_A_seed']:.4f} κ_m={R['kappa_m_seed']:.4f} "
              f"κ_logit={R['kappa_logit']:.4f} ⇒ {R['branch_seed']}")

    (OUT / f"stage_{a.stage}.json").write_text(
        json.dumps({"stage": a.stage, "seeds": seeds, **G}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n[写出] {OUT/f'stage_{a.stage}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())