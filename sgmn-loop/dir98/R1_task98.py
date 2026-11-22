#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-98 / rev 1.3 —— `O96-1`：去相关写侧键，写规则能否**够到**目标？（几何/预算/写率三分）

逐字派生 `dir97/R1_task97.py` + `dir91/probe91_kpool.py`。唯一自变量 = 键变换 `T`（`k' = s·T(k)`）；
预算 `P∈{1,10}` 只是分层。`v` 侧不动（`v_rule_mode("mean",...)` 不含 `k`，已核源码 :105-107）。

主判据量 `π`：`t_i = v_i − W⁰k_i`，`π = 1/n Σ_i [1 − ‖ΔW k_i − t_i‖/‖t_i‖]`
结果读数 `N_hit^v`（定义与 `D-127` 逐字相同；零期望 `E_random = Σm_v²/32 = 1.3125`）
"""
import argparse
import collections
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import probe91_kpool as KP                       # noqa: E402
import R2_task91 as R91                          # noqa: E402
import torch                                     # noqa: E402
from sgmn_module import LayerMem, sg             # noqa: E402

SEEDS = [20261028, 7777]
CONFIGS = [("intact", 1), ("intact", 10), ("center", 1), ("white", 1), ("white", 10)]
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
DEV = R91.DEV
CHECKS = []
_ORIG_WRITE = LayerMem.write


def rec(cid, ok, detail):
    CHECKS.append({"id": cid, "ok": bool(ok), "detail": detail})
    print(f"  {'✓' if ok else '✗'} {cid}  {detail}", flush=True)


def norm_rows(X):
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-30)


def n_hit(W, K, V, Ssets):
    """N_hit^v / N_hit^s / N_hit^v_L2 —— 与 dir97/R1_task97.py 逐字相同。"""
    R = (W @ K.T).T
    COS = norm_rows(R) @ norm_rows(V).T
    am = COS.argmax(dim=1)
    strict = int((am == torch.arange(K.shape[0], device=DEV)).sum())
    valmatch = int(sum(int(am[i]) in Ssets[i] for i in range(K.shape[0])))
    D = (R.unsqueeze(1) - V.unsqueeze(0)).pow(2).sum(-1)
    am2 = D.argmin(dim=1)
    return valmatch, strict, int(sum(int(am2[i]) in Ssets[i] for i in range(K.shape[0])))


def stable_rank(X):
    sv = torch.linalg.svdvals(X)
    return float((X ** 2).sum()) / float(sv[0]) ** 2


def progress_pi(W, W0, K, V):
    """π（推进度）+ mean‖t_i‖（所需修正的尺度）。"""
    dW = W - W0
    Tm = (V - (W0 @ K.T).T)                 # t_i = v_i − W⁰k_i  (n,d_v)
    got = (dW @ K.T).T                      # ΔW k_i             (n,d_v)
    num = (got - Tm).norm(dim=1)
    den = Tm.norm(dim=1).clamp_min(1e-30)
    return float((1.0 - num / den).mean()), float(den.mean())


# ---------------------------------------------------------------- 键变换
class IdentityTF:
    name = "intact"

    def __call__(self, k):
        return k

    def freeze(self):
        pass


class CenterTF:
    """因果滑动均值居中（μ_t 只用 t 之前的键）；`s` 由采集遍定。"""

    def __init__(self, s):
        self.s, self.total, self.n = s, None, 0
        self.first_was_zero = None

    def __call__(self, k):
        if self.first_was_zero is None:
            self.first_was_zero = (self.n == 0)          # A98-8：首调用必须 μ=0
        mu = torch.zeros_like(k) if self.n == 0 else (self.total / self.n)
        out = (k - mu) * self.s
        self.total = k.detach().clone() if self.total is None else self.total + k.detach()
        self.n += 1
        return out

    def freeze(self):
        self.mu_final = (torch.zeros_like(self.total) if self.n == 0 else self.total / self.n)
        self.__class__ = _FrozenCenter
        self._tf = self

    def _apply(self, k):
        return (k - self.mu_final) * self.s


class _FrozenCenter(CenterTF):
    """freeze() 之后：μ 固定为写相终值，不再累计（§2.5）。"""

    def __call__(self, k):
        return self._apply(k)

    def freeze(self):
        pass


class WhiteTF:
    """`A = (U√n)·K_raw⁺`：把键映到事实跨度内的正交方向（**上界臂**）。"""

    def __init__(self, Un, Kp, s):
        self.Un, self.Kp, self.s = Un, Kp, s

    def __call__(self, k):
        return self.s * (self.Un @ (self.Kp @ k))

    def freeze(self):
        pass


def _write_tf(self, hs_in, drop_sg=False):
    """= probe91_kpool._write_patched 逐字；唯一差别 = `k` 之后过一层 `self.key_tf`。"""
    if self.prev_hs is None or self.no_write:
        return
    self.write_calls += 1
    pf = self.prev_hs.reshape(-1, self.prev_hs.shape[-1])
    hf = hs_in.reshape(-1, hs_in.shape[-1])
    pk_raw = sg(pf @ self.Pk, drop_sg)
    kmode, vmode = KP.RULE[KP.ARM["name"]]
    k = self.key_tf(KP.k_rule(kmode, pk_raw, self.pk_scale, KP.VPOS["V"]))
    kp = pk_raw * self.pk_scale
    zv = sg(hf @ self.Pv, drop_sg)
    v = KP.v_rule_mode(vmode, zv, kp, k, KP.VPOS["V"])
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    G = self.alpha * torch.outer(W @ k - v, k) + self.lam * (W - self.W0)
    with torch.no_grad():
        if self.use_factorized:
            self.U.copy_(self.U - self.eta * (G @ self.V))
            self.V.copy_(self.V - self.eta * (G.T @ self.U))
        else:
            self.Wfull.copy_(self.Wfull - self.eta * G)
    self.steps += 1


def install_write():
    LayerMem.write = _write_tf


def restore_write():
    LayerMem.write = _ORIG_WRITE


def key_matrices_tf(model, st, facts, chunks, owner):
    """= probe91_kpool.key_matrices_arm 逐字；唯一差别 = K 过 `m.key_tf`（v 侧不动）。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    K = {li: [] for li in st.mems}
    Vv = {li: [] for li in st.mems}
    try:
        for fi in range(len(facts)):
            ci = owner.index(fi) + 1
            KP.VPOS["V"] = None
            model(input_ids=chunks[ci].to(DEV))
            for li, m in st.mems.items():
                hs = st._hs_in[li]
                pf = hs.reshape(-1, hs.shape[-1])
                pk_raw = sg(pf @ m.Pk)
                kmode, vmode = KP.RULE[KP.ARM["name"]]
                K[li].append(m.key_tf(KP.k_rule(kmode, pk_raw, m.pk_scale, None)).detach())
                zv = sg(pf @ m.Pv)
                Vv[li].append(KP.v_rule_mode(vmode, zv, pk_raw * m.pk_scale, None, None).detach())
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return ({li: torch.stack(K[li]).float() for li in K},
            {li: torch.stack(Vv[li]).float() for li in Vv})


# ---------------------------------------------------------------- 手算算例（§3）
def toy_example():
    V = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], device=DEV)
    K = torch.tensor([[1., 0.], [0., 1.], [0.707, 0.707]], device=DEV)
    vals = [tuple(v) for v in V.tolist()]
    Ssets = [[j for j in range(3) if vals[j] == vals[i]] for i in range(3)]
    W0 = torch.zeros(2, 2, device=DEV)
    W = torch.eye(2, device=DEV)
    pi, _ = progress_pi(W, W0, K, V)
    nv, ns, _ = n_hit(W, K, V, Ssets)
    Er = sum(len(s) for s in Ssets) / 3
    pi0, _ = progress_pi(W0, W0, K, V)
    ok = (abs(pi - 0.745) < 1e-3 and nv == 3 and ns == 2
          and abs(Er - 5 / 3) < 1e-12 and abs(pi0) < 1e-9)
    rec("A98-10", ok, f"手算算例 π={pi:.4f}(≈0.745) N_hit^v={nv}(3) N_hit^s={ns}(2) "
                      f"E_random={Er:.4f}(5/3) π(ΔW=0)={pi0:.1e}(0)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="formal", choices=["stageA", "formal"])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    tok, model = KP.model_and_tok()
    model.eval()
    torch.set_grad_enabled(True)
    facts = KP.build_facts(tok)
    chunks, owner = KP.build_chunks(tok, facts)
    n = len(facts)
    vals = [f["val"] for f in facts]
    uniq = sorted(set(vals))
    Ssets = [[j for j in range(n) if vals[j] == vals[i]] for i in range(n)]
    sum_S = sum(len(s) for s in Ssets)
    E_random = sum_S / n
    mc = collections.Counter(vals)
    print(f"[setup] facts={n} chunks={len(chunks)} distinct_vals={len(uniq)} "
          f"Σ|S_i|={sum_S} E_random={E_random:.4f} configs={CONFIGS}", flush=True)

    G = {"stage": a.stage, "n_facts": n, "E_random": E_random, "sum_S": sum_S,
         "n_distinct": len(uniq), "runs": {}, "checks": CHECKS}

    if not toy_example():
        print("[FATAL] 手算算例未通过 ⇒ 测量层有错，不开 Stage A", flush=True)
        (OUT / f"stage_{a.stage}.json").write_text(json.dumps(G, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
        return 1

    for sd in SEEDS:
        t_seed = time.time()
        # ---------- §2.2 采集遍（恒等变换、no_write）→ 定 A / s ----------
        st0 = R91.make_stack(gamma=0.0, no_write=True, seed=sd)
        for m in st0.mems.values():
            m.key_tf = IdentityTF()          # 采集遍 = 恒等变换（定义 K_raw）
        K_raw, _ = key_matrices_tf(model, st0, facts, chunks, owner)
        W0s = {li: m.W0.detach().clone() for li, m in st0.mems.items()}
        st0.remove()
        del st0
        torch.cuda.empty_cache()
        TFspec = {}
        for li in K_raw:
            Kr = K_raw[li].to(torch.float64)
            mu = Kr.mean(dim=0)
            # Kr 的行是键(32×896) ⇒ 薄 SVD 给 U(32,32) S(32) Vh(32,896)；
            # 896 维正交基是 Vh.T (896,32)，**不是** U（U 是 32 维系数量空间）
            U, S, Vh = torch.linalg.svd(Kr, full_matrices=False)
            Un = (Vh.T * (n ** 0.5)).float()                   # V√n  (896,32)
            Kp = torch.linalg.pinv(Kr.T).float()               # (32,896)：Kp@kr_i = e_i
            base = Kr.norm(dim=1).mean()
            s_c = float(base / (Kr - mu).norm(dim=1).mean())
            Aw = Un.float() @ Kp
            s_w = float(base / (Kr @ Aw.to(torch.float64).T).norm(dim=1).mean())
            TFspec[li] = dict(Un=Un, Kp=Kp, s_c=s_c, s_w=s_w)
            cr = norm_rows(Kr) @ norm_rows(Kr).T
            print(f"[seed {sd} L{li}] 采集: maxcos_raw={float(cr.fill_diagonal_(-2).max()):.4f} "
                  f"σ1/σ32={float(S[0] / S[-1]):.1f} s_center={s_c:.4f} s_white={s_w:.4f}", flush=True)

        for geom, P in CONFIGS:
            tag = f"{geom}@{P}"
            st = R91.make_stack(gamma=0.0, no_write=False, seed=sd)
            tfs = {}
            for li, m in st.mems.items():
                sp = TFspec[li]
                if geom == "intact":
                    tf = IdentityTF()
                elif geom == "center":
                    tf = CenterTF(sp["s_c"])
                else:
                    tf = WhiteTF(sp["Un"].to(DEV), sp["Kp"].to(DEV), sp["s_w"])
                m.key_tf = tf
                tfs[li] = tf
            install_write()
            try:
                for _ in range(P):
                    KP.write_phase_arm(model, st, chunks, owner, None)
                for tf in tfs.values():
                    tf.freeze()                     # 测量遍用终值 μ，不再累计
                Ws = {li: m.Wfull.detach().clone() for li, m in st.mems.items()}
                Kt, Vt = key_matrices_tf(model, st, facts, chunks, owner)
                for m in st.mems.values():
                    m.no_write = True
                st.remove()
                del st
                torch.cuda.empty_cache()
            finally:
                restore_write()

            perm = list(range(n))
            random.Random(20261120).shuffle(perm)
            RS = {}
            for li in sorted(Ws):
                W, W0, K2, V2 = Ws[li], W0s[li], Kt[li], Vt[li]
                dW = W - W0
                Kplus = torch.linalg.pinv(K2.T)
                Vperm = V2[torch.as_tensor(perm, device=DEV)]
                W_or = V2.T @ Kplus
                W_or_pi = Vperm.T @ Kplus
                pi, tscale = progress_pi(W, W0, K2, V2)
                nv, ns, nl2 = n_hit(W, K2, V2, Ssets)
                w0v, _, _ = n_hit(W0, K2, V2, Ssets)
                orv, _, _ = n_hit(W_or, K2, V2, Ssets)
                piv, _, _ = n_hit(W_or_pi, K2, V2, Ssets)
                Cn = norm_rows(K2) @ norm_rows(K2).T
                sv = torch.linalg.svdvals(K2)
                nrank = int((sv > 1e-3 * sv[0]).sum())
                Kr64 = K_raw[li].to(torch.float64)
                T_on_raw = torch.stack([tfs[li](Kr64[i].float()) for i in range(n)])
                RS[str(li)] = {
                    "pi": pi, "t_scale": tscale,
                    "need": float((W_or - W0).norm() / W0.norm()),
                    "moved": float(dW.norm() / W0.norm()),
                    "stable_rank": stable_rank(dW),
                    "n_hit_v": nv, "n_hit_s": ns, "n_hit_v_L2": nl2, "n_hit_v_W0": w0v,
                    "n_hit_v_oracle": orv, "n_hit_v_oracle_perm": piv,
                    "maxcos_keys": float(Cn.fill_diagonal_(-2.0).max()),
                    "nrank_keys": nrank, "cond_keys": float(sv[0] / sv[-1]),
                    "src_dev": float((T_on_raw - K2).abs().max()),
                    "norm_ratio_measured": float(
                        K2.norm(dim=1).mean() / K_raw[li].norm(dim=1).mean()),
                    "pi_zero_dW": progress_pi(W0, W0, K2, V2)[0],
                    "key_drift_vs_raw": float((K2 - K_raw[li]).abs().max()),
                    "norm_dev_collect": float(abs(
                        (Kr64.norm(dim=1).mean() - T_on_raw.to(torch.float64).norm(dim=1).mean())
                        / Kr64.norm(dim=1).mean())),
                    "first_mu_zero": (True if geom != "center" else tfs[li].first_was_zero),
                }
                print(f"[seed {sd} {tag} L{li}] π={pi:+.4f} need={RS[str(li)]['need']:.3f} "
                      f"moved={RS[str(li)]['moved']:.3e} sr={RS[str(li)]['stable_rank']:.4f} "
                      f"N_hit^v={nv} | maxcos={RS[str(li)]['maxcos_keys']:.4f} nrank={nrank} "
                      f"| or={orv} orpi={piv} W0={w0v}", flush=True)
            G["runs"][f"{sd}|{tag}"] = RS
        print(f"[seed {sd}] 完成，用时 {time.time() - t_seed:.0f}s", flush=True)

    # ---------------- Stage A ----------------
    layers = sorted(int(x) for x in G["runs"][f"{SEEDS[0]}|intact@1"])

    def cell(sd, tag, li):
        return G["runs"][f"{sd}|{tag}"][str(li)]

    ALL = [(sd, f"{g}@{P}", li) for sd in SEEDS for (g, P) in CONFIGS for li in layers]

    def w(key, tags=None):
        return [cell(sd, tag, li)[key] for (sd, tag, li) in ALL if tags is None or tag in tags]

    int1 = [c for c in (cell(sd, "intact@1", li) for sd in SEEDS for li in layers)]

    rec("A98-7", abs(E_random - 1.3125) < 1e-12 and sum_S == sum(m * m for m in mc.values())
        and len(uniq) == 27,
        f"E_random={E_random:.4f}==1.3125 · Σ|S_i|={sum_S}==Σm_v² · 不同值={len(uniq)}==27")
    rec("A98-0", max(c["stable_rank"] for c in int1) <= 1.05
        and max(c["n_hit_v"] for c in int1) <= 8
        and 2e-2 <= min(c["moved"] for c in int1) and max(c["moved"] for c in int1) <= 6e-2,
        f"intact@1: max sr={max(c['stable_rank'] for c in int1):.4f}(≤1.05) "
        f"max N_hit^v={max(c['n_hit_v'] for c in int1)}(≤8) "
        f"moved∈[{min(c['moved'] for c in int1):.3e},{max(c['moved'] for c in int1):.3e}]([2e-2,6e-2])")
    dec = [t for t in ["center@1", "white@1", "white@10"]]
    ok1 = [bool(c["maxcos_keys"] < 0.90 and c["nrank_keys"] >= 24)
           for c in (cell(sd, tag, li) for sd in SEEDS for tag in dec for li in layers)]
    ints_maxcos = [cell(sd, "intact@1", li)["maxcos_keys"] for sd in SEEDS for li in layers]
    rec("A98-1", all(ok1) and min(ints_maxcos) > 0.98,
        f"去相关臂 (maxcos<0.90 且 nrank≥24)：{sum(ok1)}/{len(ok1)} 通过；"
        f"intact maxcos∈[{min(ints_maxcos):.4f},{max(ints_maxcos):.4f}](>0.98)")
    rec("A98-2", max(w("src_dev")) < 1e-9,
        f"max‖K'(arm) − T(K_raw)‖ = {max(w('src_dev')):.3e} (<1e-9)")
    rec("A98-3", max(w("norm_dev_collect")) < 1e-3,   # rev 1.4：float32 分辨率（原 1e-6 不可达）
        f"max 采集遍 |mean‖s·T(k)‖/mean‖k‖ − 1| = {max(w('norm_dev_collect')):.3e} (<1e-3)；"
        f"写相后范数比仅披露 ∈[{min(w('norm_ratio_measured')):.3f},"
        f"{max(w('norm_ratio_measured')):.3f}]")
    rec("A98-4", min(w("n_hit_v_oracle")) >= 30,
        f"min N_hit^v(oracle) = {min(w('n_hit_v_oracle'))}/32 (≥30)")
    rec("A98-5", max(w("n_hit_v_oracle_perm")) <= 3,
        f"max N_hit^v(oracle^π) = {max(w('n_hit_v_oracle_perm'))}/32 (≤3)")
    rec("A98-6", max(w("n_hit_v_W0")) <= E_random + 3,
        f"max N_hit^v(W⁰) = {max(w('n_hit_v_W0'))}/32 (≤{E_random + 3:.3f})")
    rec("A98-11", max(abs(x) for x in w("pi_zero_dW")) < 1e-6,   # rev 1.4：float32 累加（原 1e-9 不可达）
        f"ΔW=0 ⇒ max|π| = {max(abs(x) for x in w('pi_zero_dW')):.3e} (<1e-6，逐格实算)")
    mu0 = [cell(sd, "center@1", li)["first_mu_zero"] for sd in SEEDS for li in layers]
    rec("A98-8", all(x is True for x in mu0),
        f"center 因果：首调用 μ=0 成立 {sum(1 for x in mu0 if x is True)}/{len(mu0)}（逐层实记）")
    drift = [cell(sd, "intact@1", li)["key_drift_vs_raw"] for sd in SEEDS for li in layers]
    rec("A98-9", max(drift) < 1e-9,
        f"γ=0 ⇒ intact@1 写相后键与 K_raw 逐位相同：max|Δ| = {max(drift):.3e} (<1e-9)")

    A_valid = [tag for tag in ["center@1", "white@1", "white@10"]
               if all(cell(sd, tag, li)["maxcos_keys"] < 0.90 and cell(sd, tag, li)["nrank_keys"] >= 24
                      for sd in SEEDS for li in layers)]
    G["A_valid"] = A_valid
    G["stageA_pass"] = all(c["ok"] for c in CHECKS)

    # ---------------- 判决（§1.3 封闭三支；只在 Stage A 全过时） ----------------
    if G["stageA_pass"]:
        av1 = [t for t in A_valid if t.endswith("@1")]
        av10 = [t for t in A_valid if t.endswith("@10")]
        pi1 = {t: min(cell(sd, t, li)["pi"] for sd in SEEDS for li in layers) for t in av1}
        pi10 = {t: min(cell(sd, t, li)["pi"] for sd in SEEDS for li in layers) for t in av10}
        if not A_valid:
            v = "INVALID"
        elif av1 and any(x >= 0.5 for x in pi1.values()):
            v = "CONFIRM"
        elif (not av1 or all(x < 0.5 for x in pi1.values())) \
                and (not av10 or all(x < 0.5 for x in pi10.values())):
            v = "FALSIFY"
        elif av1:
            v = "AMBIGUOUS"
        else:
            v = "INVALID"
        G["verdict"] = v
        G["pi_P1"] = pi1
        G["pi_P10"] = pi10
        G["pi_intact10"] = min(cell(sd, "intact@10", li)["pi"] for sd in SEEDS for li in layers)
        G["pi_intact1"] = min(cell(sd, "intact@1", li)["pi"] for sd in SEEDS for li in layers)
        print(f"\n[A98] π(P=1) = {pi1}\n[A98] π(P=10) = {pi10}"
              f"\n[A98] π(intact@1) = {G['pi_intact1']:.4f}  π(intact@10) = {G['pi_intact10']:.4f}"
              f"\n⇒ 判决 = {v}", flush=True)

    (OUT / f"stage_{a.stage}.json").write_text(json.dumps(G, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(f"\n[Stage A] {'全部通过' if G['stageA_pass'] else '有 FAIL'} → {OUT}/stage_{a.stage}.json")
    return 0 if G["stageA_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())