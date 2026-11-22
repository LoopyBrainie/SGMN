#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-99 / rev 1.0 —— 去相关键让 `W` 存下内容（`D-128`），**读侧链路能否把它变成任务准确率？**

逐字派生 `dir98/R1_task98.py`（键变换单点注入）+ `dir91/probe91_kpool.py::run_arm`（任务链路）。
★ **唯一新增**：① `q_tf`（读侧变换，与写侧 `T` 严格同源）② `W` 四条件评测
（`W_written` / `W⁰` / `0` / `W_or` 上限）③ `center_fixed`（`μ` 由标定前缀估计后冻结）。

判据（冻结，`DIRECTION_rev99_v10.md` §2）：
  `Δ_tr(arm) = acc_train(W_written) − acc_train(W⁰)`
  `Δ_ho(arm) = acc_held(W_written) − acc_held(W⁰)`
  `H99`：某去相关臂 `Δ_ho` 六格均值 ≥ **0.125**；否证线 < **0.0625**。
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
ARMS = ["intact", "center_fixed", "white", "permneg", "w_oracle"]
# permneg=A99-4 阴性对照；w_oracle=★rev1.1 真可达性上限（读侧**对着 W_or 训练**）
N_CAL = 60                                    # center_fixed 标定前缀块数（600 的 1/10）
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
DEV = R91.DEV
CHECKS = []
_ORIG_WRITE = LayerMem.write
_ORIG_READ = LayerMem.read


def rec(cid, ok, detail):
    CHECKS.append({"id": cid, "ok": bool(ok), "detail": detail})
    print(f"  {'✓' if ok else '✗'} {cid:<10} {detail}", flush=True)
    return ok


def norm_rows(X):
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


# ==================================================================== 键变换（写侧 T / 读侧同源）
def _s_calib(Tfun, Kraw, kmax=1.0, n_probe=256):
    """按 D-128 §A98-3 口径定 `s`：使 `mean‖s·T(k)‖ ≈ mean‖k‖`（超 256 行只抽样）。"""
    idx = torch.randperm(Kraw.shape[0], device=Kraw.device)[:n_probe]
    Kp = Kraw[idx]
    with torch.no_grad():
        Tk = Tfun(Kp)
    r = (Tk.norm(dim=1).mean() / Kp.norm(dim=1).mean().clamp_min(1e-30)).item()
    return float(1.0 / max(r, 1e-12))


class IdentityTF:
    name = "intact"

    def __call__(self, k):
        return k

    def qmap(self, q):
        return q

    def probe_ratio(self, Kraw):
        return 1.0


class CenterFixedTF:
    """`T(k) = s·(k − μ)`，`μ` 由标定前缀（前 `N_CAL` 块）估计后**冻结**（读写严格同源）。"""

    def __init__(self, mu, s):
        self.mu = mu
        self.s = s
        self.name = "center_fixed"

    def __call__(self, k):
        return (k - self.mu) * self.s

    def qmap(self, q):
        return (q - self.mu) * self.s


class WhiteTF:
    """`T(k) = s·A k`，`A = (V√n)·K_raw⁺`（**上界臂**，oracle）。"""

    def __init__(self, A, s):
        self.A = A
        self.s = s
        self.name = "white"

    def __call__(self, k):
        return self.s * (k @ self.A.T) if k.dim() > 1 else self.s * (self.A @ k)

    def qmap(self, q):
        return self.s * (q @ self.A.T) if q.dim() > 1 else self.s * (self.A @ q)


class _PermNeg:
    """`A99-4` 阴性对照：**破坏 k↔v 配对**（`v` 取上一块的 `hf`），键不变。"""
    name = "permneg"

    def __init__(self):
        self.hold = None

    def __call__(self, k):
        return k

    def qmap(self, q):
        return q


# ==================================================================== patch：写侧 key_tf
def _write_tf(self, hs_in, drop_sg=False):
    """= `probe91_kpool._write_patched` 逐字；唯一差别 = `k` 过 `self.key_tf`。
    另：`permneg` 臂把 `v` 换成**上一块**的 `hf` ⇒ 破坏配对（`A99-4`）。"""
    if self.prev_hs is None or self.no_write:
        return
    self.write_calls += 1
    pf = self.prev_hs.reshape(-1, self.prev_hs.shape[-1])
    hf = hs_in.reshape(-1, hs_in.shape[-1])
    pk_raw = sg(pf @ self.Pk, drop_sg)
    kmode, vmode = KP.RULE[KP.ARM["name"]]
    k = self.key_tf(KP.k_rule(kmode, pk_raw, self.pk_scale, KP.VPOS["V"]))
    kp = pk_raw * self.pk_scale
    hfv = hf
    if getattr(self, "v_neg_shift", False):
        cur = hf.detach().clone()
        hfv = self.v_hold if self.v_hold is not None else cur
        self.v_hold = cur
    zv = sg(hfv @ self.Pv, drop_sg)
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


# ==================================================================== patch：读侧 q_tf
def _read_tf(self, hs_in, drop_sg=False):
    """= `sgmn_module.LayerMem.read` 逐字；**唯一差别** = `q` 之后过 `self.q_tf`（与写侧同源）。"""
    self.read_calls += 1
    c = sg(hs_in, drop_sg)
    cf = c.reshape(-1, c.shape[-1])
    q = (self.Wq @ cf.T).T
    q = self.q_tf(q)
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    y = (sg(W, drop_sg) @ q.T).T
    m = (self.Wo @ y.T).T
    self.injection = m.reshape(hs_in.shape)
    return self.injection


def install():
    LayerMem.write = _write_tf
    LayerMem.read = _read_tf


def restore():
    LayerMem.write = _ORIG_WRITE
    LayerMem.read = _ORIG_READ


# ==================================================================== 采集遍（恒等、no_write）
def capture_keys(model, st, facts, chunks, owner):
    """= `probe91_kpool.key_matrices_arm` 逐字；唯一差别 = K 过 `m.key_tf`（v 侧不动）。"""
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


def build_tf(arm, Kraw, V, li):
    """按臂构造 `T`：`center_fixed` 的 μ 取标定前缀行（因果离线标定），`white` 用 oracle `A`。"""
    K = Kraw
    if arm == "intact":
        return IdentityTF()
    if arm == "permneg":
        return _PermNeg()
    if arm == "center_fixed":
        n = K.shape[0]
        mu = K[:min(N_CAL, n)].mean(dim=0)      # ★ 必须 1-D (d_k,)：写侧 k 是 1-D、读侧 q 是 (N,d_k)，两者都靠它广播
        s = _s_calib(lambda x: x - mu, K)
        return CenterFixedTF(mu, s)
    # white: A = Un·Kp，Un = Vhᵀ√n（键跨度的正交基，896×32）、Kp = pinv(Kᵀ)（32×896，Kp·k_i = e_i）
    # ★ 与 dir98:256-260 逐字同构；`A` 必须是真的 896×896 键→键映射（不是值空间）
    n = K.shape[0]
    _U, _S, Vh = torch.linalg.svd(K, full_matrices=False)     # Vh (32, 896)
    Un = (Vh.T * (n ** 0.5))                                  # (896, 32)
    Kp = torch.linalg.pinv(K.T)                               # (32, 896)
    Aw = Un @ Kp                                              # (896, 896)
    s = _s_calib(lambda x: x @ Aw.T, K)
    return WhiteTF(Aw, s)


# ==================================================================== 手算算例（§2.5 / A99-9）
def toy_example():
    ok = True
    ok &= rec("A99-9a", abs((3 / 4 - 1 / 4) - 0.5) < 1e-12, "Δ(3/4,1/4)=0.5")
    ok &= rec("A99-9b", abs((2 / 4 - 2 / 4) - 0.0) < 1e-12, "Δ(2/4,2/4)=0")
    V = torch.tensor([[1., 0.], [0., 1.]], device=DEV)
    K = V.clone()
    W = (torch.linalg.pinv(K) @ V).T
    got = (W @ K.T).T
    d = float((got - V).abs().max())
    ok &= rec("A99-9c", d < 1e-6, f"W_or 解算：max|W_or·K−V| = {d:.2e} (<1e-6)，shape={tuple(W.shape)}")
    return ok


# ==================================================================== 一臂
def run_arm(model, tok, facts, chunks, owner, tr, he, seed, arm):
    st = R91.make_stack(gamma=0.0, no_write=False, seed=seed)
    t0 = time.time()
    for m in st.mems.values():
        m.key_tf = IdentityTF()
        m.q_tf = IdentityTF()
        m.v_neg_shift = (arm == "permneg")
        m.v_hold = None
    install()
    try:
        # ---- 1) 采集遍（恒等）→ K_raw / V（V 与 T 无关：v_rule "mean" 只用 zv）
        Kraw, V = capture_keys(model, st, facts, chunks, owner)
        W0 = {li: m.W0.detach().clone() for li, m in st.mems.items()}
        # ---- 2) 定 T（逐层）并装上
        TF = {}
        for li in Kraw:
            tf = build_tf(arm, Kraw[li], V[li], li)
            TF[li] = tf
            st.mems[li].key_tf = tf
            st.mems[li].q_tf = tf.qmap
        # ---- 3) 真写（写侧带 T）；`w_oracle` 臂跳过写，W 直接置为下限范数精确解
        if arm == "w_oracle":
            Kl0 = Kraw
            Wor0 = {li: (torch.linalg.pinv(Kl0[li]) @ V[li]).T for li in Kl0}
            for li, m in st.mems.items():
                m.Wfull.data.copy_(Wor0[li])
            wc = {li: 0 for li in st.mems}
        else:
            wc = KP.write_phase_arm(model, st, chunks, owner, {i: None for i in range(len(facts))})
        # ---- 4) 写后采集（带 T）→ 几何 + W_or 上限
        Knew, Vnew = capture_keys(model, st, facts, chunks, owner)
        geo, Wor = {}, {}
        for li in Knew:
            Kl = Knew[li]
            nrm = norm_rows(Kl)
            C = nrm @ nrm.T
            iu = torch.triu_indices(Kl.shape[0], Kl.shape[0], offset=1)
            geo[str(li)] = {"maxcos": float(C[iu[0], iu[1]].max()),
                            "cos_median": float(C[iu[0], iu[1]].median())}
            Wor[li] = (torch.linalg.pinv(Kl) @ Vnew[li]).T        # W K^T = V^T 最小范数解
        # ---- 5) 读侧训练（训练的是 Wq/Wo/gamma，**不动 W**）
        rng = random.Random(1000 + seed % 100)
        Wwrite = {li: m.Wfull.detach().clone() for li, m in st.mems.items()}
        losses, viol, div = R91.train_read_task(model, st, tok, facts, tr, rng)
        Wafter = {li: m.Wfull.detach().clone() for li, m in st.mems.items()}
        dWrel = max(float((Wafter[li] - Wwrite[li]).norm() / Wwrite[li].norm().clamp_min(1e-30))
                    for li in Wwrite)
        snap = {li: (m.Wq.detach().clone(), m.Wo.detach().clone(), float(m.gamma))
                for li, m in st.mems.items()}
        # ---- 6) 四条件评测（只换 W；Wq/Wo/gamma 不动）
        res = {}
        for cond in ("written", "W0", "zero", "Wor"):
            for li, m in st.mems.items():
                if cond == "written":
                    m.Wfull.data.copy_(Wwrite[li])
                elif cond == "W0":
                    m.Wfull.data.copy_(W0[li])
                elif cond == "zero":
                    m.Wfull.data.zero_()
                else:
                    m.Wfull.data.copy_(Wor[li])
                m.Wq.data.copy_(snap[li][0]); m.Wo.data.copy_(snap[li][1])
                m.gamma.data.fill_(snap[li][2])
            for mm in st.mems.values():
                mm.no_write = True                      # DEF-95-01：评测期必须显式重封
            fit = KP.W91.eval_keys(model, st, tok, facts, tr, None, False, None)
            gen = KP.W91.eval_keys(model, st, tok, facts, he, None, False, None)
            res[cond] = {"acc_train": fit["acc_q"], "acc_held": gen["acc_q"],
                         "margin_train": fit["margin_mean"], "margin_held": gen["margin_mean"]}
        for li, m in st.mems.items():
            m.Wfull.data.copy_(Wwrite[li])
    finally:
        restore()
    # 组装本格
    dtr = res["written"]["acc_train"] - res["W0"]["acc_train"]
    dho = res["written"]["acc_held"] - res["W0"]["acc_held"]
    cell = {"seed": seed, "arm": arm, "gamma_final": max(float(v[2]) for v in snap.values()),
            "dW_rel_read": dWrel, "write_calls": {str(k): int(v) for k, v in wc.items()},
            "geo": geo, "acc": res, "d_tr": dtr, "d_ho": dho,
            "sec": round(time.time() - t0, 1), "violations": viol, "diverged": div,
            "final_loss": float(sum(losses[-50:]) / len(losses[-50:])) if losses else None}
    print(f"[{seed} {arm}] d_tr={dtr:+.4f} d_ho={dho:+.4f} | "
          f"acc_tr w={res['written']['acc_train']:.4f} W0={res['W0']['acc_train']:.4f} "
          f"z={res['zero']['acc_train']:.4f} or={res['Wor']['acc_train']:.4f} | "
          f"acc_ho w={res['written']['acc_held']:.4f} W0={res['W0']['acc_held']:.4f} "
          f"z={res['zero']['acc_held']:.4f} or={res['Wor']['acc_held']:.4f} | "
          f"maxcos={list(geo.values())[0]['maxcos']:.4f} dWrel={dWrel:.1e} {cell['sec']}s", flush=True)
    st.remove()
    del st
    torch.cuda.empty_cache()
    return cell


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="formal", choices=["stageA", "formal", "smoke"])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_grad_enabled(True)
    tok, model = KP.model_and_tok()
    model.eval()
    facts = KP.build_facts(tok)
    chunks, owner = KP.build_chunks(tok, facts)
    tr = list(range(16))
    he = list(range(16, 32))
    G = {"stage": a.stage, "cells": [], "checks": CHECKS}
    print(f"[setup] facts={len(facts)} chunks={len(chunks)} tr={len(tr)} he={len(he)} "
          f"arms={ARMS}", flush=True)

    if not toy_example():
        (OUT / f"stage_{a.stage}.json").write_text(
            json.dumps(G, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[FATAL] 手算算例未过 ⇒ 不开 Stage A", flush=True)
        return 1

    seeds = SEEDS if a.stage != "smoke" else SEEDS[:1]
    arms = ARMS if a.stage == "formal" else ["intact", "center_fixed"]
    for sd in seeds:
        for arm in arms:
            G["cells"].append(run_arm(model, tok, facts, chunks, owner, tr, he, sd, arm))
            (OUT / f"stage_{a.stage}.json").write_text(
                json.dumps(G, ensure_ascii=False, indent=1), encoding="utf-8")   # DEF-95-04 先落盘

    # ------------------------------- 判据与守卫
    C = G["cells"]
    def mean(f, arm):
        v = [c[f] for c in C if c["arm"] == arm]
        return (sum(v) / len(v)) if v else None
    dho = {arm: mean("d_ho", arm) for arm in arms}
    dtr = {arm: mean("d_tr", arm) for arm in arms}
    G["d_ho_mean"] = dho
    G["d_tr_mean"] = dtr

    if a.stage == "formal":
        dec = [dho[x] for x in ("center_fixed", "white")]
        # ★rev1.1：上限 = **对着 W_or 训练过**的读侧（post-hoc 换 W 会打碎复合函数，见 rev1.1 §DEF-99-04）
        ceiling = max(c["acc"]["written"]["acc_held"] for c in C if c["arm"] == "w_oracle")
        reach_ok = ceiling >= 0.125
        if reach_ok:
            rec("A99-3", True, f"可达性：ceiling acc_held(读侧对 W_or 训练) = {ceiling:.4f} ≥ 0.125")
            if max(dec) >= 0.125:
                verdict = "A-READ-OK"
            elif max(dec) < 0.0625:
                verdict = "A-READ-BLIND"
            else:
                verdict = "AMBIGUOUS"
        else:
            rec("A99-3", False, f"可达性：ceiling acc_held(读侧对 W_or 训练) = {ceiling:.4f} < 0.125 "
                               f"⇒ 判据 UNREACHABLE（不得记 FAIL）")
            verdict = "UNREACHABLE"
        G["verdict"] = verdict
        print(f"[DIR-99] d_ho 六格均值 = {dho}", flush=True)
        print(f"[DIR-99] d_tr 六格均值 = {dtr}", flush=True)
        print(f"[DIR-99] ceiling acc_held(w_oracle) = {ceiling:.4f}", flush=True)
        print(f"[DIR-99] 判决 = {verdict}", flush=True)
    (OUT / f"stage_{a.stage}.json").write_text(json.dumps(G, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    bad = [c for c in CHECKS if not c["ok"]]
    print(f"[{'Stage A] 有 FAIL' if bad else '完成'}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())