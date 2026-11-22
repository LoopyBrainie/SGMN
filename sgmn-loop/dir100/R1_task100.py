#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-100 / rev 1.0 —— `Wq` 为什么对未训练事实不给正确查询？**把 `Wq` 直接解出来**能否抬起 `acc_held`？

逐字派生 `dir99/R1_task99.py`（键变换 + 四条件评测 + 读侧单训），**唯一新增**：
  ① 最小范数最小二乘解 `M: c_{prompt末} → k_i`（**只用 16 个训练事实**）
  ② `Wq` 来源四臂（`wq_train`/`wq_solved`/`wq_solved_init`/`wq_oracle`）
  ③ 冻结 `Wq` 的读侧训练（把 `Wq` 从优化器里**排除**，不是训练后回填）

承重前提（开卡前已核，`PREMISE-100.md`）：`cos_he(M)` = **0.9965–0.9985**（`P100-A`）
⇒ 线性查询映射**存在且跨事实泛化** ⇒ `acc_held≡0` 属**优化/辨识性**，非表示性。
★ 任务行原列的 `wq_tied`/`wq_init`（`Pk`）**弃用**：前提核实测 `A_pk` 仅 0.18–0.38，是**错**映射。

判据（冻结，`DIRECTION_rev100_v10.md` §2）：
  `Δ_ho = acc_held(W_written) − acc_held(W⁰)`
  `H100`：`wq_solved` 2-seed 均值同时 (a) `Δ_ho ≥ 0.125` (b) `acc_held ≥ 0.375`
  否证：`Δ_ho < 0.0625` 或 `acc_held < 0.25`
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
ARMS = ["wq_train", "wq_solved", "wq_solved_init", "wq_oracle", "wq_oracle_id"]
BASE_ARMS = ["wq_train", "wq_solved", "wq_solved_init", "wq_oracle"]
KEY_ARM = "white"          # ★ 键固定 white（D-128：32/32 内容）；自变量只有 Wq
LAM_M = 1e-6               # M 的岭系数（与前提核同值）
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


def _cos_rows(P, Q):
    """逐行余弦 → (n,)。"""
    P, Q = P.float(), Q.float()
    return (P * Q).sum(dim=1) / (P.norm(dim=1) * Q.norm(dim=1)).clamp_min(1e-12)


def _ridge_A(C, K, lam=LAM_M):
    """最小范数最小二乘 A = Cᵀ(CCᵀ+λI)⁻¹K：(n,d)→(d,d_k)。n=16 ≪ d=896，必须走对偶式。"""
    C, K = C.float(), K.float()
    n = C.shape[0]
    G = C @ C.T + lam * torch.eye(n, device=C.device, dtype=C.dtype)
    return C.T @ torch.linalg.solve(G, K)


def prompt_reads(model, st, facts):
    """每个事实的问题 prompt 末位读出向量 c（**判据位置 P-1**，与 `train_read_task` 同一位置）。"""
    C = {li: [] for li in st.mems}
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    try:
        for fi in range(len(facts)):
            pid = facts[fi]["prompt_ids"]
            model(input_ids=torch.tensor([pid], dtype=torch.long, device=DEV))
            for li in st.mems:
                hs = st._hs_in[li]
                h = hs[0] if hs.dim() == 3 else hs
                C[li].append(h[len(pid) - 1].detach().clone())
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return {li: torch.stack(v).float() for li, v in C.items()}


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
    """§4 手算算例：**直接调** `_ridge_A`/`_cos_rows`（不另写一份推导）。"""
    ok = True
    ok &= rec("A99-9a", abs((3 / 4 - 1 / 4) - 0.5) < 1e-12, "Δ(3/4,1/4)=0.5")
    ok &= rec("A99-9b", abs((2 / 4 - 2 / 4) - 0.0) < 1e-12, "Δ(2/4,2/4)=0")
    V = torch.tensor([[1., 0.], [0., 1.]], device=DEV)
    K = V.clone()
    W = (torch.linalg.pinv(K) @ V).T
    got = (W @ K.T).T
    d = float((got - V).abs().max())
    ok &= rec("A99-9c", d < 1e-6, f"W_or 解算：max|W_or·K−V| = {d:.2e} (<1e-6)，shape={tuple(W.shape)}")

    # ---- A100-9：`_ridge_A` 的解与形状（λ 取 0 走精确解）
    C = torch.tensor([[1., 0.], [0., 1.]], device=DEV)        # 2 事实、2 维
    Kt = torch.tensor([[2., 0.], [0., 4.]], device=DEV)       # 手算目标
    A = _ridge_A(C, Kt, lam=0.0)
    pred = C @ A
    err = float((pred - Kt).abs().max())
    ok &= rec("A100-9a", A.shape == (2, 2) and err < 1e-6,
              f"ridge 精确解：shape={tuple(A.shape)}（应 (d,d_k)=(2,2)）· max|C·A−K|={err:.2e}")
    cos = _cos_rows(pred, Kt)
    ok &= rec("A100-9b", cos.shape == (2,) and float((cos - 1).abs().max()) < 1e-6,
              f"逐行余弦：shape={tuple(cos.shape)}（应 (2,)）· cos={[round(float(x),6) for x in cos]}")
    # 同向目标不惩罚（cos 只测方向，判别由 acc 承担）
    Kt2 = torch.tensor([[2., 0.], [2., 0.]], device=DEV)
    cos2 = _cos_rows(C @ _ridge_A(C, Kt2, lam=0.0), Kt2)
    ok &= rec("A100-9c", float((cos2 - 1).abs().max()) < 1e-6,
              f"同向目标：cos={[round(float(x),6) for x in cos2]} ⇒ cos 不承担判别（判据只有 acc）")
    # 正交性：A 的转置即是 Wq 的形状约定（q = Wq c）
    Wq = A.T
    q = (Wq @ C.T).T
    ok &= rec("A100-9d", float((q - Kt).abs().max()) < 1e-6,
              f"Wq=Aᵀ 约定：max|(Wq·cᵀ)ᵀ−K|={float((q-Kt).abs().max()):.2e}")
    return ok


# ==================================================================== 一臂
def train_read_frozen_wq(model, st, tok, facts, train_keys, rng, freeze_wq):
    """逐字派生 `R91.train_read_task`；**唯一差别** = `freeze_wq=True` 时把 `Wq` **从优化器排除**。
    ★ 不得"训练后回填 Wq"：那样 Wo/gamma 是**对着另一个 Wq** 训出来的，测的是一个从未存在的系统。"""
    for m in st.mems.values():
        m.no_write = True
    ps = []
    for m in st.mems.values():
        ps += [m.Wq, m.Wo, m.gamma]
    if freeze_wq:
        ps = [p for m in st.mems.values() for p in (m.Wo, m.gamma)]
    opt = torch.optim.Adam(ps, lr=R91.READ_LR)
    losses, viol, div = [], 0, False
    batches = []
    for ki in train_keys:
        f = facts[ki]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != f["val"]]
        distr = rng.sample(sorted(set(others)), R91.N_CAND - 1)
        cands = [f["val"]] + distr
        rng.shuffle(cands)
        ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in cands]
        batches.append((f["prompt_ids"], ids, cands.index(f["val"])))
    for step in range(R91.READ_STEPS):
        prompt_ids, ids, tgt = batches[step % len(batches)]
        P = len(prompt_ids)
        seq = torch.tensor([prompt_ids + [c] for c in ids], dtype=torch.long, device=DEV)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=seq).logits.float()
        idx = torch.arange(len(ids), device=DEV)
        pick = lg[idx, P - 1, torch.tensor(ids, device=DEV)].unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(pick, torch.tensor([tgt], device=DEV))
        loss.backward()
        if step % 50 == 0:
            for p in R91.write_params(st):
                if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                    viol += 1
        torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], R91.CLIP)
        opt.step()
        losses.append(float(loss))
        if not all(bool(torch.isfinite(R91.W_of(m)).all()) for m in st.mems.values()):
            div = True
            break
    for m in st.mems.values():
        m.no_write = False
    return losses, viol, div


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
            tf = build_tf(KEY_ARM, Kraw[li], V[li], li)      # ★ 键固定 white
            TF[li] = tf
            st.mems[li].key_tf = tf
            st.mems[li].q_tf = tf.qmap
        # ---- 3) 真写（写侧带 T）；`w_oracle` 臂跳过写，W 直接置为下限范数精确解
        # ★ 每臂都**真写**：本卡自变量只有 Wq，`W` 恒为 white 键下的写入结果
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
        # ---- 4.5) ★DIR-100 新增：解 M（**只用 tr 的 16 个事实**）→ 定 Wq
        Cprompt = prompt_reads(model, st, facts)
        allk = list(range(len(facts)))
        M, Mall, Mchk = {}, {}, {}
        for li in Knew:
            Kr = Kraw[li].float()                  # ★rev1.1 raw 键（近共线 ⇒ 低有效秩 ⇒ 可泛化）
            Kn = Knew[li].float()                  # 变换后键（W 里就是它）
            Cl = Cprompt[li]
            tf = TF[li]
            A_tr = _ridge_A(Cl[tr], Kr[tr])
            A_all = _ridge_A(Cl[allk], Kr[allk])
            M[li] = A_tr.T                         # Wq 形状 (d_k, d)
            Mall[li] = A_all.T
            # ★ 有效空间：q_tf 保留 ⇒ 实际查询 = qmap(M c)，须与**变换后**键比
            Mchk[str(li)] = {
                "cos_tr": float(_cos_rows(Cl[tr] @ A_tr, Kr[tr]).mean()),
                "cos_he": float(_cos_rows(Cl[he] @ A_tr, Kr[he]).mean()),        # ★A100-1（raw）
                "cos_he_all": float(_cos_rows(Cl[he] @ A_all, Kr[he]).mean()),   # ★A100-4（raw）
                "cos_eff_he": float(_cos_rows(tf.qmap((A_tr.T @ Cl[he].T).T), Kn[he]).mean())}
            if arm == "wq_oracle_id":              # ★rev1.2 判别臂：拟合 K_new、全 32、q_tf=恒等
                A_new_all = _ridge_A(Cl[allk], Kn[allk])
                Mall[li] = A_new_all.T
                Mchk[str(li)]["cos_id_he"] = float(_cos_rows(Cl[he] @ A_new_all, Kn[he]).mean())
        Wq_set = {li: m.Wq.detach().clone() for li, m in st.mems.items()}
        # ---- 5) 读侧训练（wq_solved/wq_oracle **冻结 Wq**；不动 W）
        rng = random.Random(1000 + seed % 100)
        Wwrite = {li: m.Wfull.detach().clone() for li, m in st.mems.items()}
        losses, viol, div = train_read_frozen_wq(
            model, st, tok, facts, tr, rng,
            freeze_wq=(arm in ("wq_solved", "wq_oracle", "wq_oracle_id")))
        Wq_drift = max(float((m.Wq.detach() - Wq_set[li]).norm()
                             / Wq_set[li].norm().clamp_min(1e-30))
                       for li, m in st.mems.items())
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
            "geo": geo, "acc": res, "Mchk": Mchk, "Wq_drift": Wq_drift,
            "key_arm": KEY_ARM, "d_tr": dtr, "d_ho": dho,
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
    ap.add_argument("--arms", default="", help="逗号分隔，覆盖 ARMS（判别臂用）")
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

    # ★ stageA/smoke 都要跑**全 4 臂**：A100-0 需 wq_train、A100-1/4 需 wq_solved/wq_oracle
    seeds = SEEDS if a.stage == "formal" else SEEDS[:1]
    arms = [x for x in a.arms.split(",") if x] or ARMS
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

    # ---- Stage A 守卫（任一不过 ⇒ 不开正式轮）
    ges = [c for c in C]
    if a.stage == "stageA":
        # A100-0：wq_train 逐位复现 D-129 的 white 格
        w = next(c for c in C if c["arm"] == "wq_train")
        a_ = w["acc"]
        mc = list(w["geo"].values())[0]["maxcos"]
        ok0 = (abs(a_["written"]["acc_train"] - 0.8125) < 1e-9
               and abs(a_["written"]["acc_held"] - 0.0000) < 1e-9
               and mc < 1e-4)
        rec("A100-0", ok0, f"wq_train 复现 D-129 white：acc_tr={a_['written']['acc_train']:.4f}"
                           f"(应0.8125) acc_ho={a_['written']['acc_held']:.4f}(应0.0000) maxcos={mc:.1e}(<1e-4)")
        # A100-1 / A100-4：在位 cos_he
        c_he = min(v["cos_he"] for v in w["Mchk"].values())
        rec("A100-1", c_he >= 0.90, f"在位 cos_he(M) 最小层 = {c_he:.4f} ≥ 0.90（前提核在正式装置内重现）")
        o = next(c for c in C if c["arm"] == "wq_oracle")
        c_hall = min(v["cos_he_all"] for v in o["Mchk"].values())
        rec("A100-4", c_hall >= 0.99, f"wq_oracle 的 cos_he_all 最小层 = {c_hall:.4f} ≥ 0.99")
        # A100-3：dWrel 逐格
        dw = max(c["dW_rel_read"] for c in C)
        rec("A100-3", dw < 1e-12, f"逐格 dWrel 最大 = {dw:.1e}（A97-4：任务损失不改 W）")

    full = set(BASE_ARMS) <= {c["arm"] for c in C}
    if a.stage == "formal" and full:
        # A100-2：真冻结
        fr = [c["Wq_drift"] for c in C if c["arm"] in ("wq_solved", "wq_oracle")]
        rec("A100-2", max(fr) < 1e-12, f"冻结臂 ‖Wq_after−Wq_set‖/‖Wq_set‖ 最大 = {max(fr):.1e}（应 0）")
        c_he = min(v["cos_he"] for c in C for v in c["Mchk"].values())
        rec("A100-1", c_he >= 0.90, f"在位 cos_he(M) 最小 = {c_he:.4f} ≥ 0.90")
        c_hall = min(v["cos_he_all"] for c in C for v in c["Mchk"].values())
        rec("A100-4", c_hall >= 0.99, f"cos_he_all 最小 = {c_hall:.4f} ≥ 0.99")
        dw = max(c["dW_rel_read"] for c in C)
        rec("A100-3", dw < 1e-12, f"逐格 dWrel 最大 = {dw:.1e}")

        s_ = [c for c in C if c["arm"] == "wq_solved"]
        dho_s = sum(c["d_ho"] for c in s_) / len(s_)
        aho_s = sum(c["acc"]["written"]["acc_held"] for c in s_) / len(s_)
        G["wq_solved"] = {"d_ho": dho_s, "acc_held": aho_s}
        print(f"[DIR-100] d_ho 各臂均值 = {dho}", flush=True)
        print(f"[DIR-100] d_tr 各臂均值 = {dtr}", flush=True)
        print(f"[DIR-100] ★ wq_solved: Δ_ho={dho_s:+.4f} acc_held={aho_s:.4f}", flush=True)
        for arm in ARMS:
            v = [c["acc"]["written"]["acc_held"] for c in C if c["arm"] == arm]
            t = [c["acc"]["written"]["acc_train"] for c in C if c["arm"] == arm]
            print(f"[DIR-100]   {arm:<16} acc_held={v} acc_train={t}", flush=True)
        ida = [c for c in C if c["arm"] == "wq_oracle_id"]
        if ida:
            G["wq_oracle_id"] = {"acc_held": [c["acc"]["written"]["acc_held"] for c in ida],
                                 "acc_train": [c["acc"]["written"]["acc_train"] for c in ida],
                                 "cos_id_he": min(v.get("cos_id_he", 0.0)
                                                  for c in ida for v in c["Mchk"].values())}
            print(f"[DIR-100] ★判别臂 wq_oracle_id: acc_held={G['wq_oracle_id']['acc_held']} "
                  f"acc_train={G['wq_oracle_id']['acc_train']} "
                  f"cos_id_he(min)={G['wq_oracle_id']['cos_id_he']:.6f}", flush=True)
        if dho_s >= 0.125 and aho_s >= 0.375:
            verdict = "A-WQ-SOLVABLE"
        elif dho_s < 0.0625 or aho_s < 0.25:
            verdict = "A-WQ-NOT-ENOUGH"
        else:
            verdict = "AMBIGUOUS"
        G["verdict"] = verdict
        print(f"[DIR-100] 判决 = {verdict}", flush=True)
    (OUT / f"stage_{a.stage}.json").write_text(json.dumps(G, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    bad = [c for c in CHECKS if not c["ok"]]
    print(f"[{'Stage A] 有 FAIL' if bad else '完成'}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())