#!/usr/bin/env python
"""DIR-88：η 的发散边界 —— 发散是「Hebb 项自激」还是「锚定不足」？

逐字派生自 `dir86/probe86_eta.py`。**只改**：自变量（η×λ 网格）、测点（N*/‖W−W0‖/σ1）、
以及新增 λ 与 `use_factorized`/`no_write` 两个对照轴。装置、协议、写规则**一字未改**。

主量 N*(η,λ) = 首个出现**非有限状态张量**的块号（1 基）；全 600 块有限记 None。
非有限判据**唯一**：torch.isfinite 查状态张量（U,V 或 Wfull）。不得用 NLL/梯度范数替代。
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import R1_joint_read as D16                      # noqa: E402
from sgmn_module import SGMNStack, LayerMem      # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

B = D16.N_CHUNKS
CHUNK = D16.CHUNK
SEED = D16.SEEDS[0]
GAMMA = D16.GAMMA0
ALPHA = D16.ALPHA
LAM_REF = D16.LAM
ETA_GRID = [3e-5, 1e-4, 3e-4, 1e-3]
LAM_GRID = [0.0, LAM_REF, 0.5]
SAMPLE_EVERY = 10

REC = []


def rec(cid, ok, note, **kw):
    REC.append({"id": cid, "ok": bool(ok), "note": note, **kw})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {note}", flush=True)


# ---------------------------------------------------------------- 装置
def build(model, eta, lam, gamma=GAMMA, no_write=False, use_fact=True):
    d = model.config.hidden_size
    st = SGMNStack(model, D16.MOUNT, d, d, D16.R, ALPHA, lam, eta, gamma,
                   D16.DEV, seed=SEED, use_factorized=use_fact,
                   drop_sg=False, no_write=no_write)
    for m in st.mems.values():
        m.gamma.data.fill_(float(gamma))
        m.gamma.requires_grad_(False)
    return st


def state_tensors(m):
    return [m.U, m.V] if m.use_factorized else [m.Wfull]


def W_of(m):
    return (m.W0 + m.U @ m.V.T) if m.use_factorized else m.Wfull


def drift_max(m):
    with torch.no_grad():
        return float((W_of(m) - m.W0).abs().max())


def sigma1(m, iters=15):
    """幂迭代估主奇异值（A88-10 用已知对角阵自检）。"""
    with torch.no_grad():
        W = W_of(m).detach()
        v = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
        v = v / v.norm()
        s = 0.0
        for _ in range(iters):
            u = W @ v
            nu = u.norm()
            if not torch.isfinite(nu) or nu == 0:
                return float("nan")
            u = u / nu
            w = W.T @ u
            nw = w.norm()
            if not torch.isfinite(nw) or nw == 0:
                return float("nan")
            v = w / nw
            s = float(nu)
        return s


def all_finite(st):
    with torch.no_grad():
        return all(bool(torch.isfinite(t).all()) for m in st.mems.values()
                   for t in state_tensors(m))


def digest(model):
    h = hashlib.md5()
    with torch.no_grad():
        for p in model.parameters():
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run(st, chunks, eta_lam_tag, max_blocks=B):
    """喂块，逐块查有限性；每 SAMPLE_EVERY 块记漂移/σ1。返回 N* 与曲线。"""
    curve = []
    nstar = None
    fed = 0
    try:
        for i, ck in enumerate(chunks):
            if i >= max_blocks:
                break
            model_forward(st, ck)
            fed += 1
            if not all_finite(st):
                nstar = fed
                print(f"  [{eta_lam_tag}] 发散 @ 块 {fed}", flush=True)
                break
            if fed % SAMPLE_EVERY == 0 or fed == 1:
                d = {int(li): drift_max(m) for li, m in st.mems.items()}
                curve.append({"block": fed, "drift": d,
                              "sigma1_l12": sigma1(st.mems[D16.MOUNT[1]])
                              if fed % 50 == 0 else None})
    finally:
        st.remove()
    return nstar, curve


_MODEL = None


def model_forward(st, ck):
    _MODEL(input_ids=ck.to(D16.DEV))


# ---------------------------------------------------------------- Stage A
def stage_a(model, chunks, train_docs):
    d0 = digest(model)
    rec("A88-11", B == 600 and CHUNK == 64,
        f"B={B} 块（每块 {CHUNK} token）⇒ 块 ≠ token；N* 以块计", B=B, chunk=CHUNK)

    # A88-10 σ1 自检
    class _M:  # 假 mem：只需 W_of 用到的属性
        pass
    fm = _M()
    fm.use_factorized = False
    # 线性化预测 η* ≈ 2/(α‖k‖²)；装置在 :117-119 标定 E[k^2]=5
    st0 = build(model, 1e-5, LAM_REF)
    k2 = [float((m.Pk ** 2).sum() * 5.0 * m.pk_scale ** 2) for m in st0.mems.values()]
    st0.remove()
    rec("A88-12", True, f"线性预测 η*≈2/(α‖k‖²)，按 E[k²]=5 与 d_k 估 ‖k‖²={k2[0]:.1f}"
        f" ⇒ η*≈{2.0 / k2[0]:.3e}（诊断量，不进裁决）", value=k2)
    fm.W0 = torch.zeros(3, 3, device=D16.DEV)
    fm.Wfull = torch.diag(torch.tensor([3.0, 1.0, 1.0], device=D16.DEV))
    got = sigma1(fm)
    rec("A88-10", abs(got - 3.0) <= 1e-6, f"σ1(diag(3,1,1)) 幂迭代={got:.9f}", value=got)

    # A88-2 λ 生效 + A88-5 γ
    for lam in (0.0, LAM_REF, 0.5):
        st = build(model, 1e-5, lam)
        lams = [float(m.lam) for m in st.mems.values()]
        gams = [float(m.gamma) for m in st.mems.values()]
        gr = [bool(m.gamma.requires_grad) for m in st.mems.values()]
        st.remove()
        rec(f"A88-2[λ={lam}]", all(abs(v - lam) <= 1e-6 for v in lams) and len(lams) == 3,
            f"三层 m.lam 实测={lams}（目标 {lam}，float32 容差 1e-6）", value=lams)
        rec(f"A88-5[λ={lam}]", all(abs(v - GAMMA) <= 1e-6 for v in gams)
            and not any(gr), f"三层 γ 实测={gams}，requires_grad={gr}", value=gams)

    # A88-4 非有限检测器自证（双向）
    st = build(model, 1e-5, LAM_REF)
    before = all_finite(st)
    li = D16.MOUNT[0]
    saved = st.mems[li].U.detach().clone()
    with torch.no_grad():
        st.mems[li].U[0, 0] = float("nan")
    caught = not all_finite(st)
    with torch.no_grad():
        st.mems[li].U.copy_(saved)
    restored = all_finite(st)
    st.remove()
    rec("A88-4", before and caught and restored,
        f"注入前 isfinite={before}；注入 nan 后 检出={caught}（须 True ⇒ 检出）；"
        f"复原后 isfinite={restored}（须 真/检出/真）", value=[before, caught, restored])

    # A88-8 跨轮复现：η=1e-5, λ=ref ⇒ 全有限
    st = build(model, 1e-5, LAM_REF)
    nstar_1e5, curve95 = run(st, chunks, "η=1e-5")
    rec("A88-8", nstar_1e5 is None,
        f"(λ=ref, η=1e-5) 600 块 N*={nstar_1e5}（须 None，复现 D-106 probe）",
        value=nstar_1e5,
        drift_end=max(curve95[-1]["drift"].values()) if curve95 else None)

    # A88-6 前向确定性
    st = build(model, 1e-5, LAM_REF)
    with torch.no_grad():
        lg1 = model(input_ids=chunks[0].to(D16.DEV)).logits.float().clone()
        lg2 = model(input_ids=chunks[0].to(D16.DEV)).logits.float().clone()
    st.remove()
    rec("A88-6", float((lg1 - lg2).abs().max()) == 0.0,
        f"同 seed 两次前向 logits 逐位相同 max|Δ|=0", value=0.0)

    # A88-9 工作例：手算一次 U 更新（照 :92-96 的实际顺序）
    dm, dk, dv = 6, 4, 4
    lm = LayerMem(dm, dk, dv, 2, ALPHA, LAM_REF, 1e-4, GAMMA, D16.DEV,
                  torch.Generator(device="cpu").manual_seed(999), use_factorized=True)   # 装置用 CPU 生成器（见 :110）
    hprev = torch.randn(2, 3, dm, device=D16.DEV)
    hcur = torch.randn(2, 3, dm, device=D16.DEV)
    lm.cache(hprev)
    # 手算前先快照
    U0, V0, W00 = lm.U.detach().clone(), lm.V.detach().clone(), lm.W0.detach().clone()
    Pk, Pv, pks = lm.Pk.detach().clone(), lm.Pv.detach().clone(), float(lm.pk_scale)
    lm.write(hcur)
    U_dev, V_dev = lm.U.detach().clone(), lm.V.detach().clone()
    # 手算（照实际顺序：先 U（用旧 V），再 V（用**新** U））
    pf = hprev.reshape(-1, dm)
    hf = hcur.reshape(-1, dm)
    k_h = (pf @ Pk).mean(dim=0) * pks
    v_h = (hf @ Pv).mean(dim=0)
    W_h = W00 + U0 @ V0.T
    G_h = ALPHA * torch.outer(W_h @ k_h - v_h, k_h) + LAM_REF * (W_h - W00)
    U_h = U0 - 1e-4 * (G_h @ V0)
    V_h = V0 - 1e-4 * (G_h.T @ U_h)          # ← 新 U
    eU = float((U_dev - U_h).abs().max())
    eV = float((V_dev - V_h).abs().max())
    # 若按"同步（旧 U）"算，应**不一致**——用于证明本守卫有分辨力
    V_wrong = V0 - 1e-4 * (G_h.T @ U0)
    eW = float((V_dev - V_wrong).abs().max())
    rec("A88-9", eU <= 1e-6 and eV <= 1e-6,
        f"手算(照实际顺序) max|ΔU|={eU:.2e} max|ΔV|={eV:.2e}；"
        f"若按『同步用旧U』则 max|ΔV|={eW:.2e}（须显著>1e-6 ⇒ 守卫有分辨力）",
        value={"eU": eU, "eV": eV, "e_sync_wrong": eW})

    d1 = digest(model)
    rec("A88-1", d0 == d1, f"基座摘要 d0={d0[:12]}… d1={d1[:12]}…（须逐位相同）",
        value=[d0, d1])



def a88_3(model, chunks):
    """A88-3：no_write 真跑 600 块后 U,V 逐位不变、write_calls 全 0。"""
    st = build(model, 1e-4, LAM_REF, no_write=True)
    uv0 = {li: (m.U.detach().clone(), m.V.detach().clone()) for li, m in st.mems.items()}
    for i, ck in enumerate(chunks[:B]):
        model_forward(st, ck)
    mx = max(max(float((m.U - uv0[li][0]).abs().max()),
                 float((m.V - uv0[li][1]).abs().max())) for li, m in st.mems.items())
    wc = {int(li): int(m.write_calls) for li, m in st.mems.items()}
    fin = all_finite(st)
    st.remove()
    rec("A88-3", mx == 0.0 and all(v == 0 for v in wc.values()),
        f"no_write 跑满 {B} 块后 U,V max|Δ|={mx:.1e}、write_calls={wc}（须全 0）",
        value={"max_delta": mx, "write_calls": wc, "finite": fin})



# ---------------------------------------------------------------- 主网格
def grid(model, chunks):
    res = {}
    for lam in LAM_GRID:
        for eta in ETA_GRID:
            st = build(model, eta, lam)
            tag = f"λ={lam}|η={eta:g}"
            print(f"[MEM {tag}]", flush=True)
            nstar, curve = run(st, chunks, tag)
            res[f"MEM|lam{lam}|eta{eta:g}"] = {
                "Nstar": nstar, "curve": curve,
                "drift_end": max(curve[-1]["drift"].values()) if curve else None}
            print(f"  → N*={nstar}  drift_end="
                  f"{res[f'MEM|lam{lam}|eta{eta:g}']['drift_end']}", flush=True)
            OUT.joinpath("div.json").write_text(json.dumps(
                {"dir": "DIR-88", "grid": res, "stageA": REC}, indent=1))
    # 结构性对照：全矩阵
    for eta in (1e-4, 3e-4):
        st = build(model, eta, LAM_REF, use_fact=False)
        tag = f"full|η={eta:g}"
        nstar, curve = run(st, chunks, tag)
        res[f"FULL|lam{LAM_REF}|eta{eta:g}"] = {
            "Nstar": nstar, "curve": curve,
            "drift_end": max(curve[-1]["drift"].values()) if curve else None}
        print(f"  [FULL {tag}] N*={nstar}", flush=True)
    # 分离对照：no_write
    for eta in (1e-4, 1e-3):
        st = build(model, eta, LAM_REF, no_write=True)
        nstar, curve = run(st, chunks, f"no_write|η={eta:g}")
        res[f"NOWRITE|lam{LAM_REF}|eta{eta:g}"] = {
            "Nstar": nstar, "curve": curve,
            "drift_end": max(curve[-1]["drift"].values()) if curve else None}
        print(f"  [NOWRITE η={eta:g}] N*={nstar}", flush=True)
    OUT.joinpath("div.json").write_text(json.dumps(
        {"dir": "DIR-88", "grid": res, "stageA": REC}, indent=1))
    return res


def verdict(res):
    def thr(lam):
        """最大「有限」η；全发散 ⇒ None(< 网格下界)；全有限 ⇒ 网格上界（删失）。"""
        fin = [e for e in ETA_GRID if res[f"MEM|lam{lam}|eta{e:g}"]["Nstar"] is None]
        cens_hi = len(fin) == len(ETA_GRID)
        if not fin:
            return {"value": None, "mode": "<grid_min", "censored_high": False}
        v = max(fin)
        return {"value": v, "mode": "detected" if not cens_hi else ">grid_max",
                "censored_high": cens_hi}

    t0, thi = thr(0.0), thr(0.5)
    out = {"thr_lam0": t0, "thr_lam05": thi}
    # C88-a 阳性对照
    out["C88-a"] = res["MEM|lam0.05|eta0.0001"]["Nstar"] is not None
    # C88-d 分离
    out["C88-d"] = all(res[f"NOWRITE|lam{LAM_REF}|eta{e:g}"]["Nstar"] is None
                       for e in (1e-4, 1e-3))
    # C88-b 比值
    v0, vh = t0["value"], thi["value"]
    if v0 is None or vh is None:
        out["ratio"] = None
        out["C88-b"] = "UNDEFINED" if (v0 is None) != (vh is None) else 1.0
        out["C88-b_state"] = ("INCONCLUSIVE-CENSORED"
                              if (v0 is None and vh is None) else "UNDEFINED(其一全发散)")
    else:
        r = vh / v0
        out["ratio"] = r
        cens = t0["censored_high"] or thi["censored_high"]
        if r >= 10:
            st = "FALSIFIED"
        elif r < 2:
            st = "INCONCLUSIVE-CENSORED" if cens else "CONFIRMED"
        else:
            st = "INCONCLUSIVE-BAND"
        out["C88-b_state"] = st
        out["censored"] = cens
    # C88-c 超线性：取 λ_ref 下**最大有限**η 的漂移曲线
    fin_eta = [e for e in ETA_GRID
               if res[f"MEM|lam{LAM_REF}|eta{e:g}"]["Nstar"] is None]
    key = (f"MEM|lam{LAM_REF}|eta{(max(fin_eta) if fin_eta else ETA_GRID[0]):g}")
    c = res[key]["curve"]
    sl = None
    if len(c) >= 4:
        xs = np.log([q["block"] for q in c[-11:]])
        ys = np.log([max(q["drift"].values()) for q in c[-11:]])
        m = np.isfinite(xs) & np.isfinite(ys)
        if m.sum() >= 3:
            sl = float(np.polyfit(xs[m], ys[m], 1)[0])
    out["C88-c_slope"] = sl
    out["C88-c"] = (sl is not None and sl > 1.5)
    # A88-7 单调性
    viol = []
    for lam in LAM_GRID:
        seq = [(e, res[f"MEM|lam{lam}|eta{e:g}"]["Nstar"]) for e in ETA_GRID]
        both = [(e, n) for e, n in seq if n is not None]
        for i in range(len(both) - 1):
            if both[i][1] < both[i + 1][1]:
                viol.append((lam, both[i], both[i + 1]))
    out["A88-7_violations"] = viol
    rec("A88-7", len(viol) == 0,
        f"N* 对 η 的单调性：违反 {len(viol)} 处（η↑ ⇒ N*↓）", value=viol)
    return out


def main():
    global _MODEL
    torch.manual_seed(SEED)
    tok, model = D16.model_and_tok()
    _MODEL = model
    corpus = D16.load_corpus()
    train_docs = D16.build_docs(tok, corpus["train"], 400, 512)
    chunks = D16.build_chunks(train_docs, chunk=CHUNK, n_chunks=B, seed=7)
    print(f"OK chunks={len(chunks)} B={B} CHUNK={CHUNK}", flush=True)

    stage_a(model, chunks, train_docs)
    a88_3(model, chunks)

    bad = [r for r in REC if not r["ok"]]
    OUT.joinpath("stageA.json").write_text(json.dumps(REC, indent=1))
    print(f"\n★ Stage A: {len(REC) - len(bad)}/{len(REC)} PASS", flush=True)
    if bad:
        print("FAIL 项：" + ", ".join(r["id"] for r in bad), flush=True)
        print("⇒ 阻断正式轮", flush=True)
        return

    res = grid(model, chunks)
    v = verdict(res)
    OUT.joinpath("div.json").write_text(json.dumps(
        {"dir": "DIR-88", "grid": res, "stageA": REC, "verdict": v}, indent=1))
    print("\n=== 裁决材料 ===", flush=True)
    for k in ("C88-a", "C88-d", "thr_lam0", "thr_lam05", "ratio",
              "C88-b_state", "C88-c_slope", "C88-c"):
        print(f"  {k} = {v.get(k)}", flush=True)


if __name__ == "__main__":
    main()