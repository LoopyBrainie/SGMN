#!/usr/bin/env python
"""DIR-86 R1 —— 「长期基座能力保持」：注入 SGMN 后，**基座自身**在同分布任务上的 NLL
是否随在线写累积而恶化？

★ 逐字派生自 `R1_joint_read.py`（= DIR-10 = 证据行 `D-16`，md5 57b850d4…）的
  装置与评测链路：语料/切块/写协议/评测全部 **import 复用**，不复制。
★ 只换两样东西：
   (1) **被测量**：基座同分布 NLL（`D-16` 测的是 SGMN 的收益）；
   (2) **自变量**：臂 {BASE,INJ,MEM} × γ {Γ_ref,3Γ_ref} × N {0,B/4,B/2,B} × seeds。
★ 卡片：`revolve/DIR-86/DIRECTION_rev86_v15.md`（在用；判据见 v11 §1.2，一字未改）

用法：
    python -u dir86/R1_task86.py --stage-a        # Stage A 硬门（不含正式轮）
    python -u dir86/R1_task86.py --smoke          # smoke：单 seed、只 N=150（不产生裁决）
    python -u dir86/R1_task86.py --formal         # 正式轮 3×2×4×3
    python -u dir86/R1_task86.py --analyze        # 只读 out/measure.json 算判据
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import R1_joint_read as D16                      # noqa: E402  冻结装置（D-16）
from sgmn_module import SGMNStack                # noqa: E402

# ------------------------------------------------------------------ 常量（rev 1.2/1.4 实测钉死）
GAMMA_REF = D16.GAMMA0                           # Γ_ref = 0.05
GAMMA_LO, GAMMA_HI = GAMMA_REF, 3 * GAMMA_REF    # 0.05 / 0.15
B = D16.N_CHUNKS                                 # 600 **喂入块数**（rev 1.4 §2.2）
CHUNK = D16.CHUNK                                # 64
CKPTS = [0, B // 4, B // 2, B]                   # 0/150/300/600
SEEDS = list(D16.SEEDS)                          # 3 个
ARMS = ["BASE", "INJ", "MEM"]
ETA = 1e-5                       # rev 1.6 §1 实测选定：D-16 网格内满 600 块仍全有限的最大 η
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
MEASURE = OUT / "measure.json"

D16.res["dir"] = "DIR-86"
D16.res["round"] = "R1"
D16.res["card"] = "revolve/DIR-86/DIRECTION_rev86_v15.md"
D16.res["gamma_lo"] = GAMMA_LO
D16.res["gamma_hi"] = GAMMA_HI
D16.res["B_chunks"] = B
D16.res["chunk"] = CHUNK
D16.res["ckpts"] = CKPTS
rec = D16.rec


# ------------------------------------------------------------------ 装置
def param_digest(model):
    """基座全部参数的逐位摘要（A86-2 用；要求精确不变）。"""
    h = hashlib.md5()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_stack(model, arm, gamma, seed):
    """等价于 D16.make_stack 的构造体（差别仅：复用同一个 model 实例）。

    `INJ` 臂 = 装置 **既有** 开关 `no_write=True`（rev 1.3 `DEF-86-2`）：
    W 恒为 W(0)、γ·m 照常施加 ⇒「有注入、无记忆内容」。
    """
    d = model.config.hidden_size
    st = SGMNStack(model, D16.MOUNT, d, d, D16.R, D16.ALPHA, D16.LAM, ETA,
                   GAMMA_REF, D16.DEV, seed=seed, use_factorized=True,
                   drop_sg=False, no_write=(arm == "INJ"))
    g = 0.0 if arm == "BASE" else float(gamma)
    for m in st.mems.values():
        m.gamma.data.fill_(g)
        m.gamma.requires_grad_(False)
    return st


def gamma_report(st):
    return ({int(li): float(m.gamma.detach()) for li, m in st.mems.items()},
            {int(li): bool(m.gamma.requires_grad) for li, m in st.mems.items()})


def write_calls(st):
    return {int(li): int(m.write_calls) for li, m in st.mems.items()}


def steps_report(st):
    return {int(li): int(m.steps) for li, m in st.mems.items()}


def uvt_snapshot(st):
    return {int(li): (m.U.detach().clone(), m.V.detach().clone()) for li, m in st.mems.items()}


def uvt_drift(st, snap):
    worst = 0.0
    for li, m in st.mems.items():
        u0, v0 = snap[int(li)]
        worst = max(worst,
                    float((m.U.detach() - u0).abs().max()),
                    float((m.V.detach() - v0).abs().max()))
    return worst


def run_arm(model, chunks, docs, arm, gamma, seed, ckpts=None, feed_cap=None):
    """跑一个 (arm, γ, seed) 配置，返回各检查点的逐 doc NLL 与记账。"""
    ckpts = CKPTS if ckpts is None else ckpts
    cap = B if feed_cap is None else feed_cap
    st = build_stack(model, arm, gamma, seed)
    gval, greq = gamma_report(st)
    out = {"gamma": gval, "gamma_requires_grad": greq, "nll": {}, "acc": {},
           "rel": {}, "write_calls": {}, "steps": {}}
    fed = 0
    try:
        for N in ckpts:
            target = min(N, cap)
            while fed < target:
                model(input_ids=chunks[fed].to(D16.DEV))
                fed += 1
            nll, acc, rel = D16.eval_docs(model, st, docs)
            out["nll"][str(N)] = [float(x) for x in nll]
            out["acc"][str(N)] = float(np.mean(acc))
            out["rel"][str(N)] = float(np.mean(rel))
            out["write_calls"][str(N)] = write_calls(st)
            out["steps"][str(N)] = steps_report(st)
    finally:
        st.remove()
    return out


# ------------------------------------------------------------------ 数据
def prepare():
    tok, model = D16.model_and_tok()
    corpus = D16.load_corpus()
    train_docs = D16.build_docs(tok, corpus["train"], 400, 512)
    valid_docs = D16.build_docs(tok, corpus["valid"], D16.N_EVAL_DOCS, D16.EVAL_DOC_TOK)
    chunks = D16.build_chunks(train_docs, chunk=CHUNK, n_chunks=B, seed=7)
    return tok, model, train_docs, valid_docs, chunks


# ------------------------------------------------------------------ Stage A
def stage_a(model, valid_docs, chunks):
    rec("A86-10p", B >= 128, f"B={B} 喂入块 >= 128 ⇒ 不触发 NOT-ADDRESSING", value=B)
    n_req = sum(1 for p in model.parameters() if p.requires_grad)
    rec("A86-1", n_req == 0, f"基座 requires_grad 参数数 = {n_req}（须 0，Freeze C）", value=n_req)

    d0 = param_digest(model)
    # ---- 短跑：MEM(γ_lo, seed0) 20 块 + 评测
    short = run_arm(model, chunks, valid_docs, "MEM", GAMMA_LO, SEEDS[0],
                    ckpts=[0, 20], feed_cap=20)
    d1 = param_digest(model)
    rec("A86-2", d0 == d1, f"短跑后基座参数摘要逐位不变（{d0[:12]}…）", value=d0 == d1)

    # ---- A86-11″：三层 γ 同值、已冻结（float32 下容差须 ≥1e-6，rev 1.6 §2）
    for tag, cfg in (("MEM", short),):
        g = cfg["gamma"]
        vals = list(g.values())
        ok = (len(vals) == 3 and all(abs(v - GAMMA_LO) <= 1e-6 for v in vals)
              and not any(cfg["gamma_requires_grad"].values()))
        rec("A86-11pp", ok, f"{tag} 三层 γ 实测={vals}（目标 {GAMMA_LO}，float32 容差 1e-6），"
                            f"requires_grad={cfg['gamma_requires_grad']}", tol=1e-6, value=vals)

    # ---- A86-15′/15″：写事件记账 write_calls == steps == max(0,N-1)
    ok6 = ok15 = True
    det6, det15 = [], []
    for N in (0, 20):
        wc, sp = short["write_calls"][str(N)], short["steps"][str(N)]
        exp = max(0, N - 1)
        ok6 &= all(wc[li] == sp[li] for li in wc)
        ok15 &= all(wc[li] == exp for li in wc)
        det15.append({"N": N, "write_calls": wc, "expected": exp})
    rec("A86-6", ok6, "write_calls 与 steps 逐层一致", value=det6)
    rec("A86-15p", ok15, f"喂 N 块 ⇒ write_calls == max(0,N-1)（声明条件：首块无 k_l0）",
        value=det15)
    pos = CHUNK * max(0, 20 - 1)
    rec("A86-15pp", pos != 20,
        f"N 是**块**不是 token：N=20 ⇒ 消费写位置 {pos} ≠ 20（量纲差 {pos // 20}x 量级）",
        value=pos)

    # ---- A86-3 / A86-18：BASE 注入恒 0 且 NLL 与 N 无关
    base = run_arm(model, chunks, valid_docs, "BASE", GAMMA_LO, SEEDS[0], ckpts=[0, 20, 60])
    gvals = list(base["gamma"].values())
    rec("A86-3", len(gvals) == 3 and all(v == 0.0 for v in gvals),
        f"BASE 三层 γ 精确为 0 ⇒ γ·m ≡ 0（注意 read() 返回的是 m 不是 γm）", value=gvals)
    nlls = {N: base["nll"][str(N)] for N in (0, 20, 60)}
    spread = max(float(np.max(np.abs(np.array(nlls[a]) - np.array(nlls[b]))))
                 for a in nlls for b in nlls)
    rec("A86-18", spread == 0.0,
        f"BASE 的 NLL 与 N 无关（γ=0 ⇒ 前向不依赖 W）max|Δ|={spread:.3e}", tol=0.0, value=spread)

    # ---- A86-7″ / A86-16′ / A86-17：INJ 三点相同、INJ(0)==MEM(0)、U,V 逐位不变
    inj = run_arm(model, chunks, valid_docs, "INJ", GAMMA_LO, SEEDS[0], ckpts=[0, 20, 60])
    iv = [np.array(inj["nll"][str(N)]) for N in (0, 20, 60)]
    ispread = max(float(np.max(np.abs(a - b))) for a in iv for b in iv)
    rec("A86-7pp", ispread == 0.0,
        f"INJ（no_write）三点 NLL 逐位相同 max|Δ|={ispread:.3e}"
        f"（证 no_write 生效 + 评测确定性；**不**证「INJ 未误读 W」——该职责归 A86-17）",
        tol=0.0, value=ispread)
    a0 = np.array(inj["nll"]["0"])
    m0 = np.array(short["nll"]["0"])
    same = float(np.max(np.abs(a0 - m0)))
    rec("A86-16p", same == 0.0,
        f"INJ(0) 与 MEM(0) 逐位相同 max|Δ|={same:.3e}"
        f"（N=0 两臂都未写 ⇒ W 相同；双向断言：也证 MEM 在 N=0 未误写）", tol=0.0, value=same)
    tax0 = float(np.mean(a0) - np.mean(np.array(base["nll"]["0"])))
    rec("A86-16pp", 1e-4 <= abs(tax0) <= 1.0,
        f"静态注入税 |Δ_base(0)| = {abs(tax0):.6f} nats/token（量级窗 [1e-4, 1]）", value=tax0)

    # A86-17：no_write 下跑满 B 块，U,V 逐位不变
    st = build_stack(model, "INJ", GAMMA_LO, SEEDS[0])
    snap = uvt_snapshot(st)
    try:
        for i in range(B):
            model(input_ids=chunks[i].to(D16.DEV))
        drift = uvt_drift(st, snap)
        wc = write_calls(st)
    finally:
        st.remove()
    rec("A86-17", drift == 0.0 and all(v == 0 for v in wc.values()),
        f"no_write 跑满 {B} 块后 U,V max|Δ|={drift:.3e}、write_calls={wc}（均须 0）",
        tol=0.0, value={"drift": drift, "write_calls": wc})

    # ---- A86-5′：工作例——评测仪器对一手算（参照臂须为 BASE，rev 1.6 §3）
    ids = valid_docs[0].to(D16.DEV).unsqueeze(0)
    lg = model(input_ids=ids).logits.float()[:, :-1]
    tgt = ids[:, 1:]
    manual = float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                   reduction="sum") / tgt.numel())
    via = float(base["nll"]["0"][0])          # BASE(γ=0) ≡ 未挂载（A86-13 已证逐位 0）
    rec("A86-5p", abs(manual - via) < 1e-6,
        f"eval_docs 的 BASE doc0 NLL={via:.8f} vs 手算={manual:.8f}（同一公式独立重算）",
        tol=1e-6, value={"eval_docs": via, "manual": manual})

    # ---- A86-13：BASE ≡ 完全不挂载（摘钩子后同一 stack 再评）
    st = build_stack(model, "BASE", GAMMA_LO, SEEDS[0])
    try:
        mounted, _, _ = D16.eval_docs(model, st, valid_docs)
    finally:
        st.remove()
    bare, _, _ = D16.eval_docs(model, st, valid_docs)   # 钩子已摘 ⇒ 纯基座
    d13 = float(np.max(np.abs(np.array(mounted) - np.array(bare))))
    rec("A86-13", d13 == 0.0,
        f"BASE(γ=0) 与「完全不挂载」逐位相同 max|Δ|={d13:.3e}", tol=0.0, value=d13)

    # ---- A86-14：分段推进（含中途评测）vs 一次跑完（N=B）
    staged = run_arm(model, chunks, valid_docs, "MEM", GAMMA_LO, SEEDS[0], ckpts=CKPTS)
    st = build_stack(model, "MEM", GAMMA_LO, SEEDS[0])
    try:
        for i in range(B):
            model(input_ids=chunks[i].to(D16.DEV))
        os_nll, _, _ = D16.eval_docs(model, st, valid_docs)
    finally:
        st.remove()
    d14 = float(np.max(np.abs(np.array(staged["nll"][str(B)]) - np.array(os_nll))))
    rec("A86-14", d14 == 0.0,
        f"N=B 分段（含中途评测）vs 一次跑完 逐位相同 max|Δ|={d14:.3e}"
        f"（证分段评测不改数学；依赖 eval_docs 对 prev_hs 的保存/恢复）", tol=0.0, value=d14)

    # ---- A86-9：确定性 + 种子隔离
    r1 = run_arm(model, chunks, valid_docs, "INJ", GAMMA_LO, SEEDS[0], ckpts=[0])
    r2 = run_arm(model, chunks, valid_docs, "INJ", GAMMA_LO, SEEDS[0], ckpts=[0])
    d9 = float(np.max(np.abs(np.array(r1["nll"]["0"]) - np.array(r2["nll"]["0"]))))
    sa = build_stack(model, "INJ", GAMMA_LO, SEEDS[0])
    sb = build_stack(model, "INJ", GAMMA_LO, SEEDS[1])
    try:
        seed_diff = float((sb.mems[D16.MOUNT[0]].U.detach()
                           - sa.mems[D16.MOUNT[0]].U.detach()).abs().max())
    finally:
        sa.remove()
        sb.remove()
    rec("A86-9", d9 == 0.0 and seed_diff > 0.0,
        f"同 seed 两次评测逐位相同 max|Δ|={d9:.3e}；不同 seed 的 U 不同 max|Δ|={seed_diff:.3e}",
        tol=0.0, value={"same_seed": d9, "cross_seed": seed_diff})

    # ---- A86-12：非有限不算读数（含 600 块长跑，确保发散必被拓住）
    allv = []
    for cfg in (short, base, inj):
        for N in cfg["nll"]:
            allv.extend(cfg["nll"][N])
    allv.extend(staged["nll"][str(B)])
    allv.extend(os_nll)
    bad = sum(1 for v in allv if not math.isfinite(v))
    rec("A86-12", bad == 0, f"全部 {len(allv)} 个 NLL 读数有限（非有限 {bad} 个）", value=bad)

    # ---- A86-2（复检）：以上全部短跑后基座仍逐位不变
    d2 = param_digest(model)
    rec("A86-2b", d0 == d2, f"Stage A 全部短跑后基座参数仍逐位不变", value=d0 == d2)


# ------------------------------------------------------------------ 正式轮
def formal(model, valid_docs, chunks, smoke=False):
    seeds = SEEDS[:1] if smoke else SEEDS
    ckpts = [B // 4] if smoke else CKPTS
    path = OUT / ("measure_smoke.json" if smoke else "measure.json")   # smoke 不得污染正式轮
    store = {}
    if path.exists() and not smoke:
        store = json.loads(path.read_text())
    for arm in ARMS:
        gam = [GAMMA_LO] if arm == "BASE" else [GAMMA_LO, GAMMA_HI]
        for g in gam:
            for s in seeds:
                key = f"{arm}|g{g:.4f}|s{s}"
                t0 = __import__("time").time()
                out = run_arm(model, chunks, valid_docs, arm, g, s, ckpts=ckpts)
                out["arm"], out["gamma_target"], out["seed"] = arm, g, s
                out["ckpts"] = ckpts
                store[key] = out
                path.write_text(json.dumps(store, indent=1))
                print(f"[{key}] done in {__import__('time').time() - t0:.1f}s "
                      f"nll@N={ {N: round(float(np.mean(v)), 5) for N, v in out['nll'].items()} }",
                      flush=True)
    print(f"→ {path}", flush=True)
    return store


# ------------------------------------------------------------------ 判据（v11 §1.2，一字未改）
def se3(d):
    d = np.asarray(d, dtype=float)
    n = d.size
    return 3.0 * float(d.std(ddof=1)) / math.sqrt(n), float(d.mean()), n


def spearman(x, y):
    """秩相关（不依赖 scipy）。"""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return np.array(r)
    a, b = rank(list(x)), rank(list(y))
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    return float((a * b).sum() / den) if den else 0.0


def analyze():
    store = json.loads(MEASURE.read_text())
    def get(arm, g, s, N):
        return np.array(store[f"{arm}|g{g:.4f}|s{s}"]["nll"][str(N)], dtype=float)

    seeds = [s for s in SEEDS if f"MEM|g{GAMMA_LO:.4f}|s{s}" in store and str(B) in store[f"MEM|g{GAMMA_LO:.4f}|s{s}"]["nll"]]
    if not seeds:
        print("尚无完整正式轮数据"); return
    print(f"完整 seeds = {seeds}")

    def dbase(g, N):
        return np.concatenate([get("MEM", g, s, N) - get("BASE", GAMMA_LO, s, 0) for s in seeds])

    dB = dbase(GAMMA_LO, B)
    t_p86, m_p86, n = se3(dB)
    P86 = t_p86 <= 0.01
    print(f"P86  3se(Δ_base(B))={t_p86:.6f} ({'PASS' if P86 else 'FAIL'}) ≤0.01, mean={m_p86:.6f}, n={n}")

    t_a, m_a, _ = se3(dB)
    Ca = (m_a > t_a) and (m_a > 0.02)
    print(f"C86-a Δ_base(B): mean={m_a:.6f} 3se={t_a:.6f} >0.02? {'PASS' if Ca else 'FAIL'}")

    q = np.concatenate([get("MEM", GAMMA_LO, s, B) - get("MEM", GAMMA_LO, s, B // 4) for s in seeds])
    t_b, m_b, _ = se3(q)
    Cb = m_b > t_b
    print(f"C86-b Δ(B)-Δ(B/4): mean={m_b:.6f} 3se={t_b:.6f}  -> {'PASS' if Cb else 'FAIL'}")

    c = np.concatenate([get("MEM", GAMMA_HI, s, B) - get("MEM", GAMMA_LO, s, B) for s in seeds])
    t_c, m_c, _ = se3(c)
    Cc = m_c > t_c
    print(f"C86-c γ_hi-γ_lo @B: mean={m_c:.6f} 3se={t_c:.6f}  -> {'PASS' if Cc else 'FAIL'}")

    rho = spearman(CKPTS, [float(np.mean(dbase(GAMMA_LO, N))) for N in CKPTS])
    print(f"C86-d Spearman ρ(γ_lo)={rho:.3f} (诊断, ≥0.8)")

    print("\n=== 分解（用户规则：区分通用注入 vs 内容性）===")
    print(f"  通用注入项 Δ_base(0) = {float(np.mean(dbase(GAMMA_LO, 0))):+.6f} nats/token")
    print(f"  累积/内容项 Δ_base(B)-Δ_base(0) = "
          f"{float(np.mean(dbase(GAMMA_LO, B) - dbase(GAMMA_LO, 0))):+.6f} nats/token")
    for N in CKPTS:
        print(f"  Δ_base({N}) = {float(np.mean(dbase(GAMMA_LO, N))):+.6f}")

    verdict = ("SUCCEEDED" if (P86 and Ca and Cb and Cc) else
               "INFEASIBLE-POWER" if not P86 else
               "FALSIFIED" if not Cb else "PARTIAL")
    print(f"\n★ 裁决 = {verdict}   (B={B}≥128 ⇒ 非 NOT-ADDRESSING)")
    D16.res["verdict"] = verdict
    D16.res["criteria"] = {"P86": P86, "C86a": Ca, "C86b": Cb, "C86c": Cc, "C86d_rho": rho}
    (OUT / "verdict.json").write_text(json.dumps(
        {k: D16.res[k] for k in ("dir", "criteria", "verdict")}, indent=1))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze(); return
    tok, model, train_docs, valid_docs, chunks = prepare()
    print(f"语料 OK: train_docs={len(train_docs)} valid_docs={len(valid_docs)} "
          f"chunks={len(chunks)} dev={D16.DEV}", flush=True)
    if a.stage_a:
        stage_a(model, valid_docs, chunks)
        (OUT / "stageA.json").write_text(json.dumps(D16.res, indent=1, ensure_ascii=False))
        bad = [c for c in D16.res["checks"] if not c["passed"]]
        print(f"\n★ Stage A: {len(D16.res['checks']) - len(bad)}/{len(D16.res['checks'])} PASS"
              f"{'' if not bad else '  FAIL: ' + ', '.join(c['name'] for c in bad)}", flush=True)
        return
    formal(model, valid_docs, chunks, smoke=a.smoke)


if __name__ == "__main__":
    main()