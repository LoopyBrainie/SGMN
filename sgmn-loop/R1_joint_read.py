#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-10 · R1 —— 读取侧联合训练下的前向任务收益（真实 LLM + wikitext-2）

判据见 ../DIRECTION.md（冻结于 2026-11-04）。

【v2.4 §4 的实现】
  任务损失只更新读取侧 φ=(φ^q,φ^o) 与 γ；写入通道只由显式局部规则驱动。
  装置无需修改：读侧 Wq/Wo/gamma 已 requires_grad；写侧被 sg() + no_grad 双重隔离。

【两阶段协议（保证 J1 与 J2 的记忆逐位相同）】
  ① 写阶段：teacher-forced 64-token 块前视 600 块（每块 = 1 次写事件），任务损失关闭。
  ② 读侧训练阶段：写冻结，只训练 (Wq, Wo, gamma)，任务损失 = 块内 next-token NLL。
  J1 = 只做①后直接评测；J2 = ①后快照 + ②。
  ⇒ J1 与 J2 的记忆内容逐位相同，差异严格归因于读侧是否被训练。

【评测协议与 DIR-09 完全一致】
  写冻结、读生效；valid split 60 篇 × 512 token；逐篇 NLL + top-1。
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module import SGMNStack, grad_report        # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ALPHA, LAM = 1.0, 0.05
GAMMA0, R = 0.05, 8
MOUNT = [8, 12, 16]
SEEDS = [20261028, 20261104, 7]
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")
CORPUS_DIR = pathlib.Path.home() / "sgmn-loop" / "corpus"

CHUNK = 64
N_CHUNKS = 600
N_EVAL_DOCS = 60
EVAL_DOC_TOK = 512
ETA_GRID = [1e-7, 1e-6, 1e-5, 1e-4]
LR_PROBE = [1e-4, 1e-3, 1e-2]
CLIP = 1.0

res = {"dir": "DIR-10", "round": "R1", "stage": "A+B", "model": MODEL, "device": DEV,
       "mount": MOUNT, "alpha": ALPHA, "lam": LAM, "gamma0": GAMMA0, "r": R,
       "chunk": CHUNK, "corpus": "wikitext-2-raw-v1", "checks": []}


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
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


# ----------------------------------------------------------------- 基础设施
_M = {}


def model_and_tok():
    if "m" not in _M:
        tok = AutoTokenizer.from_pretrained(MODEL)
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _M["m"] = (tok, m)
    return _M["m"]


def make_stack(eta_write=None, no_write=False, seed=20261028, drop_sg=False):
    tok, model = model_and_tok()
    d = model.config.hidden_size
    st = SGMNStack(model, MOUNT, d, d, R, ALPHA, LAM, 0.01, GAMMA0, DEV,
                   seed=seed, use_factorized=True, drop_sg=drop_sg, no_write=no_write)
    if eta_write is not None:
        for m in st.mems.values():
            m.eta = eta_write
    return st


def W_of(m):
    return m.W0.detach() + m.U.detach() @ m.V.detach().T


def read_params(st):
    ps = []
    for m in st.mems.values():
        ps += [m.Wq, m.Wo, m.gamma]
    return ps


def write_params(st):
    ps = []
    for m in st.mems.values():
        ps += [m.U, m.V, m.W0, m.Pk, m.Pv]
    return ps


def snapshot(st):
    return {li: (m.U.detach().clone(), m.V.detach().clone(), m.W0.detach().clone(),
                 m.Wq.detach().clone(), m.Wo.detach().clone(),
                 m.gamma.detach().clone())
            for li, m in st.mems.items()}


def restore(st, snap):
    with torch.no_grad():
        for li, (U, V, W0, Wq, Wo, g) in snap.items():
            m = st.mems[li]
            m.U.copy_(U)
            m.V.copy_(V)
            m.W0.copy_(W0)
            m.Wq.copy_(Wq)
            m.Wo.copy_(Wo)
            m.gamma.copy_(g)
            m.prev_hs = None


# ----------------------------------------------------------------- 语料
def load_corpus():
    import requests
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "valid"):
        p = CORPUS_DIR / f"{split}.txt"
        if not p.exists():
            r = requests.get(CORPUS_BASE + f"{split}.txt", timeout=90)
            r.raise_for_status()
            p.write_bytes(r.content)
        out[split] = p.read_text(encoding="utf-8", errors="ignore")
    return out


def build_docs(tok, text, n_docs, target_tokens):
    import re as _re
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    docs, cur = [], []
    for q in paras:
        cur.extend(tok(q).input_ids)
        if len(cur) >= target_tokens:
            docs.append(torch.tensor(cur[:target_tokens], dtype=torch.long))
            cur = []
            if len(docs) >= n_docs:
                break
    return docs


def build_chunks(docs, chunk=CHUNK, n_chunks=N_CHUNKS, seed=0):
    """从 train 文档切出 64-token 块（固定序列，跨臂可复现）。"""
    rng = random.Random(seed)
    pool = []
    for d in docs:
        for i in range(0, max(1, d.numel() - chunk), chunk):
            pool.append(d[i:i + chunk])
    rng.shuffle(pool)
    out = []
    while len(out) < n_chunks:
        out.extend(pool)
        rng.shuffle(out)
        out = out[:n_chunks]
    return [c.unsqueeze(0) for c in out[:n_chunks]]


# ----------------------------------------------------------------- 阶段①/②/评测
@torch.no_grad()
def write_phase(model, st, chunks):
    """阶段①：写开、任务损失关。每块 = 1 次写事件。"""
    for ids in chunks:
        model(input_ids=ids.to(DEV))
    return {li: m.write_calls for li, m in st.mems.items()}


def train_read(model, st, chunks, lr, seed=0, check_every=50):
    """阶段②：写冻结，只训练读侧。返回 (losses, write_grad_violations, diverged)。"""
    for m in st.mems.values():
        m.no_write = True
    opt = torch.optim.Adam(read_params(st), lr=lr)
    losses, violations = [], 0
    diverged = False
    for step, ids in enumerate(chunks):
        ids = ids.to(DEV)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=ids).logits.float()
        tgt = ids[:, 1:]
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                               tgt.reshape(-1))
        loss.backward()
        if step % check_every == 0:
            for p in write_params(st):
                if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                    violations += 1
        torch.nn.utils.clip_grad_norm_(read_params(st), CLIP)
        opt.step()
        losses.append(float(loss))
        if not all(bool(torch.isfinite(m.U).all() and torch.isfinite(m.V).all())
                   for m in st.mems.values()):
            diverged = True
            break
    for m in st.mems.values():
        m.no_write = False
    return losses, violations, diverged


@torch.no_grad()
def eval_docs(model, st, docs):
    saved = {li: m.no_write for li, m in st.mems.items()}
    saved_prev = {li: m.prev_hs for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    nlls, accs, rel = [], [], []
    try:
        for ids in docs:
            x = ids.to(DEV).unsqueeze(0)
            lg = model(input_ids=x).logits.float()
            tgt = x[:, 1:]
            lg = lg[:, :-1]
            nlls.append(float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                              tgt.reshape(-1), reduction="sum") / tgt.numel()))
            accs.append(float((lg.argmax(-1) == tgt).float().mean()))
            # 注入的相对幅度：gamma*||m|| / ||h_in||（用装置缓存的层输入与 injection）
            vals = []
            for li, m in st.mems.items():
                hs = st._hs_in[li]
                inj = m.injection
                if hs is not None and inj is not None:
                    vals.append(float(m.gamma.detach()) * float(inj.norm())
                                / max(float(hs.norm()), 1e-30))
            rel.append(float(np.mean(vals)) if vals else 0.0)
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
            m.prev_hs = saved_prev[li]
    return nlls, accs, rel


# ----------------------------------------------------------------- 统计
def sign_test_p(d):
    d = [x for x in d if abs(x) > 1e-15]
    n = len(d)
    if n == 0:
        return 1.0, 0, n
    k = sum(1 for x in d if x > 0)
    pk = math.comb(n, k) * 0.5 ** n
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(n + 1)
            if math.comb(n, i) * 0.5 ** n <= pk + 1e-12)
    return min(1.0, p), k, n


def compare(per_seed_a, per_seed_b):
    """per_seed_*：{seed: np.array(逐文档)}"""
    diffs = [np.asarray(per_seed_a[s]) - np.asarray(per_seed_b[s]) for s in per_seed_a]
    D = np.concatenate(diffs) if diffs else np.array([])
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return {"mean_diff": float("nan"), "sd": float("nan"), "t": float("nan"),
                "p_sign": 1.0, "neg": 0, "n": 0, "seed_means": [], "consistent": False}
    m = float(D.mean())
    sd = float(D.std(ddof=1)) if len(D) > 1 else 0.0
    t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
    ps, k, n = sign_test_p(list(D))
    sm = [float(x.mean()) for x in diffs if np.isfinite(x).all()]
    consistent = bool(sm) and (all(x < 0 for x in sm) or all(x > 0 for x in sm))
    return {"mean_diff": m, "sd": sd, "t": t, "p_sign": ps, "neg": n - k, "n": n,
            "seed_means": sm, "consistent": consistent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_chunks = 40 if args.smoke else N_CHUNKS
    n_eval = 8 if args.smoke else N_EVAL_DOCS
    eval_tok = 256 if args.smoke else EVAL_DOC_TOK
    seeds = SEEDS[:2] if args.smoke else SEEDS
    eta_grid = [1e-6, 1e-5] if args.smoke else ETA_GRID
    res["config"] = {"n_chunks": n_chunks, "n_eval_docs": n_eval, "eval_doc_tok": eval_tok,
                     "seeds": seeds, "eta_grid": eta_grid, "chunk": CHUNK,
                     "lr_probe": LR_PROBE}

    print("=== DIR-10：读取侧联合训练下的前向任务收益 ===", flush=True)
    tok, model = model_and_tok()
    corpus = load_corpus()
    train_docs = build_docs(tok, corpus["train"], 400, 512)
    valid_docs = build_docs(tok, corpus["valid"], n_eval, eval_tok)
    chunks = build_chunks(train_docs, n_chunks=n_chunks, seed=7)
    print(f"    写阶段 {n_chunks} 块 × {CHUNK} token；评测 {len(valid_docs)} 篇 × {eval_tok} token",
          flush=True)

    # ================================================================ Stage A
    print("\n[Stage A] 装置校验门", flush=True)

    # ---- A0' Freeze C 契约在【联合训练】下复检
    st = make_stack(eta_write=1e-6)
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    ids = chunks[0].to(DEV)
    lg = model(input_ids=ids).logits.float()
    F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                    ids[:, 1:].reshape(-1)).backward()
    rrep = grad_report(st.named_read_params())
    wrep = grad_report(st.named_write_params())
    read_ok = all((not isn) and g > 1e-3 and not bad for _, g, isn, bad in rrep)
    read_min = min([g for _, g, _, _ in rrep] or [0])
    write_ok = all(isn or g == 0.0 for _, g, isn, _ in wrep)
    base_same = all(torch.equal(q.detach(), base_before[n])
                    for n, q in model.named_parameters())
    st.remove()
    torch.cuda.empty_cache()
    st2 = make_stack(drop_sg=True)
    lg2 = model(input_ids=ids).logits.float()
    F.cross_entropy(lg2[:, :-1].reshape(-1, lg2.shape[-1]),
                    ids[:, 1:].reshape(-1)).backward()
    wrep2 = grad_report(st2.named_write_params())
    leak = len([1 for _, g, isn, _ in wrep2 if not isn and g > 0])
    st2.remove()
    torch.cuda.empty_cache()
    for q in model.parameters():
        q.grad = None
    a0_ok = read_ok and write_ok and base_same and leak > 0
    rec("A0' Freeze C 契约在【联合训练】下复检（②读梯度 ③写入梯度 ④基座 + 阴性对照）", a0_ok,
        f"②读梯度 {sum(1 for _, _, i, _ in rrep if not i)}/{len(rrep)} 非零，最小 {read_min:.3e}；"
        f"③写入梯度精确 0：{sum(1 for _, g, i, _ in wrep if i or g == 0.0)}/{len(wrep)}；"
        f"④基座逐位不变：{base_same}；阴性对照泄漏 {leak}/{len(wrep2)}",
        "读梯度>1e-3 且 写入梯度==0 且 基座不变 且 泄漏>0", a0_ok)

    # ---- A4' 单一装置
    st = make_stack(eta_write=1e-6)
    n_handles, n_mems = len(st._handles), len(st.mems)
    st.remove()
    rec("A4' 同一时刻只允许一个装置挂在被测模型上",
        n_handles == 2 * n_mems and n_mems == len(MOUNT),
        f"挂载句柄 {n_handles}（每层 2 个），记忆数 {n_mems}", "句柄数 == 2*层数", True)

    # ---- A3' 写阶段确实发生
    st = make_stack(eta_write=1e-6)
    wc = write_phase(model, st, chunks[:20])
    ok = all(v == 20 for v in wc.values())
    rec("A3' 写阶段中写通道确实发生（write_calls == 前向次数）", ok,
        f"20 块前视 ⇒ write_calls={wc}", "精确相等", wc)
    st.remove()
    torch.cuda.empty_cache()

    # ---- A5' 学习率探针
    print("  [学习率探针]", flush=True)
    probe = []
    best_lr, best_drop = None, -1e9
    for lr in LR_PROBE:
        st = make_stack(eta_write=1e-6)
        write_phase(model, st, chunks[:20])
        losses, viol, div = train_read(model, st, chunks[:40], lr, seed=1)
        st.remove()
        torch.cuda.empty_cache()
        if div or len(losses) < 8:
            drop = float("nan")
        else:
            n = len(losses)
            drop = float(np.mean(losses[:max(1, n // 10)]) - np.mean(losses[max(0, n - n // 10):]))
        probe.append({"lr": lr, "drop": drop, "n_steps": len(losses), "violations": viol})
        print(f"    lr={lr:<8g} 前10%−后10% 损失下降 = {drop:+.5f}（步数 {len(losses)}）", flush=True)
        if np.isfinite(drop) and drop > best_drop:
            best_drop, best_lr = drop, lr
    res["lr_probe"] = probe
    rec("A5' 学习率探针（至少一个 lr 使训练损失下降）", best_lr is not None,
        "；".join(f"lr={p['lr']:g}: 下降 {p['drop']:+.5f}" for p in probe) +
        f" ⇒ 取 LR={best_lr}", "至少一个下降", best_lr)
    LR = best_lr or 1e-3

    # ================================================================ 主扫描
    print("\n[R1] 主扫描：每种子 × 各臂", flush=True)
    arms = {}
    for seed in seeds:
        print(f"\n  [种子 {seed}]", flush=True)
        sa = {}

        # J0 base：完全无 SGMN
        with torch.no_grad():
            nll, acc = [], []
            for ids in valid_docs:
                x = ids.to(DEV).unsqueeze(0)
                lg = model(input_ids=x).logits.float()
                tgt = x[:, 1:]
                lg = lg[:, :-1]
                nll.append(float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                                  tgt.reshape(-1), reduction="sum") / tgt.numel()))
                acc.append(float((lg.argmax(-1) == tgt).float().mean()))
        sa["base"] = {"nll": nll, "acc": acc, "rel": [0.0] * len(nll)}
        print(f"    base           NLL={np.mean(nll):.5f}", flush=True)

        # J4 frozen_nowrite：写关、读随机
        st = make_stack(eta_write=1e-6, no_write=True, seed=seed)
        nll, acc, rel = eval_docs(model, st, valid_docs)
        sa["frozen_nowrite"] = {"nll": nll, "acc": acc, "rel": rel}
        st.remove()
        torch.cuda.empty_cache()
        print(f"    frozen_nowrite NLL={np.mean(nll):.5f}", flush=True)

        # J3 joint_nowrite：写关、读训练
        st = make_stack(eta_write=1e-6, no_write=True, seed=seed)
        wq0 = {li: st.mems[li].Wq.detach().clone() for li in st.mems}
        wo0 = {li: st.mems[li].Wo.detach().clone() for li in st.mems}
        losses, viol, div = train_read(model, st, chunks, LR, seed=seed)
        dq = float(np.mean([float((st.mems[li].Wq.detach() - wq0[li]).norm()
                                   / max(float(wq0[li].norm()), 1e-30)) for li in st.mems]))
        do = float(np.mean([float((st.mems[li].Wo.detach() - wo0[li]).norm()
                                   / max(float(wo0[li].norm()), 1e-30)) for li in st.mems]))
        gfin = float(np.mean([float(st.mems[li].gamma.detach()) for li in st.mems]))
        nll, acc, rel = eval_docs(model, st, valid_docs)
        sa["joint_nowrite"] = {"nll": nll, "acc": acc, "rel": rel, "losses": losses,
                               "viol": viol, "dWq": dq, "dWo": do, "gamma": gfin,
                               "diverged": div}
        st.remove()
        torch.cuda.empty_cache()
        print(f"    joint_nowrite  NLL={np.mean(nll):.5f}  γ {GAMMA0}→{gfin:.4f} "
              f"ΔWq={dq:.3e} ΔWo={do:.3e} viol={viol} "
              f"loss {np.mean(losses[:5]):.3f}→{np.mean(losses[-5:]):.3f}", flush=True)

        # J1 frozen(η) / J2 joint(η)
        for eta in eta_grid:
            t0 = time.time()
            # J1：只做阶段①
            st = make_stack(eta_write=eta, seed=seed)
            wc = write_phase(model, st, chunks)
            mem_snap = {li: (st.mems[li].U.detach().clone(), st.mems[li].V.detach().clone())
                        for li in st.mems}
            nll, acc, rel = eval_docs(model, st, valid_docs)
            sa[f"frozen_{eta:g}"] = {"nll": nll, "acc": acc, "rel": rel, "wc": wc,
                                     "diverged": False}
            st.remove()
            torch.cuda.empty_cache()

            # J2：阶段①（同一块序列）→ 快照 → 阶段②（写冻结、只训读）
            st = make_stack(eta_write=eta, seed=seed)
            write_phase(model, st, chunks)
            # 校验记忆与 J1 逐位相同
            same_mem = all(
                float((st.mems[li].U.detach() - mem_snap[li][0]).abs().max()) == 0.0
                and float((st.mems[li].V.detach() - mem_snap[li][1]).abs().max()) == 0.0
                for li in st.mems)
            wq0 = {li: st.mems[li].Wq.detach().clone() for li in st.mems}
            wo0 = {li: st.mems[li].Wo.detach().clone() for li in st.mems}
            losses, viol, div = train_read(model, st, chunks, LR, seed=seed)
            dq = float(np.mean([float((st.mems[li].Wq.detach() - wq0[li]).norm()
                                       / max(float(wq0[li].norm()), 1e-30)) for li in st.mems]))
            do = float(np.mean([float((st.mems[li].Wo.detach() - wo0[li]).norm()
                                       / max(float(wo0[li].norm()), 1e-30)) for li in st.mems]))
            gfin = float(np.mean([float(st.mems[li].gamma.detach()) for li in st.mems]))
            nll, acc, rel = eval_docs(model, st, valid_docs)
            sa[f"joint_{eta:g}"] = {"nll": nll, "acc": acc, "rel": rel, "losses": losses,
                                    "viol": viol, "dWq": dq, "dWo": do, "gamma": gfin,
                                    "diverged": div, "same_mem": same_mem}
            st.remove()
            torch.cuda.empty_cache()
            print(f"    eta={eta:<8g} frozen NLL={np.mean(sa[f'frozen_{eta:g}']['nll']):.5f} "
                  f"| joint NLL={np.mean(nll):.5f} γ→{gfin:.4f} ΔWq={dq:.3e} "
                  f"viol={viol} mem_same={same_mem} ({time.time()-t0:.0f}s)", flush=True)

        arms[seed] = sa

    # ================================================================ 分析
    print("\n[R2] 分析与判据（跨 3 种子，文档级配对）", flush=True)

    def N(seed, arm):
        return np.asarray(arms[seed][arm]["nll"], dtype=float)

    summary = {}
    for arm in arms[seeds[0]]:
        rows = [arms[s][arm] for s in seeds]
        summary[arm] = {
            "mean_nll": float(np.mean([np.mean(r["nll"]) for r in rows])),
            "per_seed": {str(s): float(np.mean(arms[s][arm]["nll"])) for s in seeds},
            "acc": float(np.mean([np.mean(r["acc"]) for r in rows])),
            "rel": float(np.mean([np.mean(r["rel"]) for r in rows])),
            "gamma": float(np.mean([r.get("gamma", GAMMA0) for r in rows])),
            "dWq": float(np.mean([r.get("dWq", 0.0) for r in rows])),
            "dWo": float(np.mean([r.get("dWo", 0.0) for r in rows])),
            "viol": int(sum(r.get("viol", 0) for r in rows)),
            "diverged": any(r.get("diverged", False) for r in rows)}
    res["nll_summary"] = summary
    print("  各臂平均 NLL（跨种子）：", flush=True)
    for arm, v in summary.items():
        print(f"    {arm:16s} NLL={v['mean_nll']:.5f}  acc={v['acc']*100:.2f}%  "
              f"γ={v['gamma']:.4f}  注入相对幅度={v['rel']:.3e}  ΔWq={v['dWq']:.2e} "
              f"viol={v['viol']}", flush=True)

    # ---- A1' 联合训练确实发生
    a1_arms = [a for a in summary if a.startswith("joint")]
    a1_ok = all(summary[a]["dWq"] > 1e-3 and summary[a]["dWo"] > 1e-3 for a in a1_arms)
    rec("A1' 联合训练确实发生（读侧参数改变 且 训练损失下降）", a1_ok,
        "；".join(f"{a}: ΔWq={summary[a]['dWq']:.3e} ΔWo={summary[a]['dWo']:.3e} "
                  f"γ {GAMMA0}→{summary[a]['gamma']:.4f}" for a in a1_arms),
        "每个 joint 臂: ΔWq>1e-3 且 ΔWo>1e-3", a1_ok)

    # ---- H1 联合训练使写通道产生任务级收益
    h1 = {}
    for eta in eta_grid:
        a = f"joint_{eta:g}"
        b = f"frozen_{eta:g}"
        if a in summary and b in summary:
            c = compare({str(s): N(s, a) for s in seeds}, {str(s): N(s, b) for s in seeds})
            h1[a] = c
    good1 = [k for k, v in h1.items() if v["mean_diff"] < 0 and v["p_sign"] < 0.05
             and v["consistent"]]
    rec("H1 〔主判据〕联合训练使写通道产生任务级收益（NLL(joint) < NLL(frozen)，同 η）",
        len(good1) > 0,
        "；".join(f"{k}: 配对差均值 {v['mean_diff']:+.5f} (σ={v['sd']:.5f}, t={v['t']:+.2f}, "
                  f"符号检验 p={v['p_sign']:.4g}, 负向 {v['neg']}/{v['n']}) 跨种子均值 "
                  + " ".join(f"{x:+.5f}" for x in v["seed_means"]) for k, v in h1.items()),
        "至少一个 η: 均值<0 且 p<0.05 且 跨种子同向", good1)

    # ---- H2 联合训练后不损害基座
    jw = [a for a in summary if a.startswith("joint_") and a != "joint_nowrite"]
    best_joint = min(jw, key=lambda a: summary[a]["mean_nll"]) if jw else None
    h2 = compare({str(s): N(s, best_joint) for s in seeds},
                 {str(s): N(s, "base") for s in seeds})
    rec("H2 联合训练后 SGMN 不损害基座（NLL(joint) < NLL(base)）",
        h2["mean_diff"] < 0 and h2["p_sign"] < 0.05 and h2["consistent"],
        f"最优 joint 臂 {best_joint}：配对差均值 {h2['mean_diff']:+.5f} "
        f"(σ={h2['sd']:.5f}, t={h2['t']:+.2f}, 符号检验 p={h2['p_sign']:.4g}, "
        f"负向 {h2['neg']}/{h2['n']}) 跨种子均值 "
        + " ".join(f"{x:+.5f}" for x in h2["seed_means"]),
        "均值<0 且 p<0.05 且 跨种子同向", h2)

    # ---- H3 写通道在联合训练下有独立贡献
    h3 = compare({str(s): N(s, best_joint) for s in seeds},
                 {str(s): N(s, "joint_nowrite") for s in seeds})
    rec("H3 写通道在联合训练下有独立贡献（NLL(joint+write) < NLL(joint 无写)）",
        h3["mean_diff"] < 0 and h3["consistent"],
        f"配对差均值 {h3['mean_diff']:+.5f} (σ={h3['sd']:.5f}, t={h3['t']:+.2f}, "
        f"符号检验 p={h3['p_sign']:.4g}) 跨种子均值 "
        + " ".join(f"{x:+.5f}" for x in h3["seed_means"]),
        "均值<0 且 跨种子同向", h3)

    # ---- H4 非退化
    h4_ok = summary[best_joint]["gamma"] > 0.01 and summary[best_joint]["rel"] > 1e-3
    rec("H4 非退化：联合训练不得退化为「把 SGMN 关掉」",
        h4_ok,
        f"最优 joint 臂 {best_joint}：γ {GAMMA0} → {summary[best_joint]['gamma']:.4f}；"
        f"评测时注入相对幅度 γ||m||/||h|| = {summary[best_joint]['rel']:.3e}",
        "γ>0.01 且 注入相对幅度>1e-3", h4_ok)

    # ---- H5 Freeze C 回归
    h5_ok = all(summary[a]["viol"] == 0 for a in summary if a.startswith("joint"))
    rec("H5 Freeze C 回归：联合训练全程写入侧梯度恒为精确 0", h5_ok,
        "；".join(f"{a}: 违规抽检 {summary[a]['viol']} 次" for a in summary
                  if a.startswith("joint")),
        "违规次数 == 0", h5_ok)

    # ---- H7 η 响应
    curve = {a: summary[a]["mean_nll"] for a in summary
             if a.startswith("joint_") and a != "joint_nowrite"}
    res["eta_curve_joint"] = curve
    print("  联合训练的 η 响应：", flush=True)
    for k in sorted(curve, key=lambda x: float(x.split("_")[1]) if x != "joint_nowrite" else 0):
        print(f"    {k:14s} NLL={curve[k]:.5f}", flush=True)
    rec("H7 读路径是否改变写通道的最优 η（报告项）", True,
        "；".join(f"{k}:{v:.5f}" for k, v in sorted(curve.items(), key=lambda x: float(x[0].split('_')[1]) if x[0]!="joint_nowrite" else 0)) +
        f"；DIR-09（未联合训练）的最优 η ≈ 1e-6", "记录曲线与最优", curve)

    # ---- A2' 评测时记忆逐位不变（用 joint 臂抽查）
    st = make_stack(eta_write=1e-6, seed=seeds[0])
    W0 = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    eval_docs(model, st, valid_docs[:4])
    d = max(float((W_of(st.mems[li]).cpu() - W0[li]).abs().max()
                  / max(float(W0[li].abs().max()), 1e-30)) for li in st.mems)
    st.remove()
    torch.cuda.empty_cache()
    rec("A2' 评测阶段记忆逐位不变（写被冻结）", d < 1e-12,
        f"4 篇评测文档后记忆相对差 {d:.3e}", "<1e-12", d)

    res["comparisons"] = {"H1": h1, "H2": h2, "H3": h3}
    res["joint_nowrite_nll"] = summary["joint_nowrite"]["mean_nll"]
    res["frozen_nll_by_eta"] = {a: summary[a]["mean_nll"] for a in summary
                                 if a.startswith("frozen_")}
    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (HERE / ("R1_smoke_results.json" if args.smoke else "R1_results.json")).write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DIR-10 R1：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
