#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-12 · R2 —— ★ DIR-10 协议的重测：新读接口下「写 vs 写关」是否翻转

判据见 ../DIRECTION.md（H6）。

【协议完全沿用 DIR-10】两阶段：① 写阶段 = teacher-forced 64-token 块 × 600（任务损失关）；
② 读侧训练 = 写冻结，优化器**只含 (Wq, Wo, gamma)**，任务损失 = 块内 next-token NLL，
Adam lr=1e-3（DIR-10 探针胜者），clip 1.0，600 步。
评测：valid 60 篇 × 512 token、逐篇 NLL + top-1、3 种子（与 DIR-10 逐位可比）。

【与 DIR-10 的差别（全部显式声明）】
  1) 写入：**与 DIR-10 完全相同**（因子化 §3.3 + 块级平均写，η=1e-6）⇒ 唯一变量是【读算子】；
  2) 读取：新增**竞争式检索算子**（λ 凸组合 + 余弦竞争），λ=0 精确等价于 DIR-10 的线性读
     ⇒ λ=0 的臂（T1/T4）必须复现 DIR-10 的数值（joint_1e-6 = 2.46297 / joint_nowrite = 2.46294），
       这一条作为**装置等价性硬检查**写进判据；
  3) 任务①（合成回忆）用的是位置级 + 全矩阵（那是另一个方向的问题），与本轮 H6 无关。
⇒ **写开 vs 写关 的比较在【同一配置内】进行 ⇒ 不受上述差别影响 ⇒ H6 是干净的对照。"""
import argparse
import json
import math
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module_retrieval import SGMNStack as SGRet   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
SEEDS = [20261028, 20261104, 7]
CHUNK, N_CHUNKS = 64, 600
N_EVAL_DOCS, EVAL_DOC_TOK = 60, 512
LR, CLIP, ETA_WRITE = 1e-3, 1.0, 1e-6
BETA = 4.0
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-12", "round": "R2", "stage": "DIR-10 协议重测", "checks": [], "arms": {}}


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


MODEL_TOK = None


def make_stack(eta_write, no_write=False, seed=20261028, beta=16.0, lam=0.0):
    tok, model = MODEL_TOK
    d = model.config.hidden_size
    # ★ 装置对齐：与 DIR-10 **完全相同的写入路径**（因子化 §3.3 + 块级平均写 §2.3/§3.1，η=1e-6），
    #   唯一差别是【读算子】（λ）。⇒ λ=0 的臂必须逐位复现 DIR-10 的数值（装置等价性硬检查）。
    st = SGRet(model, MOUNT, d, d, R, ALPHA, LAM, eta_write, GAMMA0, DEV,
               seed=seed, use_factorized=True, no_write=no_write, perpos=False)
    for m in st.mems.values():
        m.read_beta = beta          # 【D9】锐度必须显式给 16，否则 logits≡0 ⇒ 注意力均匀
        m.read_lambda = lam
    return st


def read_params(st):
    ps = []
    for m in st.mems.values():
        ps += [m.Wq, m.Wo, m.gamma]
    return ps


def write_params(st):
    ps = []
    for m in st.mems.values():
        ps += [m.U, m.V, m.W0, m.Pk, m.Pv]
        if getattr(m, "Wfull", None) is not None:
            ps.append(m.Wfull)
    return ps


def mem_state(st):
    out = {}
    for li, m in st.mems.items():
        Wf = (m.Wfull.detach().clone() if getattr(m, "Wfull", None) is not None else None)
        out[li] = (m.U.detach().clone(), m.V.detach().clone(),
                   m.W0.detach().clone(), Wf)
    return out


def reset_mem(st, snap):
    with torch.no_grad():
        for li, (U, V, W0, Wf) in snap.items():
            m = st.mems[li]
            m.U.copy_(U)
            m.V.copy_(V)
            m.W0.copy_(W0)
            if Wf is not None and getattr(m, "Wfull", None) is not None:
                m.Wfull.copy_(Wf)
            m.prev_hs = None
            m.write_calls = 0


def load_corpus():
    import requests
    cdir = pathlib.Path.home() / "sgmn-loop" / "corpus"
    cdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "valid"):
        p = cdir / f"{split}.txt"
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


def build_chunks(docs, chunk=CHUNK, n_chunks=N_CHUNKS, seed=7):
    rng = random.Random(seed)
    pool = []
    for d in docs:
        for i in range(0, max(1, d.numel() - chunk), chunk):
            pool.append(d[i:i + chunk])
    rng.shuffle(pool)
    out = []
    while len(out) < n_chunks:
        out.extend(pool)
        out = out[:n_chunks]
    return [c.unsqueeze(0) for c in out[:n_chunks]]


@torch.no_grad()
def write_phase(model, st, chunks):
    for ids in chunks:
        model(input_ids=ids.to(DEV))


def train_read(model, st, chunks, lr, check_every=50):
    for m in st.mems.values():
        m.no_write = True
    opt = torch.optim.Adam(read_params(st), lr=lr)
    losses, viol = [], 0
    for step, ids in enumerate(chunks):
        ids = ids.to(DEV)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=ids).logits.float()
        tgt = ids[:, 1:]
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1))
        loss.backward()
        if step % check_every == 0:
            for p in write_params(st):
                if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                    viol += 1
        torch.nn.utils.clip_grad_norm_(read_params(st), CLIP)
        opt.step()
        losses.append(float(loss.detach()))
    for m in st.mems.values():
        m.no_write = False
    return losses, viol


@torch.no_grad()
def eval_docs(model, st, docs):
    saved = {li: m.no_write for li, m in st.mems.items()}
    saved_prev = {li: m.prev_hs for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    nll, acc = [], []
    try:
        for ids in docs:
            x = ids.to(DEV).unsqueeze(0)
            lg = model(input_ids=x).logits.float()
            tgt = x[:, 1:]
            lg = lg[:, :-1]
            nll.append(float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                              tgt.reshape(-1), reduction="sum") / tgt.numel()))
            acc.append(float((lg.argmax(-1) == tgt).float().mean()))
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
            m.prev_hs = saved_prev[li]
    return nll, acc


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


def compare(a, b):
    D = np.concatenate([np.asarray(a[s]) - np.asarray(b[s]) for s in a])
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "t": float("nan"),
                "p_sign": 1.0, "pos": 0, "n": 0, "consistent": False}
    m = float(D.mean())
    sd = float(D.std(ddof=1))
    t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
    ps, k, n = sign_test_p(list(D))
    sm = [float(np.mean(a[s]) - np.mean(b[s])) for s in a]
    return {"mean": m, "sd": sd, "t": t, "p_sign": ps, "pos": k, "n": n,
            "seed_means": sm, "consistent": bool(k == n or k == 0)}


def main():
    global MODEL_TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_chunks = 40 if args.smoke else N_CHUNKS
    n_eval = 8 if args.smoke else N_EVAL_DOCS
    seeds = SEEDS[:2] if args.smoke else SEEDS
    res["config"] = {"n_chunks": n_chunks, "n_eval_docs": n_eval, "seeds": seeds,
                     "eta_write": ETA_WRITE, "beta": BETA, "lr": LR,
                     "write": "位置级 + 全矩阵（§3.2）"}

    print("=== DIR-12 R2：DIR-10 协议的重测（新读接口）===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    MODEL_TOK = (tok, model)
    corpus = load_corpus()
    train_docs = build_docs(tok, corpus["train"], 400, 512)
    valid_docs = build_docs(tok, corpus["valid"], n_eval, EVAL_DOC_TOK)
    chunks = build_chunks(train_docs, n_chunks=n_chunks)
    print(f"    写阶段 {n_chunks} 块 × {CHUNK} token；评测 {len(valid_docs)} 篇", flush=True)

    # ---------------- Stage A：Freeze C 在【新读接口 + 位置级写】下复检
    print("\n[Stage A] Freeze C 复检", flush=True)
    st = make_stack(ETA_WRITE, seed=20261028, beta=16.0, lam=1.0)
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    ids = chunks[0].to(DEV)
    lg = model(input_ids=ids).logits.float()
    F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    from sgmn_module_retrieval import grad_report
    rrep = grad_report(st.named_read_params())
    wrep = grad_report(st.named_write_params())
    gam = [g for n, g, i, b in rrep if n.endswith("gamma")]
    read_ok = (all((not bad) and (isn or g >= 0) for _, g, isn, bad in rrep)
               and all(g > 0 for g in gam) and max(g for _, g, _, _ in rrep) > 1e-6)
    write_ok = all(isn or g == 0.0 for _, g, isn, _ in wrep)
    base_same = all(torch.equal(q.detach(), base_before[n]) for n, q in model.named_parameters())
    st.remove()
    torch.cuda.empty_cache()
    st2 = SGRet(model, MOUNT, model.config.hidden_size, model.config.hidden_size,
                R, ALPHA, LAM, ETA_WRITE, GAMMA0, DEV, seed=20261028,
                use_factorized=True, perpos=False, drop_sg=True)
    for m in st2.mems.values():
        m.read_beta = 16.0
        m.read_lambda = 1.0
    lg2 = model(input_ids=ids).logits.float()
    F.cross_entropy(lg2[:, :-1].reshape(-1, lg2.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    wrep2 = grad_report(st2.named_write_params())
    leak = len([1 for _, g, isn, _ in wrep2 if not isn and g > 0])
    st2.remove()
    torch.cuda.empty_cache()
    for q in model.parameters():
        q.grad = None
    a1_ok = read_ok and write_ok and base_same and leak > 0
    rec("A1' Freeze C 复检（新读接口 + 位置级写；含阴性对照）", a1_ok,
        f"②读梯度 {sum(1 for _,_,i,_ in rrep if not i)}/{len(rrep)} 非零（最小 "
        f"{min(g for _,g,_,_ in rrep):.3e}）；③写入梯度精确0 {sum(1 for _,g,i,_ in wrep if i or g==0.0)}/{len(wrep)}；"
        f"④基座逐位不变 {base_same}；阴性对照泄漏 {leak}/{len(wrep2)}",
        "读梯度>1e-3 且 写入梯度==0 且 基座不变 且 泄漏>0", a1_ok)

    # ---------------- 主测量
    arms = {}
    # 第 3 列 = λ（竞争强度）：T0/T1/T4 用 λ=0（线性读），T2/T3 用 λ=1（纯竞争读）
    cfgs = [("T0_frozen", False, 0.0, False),      # 读不训练、线性读、写 on
            ("T1_joint_linear", True, 0.0, False),  # 读训练、线性读、写 on
            ("T2_joint_attn", True, 1.0, False),   # 读训练、检索读、写 on  ★
            ("T3_joint_attn_nowrite", True, 1.0, True),   # ★ 写关对照
            ("T4_joint_nowrite", True, 0.0, True)]   # 写关、线性读
    for seed in seeds:
        print(f"\n  [种子 {seed}]", flush=True)
        sa = {}
        with torch.no_grad():
            nll, acc = [], []
            for ids_ in valid_docs:
                x = ids_.to(DEV).unsqueeze(0)
                lg = model(input_ids=x).logits.float()
                tgt = x[:, 1:]
                lg = lg[:, :-1]
                nll.append(float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                                  tgt.reshape(-1), reduction="sum") / tgt.numel()))
                acc.append(float((lg.argmax(-1) == tgt).float().mean()))
        sa["base"] = {"nll": nll, "acc": acc}
        print(f"    base             NLL={np.mean(nll):.5f} acc={np.mean(acc)*100:.2f}%", flush=True)
        for name, train, lam, nowrite in cfgs:
            st = make_stack(ETA_WRITE, no_write=nowrite, seed=seed, beta=16.0, lam=lam)
            snap = mem_state(st)
            reset_mem(st, snap)
            if not nowrite:
                write_phase(model, st, chunks)
            viol = 0
            if train:
                losses, viol = train_read(model, st, chunks, LR)
            else:
                losses = []
            nll, acc = eval_docs(model, st, valid_docs)
            st.remove()
            torch.cuda.empty_cache()
            sa[name] = {"nll": nll, "acc": acc, "viol": viol,
                        "loss_last": float(np.mean(losses[-20:])) if losses else 0.0}
            print(f"    {name:22s} NLL={np.mean(nll):.5f} acc={np.mean(acc)*100:.2f}% "
                  f"viol={viol} loss末={np.mean(losses[-20:]) if losses else 0:.3f}", flush=True)
        arms[seed] = sa

    # ---------------- 分析
    print("\n[R2] 分析（逐文档配对 + 符号检验 + 跨 3 种子一致性）", flush=True)
    summ = {a: float(np.mean([np.mean(arms[s][a]["nll"]) for s in seeds]))
            for a in arms[seeds[0]]}
    res["nll_summary"] = summ
    print("  各臂 NLL：", flush=True)
    for a, v in summ.items():
        print(f"    {a:22s} {v:.5f}", flush=True)

    def cmps(a, b):
        A = {str(s): arms[s][a]["nll"] for s in seeds}
        B = {str(s): arms[s][b]["nll"] for s in seeds}
        return compare(A, B)

    # 装置等价性硬检查：λ=0 的臂必须复现 DIR-10 的数值
    d10_ref = {"T1_joint_linear": 2.46297, "T4_joint_nowrite": 2.46294}
    for arm, ref in d10_ref.items():
        got = summ[arm]
        rec(f"〔装置等价性〕{arm} 复现 DIR-10 的 {ref:.5f}（唯一差别：读算子 λ=0）",
            abs(got - ref) < 5e-3,
            f"实测 {got:.5f}（差 {got-ref:+.5f}）", "|差| < 5e-3", got)

    # H6：写 vs 写关（检索读）
    h6 = cmps("T2_joint_attn", "T3_joint_attn_nowrite")
    rec("H6 ★DIR-10 协议重测：新读接口下「写 vs 写关」的配对差翻转为正",
        h6["mean"] > 0 and h6["p_sign"] < 0.05 and h6["consistent"],
        f"T2-T3 配对差 {h6['mean']:+.5f}（sd {h6['sd']:.5f}, t={h6['t']:+.2f}, "
        f"符号检验 p={h6['p_sign']:.4g}, 正向 {h6['pos']}/{h6['n']}）跨种子 "
        + " ".join(f"{x:+.5f}" for x in h6["seed_means"]),
        "均值>0 且 p<0.05 且 跨种子同向 ⇒ DIR-10 的 H3 被推翻", h6)
    # 对照：线性读下的同比较（= DIR-10 的 H3）
    h6b = cmps("T1_joint_linear", "T4_joint_nowrite")
    rec("〔对照〕线性读（β=0）下的同比较（= DIR-10 的 H3 的复现）",
        h6b["mean"] > 0 and h6b["p_sign"] < 0.05 and h6b["consistent"],
        f"T1-T4 配对差 {h6b['mean']:+.5f}（t={h6b['t']:+.2f}, p={h6b['p_sign']:.4g}, "
        f"正向 {h6b['pos']}/{h6b['n']}）跨种子 "
        + " ".join(f"{x:+.5f}" for x in h6b["seed_means"]),
        "均值>0 且 p<0.05 且 跨种子同向", h6b)
    # 检索读 vs 线性读（都在写开下）
    h7 = cmps("T2_joint_attn", "T1_joint_linear")
    rec("〔补充〕检索读 vs 线性读（写开，测读接口本身的任务效应）",
        h7["mean"] > 0 and h7["consistent"],
        f"T2-T1 配对差 {h7['mean']:+.5f}（t={h7['t']:+.2f}, p={h7['p_sign']:.4g}）跨种子 "
        + " ".join(f"{x:+.5f}" for x in h7["seed_means"]),
        "均值>0 且 跨种子同向", h7)
    # 写 vs 写关（线性读，写关对照）
    h8 = cmps("T1_joint_linear", "base")
    rec("〔补充〕线性读联合训练 vs 纯基座（DIR-10 的 H2 复现）",
        h8["mean"] < 0 and h8["consistent"],
        f"T1-base 配对差 {h8['mean']:+.5f}（t={h8['t']:+.2f}, p={h8['p_sign']:.4g}）",
        "均值<0 且 跨种子同向", h8)

    res["comparisons"] = {"H6": h6, "H6b": h6b, "H7": h7, "H8": h8}
    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary_pass"] = {"passed": npass, "total": len(res["checks"])}
    (HERE / ("R2_smoke_results.json" if args.smoke else "R2_results.json")).write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DIR-12 R2：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
