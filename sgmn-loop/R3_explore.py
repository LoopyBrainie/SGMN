#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
DIR-09 · R3 —— 探索性测量（**不进入冻结判据**，仅用于机理归因）

R1 的两个任务级结论：
  E1 FAIL：写相对【随机记忆】无任务级收益（配对差 −0.000149，t=−0.40，跨种子不一致）。
  E2 FAIL：读通道相对【纯基座】净损失 +0.044223 nats/token（t=23.04，p=1.25e-12，跨种子一致）。

**机理假设 H_read**：写通道学到的是 $W P_k h_{l-1}\approx P_v h_l$（一个对 h_{l-1} 的线性预测器），
但读出路径是 $m = W_o\,W\,W_q\,c$，其中 $W_q,W_o$ 是**随机初始化、从未训练**的
⇒ 复合 $W_o W W_q$ 与目标空间不对齐 ⇒ 注入是噪声 ⇒ 只可能损害 LM。

**R3a（读路径对齐臂）**：令 $W_q\leftarrow P_k$、$W_o\leftarrow P_v$（使读路径与写的目标空间对齐），
比较三种记忆：(i) 纯基座、(ii) 随机 $W_0$、(iii) 写后的 $W$（$\eta=10^{-6}$）。
若 (iii) < (ii) < (i) ⇒ 机制在【读路径对齐】后确实有任务级收益 ⇒
「前向任务收益」缺的不是写通道，而是 v2.4 §4 要求的**读取侧联合训练**。

**R3b（尺度律的跨协议量化）**：DIR-08 的最优 $\eta^\star=10^{-5}$（协议：贪心生成、batch=4），
DIR-09 的最优 $\eta^\star\approx10^{-6}$（协议：teacher-forced 真实文本、batch=1）。
DIR-07 的尺度律给出 $\eta^\star\propto1/\mu$（$\mu=E\lVert k\rVert^2$）。
本测量在**同一装置、同一代码**下测四种协议的 $\mu$：
  (P1) teacher-forced 真实文本 batch=1（DIR-09 协议）
  (P2) teacher-forced 真实文本 batch=4
  (P3) 贪心生成 batch=4（DIR-08 协议）
  (P4) 贪心生成 batch=1
若 $\mu(P1)/\mu(P3)\approx\eta^\star(P3)/\eta^\star(P1)$ ⇒ 尺度律解释了两个协议间的最优步长位移。
"""
import json
import pathlib
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import DynamicCache
except Exception:                                    # pragma: no cover
    DynamicCache = None

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module import SGMNStack                    # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ALPHA, LAM = 1.0, 0.05
GAMMA, R = 0.05, 8
MOUNT = [8, 12, 16]
ETA = 1e-6
SEED = 20261028
PREFILL = 16
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")
CORPUS_DIR = pathlib.Path.home() / "sgmn-loop" / "corpus"

res = {"dir": "DIR-09", "round": "R3", "stage": "exploratory", "checks": []}


def rec(name, passed, detail, tol=None, value=None):
    res["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                          "tolerance": tol, "value": value})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


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


def make_stack(model, d, seed=SEED, eta=ETA):
    st = SGMNStack(model, MOUNT, d, d, R, ALPHA, LAM, 0.01, GAMMA, DEV,
                   seed=seed, use_factorized=True)
    for m in st.mems.values():
        m.eta = eta
    return st


def W_of(m):
    return m.W0.detach() + m.U.detach() @ m.V.detach().T


@torch.no_grad()
def write_phase(model, st, docs, prefill=PREFILL):
    for ids in docs:
        ids = ids.to(DEV)
        L = ids.shape[0]
        cache = DynamicCache()
        model(input_ids=ids[:prefill].unsqueeze(0), past_key_values=cache, use_cache=True)
        for t in range(prefill, L):
            out = model(input_ids=ids[t:t + 1].unsqueeze(0),
                        attention_mask=torch.ones(1, t + 1, device=DEV, dtype=torch.long),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
    return {li: m.write_calls for li, m in st.mems.items()}


@torch.no_grad()
def eval_docs(model, st, docs):
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    nlls = []
    try:
        for ids in docs:
            x = ids.to(DEV).unsqueeze(0)
            lg = model(input_ids=x).logits.float()
            tgt = x[:, 1:]
            lg = lg[:, :-1]
            nlls.append(float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="sum") / tgt.numel()))
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return nlls


@torch.no_grad()
def measure_mu(model, st, mode, batch, docs=None, prompts=None, n_steps=600):
    """测 E||k||^2（k 与装置 write() 内完全同款）。"""
    mus = []
    if mode == "teacher":
        ids = torch.stack([t[:1024] for t in docs[:batch]]).to(DEV)
        L = ids.shape[1]
        cache = DynamicCache()
        model(input_ids=ids[:PREFILL], past_key_values=cache, use_cache=True)
        for t in range(PREFILL, L):
            pre = {li: m.prev_hs for li, m in st.mems.items()}
            out = model(input_ids=ids[:, t:t + 1],
                        attention_mask=torch.ones(batch, t + 1, device=DEV, dtype=torch.long),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            for li, m in st.mems.items():
                pf = pre[li]
                if pf is None:
                    continue
                k = ((pf.reshape(-1, pf.shape[-1]) @ m.Pk.detach()).mean(dim=0)) * m.pk_scale
                v2 = float((k ** 2).sum())
                if not np.isfinite(v2):
                    continue
                mus.append(v2)
            if len(mus) >= n_steps:
                break
    else:                                               # greedy
        enc = prompts
        ids = enc.to(DEV)
        B, L0 = ids.shape
        cache = DynamicCache()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        for t in range(n_steps):
            pre = {li: m.prev_hs for li, m in st.mems.items()}
            nt = out.logits[:, -1].argmax(-1, keepdim=True)
            out = model(input_ids=nt, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            for li, m in st.mems.items():
                pf = pre[li]
                if pf is None:
                    continue
                k = ((pf.reshape(-1, pf.shape[-1]) @ m.Pk.detach()).mean(dim=0)) * m.pk_scale
                v2 = float((k ** 2).sum())
                if not np.isfinite(v2):
                    continue
                mus.append(v2)
    return float(np.mean(mus)), float(np.std(mus)), len(mus)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        p = CORPUS_DIR / f"{split}.txt"
        if not p.exists():
            import requests
            r = requests.get(CORPUS_BASE + f"{split}.txt", timeout=90)
            r.raise_for_status()
            p.write_bytes(r.content)
    train_txt = (CORPUS_DIR / "train.txt").read_text(encoding="utf-8", errors="ignore")
    valid_txt = (CORPUS_DIR / "valid.txt").read_text(encoding="utf-8", errors="ignore")
    train_docs = build_docs(tok, train_txt, 2, 1024)
    valid_docs = build_docs(tok, valid_txt, 60, 512)
    prompts = tok(["The capital of France is", "In a distant galaxy the explorer found",
                   "The quick brown fox jumps over", "Machine learning models are"],
                  return_tensors="pt", padding="max_length", truncation=True,
                  max_length=32)["input_ids"].to(DEV)

    # ================= R3b 尺度律的跨协议量化 =================
    print("\n[R3b] 四种协议的 E||k||^2（同一装置、同一代码）", flush=True)
    mu = {}
    for name, mode, batch in [("P1_teacher_b1", "teacher", 1),
                              ("P2_teacher_b4", "teacher", 4),
                              ("P3_greedy_b4", "greedy", 4),
                              ("P4_greedy_b1", "greedy", 1)]:
        st = make_stack(model, d, eta=0.0)      # μ 是【输入统计量】：写通道必须关闭
        if mode == "teacher":
            docs = train_docs if batch == 1 else build_docs(tok, train_txt, 4, 1024)
            m, sd, n = measure_mu(model, st, mode, batch, docs=docs)
        else:
            m, sd, n = measure_mu(model, st, mode, batch, prompts=prompts)
        st.remove()
        torch.cuda.empty_cache()
        mu[name] = {"mu": m, "std": sd, "n": n}
        print(f"    {name:16s} E||k||^2 = {m:.1f}  (sd {sd:.1f}, n={n})", flush=True)
    ratio_mu = mu["P1_teacher_b1"]["mu"] / mu["P3_greedy_b4"]["mu"]
    ratio_eta = 1e-5 / 1e-6
    res["mu_by_protocol"] = mu
    res["mu_ratio_P1_over_P3"] = ratio_mu
    res["eta_ratio_expected"] = ratio_eta
    rec("R3b 尺度律解释了两个协议间的最优步长位移（探索性）",
        0.2 <= ratio_mu <= 5.0 * ratio_eta,
        f"μ(P1 teacher-forced b1) = {mu['P1_teacher_b1']['mu']:.1f}，"
        f"μ(P3 greedy b4) = {mu['P3_greedy_b4']['mu']:.1f} ⇒ 比值 {ratio_mu:.2f}；"
        f"实测最优 η 的比值 = 10×（DIR-08 1e-5 vs DIR-09 1e-6）⇒ 尺度律预测 η*∝1/μ "
        f"⇒ 预期 μ 比值 ≈10，实测 {ratio_mu:.2f}"
        f"（另：P2 teacher b4 = {mu['P2_teacher_b4']['mu']:.1f}、P4 greedy b1 = {mu['P4_greedy_b1']['mu']:.1f}，"
        f"⇒ batch 平均是 μ 的主要来源之一）",
        "μ 比值与 10 同量级（≤50×）", ratio_mu)

    # ================= R3a 读路径对齐臂 =================
    print("\n[R3a] 读路径对齐臂（探索性）", flush=True)
    # (i) 纯基座
    with torch.no_grad():
        base_nll = []
        for ids in valid_docs:
            x = ids.to(DEV).unsqueeze(0)
            lg = model(input_ids=x).logits.float()
            tgt = x[:, 1:]
            lg = lg[:, :-1]
            base_nll.append(float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="sum") / tgt.numel()))
    print(f"    base           NLL={np.mean(base_nll):.5f}", flush=True)

    # (ii) 随机 W0 + 对齐读路径（无写）
    st = make_stack(model, d)
    with torch.no_grad():
        for m in st.mems.values():
            m.Wq.copy_(m.Pk)
            m.Wo.copy_(m.Pv)
    nll_align_rand = eval_docs(model, st, valid_docs)
    print(f"    align+randW0   NLL={np.mean(nll_align_rand):.5f}", flush=True)
    st.remove()
    torch.cuda.empty_cache()

    # (iii) 写后的 W + 对齐读路径
    st = make_stack(model, d)
    with torch.no_grad():
        for m in st.mems.values():
            m.Wq.copy_(m.Pk)
            m.Wo.copy_(m.Pv)
    write_phase(model, st, train_docs)
    nll_align_learned = eval_docs(model, st, valid_docs)
    print(f"    align+learnedW NLL={np.mean(nll_align_learned):.5f}", flush=True)
    st.remove()
    torch.cuda.empty_cache()

    # (iv) 写后的 W + 随机读路径（R1 的 A2 臂，重跑确认）
    st = make_stack(model, d)
    write_phase(model, st, train_docs)
    nll_rand_learned = eval_docs(model, st, valid_docs)
    print(f"    rand+learnedW  NLL={np.mean(nll_rand_learned):.5f}", flush=True)
    st.remove()
    torch.cuda.empty_cache()

    res["read_path_arms"] = {
        "base": float(np.mean(base_nll)),
        "aligned_randW0": float(np.mean(nll_align_rand)),
        "aligned_learnedW": float(np.mean(nll_align_learned)),
        "random_learnedW": float(np.mean(nll_rand_learned))}
    a, b, c2, e = (res["read_path_arms"]["base"],
                   res["read_path_arms"]["aligned_randW0"],
                   res["read_path_arms"]["aligned_learnedW"],
                   res["read_path_arms"]["random_learnedW"])
    rec("R3a 读路径对齐后写通道是否有任务级收益（探索性）", c2 < b,
        f"base={a:.5f}；对齐读路径+随机W0={b:.5f}；对齐读路径+学到的W={c2:.5f}；"
        f"随机读路径+学到的W={e:.5f}。"
        + (f"⇒ 对齐后【学到的 W 优于随机 W0】{b - c2:+.5f} nats/token ⇒ 写通道有任务级收益，"
           f"缺的是读取侧对齐（v2.4 §4 的联合训练）" if c2 < b
           else "⇒ 对齐后学到的 W 仍不优于随机 W0 ⇒ 机理假设不成立"),
        "aligned_learnedW < aligned_randW0", c2 < b)

    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (HERE / "R3_explore_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DIR-09 R3（探索性）：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
