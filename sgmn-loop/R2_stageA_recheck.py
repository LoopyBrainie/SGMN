#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-09 · R2 —— Stage A 重检（针对 R1 记录的两项 FAIL）

R1 的两项 FAIL 均为【脚本侧缺陷】，不是装置缺陷，此处用严格单一装置纪律重检：

  * A1' FAIL：write_calls = 2017 而"在线步数"记为 2016。
    根因：我的步数记账漏了【每篇文档的预填前视】—— 第 2 篇文档的预填时 prev_hs 非空，
    因此它**也**产生一次写。正确关系是：write_calls == 前向调用总数 − 1
    （只有整个写阶段的第一次预填没有 k_{l,0}，故不能写）。
  * A3' FAIL：两次写阶段结果不一致（W 差 7.175e-02）。
    根因：R1 的 A3' 期间**同时挂着 3 个 SGMNStack**（A1' 的栈未卸载就新建了 st_b/st_c），
    三个记忆的注入叠加 ⇒ 隐状态被污染 ⇒ 两次写阶段不可能逐位相同。
    这是我自己违反了机制级纪律 #6（同一时刻只允许一个装置挂在被测模型上），
    纪律 #6 本应在 Stage A 当场抓住它 —— 检查确实抓到了，只是抓的是"结果"不是"原因"。
    本脚本严格「建栈 → 测 → remove() → 再建栈」，逐次串行。

判据（沿用 R1 的阈值，未改动）：
  A1'' write_calls == 前向调用总数 − 1
  A3'' 两次写阶段后的 W 逐位相同（max|Δ| == 0.0）
"""
import json
import pathlib
import sys
import time

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
ETA = 1e-5
SEED = 20261028
PREFILL = 16
N_WRITE_DOCS = 2
WRITE_DOC_TOK = 1024
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")
CORPUS_DIR = pathlib.Path.home() / "sgmn-loop" / "corpus"

res = {"dir": "DIR-09", "round": "R2", "stage": "A-recheck", "checks": []}


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


def make_stack(model, d, seed=SEED):
    st = SGMNStack(model, MOUNT, d, d, R, ALPHA, LAM, 0.01, GAMMA, DEV,
                   seed=seed, use_factorized=True)
    for m in st.mems.values():
        m.eta = ETA
    return st


def W_of(m):
    return m.W0.detach() + m.U.detach() @ m.V.detach().T


@torch.no_grad()
def write_phase(model, st, docs, prefill=PREFILL):
    """返回 (前向调用总数, write_calls)。"""
    fwd = 0
    for ids in docs:
        ids = ids.to(DEV)
        L = ids.shape[0]
        cache = DynamicCache()
        model(input_ids=ids[:prefill].unsqueeze(0), past_key_values=cache, use_cache=True)
        cache = out_pkv = cache
        fwd += 1
        for t in range(prefill, L):
            out = model(input_ids=ids[t:t + 1].unsqueeze(0),
                        attention_mask=torch.ones(1, t + 1, device=DEV, dtype=torch.long),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            fwd += 1
    return fwd, {li: m.write_calls for li, m in st.mems.items()}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    p = CORPUS_DIR / "train.txt"
    if not p.exists():
        import requests
        r = requests.get(CORPUS_BASE + "train.txt", timeout=90)
        r.raise_for_status()
        p.write_bytes(r.content)
    text = p.read_text(encoding="utf-8", errors="ignore")
    train_docs = build_docs(tok, text, N_WRITE_DOCS, WRITE_DOC_TOK)
    print(f"写阶段文档 {len(train_docs)} 篇 × {WRITE_DOC_TOK} token", flush=True)

    # ---------- A1''：前向调用总数与 write_calls 的关系（严格单一装置）
    st = make_stack(model, d)
    fwd, wc = write_phase(model, st, train_docs)
    st.remove()                       # ← 立即卸载，绝不与下一个栈并存
    torch.cuda.empty_cache()
    ok = all(v == fwd - 1 for v in wc.values())
    rec("A1'' write_calls == 前向调用总数 − 1（每次写阶段独立建栈并立即卸载）", ok,
        f"前向调用总数 {fwd}；write_calls={wc} ⇒ 关系 write_calls = 前向 − 1 "
        f"（整个写阶段的第一次预填没有 k_{{l,0}}，故不能写；此后每次前视都写一次，"
        f"包括每篇文档的预填）", "write_calls == fwd − 1", ok)

    # ---------- A3''：两次写阶段逐位可重复（严格串行，建栈→测→卸载）
    st1 = make_stack(model, d)
    write_phase(model, st1, train_docs)
    W1 = {li: W_of(st1.mems[li]).cpu().clone() for li in st1.mems}
    st1.remove()
    torch.cuda.empty_cache()

    st2 = make_stack(model, d)
    write_phase(model, st2, train_docs)
    W2 = {li: W_of(st2.mems[li]).cpu().clone() for li in st2.mems}
    st2.remove()
    torch.cuda.empty_cache()

    rep = max(float((W1[li] - W2[li]).abs().max()) for li in st1.mems)
    rec("A3'' 两次写阶段逐位可重复（严格单一装置：建栈→测→卸载→再建栈）", rep == 0.0,
        f"两次写阶段（{fwd} 次前视）后的 W 最大逐位差 {rep:.3e}", "精确 0", rep)

    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (HERE / "R2_recheck_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DIR-09 R2 重检：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
