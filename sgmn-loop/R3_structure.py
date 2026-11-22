#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-12 · R3 —— 【探索性·结构诊断】为什么检索不可能？

R1 的结论（预判）：竞争式检索算子（λ）也不能把写入的内容取出来 ⇒ H2 不成立。
R3 回答**为什么**，测四个结构性量（全部为**探索性报告量**，不进入判据）：

  R3-a **状态的有效秩**：对 W = U V^T（因子化，§3.3）与 W_full（§3.2）分别算奇异值谱，
       报告参与比 PR = (Σλ)²/Σλ²（= 有效秩）与"超过阈值 1e-6·λ_max 的秩"。
       ⇒ 若 PR(因子化) ≤ r=8 而事实数 = 60 ⇒ **可寻址槽位数 ≤ r** ⇒ 结构上不可能按键取回单条关联。

  R3-b **键的有效槽位数**：把写阶段记录的键堆成 K（N×d_k），算 Gram 矩阵的特征值参与比
       ⇒ PR = 有效正交方向数 ⇒ 若 PR ≪ N（键高度相关）⇒ 读出必然是"核加权平均"而非检索。

  R3-c **键间余弦分布**：写入键两两余弦的均值/95 分位（LLM 隐状态天然相关 ⇒ 余弦高）。

  R3-d **写保真度**：对每条被写入的关联 (k_i, v_i)，算 cos(W k_i, v_i)（W 为写后状态）
       ⇒ 若 ≈0 ⇒ 写本身没有把 v_i 存到"用 k_i 查询会取到"的槽位上。

协议：与 R1 相同的合成回忆语料（wikitext-2 填充 30 token + 4 位码事实句，N=60 篇），
写入用**块级平均写**（DIR-04 资产语义，= DIR-10 同一路径），η=1e-6；
两个分支各测一次：因子化（§3.3）与全矩阵（§3.2）。
"""
import json
import pathlib
import random
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module_retrieval import SGMNStack as SGRet   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R, ETA = 1.0, 0.05, 0.05, 8, 1e-6
N_DOCS, FILLER_TOK = 60, 30
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-12", "round": "R3", "type": "探索性·结构诊断（不进入判据）",
       "n_docs": N_DOCS, "r": R, "eta": ETA, "branches": {}}


def load_filler():
    import requests
    p = pathlib.Path.home() / "sgmn-loop" / "corpus" / "train.txt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(CORPUS_BASE + "train.txt", timeout=90)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p.read_text(encoding="utf-8", errors="ignore")


def build_facts(tok, seed):
    import re as _re
    text = load_filler()
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    rng = random.Random(seed)
    rng.shuffle(paras)
    pool = []
    for q in paras:
        pool.extend(tok(q).input_ids)
        if len(pool) > 40000:
            break
    docs, used = [], set()
    while len(docs) < N_DOCS:
        i = rng.randrange(0, len(pool) - 200)
        fill = pool[i:i + FILLER_TOK]
        code = f"{rng.randrange(1000, 9999)}"
        if code in used:
            continue
        used.add(code)
        pre = " The vault code is"
        pre_ids = tok(pre).input_ids
        full_ids = tok(pre + " " + code + ".").input_ids
        if full_ids[:len(pre_ids)] != pre_ids:
            continue
        docs.append({"ids": fill + full_ids, "code_ids": full_ids[len(pre_ids):],
                     "prefix_len": len(fill) + len(pre_ids)})
    return docs


def participation_ratio(lam):
    lam = np.asarray(lam, dtype=float)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0, 0
    return float(lam.sum() ** 2 / (lam ** 2).sum()), int((lam > 1e-6 * lam.max()).sum())


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    docs = build_facts(tok, 20261028)

    for branch, fact in (("factorized", True), ("full", False)):
        print(f"\n=== R3 · 分支 {branch} ===", flush=True)
        st = SGRet(model, MOUNT, d, d, R, ALPHA, LAM, ETA, GAMMA0, DEV,
                   seed=20261028, use_factorized=fact, perpos=False)
        keys = {li: [] for li in st.mems}
        vals = {li: [] for li in st.mems}
        for doc in docs:
            ids = torch.tensor(doc["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
            with torch.no_grad():
                model(input_ids=ids)
            for li, m in st.mems.items():
                h = st._hs_in[li][0]
                keys[li].append((h @ m.Pk.detach().T).mean(dim=0))
                vals[li].append((h @ m.Pv.detach().T).mean(dim=0))
        out = {"layers": {}}
        for li, m in st.mems.items():
            K = torch.stack(keys[li]).double()                    # (N,d_k)
            V = torch.stack(vals[li]).double()                    # (N,d_v)
            W = (m.W0.detach() + m.U.detach() @ m.V.detach().T) if fact else m.Wfull.detach()
            W = W.double()
            # --- R3-a 状态有效秩
            sv = torch.linalg.svdvals(W)
            pr_w, rank_w = participation_ratio(sv.cpu().numpy())
            # --- R3-b 键的有效槽位数
            Kn = K / K.norm(dim=1, keepdim=True).clamp_min(1e-30)
            ev = torch.linalg.eigvalsh(Kn @ Kn.T)
            pr_k, rank_k = participation_ratio(ev.cpu().numpy())
            # --- R3-c 键间余弦
            G = (Kn @ Kn.T).cpu().numpy()
            off = G[~np.eye(G.shape[0], dtype=bool)]
            # --- R3-d 写保真度
            Wk = (W @ K.T)                                        # (d_v,N)
            cos_fid = torch.nn.functional.cosine_similarity(Wk.T, V, dim=1)
            out["layers"][str(li)] = {
                "state_pr": pr_w, "state_rank_1e-6": rank_w,
                "sv_top": [float(x) for x in sv[:12].cpu().numpy()],
                "key_pr": pr_k, "key_rank_1e-6": rank_k, "n_keys": int(K.shape[0]),
                "key_cos_mean": float(off.mean()), "key_cos_p95": float(np.percentile(off, 95)),
                "write_fidelity_cos_mean": float(cos_fid.mean()),
                "write_fidelity_cos_max": float(cos_fid.max()),
            }
            print(f"  层 {li}: 状态有效秩 PR={pr_w:.3f}（秩 {rank_w}）| "
                  f"键有效槽位 PR={pr_k:.3f}（秩 {rank_k}，共 {K.shape[0]} 键）| "
                  f"键间余弦 均值 {off.mean():+.3f} p95 {np.percentile(off,95):+.3f} | "
                  f"写保真度 cos(Wk,v) 均值 {float(cos_fid.mean()):+.4f} 最大 {float(cos_fid.max()):+.4f}",
                  flush=True)
        res["branches"][branch] = out
        st.remove()
        torch.cuda.empty_cache()

    (HERE / "R3_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("\n=== R3 完成 ===", flush=True)


if __name__ == "__main__":
    main()
