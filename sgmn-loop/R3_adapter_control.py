#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-10 · R3 —— 【探索性】读路径对照：SGMN 的联合训练收益 vs 纯残差适配器。

【为什么做】R1 的 H3 失败：联合训练下「写」与「不写」的任务差异为 +0.00003（不显著），
即【写入的记忆内容对任务收益没有贡献】；收益全部来自可训练的读侧。
但 SGMN 的读侧本身就是一个可训练的残差变换（且其输入被 stop-gradient 隔离：
读上下文 c = sg(h_l^-)，故它等价于"分离输入的残差适配器"）。
⇒ 必须排除一个更平凡的解释：**任意可训练残差适配器都能拿到同样的收益**。

【对照设计】完全镜像 SGMN 的读侧结构与训练协议，只把中间的"记忆"拿掉：
    SGMN 读:  m = Wo · (sg(W) · Wq · sg(c))      —— 三矩阵双线性，中间是记忆 W
    适配器:   m = Bo · (Ba · sg(c))               —— 两矩阵，中间无记忆
  * 挂载层相同（8/12/16），输入同样 stop-gradient，γ 初值相同（0.05，可训练）；
  * 初始化尺度镜像装置：Ba ~ N(0,1/d)，Bo ~ N(0,1/rank)；
  * 秩 = {8, 64, 896}：8 = 记忆秩（低秩瓶颈），896 = 全秩（参数数与 SGMN 读侧 Wq+Wo 相同）；
  * 训练协议、语料、评测集、种子与 R1 完全一致。

【纪律】本轮是【探索性测量，不进入判据】（机制级纪律 #8）；数值仅用于解释 H3 的失败，
不改写 R1 的任何判据结论。
"""
import json
import math
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
CHUNK = 64
N_CHUNKS = 600
N_EVAL_DOCS = 60
EVAL_DOC_TOK = 512
SEEDS = [20261028, 20261104, 7]
RANKS = [8, 64, 896]
GAMMA0 = 0.05
CLIP = 1.0
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

HERE = pathlib.Path(__file__).resolve().parent

def _find_r1():
    for cand in (HERE / "R1_results.json", HERE.parent / "R1_results.json",
                 HERE.parent / "sgmn-loop" / "R1_results.json",
                 pathlib.Path.home() / "sgmn-loop" / "R1_results.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError("R1_results.json not found")


class AdapterLayer:
    """纯残差适配器（无记忆），镜像 SGMN 读侧的可训练形状与注入方式。"""

    def __init__(self, d, rank, gamma, gen):
        # 形状镜像装置：Wq:(d_k,d_model)、Wo:(d_model,d_v) ⇒ Ba:(rank,d)、Bo:(d,rank)
        # 尺度镜像装置的 mk():Ba 的元素 ~ N(0,1/d)、Bo 的元素 ~ N(0,1/rank)
        # ⇒ 任意秩下 ||m|| ≈ ||c||（两个缩放相消），与 SGMN 读侧的注入量级一致
        self.Ba = (torch.randn(rank, d, generator=gen) / math.sqrt(d)).to(DEV).requires_grad_(True)
        self.Bo = (torch.randn(d, rank, generator=gen) / math.sqrt(rank)).to(DEV).requires_grad_(True)
        self.gamma = torch.tensor(float(gamma), device=DEV, requires_grad=True)
        self.injection = None


class AdapterStack:
    def __init__(self, model, mount_layers, rank, gamma=GAMMA0, seed=0):
        self.model = model
        self.layers_ = {}
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        d = model.config.hidden_size
        for li in mount_layers:
            self.layers_[li] = AdapterLayer(d, rank, gamma, gen)
        self._hs_in = {}
        self._handles = []
        self._install()

    def _layers(self):
        for attr in ("model", "transformer"):
            base = getattr(self.model, attr, None)
            if base is not None and hasattr(base, "layers"):
                return base.layers
        return self.model.layers

    def _install(self):
        layers = self._layers()
        for li, ad in self.layers_.items():
            layer = layers[li]

            def pre_hook(module, args, kwargs, li=li):
                self._hs_in[li] = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                return None

            def fwd_hook(module, args, kwargs, output, li=li):
                ad = self.layers_[li]
                hs_in = self._hs_in[li]
                hs_out = output[0] if isinstance(output, tuple) else output
                c = hs_in.detach()                                  # 与 SGMN 读侧一致：stop-gradient 输入
                cf = c.reshape(-1, c.shape[-1])
                m = (ad.Bo @ (ad.Ba @ cf.T)).T                        # (rank,N)->(d,N)->(N,d)
                ad.injection = m.reshape(hs_in.shape)
                hs_out = hs_out + ad.gamma * ad.injection
                if isinstance(output, tuple):
                    return (hs_out,) + tuple(output[1:])
                return hs_out

            self._handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            self._handles.append(layer.register_forward_hook(fwd_hook, with_kwargs=True))

    def params(self):
        ps = []
        for ad in self.layers_.values():
            ps += [ad.Ba, ad.Bo, ad.gamma]
        return ps

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


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


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 与 R1 完全相同的语料切分
    corpus = load_corpus()
    train_docs = build_docs(tok, corpus["train"], 400, 512)
    valid_docs = build_docs(tok, corpus["valid"], N_EVAL_DOCS, EVAL_DOC_TOK)
    chunks = build_chunks(train_docs)

    # 学习率取 R1 探针的胜者（同一协议）
    r1 = json.loads(_find_r1().read_text(encoding="utf-8"))
    probe = r1.get("lr_probe", [])
    LR = max([p for p in probe if np.isfinite(p["drop"])],
             key=lambda p: p["drop"])["lr"] if probe else 1e-3
    print(f"=== DIR-10 R3（探索性）：适配器对照，LR={LR}（取自 R1 探针）===", flush=True)

    res = {"dir": "DIR-10", "round": "R3", "exploratory": True,
           "note": "探索性测量，不进入判据；用于解释 H3 失败", "lr": LR, "arms": {}}

    def train_and_eval(rank, seed):
        st = AdapterStack(model, MOUNT, rank, GAMMA0, seed)
        opt = torch.optim.Adam(st.params(), lr=LR)
        losses = []
        for ids in chunks:
            ids = ids.to(DEV)
            opt.zero_grad(set_to_none=True)
            lg = model(input_ids=ids).logits.float()
            tgt = ids[:, 1:]
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.params(), CLIP)
            opt.step()
            losses.append(float(loss))
        with torch.no_grad():
            nll, acc, rel = [], [], []
            for ids in valid_docs:
                x = ids.to(DEV).unsqueeze(0)
                lg = model(input_ids=x).logits.float()
                tgt = x[:, 1:]
                lg = lg[:, :-1]
                nll.append(float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                                  tgt.reshape(-1), reduction="sum") / tgt.numel()))
                acc.append(float((lg.argmax(-1) == tgt).float().mean()))
                vals = [float(ad.gamma.detach()) * float(ad.injection.norm())
                        / max(float(st._hs_in[li].norm()), 1e-30)
                        for li, ad in st.layers_.items() if ad.injection is not None]
                rel.append(float(np.mean(vals)) if vals else 0.0)
        gfin = float(np.mean([float(ad.gamma.detach()) for ad in st.layers_.values()]))
        st.remove()
        torch.cuda.empty_cache()
        return {"nll": nll, "acc": acc, "rel": rel, "gamma": gfin,
                "loss_first5": float(np.mean(losses[:5])),
                "loss_last5": float(np.mean(losses[-5:])), "n_steps": len(losses)}

    for rank in RANKS:
        per_seed = {}
        for seed in SEEDS:
            per_seed[str(seed)] = train_and_eval(rank, seed)
            print(f"  rank={rank:<4d} 种子 {seed}: NLL={np.mean(per_seed[str(seed)]['nll']):.5f} "
                  f"γ→{per_seed[str(seed)]['gamma']:.5f} "
                  f"注入相对幅度={np.mean(per_seed[str(seed)]['rel']):.3e} "
                  f"loss {per_seed[str(seed)]['loss_first5']:.3f}→"
                  f"{per_seed[str(seed)]['loss_last5']:.3f}", flush=True)
        res["arms"][f"adapter_rank{rank}"] = per_seed
        (HERE / "R3_explore_results.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------ 对比 R1 的 joint 臂
    def perdoc(per_seed):
        return {s: np.asarray(v["nll"], dtype=float) for s, v in per_seed.items()}

    def compare(a, b):
        A, B = perdoc(a), perdoc(b)
        D = np.concatenate([A[s] - B[s] for s in A])
        D = D[np.isfinite(D)]
        if len(D) == 0:
            return None
        m = float(D.mean())
        sd = float(D.std(ddof=1))
        t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
        ps, k, n = sign_test_p(list(D))
        sm = [float(A[s].mean() - B[s].mean()) for s in A]
        return {"mean_diff": m, "sd": sd, "t": t, "p_sign": ps, "n": n,
                "seed_means": sm,
                "consistent": bool(all(x < 0 for x in sm) or all(x > 0 for x in sm))}

    print("\n  【对照表】平均 NLL（跨 3 种子，60 篇 valid）", flush=True)
    summ = {a: float(np.mean([np.mean(v["nll"]) for v in per_seed.values()]))
            for a, per_seed in res["arms"].items()}
    r1s = r1["nll_summary"]
    for a in ("base", "joint_nowrite", "joint_1e-07", "joint_1e-06", "joint_1e-05",
              "frozen_1e-06"):
        if a in r1s:
            summ[f"[R1] {a}"] = r1s[a]["mean_nll"]
    for a in sorted(summ, key=lambda x: summ[x]):
        print(f"    {a:26s} NLL={summ[a]:.5f}", flush=True)
    res["nll_summary"] = summ

    # 适配器 vs R1 的 joint 臂（配对）
    cmp_res = {}
    r1_joint_per_seed = {}
    # R1 的逐文档数据没有单独存盘；用摘要里的均值做【非配对】对照（跨种子均值）
    for a, per_seed in res["arms"].items():
        cmp_res[a] = {"mean_nll": summ[a],
                      "vs_joint_1e-06_mean_diff": summ[a] - r1s["joint_1e-06"]["mean_nll"],
                      "vs_joint_nowrite_mean_diff": summ[a] - r1s["joint_nowrite"]["mean_nll"],
                      "vs_base_mean_diff": summ[a] - r1s["base"]["mean_nll"]}
    res["comparisons"] = cmp_res
    print("\n  【对照差值】（非配对，跨种子均值之差，负向=更好）", flush=True)
    for a, c in cmp_res.items():
        print(f"    {a:22s} vs joint_1e-06: {c['vs_joint_1e-06_mean_diff']:+.5f} | "
              f"vs joint_nowrite: {c['vs_joint_nowrite_mean_diff']:+.5f} | "
              f"vs base: {c['vs_base_mean_diff']:+.5f}", flush=True)

    (HERE / "R3_explore_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== DIR-10 R3 完成（探索性，不进入判据）===", flush=True)


if __name__ == "__main__":
    main()
