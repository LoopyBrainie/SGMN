#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-10 · R4 —— 复检轮：A3' 记账口径修正 + H4 退化机制的 γ 轨迹刻画。

【A3' 的性质】R1 的 A3' 记为 FAIL：20 块前视 ⇒ write_calls=19（实际值正确），
而检查写成 `write_calls == 20`。这是【判据表达式的记账口径错误】，不是装置缺陷：
v2.4 §2.3 的 k_{l,t}=sg(P_k(h_{l,t-1})) 需要前驱隐状态，故**首个预填无 k_{l,0}、不发生写入**；
于是 write_calls = 前向次数 − 1（DIR-06 已确立的同一条记账规则）。
本轮按该口径复检，并补一条【有区分力】的对照：1 次前视 ⇒ 0 次写；2 次前视 ⇒ 1 次写。
判据本身（"写通道确实发生"）不变；口径变更在此显式记录。

【H4 的机制刻画】R1 观测到联合训练把 γ 从 0.05 压到 ~0.005（10×），但评测时注入相对幅度
仍有 ~4e-2（读侧同时放大了 m）。本轮记录 γ 与注入相对幅度随训练步的轨迹，
把"部分退化"从现象变成可解释的轨迹。
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module import SGMNStack      # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
CHUNK = 64
N_CHUNKS = 600
GAMMA0 = 0.05
LR = 1e-3
CLIP = 1.0
SEED = 20261028
ETA = 1e-6

res = {"dir": "DIR-10", "round": "R4", "recheck_A3": {}, "gamma_traj": {}}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import requests, re as _re, random
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cdir = pathlib.Path.home() / "sgmn-loop" / "corpus"
    cdir.mkdir(parents=True, exist_ok=True)
    p = cdir / "train.txt"
    if not p.exists():
        r = requests.get("https://raw.githubusercontent.com/pytorch/examples/main/"
                         "word_language_model/data/wikitext-2/train.txt", timeout=90)
        r.raise_for_status()
        p.write_bytes(r.content)
    text = p.read_text(encoding="utf-8", errors="ignore")
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    ids = []
    for q in paras:
        ids.extend(tok(q).input_ids)
        if len(ids) >= 400 * 512:
            break
    t = torch.tensor(ids[:400 * 512], dtype=torch.long)
    rng = random.Random(7)
    pool = [t[i:i + CHUNK] for i in range(0, t.numel() - CHUNK, CHUNK)]
    rng.shuffle(pool)
    chunks = []
    while len(chunks) < N_CHUNKS:
        chunks.extend(pool)
        chunks = chunks[:N_CHUNKS]
    chunks = [c.unsqueeze(0) for c in chunks[:N_CHUNKS]]

    # ================================================== A3' 复检（有区分力对照）
    print("=== DIR-10 R4：A3' 复检（口径：write_calls = 前向次数 − 1）===", flush=True)
    cases = []
    for n in (1, 2, 20):
        st = SGMNStack(model, MOUNT, model.config.hidden_size, model.config.hidden_size,
                       8, 1.0, 0.05, ETA, GAMMA0, DEV, seed=SEED, use_factorized=True)
        with torch.no_grad():
            for ids_ in chunks[:n]:
                model(input_ids=ids_.to(DEV))
        wc = {li: st.mems[li].write_calls for li in st.mems}
        ok = all(v == n - 1 for v in wc.values())
        cases.append({"forwards": n, "write_calls": wc, "expected": n - 1, "passed": ok})
        print(f"  前向 {n:>3} 次 ⇒ write_calls={list(wc.values())[0]}（期望 {n-1}）"
              f" [{'PASS' if ok else 'FAIL'}]", flush=True)
        st.remove()
        torch.cuda.empty_cache()
    a3 = all(c["passed"] for c in cases)
    res["recheck_A3"] = {"cases": cases, "passed": a3,
                         "note": "首个预填无 k_{l,0} ⇒ 不写入；write_calls = 前向次数 − 1"}
    print(f"  A3' 复检：{'PASS' if a3 else 'FAIL'}", flush=True)

    # ================================================== γ 轨迹
    print("\n=== DIR-10 R4：联合训练的 γ / 注入相对幅度 / 损失 轨迹 ===", flush=True)

    def run(arm, write_on, eta):
        st = SGMNStack(model, MOUNT, model.config.hidden_size, model.config.hidden_size,
                       8, 1.0, 0.05, eta, GAMMA0, DEV, seed=SEED,
                       use_factorized=True, no_write=not write_on)
        if write_on:
            with torch.no_grad():
                for ids_ in chunks:
                    model(input_ids=ids_.to(DEV))
        for m in st.mems.values():
            m.no_write = True
        opt = torch.optim.Adam(st.read_params(), lr=LR)
        traj, losses = [], []
        for step, ids_ in enumerate(chunks):
            ids_ = ids_.to(DEV)
            opt.zero_grad(set_to_none=True)
            lg = model(input_ids=ids_).logits.float()
            tgt = ids_[:, 1:]
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.read_params(), CLIP)
            opt.step()
            losses.append(float(loss))
            if step % 25 == 0 or step == len(chunks) - 1:
                g = float(np.mean([float(st.mems[li].gamma.detach()) for li in st.mems]))
                # 注入相对幅度（用当前隐状态估计）
                with torch.no_grad():
                    x = chunks[0].to(DEV)
                    model(input_ids=x)
                    vals = []
                    for li, m in st.mems.items():
                        hs = st._hs_in[li]
                        inj = m.injection
                        if hs is not None and inj is not None:
                            vals.append(float(m.gamma.detach()) * float(inj.norm())
                                        / max(float(hs.norm()), 1e-30))
                traj.append({"step": step + 1, "gamma": g,
                             "rel_inj": float(np.mean(vals)) if vals else 0.0,
                             "loss": float(loss)})
        st.remove()
        torch.cuda.empty_cache()
        return traj, losses

    for arm, write_on, eta in (("joint_1e-06", True, ETA),
                                ("joint_nowrite", False, ETA)):
        traj, losses = run(arm, write_on, eta)
        res["gamma_traj"][arm] = {"traj": traj,
                                  "loss_first": float(np.mean(losses[:20])),
                                  "loss_last": float(np.mean(losses[-20:])),
                                  "n": len(losses)}
        print(f"\n  [{arm}] 损失 {np.mean(losses[:20]):.4f} → {np.mean(losses[-20:]):.4f}",
              flush=True)
        print(f"    {'步':>5} {'γ':>9} {'注入相对幅度':>14} {'损失':>9}", flush=True)
        for r in traj[::4] + [traj[-1]]:
            print(f"    {r['step']:>5} {r['gamma']:>9.5f} {r['rel_inj']:>14.4e} "
                  f"{r['loss']:>9.4f}", flush=True)

    (HERE / "R4_recheck_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== DIR-10 R4 完成 ===", flush=True)


if __name__ == "__main__":
    main()
