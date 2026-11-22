#!/usr/bin/env python
"""DIR-86 前置探针：**选定 η 的工作点**（不是抄构造默认值）。

`R1_joint_read.py:96` 的 `0.01` 是**被覆盖的占位默认**；D-16 的全部真实调用是
`eta_write=1e-6`（:317/:350/:358/:371/:413/:421/:583），正式轮则在
`ETA_GRID=[1e-7,1e-6,1e-5,1e-4]` 上扫描（:444/:455）。
⇒ 「D-16 的参考 η」**不存在**，存在的是一个**区间**。本探针在 D-16 自己的网格上
实测：**取使状态在满 600 块预算内保持有限的最大 η**（最大化效应量，同时不发散）。

同时记录 Δ_base(600) 与静态注入税 Δ_base(0)，用于判断 P86 功效。
"""
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import R1_joint_read as D16                      # noqa: E402
from sgmn_module import SGMNStack                # noqa: E402

GAMMA_REF = D16.GAMMA0
B = D16.N_CHUNKS
CHUNK = D16.CHUNK
SEED = D16.SEEDS[0]
ETAS = [1e-6, 1e-5, 1e-4, 1e-3]                  # D-16 网格 + 一个越界点探边界
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def build(model, eta, gamma, no_write=False):
    d = model.config.hidden_size
    st = SGMNStack(model, D16.MOUNT, d, d, D16.R, D16.ALPHA, D16.LAM, eta,
                   GAMMA_REF, D16.DEV, seed=SEED, use_factorized=True,
                   drop_sg=False, no_write=no_write)
    for m in st.mems.values():
        m.gamma.data.fill_(float(gamma))
        m.gamma.requires_grad_(False)
    return st


def wdrift(st, w0):
    return max(float((m.W0 + m.U @ m.V.T - w0[int(li)]).abs().max())
               for li, m in st.mems.items())


def wsnap(st):
    return {int(li): (m.W0 + m.U @ m.V.T).detach().clone() for li, m in st.mems.items()}


def run(model, chunks, docs, eta, gamma, no_write=False, checkpoints=(0, B)):
    st = build(model, eta, gamma, no_write)
    snap = wsnap(st)
    rows = {}
    fed = 0
    try:
        for N in checkpoints:
            while fed < N:
                model(input_ids=chunks[fed].to(D16.DEV))
                fed += 1
            nll, _, _ = D16.eval_docs(model, st, docs)
            nll = np.array(nll, dtype=float)
            rows[N] = {"mean_nll": float(np.mean(nll[np.isfinite(nll)]))
                       if np.isfinite(nll).any() else float("nan"),
                       "n_nonfinite": int((~np.isfinite(nll)).sum()),
                       "n_docs": int(nll.size),
                       "w_drift_max": wdrift(st, snap),
                       "write_calls": {int(li): int(m.write_calls) for li, m in st.mems.items()}}
    finally:
        st.remove()
    return rows


def main():
    tok, model = D16.model_and_tok()
    corpus = D16.load_corpus()
    train_docs = D16.build_docs(tok, corpus["train"], 400, 512)
    valid_docs = D16.build_docs(tok, corpus["valid"], D16.N_EVAL_DOCS, D16.EVAL_DOC_TOK)
    chunks = D16.build_chunks(train_docs, chunk=CHUNK, n_chunks=B, seed=7)
    print(f"OK chunks={len(chunks)} docs={len(valid_docs)}", flush=True)

    base = run(model, chunks, valid_docs, 1e-6, 0.0)          # γ=0 ⇒ η 无关
    b0, bB = base[0]["mean_nll"], base[B]["mean_nll"]
    print(f"BASE  N=0 {b0:.6f}  N={B} {bB:.6f}  (应相等)", flush=True)

    res = {"dir": "DIR-86", "probe": "eta", "base": {str(k): v for k, v in base.items()},
           "etas": {}}
    for eta in ETAS:
        rows = run(model, chunks, valid_docs, eta, GAMMA_REF)
        tax0 = rows[0]["mean_nll"] - b0
        d600 = rows[B]["mean_nll"] - bB
        res["etas"][str(eta)] = {str(k): v for k, v in rows.items()}
        print(f"η={eta:.0e}  N0 nll={rows[0]['mean_nll']:.6f} (税 {tax0:+.6f})  "
              f"N{B} nll={rows[B]['mean_nll']:.6f} (Δ_base {d600:+.6f})  "
              f"|W−W0|={rows[B]['w_drift_max']:.3e}  非有限={rows[B]['n_nonfinite']}/{rows[B]['n_docs']}",
              flush=True)
        OUT.joinpath("eta_probe.json").write_text(json.dumps(res, indent=1))

    # 判定：取网格内「满预算仍全有限」的最大 η
    ok = [e for e in ETAS if res["etas"][str(e)][str(B)]["n_nonfinite"] == 0]
    print(f"\n★ 满 {B} 块全有限的 η：{ok}")
    print(f"★ 选定 η = {max(ok) if ok else None}（网格 {ETAS}）")
    res["chosen_eta"] = max(ok) if ok else None
    OUT.joinpath("eta_probe.json").write_text(json.dumps(res, indent=1))
    print(f"→ {OUT / 'eta_probe.json'}")


if __name__ == "__main__":
    main()