#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-07 · R4 —— 真实 LLM 的 (k -> v) 回归信噪比

R3 的结论：窗口【是否为空】只由 SNR 决定，与输入尺度无关（临界 SNR ≲ 0.05 空、≳ 0.1 非空）。
本轮测量真实 LLM 上、v2.4 §2.3 所定义的 (k_t, v_t) 对的回归信噪比：
    k_t = P_k(h_{l,t-1}) ,  v_t = P_v(h_{l,t})   （写入实际使用的对）
SNR := 被 k 线性解释的 Var(v) 比例 = 岭回归 R^2（最优线性预测器）。

若真实 LLM 的 SNR < 0.05  => R3 的空窗口结论直接适用
                                 => 低秩可塑记忆的在线写入需要【重新设计】，而非调参。
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sgmn_module_instr import SGMNStack

OUT = pathlib.Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
NCH = 4
SNR_EMPTY = 0.05        # R3 判定为「窗口空」的 SNR
SNR_NONEMPTY = 0.1      # R3 判定为「窗口非空」的 SNR

res = {"dir": "DIR-07", "round": "R4", "stage": "B", "model": MODEL, "checks": []}


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
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def ridge_r2(X, Y, lam=1e-2):
    """最优线性预测 Y ≈ X W 的 R^2（中心化 + 岭正则，用 lstsq 求解以抗奇异）。"""
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[ok], Y[ok]
    X = X - X.mean(axis=0); Y = Y - Y.mean(axis=0)
    n, dk = X.shape
    if n < dk + 5:
        return float("nan"), None
    # 岭：增广 sqrt(lam*n) * I
    A = np.vstack([X, np.sqrt(lam * n) * np.eye(dk)])
    B = np.vstack([Y, np.zeros((dk, Y.shape[1]))])
    W, *_ = np.linalg.lstsq(A, B, rcond=None)
    pred = X @ W
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum(Y ** 2))
    return float(1.0 - ss_res / ss_tot), W


def main():
    print("=== DIR-07 R4：真实 LLM 的 (k -> v) 回归信噪比 ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    texts = ["The capital of France is", "In a distant galaxy the explorer found",
             "The quick brown fox jumps over", "Machine learning models are",
             "She opened the old wooden door and found", "Every morning the village",
             "Scientists announced that the new", "He remembered the summer of"]
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    ids = enc["input_ids"].to(DEV); mask = enc["attention_mask"].to(DEV)
    nl = model.config.num_hidden_layers; d = model.config.hidden_size
    mount = [nl // 3, nl // 2, 2 * nl // 3]

    st = SGMNStack(model, mount, d, d, 8, 1.0, 0.05, 3e-5, 0.05, DEV,
                   seed=20261029, use_factorized=True)

    # 收集写入实际使用的 (k_t, v_t) 对：对每个挂载层，取每个时间步的
    #   k_t = P_k(h_{l,t-1}) , v_t = P_v(h_{l,t})
    pairs = {li: [] for li in mount}
    prev_h = {li: None for li in mount}
    captured = {}

    def make_pre_hook(idx):
        def pre_hook(module, args, kwargs):
            captured[idx] = args[0].detach()
        return pre_hook

    handles = []
    for li in mount:
        handles.append(st._layers()[li].register_forward_pre_hook(make_pre_hook(li), with_kwargs=True))

    # 单次整段前视：捕获每个挂载层的逐 token 隐状态 h（写通道在段内惰性，不影响 h）
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask)
        for li in mount:
            h = captured[li]                                   # (B, L, d)
            v = h @ st.mems[li].Pv.detach()                     # (B, L, d_v)
            k = torch.cat([h[:, :-1], h[:, :1]], dim=1) @ st.mems[li].Pk.detach()  # k_t 用 h_{t-1}
            pairs[li].append((k[:, :-1].reshape(-1, d).cpu().numpy(),
                              v[:, 1:].reshape(-1, d).cpu().numpy()))

    for h in handles:
        h.remove()
    st.remove()
    torch.cuda.empty_cache()

    # ---- 计算每个挂载层的 SNR ----
    rows = []
    for li in mount:
        K = np.concatenate([p[0] for p in pairs[li]], axis=0)
        V = np.concatenate([p[1] for p in pairs[li]], axis=0)
        r2, W = ridge_r2(K, V)
        # 对照：只用截距（不学）的 R^2
        r2_null = 1.0 - float(np.sum((V - V.mean(axis=0)) ** 2)) / float(
            np.sum((V - V.mean(axis=0)) ** 2))
        rows.append({"layer": li, "n_pairs": int(K.shape[0]), "snr_r2": r2,
                     "r2_null": r2_null, "var_v": float(np.var(V)),
                     "mu_k": float(np.mean(K ** 2))})
        print(f"  层 {li}: n={K.shape[0]}  SNR(R^2)={r2:+.5f}  E[k^2]={rows[-1]['mu_k']:.3f}  "
              f"Var(v)={rows[-1]['var_v']:.3f}")
    res["per_layer_snr"] = rows
    snr_vals = [r["snr_r2"] for r in rows]
    snr_mean = float(np.mean(snr_vals))
    res["snr_mean"] = snr_mean

    # ---- 判定：真实 LLM 的 SNR 落在 R3 的哪一档 ----
    if snr_mean < SNR_EMPTY:
        verdict_txt = ("真实 LLM 的 SNR < 0.05 —— 落在 R3 的【窗口空】区间 => "
                       "低秩可塑记忆的在线写入需要【重新设计】")
        passed = True
    elif snr_mean < SNR_NONEMPTY:
        verdict_txt = ("真实 LLM 的 SNR ∈ [0.05, 0.10) —— 落在 R3 的【边界】区间 => "
                       "窗口是否存在取决于未测的细节")
        passed = False
    else:
        verdict_txt = ("真实 LLM 的 SNR >= 0.10 —— 落在 R3 的【窗口非空】区间 => "
                       "存在可行窗口（但位置随尺度左移）")
        passed = True
    rec("R4 真实 LLM 的 (k->v) 回归信噪比已测定", passed, verdict_txt,
        f"SNR = {snr_mean:.5f}（分层：{[round(s,5) for s in snr_vals]}）", snr_mean)

    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (OUT / "R4_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\n=== DIR-07 R4：{npass}/{len(res['checks'])} 通过 ===")


if __name__ == "__main__":
    main()
