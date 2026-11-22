#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-04 · R2 —— ① 前向记忆敏感性（真实预训练 LLM）
卡片 §6.1：① 固定输入下扰动记忆 Delta W，测输出 logits 的变化。
判定：eps ∈ {1e-3,1e-2,1e-1} 上变化非零且【严格单调递增】（所有 seed 一致），
      且 ||dlogits||/eps 在小 eps 下近似常数（线性性）。
H3：退化点（gamma=0、U=V=0）上敏感性必须失效。
"""
import json
import pathlib
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sgmn_module import SGMNStack

OUT = pathlib.Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ALPHA, LAM, ETA, GAMMA = 1.0, 0.05, 0.01, 0.05
R = 8
EPS = [1e-3, 1e-2, 1e-1]
SEEDS = [1, 2, 3, 4, 5]

res = {"dir": "DIR-04", "round": "R2", "stage": "B", "model": MODEL, "checks": []}


def rec(name, passed, detail, tol=None, value=None):
    res["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                          "tolerance": tol, "value": value})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def build(seed, gamma=GAMMA, zero_uv=False, use_factorized=True):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = model.config.num_hidden_layers
    mount = [n // 3, n // 2, 2 * n // 3]
    d = model.config.hidden_size
    stack = SGMNStack(model, mount, d, d, R, ALPHA, LAM, ETA, gamma, DEV,
                      seed=seed, use_factorized=use_factorized)
    if zero_uv:
        with torch.no_grad():
            for m in stack.mems.values():
                m.U.zero_(); m.V.zero_()
    return tok, model, stack, mount, d


def make_batch(tok, bs=4, L=32):
    texts = ["The capital of France is", "In a distant galaxy the explorer found",
             "The quick brown fox jumps over", "Machine learning models are"]
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=L)
    return enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)


def perturb_U(stack, eps, gen):
    """把可塑状态 Delta W = U V^T 沿单位随机方向 u 扰动，使 ||d(Delta W)||_F = eps。
    实现：取 V 的第一列 v（范数 nv），令 U[:,0] += (eps/nv) * u，
    则 d(DeltaW) = (eps/nv) u v^T，其 Frobenius 范数 = (eps/nv)*||u||*||v|| = eps。
    （U 是 (d_v, r)，故扰动必须加在某一列上，不能整块加。）"""
    for m in stack.mems.values():
        v0 = m.V[:, 0].detach()
        nv = float(v0.norm())
        u = torch.randn(m.d_v, generator=gen).to(DEV)
        u = u / u.norm()
        with torch.no_grad():
            m.U[:, 0].add_(u, alpha=eps / max(nv, 1e-12))


def snapshot_mem(stack):
    return {li: (m.U.detach().clone(), m.V.detach().clone(), m.W0.detach().clone())
            for li, m in stack.mems.items()}


def restore_mem(stack, snap):
    with torch.no_grad():
        for li, (U, V, W0) in snap.items():
            m = stack.mems[li]
            m.U.copy_(U); m.V.copy_(V); m.W0.copy_(W0)
            m.prev_hs = None          # 每次前视作独立的一步（首步无 k_{l,0}）


@torch.no_grad()
def logits_of(model, ids, mask):
    return model(input_ids=ids, attention_mask=mask).logits.float()


def main():
    print("=== DIR-04 R2：① 前向记忆敏感性 ===")
    gen = torch.Generator(device="cpu"); gen.manual_seed(4242)
    rows = []
    for seed in SEEDS:
        tok, model, stack, mount, d = build(seed)
        ids, mask = make_batch(tok)
        snap = snapshot_mem(stack)
        restore_mem(stack, snap)
        base = logits_of(model, ids, mask)
        g = torch.Generator(device="cpu"); g.manual_seed(seed * 977)
        row = {"seed": seed, "eps": []}
        for eps in EPS:
            restore_mem(stack, snap)
            perturb_U(stack, eps, g)
            lp = logits_of(model, ids, mask)
            delta = float((lp - base).norm())
            row["eps"].append({"eps": eps, "delta_logits_norm": delta,
                               "ratio_over_eps": delta / eps})
        rows.append(row)
        print(f"  seed={seed}: " + "  ".join(
            f"eps={e['eps']:g}:|dL|={e['delta_logits_norm']:.4e}" for e in row["eps"]))
        del model, stack
        torch.cuda.empty_cache()

    res["per_seed"] = rows

    # ---- 判定 ①：非零 + 严格单调递增（所有 seed 一致）----
    all_nonzero = all(e["delta_logits_norm"] > 0 for r in rows for e in r["eps"])
    all_mono = all(all(r["eps"][i]["delta_logits_norm"] < r["eps"][i + 1]["delta_logits_norm"]
                       for i in range(len(r["eps"]) - 1)) for r in rows)
    rec("①a 记忆扰动引起非零输出变化", all_nonzero,
        f"所有 seed × eps 的 |dlogits| 均 > 0（共 {sum(len(r['eps']) for r in rows)} 点）",
        "全部 > 0", None)
    rec("①b 输出变化随扰动幅度严格单调递增（所有 seed 一致）", all_mono,
        f"每个 seed 的 eps=1e-3<1e-2<1e-1 序列严格递增: {all_mono}", "全部严格递增", None)

    # ---- 线性性：|dlogits|/eps 在小 eps 区间近似常数 ----
    lin = []
    for r in rows:
        a, b = r["eps"][0]["ratio_over_eps"], r["eps"][1]["ratio_over_eps"]
        lin.append(abs(b - a) / max(abs(a), 1e-30))
    maxlin = max(lin)
    rec("①c 小 eps 区间近似线性（|dlogits|/eps 相对变化 <= 30%）", maxlin <= 0.30,
        f"各 seed 在 eps=1e-3 vs 1e-2 的比值相对变化最大 {maxlin*100:.1f}%",
        "<=30%", maxlin)

    # ---- H3 退化点 ----
    print("  · 退化点检查")
    tok, model, stack, mount, d = build(11, gamma=0.0)     # 退化点 1: gamma=0
    ids, mask = make_batch(tok)
    snap = snapshot_mem(stack); restore_mem(stack, snap)
    base = logits_of(model, ids, mask)
    g = torch.Generator(device="cpu"); g.manual_seed(7)
    restore_mem(stack, snap)
    perturb_U(stack, 1e-1, g)
    dlt = float((logits_of(model, ids, mask) - base).norm())
    del model, stack
    torch.cuda.empty_cache()
    rec("H3a 退化点 gamma=0：记忆扰动不改变输出（逐位相同）", dlt == 0.0,
        f"|dlogits| = {dlt:.3e}（应为 0）", "精确 0", dlt)

    tok, model, stack, mount, d = build(12, zero_uv=True)   # 退化点 2: U=V=0
    ids, mask = make_batch(tok)
    snap = snapshot_mem(stack); restore_mem(stack, snap)
    base = logits_of(model, ids, mask)
    g = torch.Generator(device="cpu"); g.manual_seed(8)
    restore_mem(stack, snap)
    perturb_U(stack, 1e-1, g)
    dlt2 = float((logits_of(model, ids, mask) - base).norm())
    del model, stack
    torch.cuda.empty_cache()
    rec("H3b 退化点 U=V=0：记忆扰动不改变输出（逐位相同）", dlt2 == 0.0,
        f"|dlogits| = {dlt2:.3e}（应为 0）", "精确 0", dlt2)

    res["summary"] = {"n_checks": len(res["checks"]),
                      "passed": sum(1 for c in res["checks"] if c["passed"])}
    (OUT / "R2_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\n=== DIR-04 R2：{res['summary']['passed']}/{res['summary']['n_checks']} 通过 ===")


if __name__ == "__main__":
    main()
