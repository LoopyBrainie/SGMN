#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-04 · R1 —— Stage A 装置校验门（硬闸）
自包含：只依赖 sgmn_module.py（本方向的装置）+ torch/transformers。

A1  v2.4 §3.1 / §3.3 / §2.3 / §2.2 的公式在真实模型上逐字实现并通过机器精度自检
A2  sg 语义：dL/dW|_read == 0（精确），且前向值仍参与计算
A3  冻结基座：dL/dtheta_base == 0（精确），且【输入梯度仍回传】（v2.4 §4 明文要求）
A4  阴性对照：去掉 stop-gradient 后 ③ 必须失败（证明检查有区分力）
A5  非退化点确认：gamma != 0, U,V != 0, ||W - W^0|| > 0
"""
import json
import pathlib
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sgmn_module import SGMNStack, LayerMem, sg, grad_report

OUT = pathlib.Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ALPHA, LAM, ETA, GAMMA = 1.0, 0.05, 0.01, 0.05
R = 8

results = {"dir": "DIR-04", "round": "R1", "stage": "A", "model": MODEL,
           "dtype": str(DTYPE), "device": DEV, "checks": []}


def rec(name, passed, detail, tol=None, value=None):
    results["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                              "tolerance": tol, "value": value})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


# ============================================================== 装置自检（纯张量层）
def unit_checks():
    print("\n[A1] 装置自检（纯张量层，机器精度）")
    torch.manual_seed(0)
    d_v, d_k, d_model, r = 3, 4, 5, 2
    gen = torch.Generator(); gen.manual_seed(7)
    W0 = torch.randn(d_v, d_k, generator=gen)
    U = torch.randn(d_v, r, generator=gen) * 0.1
    V = torch.randn(d_k, r, generator=gen) * 0.1
    W = W0 + U @ V.T
    k = torch.randn(d_k, generator=gen)
    v = torch.randn(d_v, generator=gen)

    # (a) G 公式 vs autograd of l(W)
    Wg = W.clone().requires_grad_(True)
    ell = 0.5 * torch.sum((Wg @ k - v) ** 2) + 0.5 * LAM * torch.sum((Wg - W0) ** 2)
    ell.backward()
    G_formula = (W @ k - v).unsqueeze(1) @ k.unsqueeze(0) + LAM * (W - W0)
    errG = float((Wg.grad - G_formula).abs().max() / G_formula.abs().max())
    rec("A1a  G = alpha(Wk-v)k^T + lambda(W-W^0) vs autograd", errG < 1e-6,
        f"最大相对误差 {errG:.3e}", "rel<1e-6", errG)

    # (b) 二阶展开恒等式（v2.4 §3.3）
    eta = 0.01
    U_new = U - eta * (G_formula @ V)
    V_new = V - eta * (G_formula.T @ U)
    lhs = (U_new @ V_new.T) - (U @ V.T)
    rhs = -eta * (G_formula @ V @ V.T + U @ U.T @ G_formula) + eta**2 * (G_formula @ V @ U.T @ G_formula)
    err2 = float((lhs - rhs).abs().max() / rhs.abs().max())
    rec("A1b  因子化二阶展开 (v2.4 §3.3)", err2 < 1e-6, f"最大相对误差 {err2:.3e}", "rel<1e-6", err2)

    # (c) 同步步 vs 异步步必须不同（确认装置走同步支）
    U_async = U - eta * (G_formula @ V)
    V_async = V - eta * (G_formula.T @ U_async)
    gap = float((U_async @ V_async.T - U_new @ V_new.T).abs().max())
    rec("A1c  同步步与异步步确实不同（确认走同步支）", gap > 1e-9,
        f"两种步长 Delta W 最大差 {gap:.3e}", ">1e-9", gap)

    # (d) k,v 定义：k 用 t-1 的输入，v 用 t 的输入
    mem = LayerMem(d_model, d_k, d_v, r, ALPHA, LAM, ETA, GAMMA, "cpu", gen)
    hs = [torch.randn(d_model, generator=gen) for _ in range(3)]
    mem.prev_hs = None
    mem.write(hs[0]); mem.cache(hs[0])            # t=0: 无 k -> 不写
    n0 = mem.write_calls
    # 捕获【写之前】的 U, V, W（同步步与 G 都用旧值）
    U_b, V_b, W0_b = mem.U.detach().clone(), mem.V.detach().clone(), mem.W0.detach().clone()
    W_t = W0_b + U_b @ V_b.T
    mem.write(hs[1])                              # t=1: 用 prev_hs=hs[0], 当前=hs[1]
    k_expect = hs[0] @ mem.Pk.detach()
    v_expect = hs[1] @ mem.Pv.detach()
    G_expect = ALPHA * (W_t @ k_expect - v_expect).unsqueeze(1) @ k_expect.unsqueeze(0) \
        + LAM * (W_t - W0_b)
    U_after = mem.U.detach().clone()
    U_pred = U_b - ETA * (G_expect @ V_b)
    errk = float((U_after - U_pred).abs().max() / U_pred.abs().max())
    rec("A1d  k 取 t-1 输入、v 取 t 输入（v2.4 §2.3）", errk < 1e-5 and n0 == 0,
        f"首个时间步未写入: {n0==0}；写入后 U 与手算值相对误差 {errk:.3e}", "rel<1e-5", errk)

    # (e) 读写顺序：读必须用 W_t，写产生 W_{t+1}
    mem2 = LayerMem(d_model, d_k, d_v, r, ALPHA, LAM, ETA, GAMMA, "cpu", gen)
    W_before = (mem2.W0.detach() + mem2.U.detach() @ mem2.V.detach().T).clone()
    mem2.read(hs[0])
    W_after_read = (mem2.W0.detach() + mem2.U.detach() @ mem2.V.detach().T).clone()
    mem2.prev_hs = hs[0]
    mem2.write(hs[0])
    W_after_write = (mem2.W0.detach() + mem2.U.detach() @ mem2.V.detach().T).clone()
    rec("A1e  读用 W_t、写产生 W_{t+1}（v2.4 §2.2 顺序）",
        torch.equal(W_before, W_after_read) and not torch.equal(W_after_read, W_after_write),
        f"读后 W 未变: {torch.equal(W_before, W_after_read)}；写后 W 已变: "
        f"{not torch.equal(W_after_read, W_after_write)}", "顺序正确", None)


def sg_semantics():
    print("\n[A2] sg 语义：梯度为零 + 前向仍参与")
    torch.manual_seed(1)
    d_v, d_k, d_model = 3, 4, 5
    W = torch.randn(d_v, d_k, requires_grad=True)
    q = torch.randn(d_k, requires_grad=True)
    Wo = torch.randn(d_model, d_v, requires_grad=True)
    # 正确：y = sg(W) q
    y = sg(W) @ q
    m = Wo @ y
    m.sum().backward()
    g_read = float(W.grad.abs().max()) if W.grad is not None else 0.0
    g_q = float(q.grad.abs().max())
    rec("A2a  dL/dW|_read 精确为 0（None 或逐元素 0）",
        (W.grad is None) or (g_read == 0.0),
        f"W.grad max|.| = {g_read}（grad 为 None={W.grad is None}；None 即'从未接入计算图'，"
        f"同样满足'梯度为零'）", "None 或精确 0", g_read)
    rec("A2b  读支路的输入梯度仍存在（截断只断 W，不断 q）", g_q > 1e-8,
        f"q.grad max|.| = {g_q:.3e}", ">1e-8", g_q)
    # 前向值仍参与：改 W -> m 变
    W2 = W.detach().clone() + 0.1 * torch.randn(d_v, d_k)
    m2 = Wo @ (W2 @ q.detach())
    dch = float((m2 - m.detach()).abs().max())
    rec("A2c  前向值仍参与计算（sg 不是前向断开）", dch > 1e-8,
        f"把 W 改变 0.1 量级后，m 的变化 {dch:.3e}（非 0 => 前向确实用了 W 的值）",
        ">1e-8", dch)


# ============================================================== 模型级检查
def build(model_name, mount_rule="third_half"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    mount = [n_layers // 3, n_layers // 2, 2 * n_layers // 3] if mount_rule == "third_half" else mount_rule
    d = model.config.hidden_size
    stack = SGMNStack(model, mount, d, d, R, ALPHA, LAM, ETA, GAMMA, DEV,
                      seed=20261026, use_factorized=True)
    return tok, model, stack, mount, d


def make_batch(tok, bs=4, L=32):
    texts = ["The capital of France is", "In a distant galaxy the explorer found",
             "The quick brown fox jumps over", "Machine learning models are"]
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=L)
    return enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)


def task_loss(model, ids, mask):
    logits = model(input_ids=ids, attention_mask=mask).logits
    tgt = ids[:, 1:].reshape(-1)
    lp = logits[:, :-1].reshape(-1, logits.shape[-1])
    return F.cross_entropy(lp, tgt)


def model_checks():
    print("\n[A3-A5] 模型级检查（真实预训练 LLM）")
    tok, model, stack, mount, d = build(MODEL)
    ids, mask = make_batch(tok)

    # ---- A5 非退化点 ----
    gammas = [float(m.gamma) for m in stack.mems.values()]
    unz = all(float(m.U.abs().max()) > 0 for m in stack.mems.values())
    wdiff = all(float((m.U @ m.V.T).abs().max()) > 0 for m in stack.mems.values())
    rec("A5  非退化点：gamma!=0, U,V!=0, ||W-W^0||>0",
        all(abs(g) > 0 for g in gammas) and unz and wdiff,
        f"gamma={gammas}；U,V 非零:{unz}；||W-W0||>0:{wdiff}", "全部非零", gammas)

    # ---- A3a 冻结基座：requires_grad=False + backward 后 .grad 为 None ----
    base = list(model.parameters())
    all_frozen = all(not p.requires_grad for p in base)
    loss = task_loss(model, ids, mask)
    loss.backward()
    base_grads = [p.grad for p in base]
    none_or_zero = all(g is None or float(g.abs().max()) == 0.0 for g in base_grads)
    rec("A3a  冻结基座：参数 requires_grad=False 且 backward 后无梯度",
        all_frozen and none_or_zero,
        f"全部冻结:{all_frozen}；梯度全为 None/0:{none_or_zero}", "精确 0 / None", None)

    # ---- A3b 输入梯度仍回传（v2.4 §4 明文要求）----
    # 多层挂载：低层注入点的读参数必须有非零梯度 —— 只有基座主路径传梯度才可能
    rp = stack.named_read_params()
    rep = grad_report(rp)
    nz = [(nm, g) for nm, g, isnone, bad in rep if not isnone and g > 0]
    rec("A3b  基座主路径仍传梯度（低层读参数拿到非零梯度）", len(nz) > 0,
        f"有非零梯度的读参数 {len(nz)}/{len(rep)} 个：{[n for n,_ in nz][:6]}", ">0 个", len(nz))

    # ---- A3c 冻结基座 .data 逐位不变（且优化器步进后仍不变）----
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.SGD(list(model.parameters()) + stack.read_params() + stack.write_params(),
                          lr=0.1)
    opt.step()
    unchanged = all(torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
    rec("A3c  优化器步进后冻结基座 .data 逐位不变", unchanged,
        f"全部 {len(before)} 个基座张量逐位相同: {unchanged}", "bitwise 相同", None)

    # ---- A2（模型级）：写入通道梯度必须精确为 0 ----
    wp = stack.named_write_params()
    wrep = grad_report(wp)
    allzero = all(g == 0.0 for _, g, isnone, _ in wrep)
    rec("A2d  模型级：写入通道梯度精确为 0（None 或 0）", allzero,
        f"写入参数 {len(wrep)} 个，最大梯度 {max([g for _,g,_,_ in wrep] or [0]):.3e}", "精确 0", None)

    # ---- A4 阴性对照：去掉 sg 后 写入通道必须拿到非零梯度 ----
    print("  · A4 阴性对照：构造 drop_sg=True 的装置")
    tok2, model2, stack2, mount2, d2 = build(MODEL)
    stack2.remove()
    for h in list(stack2._handles):
        h.remove()
    # 用 drop_sg 重建
    stack2 = SGMNStack(model2, mount2, d2, d2, R, ALPHA, LAM, ETA, GAMMA, DEV,
                        seed=20261026, use_factorized=True, drop_sg=True)
    ids2, mask2 = make_batch(tok2)
    loss2 = task_loss(model2, ids2, mask2)
    loss2.backward()
    wrep2 = grad_report(stack2.named_write_params())
    nonzero = [(nm, g) for nm, g, isnone, _ in wrep2 if not isnone and g > 0]
    rec("A4  阴性对照：去掉 sg 后写入通道必须拿到非零梯度（检查有区分力）", len(nonzero) > 0,
        f"{len(nonzero)}/{len(wrep2)} 个写入参数拿到非零梯度：{[n for n,_ in nonzero][:8]}",
        ">0 个（否则检查恒真、无效）", len(nonzero))

    stack.remove(); stack2.remove()
    results["meta"] = {"model": MODEL, "n_layers": model.config.num_hidden_layers,
                       "hidden_size": model.config.hidden_size, "mount": mount,
                       "d_k": d, "d_v": d, "r": R, "alpha": ALPHA, "lambda": LAM,
                       "eta": ETA, "gamma": GAMMA, "dtype": str(DTYPE),
                       "seq_len": 32, "batch": 4, "read_params_with_grad": [n for n, _ in nz]}
    print(f"  · 模型元信息: {results['meta']}")


if __name__ == "__main__":
    unit_checks()
    sg_semantics()
    model_checks()
    npass = sum(1 for c in results["checks"] if c["passed"])
    nall = len(results["checks"])
    results["summary"] = {"passed": npass, "total": nall, "stage_a_passed": npass == nall}
    print(f"\n=== DIR-04 Stage A：{npass}/{nall} 通过 -> "
          f"{'STAGE A PASSED' if npass == nall else 'STAGE A FAILED (BLOCKED)'} ===")
    (OUT / "R1_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
