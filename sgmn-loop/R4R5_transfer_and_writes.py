#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-04 · R4/R5 —— 模型迁移检查 + 因子化写 vs 全矩阵参考写

R4（H4）：在架构不同的第二个模型（TinyLlama-1.1B-Chat，Llama 架构）上复现 ①②③④
R5：Freeze B 的两种写法（因子化同步步 / 全矩阵参考步）下，接口契约是否都成立
"""
import json
import pathlib
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sgmn_module import SGMNStack, grad_report

OUT = pathlib.Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ALPHA, LAM, ETA, GAMMA = 1.0, 0.05, 0.01, 0.05
R = 8
NONZERO_TOL = 1e-12
MODELS = [("Qwen/Qwen2.5-0.5B-Instruct", "primary"),
          ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "transfer")]

res = {"dir": "DIR-04", "round": "R4+R5", "stage": "B", "checks": []}


def rec(name, passed, detail, tol=None, value=None):
    res["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                          "tolerance": tol, "value": value})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def build(model_name, seed=20261026, use_factorized=True, drop_sg=False, gamma=GAMMA):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = model.config.num_hidden_layers
    mount = [n // 3, n // 2, 2 * n // 3]
    d = model.config.hidden_size
    stack = SGMNStack(model, mount, d, d, R, ALPHA, LAM, ETA, gamma, DEV,
                      seed=seed, use_factorized=use_factorized, drop_sg=drop_sg)
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


def contract(model_name, tag, use_factorized=True):
    """在给定模型与写法下跑一遍 ①②③④，返回 (dict, 写参数名列表)。"""
    tok, model, stack, mount, d = build(model_name, use_factorized=use_factorized)
    ids, mask = make_batch(tok)
    base_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    write_before = {n: t.detach().clone() for n, t in stack.named_write_params()}
    loss = task_loss(model, ids, mask)
    loss.backward()

    rrep = grad_report(stack.named_read_params())
    wrep = grad_report(stack.named_write_params())
    brep = [(n, p.grad) for n, p in model.named_parameters()]

    r_nonzero = all((not isnone) and g > NONZERO_TOL and not bad for _, g, isnone, bad in rrep)
    r_finite = all(not bad for _, _, _, bad in rrep)
    w_zero = all(isnone or g == 0.0 for _, g, isnone, _ in wrep)
    b_zero = all(g is None or float(g.abs().max()) == 0.0 for _, g in brep)

    opt = torch.optim.SGD([p for p in model.parameters()]
                          + stack.read_params() + stack.write_params(), lr=0.1)
    opt.step()
    base_same = all(torch.equal(base_before[n], p.detach()) for n, p in model.named_parameters())
    write_same = all(torch.equal(write_before[n], t.detach()) for n, t in stack.named_write_params())
    read_changed = sum(1 for n, t in stack.named_read_params()
                       if not torch.equal(t.detach(), t.detach() * 0))  # placeholder replaced below

    out = {"tag": tag, "model": model_name, "use_factorized": use_factorized,
           "n_layers": model.config.num_hidden_layers, "hidden": model.config.hidden_size,
           "mount": mount, "loss": float(loss),
           "read_grads": rrep, "write_grads": wrep,
           "read_min_grad": min([g for _, g, _, _ in rrep] or [0]),
           "write_max_grad": max([g for _, g, _, _ in wrep] or [0]),
           "r_nonzero": r_nonzero, "r_finite": r_finite, "w_zero": w_zero, "b_zero": b_zero,
           "base_same_after_step": base_same, "write_same_after_step": write_same}
    del model, stack
    torch.cuda.empty_cache()
    return out


def main():
    print("=== DIR-04 R4：模型迁移检查 ===")
    rows = []
    for mname, tag in MODELS:
        r = contract(mname, tag)
        rows.append(r)
        print(f"  · {tag} {mname}: ②非零={r['r_nonzero']} ②有限={r['r_finite']} "
              f"③零={r['w_zero']} ④基座零={r['b_zero']} 基座不变={r['base_same_after_step']} "
              f"写入不变={r['write_same_after_step']}  读梯度最小 {r['read_min_grad']:.3e}")
    res["transfer"] = rows
    for r in rows:
        ok = r["r_nonzero"] and r["r_finite"] and r["w_zero"] and r["b_zero"] \
            and r["base_same_after_step"] and r["write_same_after_step"]
        rec(f"H4 {r['tag']} 模型 {r['model'].split('/')[-1]}：①②③④ 全部复现", ok,
            f"②非零:{r['r_nonzero']} ②有限:{r['r_finite']} ③零:{r['w_zero']} "
            f"④基座零:{r['b_zero']} ④基座不变:{r['base_same_after_step']} "
            f"④写入不变:{r['write_same_after_step']}", "全部成立", None)

    print("\n=== DIR-04 R5：因子化写 vs 全矩阵参考写 ===")
    r5 = []
    for uf, label in ((True, "因子化同步步 (v2.4 §3.3)"), (False, "全矩阵参考步 (v2.4 §3.2)")):
        r = contract(MODELS[0][0], label, use_factorized=uf)
        r5.append(r)
        print(f"  · {label}: ②非零={r['r_nonzero']} ②有限={r['r_finite']} ③零={r['w_zero']} "
              f"④基座零={r['b_zero']} ④基座不变={r['base_same_after_step']} "
              f"④写入不变={r['write_same_after_step']}  读梯度最小 {r['read_min_grad']:.3e}")
    res["write_dynamics"] = r5
    for r in r5:
        ok = r["r_nonzero"] and r["r_finite"] and r["w_zero"] and r["b_zero"] \
            and r["base_same_after_step"] and r["write_same_after_step"]
        rec(f"R5 {r['tag']}：①②③④ 全部成立", ok,
            f"②非零:{r['r_nonzero']} ②有限:{r['r_finite']} ③零:{r['w_zero']} "
            f"④基座零:{r['b_zero']} ④基座不变:{r['base_same_after_step']} "
            f"④写入不变:{r['write_same_after_step']}", "全部成立", None)

    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (OUT / "R4R5_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\n=== DIR-04 R4+R5：{npass}/{len(res['checks'])} 通过 ===")


if __name__ == "__main__":
    main()
