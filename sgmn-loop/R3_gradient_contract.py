#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-04 · R3 —— ②③④ 梯度契约检查（真实预训练 LLM）

卡片 §6.1：
  ② 读取参数的非零有限梯度
  ③ 任务梯度不进入写入通道
  ④ 冻结基座不发生意外更新
另加【区分力检查】：优化器步进后【读参数确实变了】——否则 ④ 会因优化器根本没动而恒真。
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
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ALPHA, LAM, ETA, GAMMA = 1.0, 0.05, 0.01, 0.05
R = 8
NONZERO_TOL = 1e-12

res = {"dir": "DIR-04", "round": "R3", "stage": "B", "model": MODEL, "checks": []}


def rec(name, passed, detail, tol=None, value=None):
    res["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                          "tolerance": tol, "value": value})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def build(model_name=MODEL, seed=20261026, use_factorized=True, drop_sg=False, gamma=GAMMA):
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


def main():
    print("=== DIR-04 R3：②③④ 梯度契约检查 ===")
    tok, model, stack, mount, d = build()
    ids, mask = make_batch(tok)

    base_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    read_before = {n: t.detach().clone() for n, t in stack.named_read_params()}
    write_before = {n: t.detach().clone() for n, t in stack.named_write_params()}

    loss = task_loss(model, ids, mask)
    loss.backward()

    # ---------------- ② 读取参数：非零且有限 ----------------
    rrep = grad_report(stack.named_read_params())
    bad_nonzero = [(n, g) for n, g, isnone, _ in rrep if isnone or g <= NONZERO_TOL]
    bad_finite = [(n, g) for n, g, _, bad in rrep if bad]
    rec("②a 读取参数梯度非零（每个挂载层的 Wq/Wo/gamma）", len(bad_nonzero) == 0,
        f"{len(rrep)} 个读参数全部非零；最小梯度 {min([g for _,g,_,_ in rrep]):.3e}；"
        f"异常 {[b[0] for b in bad_nonzero][:5]}",
        f"全部 > {NONZERO_TOL}", min([g for _, g, _, _ in rrep] or [0]))
    rec("②b 读取参数梯度有限（无 inf/nan）", len(bad_finite) == 0,
        f"非有限梯度的读参数：{[b[0] for b in bad_finite] or '无'}", "全部有限", None)
    per_layer = {}
    for n, g, isnone, _ in rrep:
        per_layer.setdefault(n.split(".")[0], []).append(g)
    print("  · 逐层读参数梯度范数：")
    for li, gs in per_layer.items():
        print(f"      {li}: " + "  ".join(f"{x:.3e}" for x in gs))

    # ---------------- ③ 写入通道：梯度精确为 0 ----------------
    wrep = grad_report(stack.named_write_params())
    nonzero_w = [(n, g) for n, g, isnone, _ in wrep if not isnone and g != 0.0]
    rec("③a 任务梯度不进入写入通道（梯度为 None 或精确 0）", len(nonzero_w) == 0,
        f"{len(wrep)} 个写入参数，最大梯度 {max([g for _,g,_,_ in wrep] or [0]):.3e}；"
        f"非零者 {[b[0] for b in nonzero_w][:5] or '无'}", "精确 0 / None", None)

    # ---------------- ④ 冻结基座：无梯度 + .data 逐位不变 ----------------
    base_grads = [(n, p.grad) for n, p in model.named_parameters()]
    bad_base = [n for n, g in base_grads if g is not None and float(g.abs().max()) != 0.0]
    rec("④a 冻结基座参数无梯度（None 或精确 0）", len(bad_base) == 0,
        f"{len(base_grads)} 个基座参数，非零梯度者 {bad_base[:5] or '无'}", "None 或精确 0", None)

    # 优化器步进：读参数应变，基座与写入通道必须逐位不变
    opt = torch.optim.SGD([p for p in model.parameters()]
                          + stack.read_params() + stack.write_params(), lr=0.1)
    opt.step()
    base_same = all(torch.equal(base_before[n], p.detach()) for n, p in model.named_parameters())
    write_same = all(torch.equal(write_before[n], t.detach()) for n, t in stack.named_write_params())
    read_changed = [n for n, t in stack.named_read_params()
                    if not torch.equal(read_before[n], t.detach())]
    rec("④b 优化器步进后冻结基座 .data 逐位不变", base_same,
        f"全部 {len(base_before)} 个基座张量逐位相同: {base_same}", "bitwise 相同", None)
    rec("④c 写入通道也逐位不变（未被任务梯度驱动）", write_same,
        f"全部 {len(write_before)} 个写入张量逐位相同: {write_same}", "bitwise 相同", None)
    rec("④d【区分力】读参数在同一次优化器步进后确实改变", len(read_changed) > 0,
        f"{len(read_changed)}/{len(read_before)} 个读参数发生改变（说明优化器确实在动，"
        f"故 ④ 的'不变'不是因优化器空转）", ">0 个", len(read_changed))

    # ---------------- H2：②与③ 相容性 ----------------
    rec("H2  ②与③ 相容：读参数拿到非零有限梯度，同时写入通道精确为 0",
        len(bad_nonzero) == 0 and len(bad_finite) == 0 and len(nonzero_w) == 0,
        f"② 通过:{len(bad_nonzero)==0 and len(bad_finite)==0}；③ 通过:{len(nonzero_w)==0}",
        "两者同时成立", None)

    res["read_grad_report"] = rrep
    res["write_grad_report"] = wrep
    res["meta"] = {"model": MODEL, "n_layers": model.config.num_hidden_layers,
                   "hidden_size": model.config.hidden_size, "mount": mount, "r": R,
                   "loss": float(loss), "read_changed": read_changed}
    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (OUT / "R3_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\n=== DIR-04 R3：{npass}/{len(res['checks'])} 通过 ===")


if __name__ == "__main__":
    main()
