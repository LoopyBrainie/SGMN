#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-105 诊断：**跨格污染定位**（report-only，不进判据）

目的：`A105-1` 在正式轮里失败（seed 7777 lin@600 实测 25/32，隔离跑 28/32）。
      本探针回答一个二选一的问题：**基座模型在格与格之间是否被改动？**

做法：同一进程内跑若干格，每格前后对**基座模型全部参数**做 sha1，
      并把每格读到的 `acc_all32` 一并打印。若基座 sha1 变化 ⇒ 基座被污染；
      若基座不变而 acc 变 ⇒ 污染在 `sgmn` 侧或来自数值非确定性。

用法：/opt/pytorch/bin/python diag105_state.py [--fams lin,gate,mlp1] [--steps 600,3000] [--seeds 20261028,7777]
"""
import argparse
import hashlib
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "dir91"))
sys.path.insert(0, str(_HERE.parent / "dir99"))
sys.path.insert(0, str(_HERE.parent / "dir100"))
sys.path.insert(0, str(_HERE.parent / "dir101"))
sys.path.insert(0, str(_HERE.parent / "dir103"))
sys.path.insert(0, str(_HERE.parent / "dir105"))

import torch                                     # noqa: E402
import R1_task105 as T                           # noqa: E402


def model_hash(model):
    """基座参数 + buffer 的 sha1（逐张量，按名字排序）。"""
    h = hashlib.sha1()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    for name, b in sorted(model.named_buffers()):
        h.update(name.encode())
        h.update(b.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fams", default="lin,gate,mlp1")
    ap.add_argument("--steps", default="600,3000")
    ap.add_argument("--seeds", default="20261028,7777")
    a = ap.parse_args()
    fams = [x for x in a.fams.split(",") if x]
    grids = [int(x) for x in a.steps.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x]

    # ★ 与 R1_task105.main 完全同款的初始化顺序
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    tok, model = T.KP.model_and_tok()
    model.eval()
    facts = T.R91.build_facts(tok)
    chunks, owner = T.R91.build_chunks(tok, facts)

    h0 = model_hash(model)
    print(f"[base] 起始模型 sha1 = {h0}", flush=True)
    print(f"{'#':>2} {'seed':>8} {'fam':>5} {'steps':>5} {'acc':>7} {'base_sha1':>17} {'changed':>7}",
          flush=True)

    i = 0
    for seed in seeds:
        for fam in fams:
            for stp in grids:
                i += 1
                hb = model_hash(model)
                c = T.run_cell(model, tok, facts, seed, fam, stp, "diag", chunks, owner)
                ha = model_hash(model)
                print(f"{i:>2} {seed:>8} {fam:>5} {stp:>5} {c['acc_all32']:>7.4f} "
                      f"{ha:>17} {'★YES' if ha != hb else 'no':>7}", flush=True)
                if ha != hb:
                    print(f"   ★★ 基座在格 {i}（{fam}@{stp} seed={seed}）内被改动："
                          f"{hb} -> {ha}", flush=True)
    print(f"[base] 结束模型 sha1 = {model_hash(model)}   起始 = {h0}   "
          f"{'★★ 基座被污染' if model_hash(model) != h0 else '基座未被改动'}", flush=True)


if __name__ == "__main__":
    main()