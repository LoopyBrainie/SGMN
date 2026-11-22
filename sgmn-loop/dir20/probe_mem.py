#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-20 · 显存剖面探针：在**训练机**上跑（不污染本地环境）。

目的：定位 DIR-20 smoke 轮 A20-8 处 CUDA OOM 的显存占用来源。
"""
import gc
import sys

import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir20")
from sgmn_module_hebb_v4 import SGMNStack as SGS  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda"
MOUNT = [8, 12, 16]


def mem(tag):
    print("%-42s alloc=%8.1f MiB  reserved=%8.1f MiB  n_alloc=%d"
          % (tag, torch.cuda.memory_allocated() / 2**20,
             torch.cuda.memory_reserved() / 2**20,
             torch.cuda.memory_allocated() // 2**20), flush=True)


def main():
    from transformers import AutoModelForCausalLM
    mem("0 起点")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    mem("1 载入基座（fp32）")
    d = model.config.hidden_size

    stacks = []
    for i in range(12):
        st = SGS(model, MOUNT, d, d, 8, 1.0, 0.05, 1e-1, 0.05, DEV, seed=20261028,
                 use_factorized=False, no_write=False, perpos=True,
                 mem_kind="hebbian_full", n_entries=256, key_mode="ortho")
        stacks.append(st)
    mem("2 建 12 个装置（同时存活）")

    ids = torch.randint(100, 5000, (1, 36), device=DEV)
    with torch.no_grad():
        stacks[0].model(input_ids=ids)
    mem("3 一次前向（36 token）")

    snap = stacks[0].snapshot_all()
    mem("4 snapshot_all")
    stacks[0].restore_all(snap)
    mem("5 restore_all")

    state = stacks[0].export_state(include_read_map=True, include_addr=True)
    mem("6 export_state（CPU 副本）")
    del state
    gc.collect()
    torch.cuda.empty_cache()
    mem("7 释放 export")

    for st in stacks:
        st.remove()
    mem("8 全部 remove()（钩子）")
    stacks.clear()
    gc.collect()
    torch.cuda.empty_cache()
    mem("9 释放装置对象")

    m2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    mem("10 再载入一个基座")
    del m2
    gc.collect()
    torch.cuda.empty_cache()
    mem("11 释放")


if __name__ == "__main__":
    main()
