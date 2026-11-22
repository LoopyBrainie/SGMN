#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""补充：核对影响锥各 d 上究竟是哪个 token（解释 d=-5 的 0.40 是否为 token 身份混淆）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import build_facts, build_chunks
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
facts = build_facts(tok)
chunks, owner = build_chunks(tok, facts)
DS = list(range(-8, 4))
rows = {d: [] for d in DS}
for f in facts:
    enc = tok(f["sent"] + " ", add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ch = len(f["prompt"]) + 1
    j = next(t for t, (s0, e0) in enumerate(offs) if s0 <= ch < e0)
    per = len(ids)
    for d in DS:
        rows[d].append(tok.decode([ids[(j + d) % per]]))
print("值 token =", repr(tok.decode([ids[j]])), " period =", per)
for d in DS:
    uniq = sorted(set(rows[d]))
    print(f"  d={d:+d}: 不同 token 数={len(uniq):2d}  {uniq[:6]}")
