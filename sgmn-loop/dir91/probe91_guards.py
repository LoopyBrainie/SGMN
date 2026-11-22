#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只跑新增的三条 Stage A 守卫（A91-19/20/21），验证它们真的能判（DEF-91-23/24）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (build_chunks, build_facts, dump, make_stack, model_and_tok,
                       stage_a, storage_gamma0, W_of, write_phase)

tok, model = model_and_tok()
facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)

# --- A91-19 复算：两种 W 候选的可分辨性 ---
st = make_stack(no_write=True)
try:
    for li, m in st.mems.items():
        Wf = W_of(m).float(); Wx = (m.W0 + m.U @ m.V.T).detach().float()
        print(f"[A91-19] L{li} use_factorized={m.use_factorized} "
              f"rel_diff={float((Wx-Wf).norm()/(Wf.norm()+1e-30)):.4e}", flush=True)
finally:
    st.remove()

# --- A91-20 复算：write_phase 入口 prev_hs 与写入次数 ---
st = make_stack(no_write=False)
try:
    pre = [li for li, m in st.mems.items() if m.prev_hs is not None]
    write_phase(model, st, chunks)
    nw = [int(m.write_calls) for m in st.mems.values()]
    print(f"[A91-20] pre_nonnull={pre} writes={nw} expect={len(chunks)-1} "
          f"=> {'PASS' if (not pre and all(n==len(chunks)-1 for n in nw)) else 'FAIL'}", flush=True)
finally:
    st.remove()

# --- A91-21 复算：聚合口径 ---
s = storage_gamma0(model, facts, chunks, owner)
print(f"[A91-21] per_layer_ratios={s['ratio_per_layer']}", flush=True)
print(f"[A91-21] median_of_ratios={s['ratio_median_of_layer_ratios']:.3f} "
      f"min={s['ratio_min']:.3f} max={s['ratio_max']:.3f} "
      f"| OLD cross-layer improve_ratio={s['improve_ratio']:.3f}", flush=True)
print(f"[A91-21] written={s['written']}", flush=True)
print(f"[A91-21] written_mean={s['written_mean']}", flush=True)
print(f"[A91-21] baseline={s['baseline']}", flush=True)
print("\n★ 守卫复算完成", flush=True)
