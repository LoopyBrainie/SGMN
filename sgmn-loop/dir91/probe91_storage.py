#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.9 附二：**存储 rel.err 的可复现性裁决**（诊断）

矛盾：`D-110`/rev1.7 报装置口径存储 rel.err **0.147/0.191**（等价 cos≈0.98），
      `probe91_write.py` 用 hook 抓到的真写入对只得到 cos **0.067/0.014/-0.008**（等价 rel.err≈1.0）。
二者不可共存，必须判定谁对。本探针在**同一进程、同一装置状态**下并排报：

  P1  hook 抓的真写入对 (k_prev, v_in)         → rel.err, cos
  P2  `key_matrices` 口径 (_hs_in[li], Pv)      → rel.err, cos
  P3  Σ 全体写入的**平均** rel.err（每次 write 都记） → 收敛轨迹
  P4  `||k||`, `||v||`, `||Wk||`，检查 lr·alpha·||k||² 是否 ≈1（Hebb 收敛条件）

判读：若 P3 的**末值**小（≈0.19）而 P1 大（≈1.0）⇒ 说明 W 在**后续写入**中把这一对覆盖了，
      「0.98」描述的是**写入瞬间**、不是终态 ⇒ 两者都对，但含义不同，须在文档里分开写。

★ 只做测量。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, N_FACTS, MOUNT, build_chunks, build_facts, dump, key_matrices,
                       make_stack, model_and_tok, write_phase)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05


def cos(a, b):
    a = a.detach().float().reshape(-1); b = b.detach().float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rel(W, k, v):
    return float(((W @ k) - v).norm() / (v.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    res = {"seed": a.seed}

    st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
    HIST = {li: [] for li in MOUNT}
    for li, m in st.mems.items():
        orig = m.write
        def wrap(hs_in, drop_sg=False, li=li, m=m, orig=orig):
            kb = None if m.prev_hs is None else m.prev_hs.reshape(-1, m.prev_hs.shape[-1])
            Wb = None
            if kb is not None:
                Wb = (m.W0 + m.U @ m.V.T).detach().float().clone()
                kt = ((kb @ m.Pk).mean(0)) * m.pk_scale
                vt = (hs_in.reshape(-1, hs_in.shape[-1]) @ m.Pv).mean(0)
            r = orig(hs_in, drop_sg)
            if kb is not None:
                HIST[li].append((float(kt.norm()), float(vt.norm()),
                                 rel(Wb, kt.float(), vt.float()), cos(Wb @ kt.float(), vt.float())))
            return r
        m.write = wrap
    write_phase(model, st, chunks)

    print(f"写入次数/层 = {len(HIST[MOUNT[0]])}", flush=True)
    res["traj"] = {}
    for li in MOUNT:
        h = np.array(HIST[li])
        res["traj"][li] = {"n": len(h), "kn": float(h[:, 0].mean()), "vn": float(h[:, 1].mean()),
                           "rel_first": float(h[0, 2]), "rel_last": float(h[-1, 2]),
                           "cos_first": float(h[0, 3]), "cos_last": float(h[-1, 3]),
                           "rel_min": float(h[:, 2].min()), "rel_mean_last10": float(h[-10:, 2].mean()),
                           "cos_mean_last10": float(h[-10:, 3].mean())}
        t = res["traj"][li]
        print(f" L{li}: ||k||={t['kn']:.1f} ||v||={t['vn']:.1f}", flush=True)
        print(f"   {'写入瞬间':<10} 首 rel={t['rel_first']:.4f} cos={t['cos_first']:+.4f} | "
              f"末 rel={t['rel_last']:.4f} cos={t['cos_last']:+.4f}", flush=True)
        print(f"   {'末10次均值':<10} rel={t['rel_mean_last10']:.4f} cos={t['cos_mean_last10']:+.4f} "
              f"| 全程最小 rel={t['rel_min']:.4f}", flush=True)

    # P2: key_matrices 口径 + 终态 W
    for m in st.mems.values():
        m.no_write = True
    K, V = key_matrices(model, st, facts, chunks, owner)
    res["keymat"] = {}
    for li in MOUNT:
        m = st.mems[li]
        W = (m.W0 + m.U @ m.V.T).detach().float()
        rs = [rel(W, K[li][i], V[li][i]) for i in range(N_FACTS)]
        cs = [cos(W @ K[li][i], V[li][i]) for i in range(N_FACTS)]
        res["keymat"][li] = {"rel_mean": float(np.mean(rs)), "cos_mean": float(np.mean(cs))}
        print(f" [key_matrices 口径] L{li}: rel={np.mean(rs):.4f} cos={np.mean(cs):+.4f}", flush=True)
    # Hebb 收敛条件
    for li in MOUNT:
        p = st.mems[li]
        t = res["traj"][li]
        prod = float(p.lr) * float(p.alpha) * t["kn"] ** 2 if hasattr(p, "lr") and hasattr(p, "alpha") else None
        print(f" [收敛条件] L{li}: lr*alpha*||k||^2 = {prod}", flush=True)
        res["traj"][li]["hebb_step"] = prod
    st.remove()
    dump(res, "probe91_storage.json")
    print("\n★ 诊断完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
