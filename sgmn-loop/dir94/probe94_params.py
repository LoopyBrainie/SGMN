#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-94 · 机制补充读数：读出侧参数**训练后**的量级（`γ`/`Wq`/`Wo`/`W`/`W0`）。

**不是判据**（判据已在 `R1_task94.py --stage formal` 冻结并跑完）。
为什么需要：主轮实测 `γ×20` 只把 NLL 动了 ≤0.0045 nats。这有两种可能：
  (a) **记忆通道**本来就没有杠杆（注入相对残差流过小）；
  (b) 读出侧**学会把 `γ` 压小**（"最优策略是忽略记忆"）。
二者对机制的叙述完全不同 ⇒ 必须直接读出训练后的 `γ` 才能区分。
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir93")
import R1_task93 as R                            # noqa: E402

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir94/out")
OUT.mkdir(exist_ok=True)

tok, model = R.model_and_tok()
facts = R.build_facts(tok)
chunks, owner = R.build_chunks(tok, facts)
order = list(range(len(facts)))
random.Random(4242).shuffle(order)
tk = order[:R.N_TRAIN_KEYS]

res = {"gamma0": R.GAMMA0, "per_seed": {}}
for sd in (20261028, 20261104, 7):
    st = R.make_stack(gamma=R.GAMMA0, seed=sd)
    R.write_phase(model, st, chunks)
    pre = {li: float(m.gamma) for li, m in st.mems.items()}
    R.train_read_task(model, st, tok, facts, tk, random.Random(1000 + sd % 100))
    res["per_seed"][str(sd)] = {
        "gamma_pre": pre,
        "layers": {li: {"gamma_post": float(m.gamma),
                        "Wq_fro": float(m.Wq.detach().norm()),
                        "Wo_fro": float(m.Wo.detach().norm()),
                        "W_fro": float(m.Wfull.detach().norm()),
                        "W0_fro": float(m.W0.norm())}
                   for li, m in st.mems.items()},
    }
    st.remove()
    del st

(OUT / "params.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))