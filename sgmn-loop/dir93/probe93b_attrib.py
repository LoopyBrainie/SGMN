#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-93 · `S90` 漂移的**归因**诊断（**不是判据**）。

`probe93_s90state.py` 已证：写后 `S90 = 0.19993`，读侧训练后 `S90 = 1.7312`，而 **`W` 逐位未动**（`dW = 0.0`）。
`read_params(st) = [Wq, Wo, gamma]` ⇒ 训练只可能通过这三者改变 `S90`。
本探针逐个**还原**它们，把 1.73 → 0.20 的位移**归因到具体参数**（而不是靠推理）。

读数：`s90_b`（全训练）→ 逐个还原 4 组 → `s90_restore_all`（全还原，须回到 0.19993）。
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dir90"))
import R1_task90 as R                            # noqa: E402

tok, model = R.model_and_tok()
facts = R.build_facts(tok)
chunks, owner = R.build_chunks(tok, facts)
train_keys = list(range(R.N_TRAIN_KEYS))

st = R.make_stack(gamma=R.GAMMA0)
R.write_phase(model, st, chunks)

snap = {(li, nm): t.detach().clone()
        for li, m in st.mems.items()
        for nm, t in (("Wq", m.Wq), ("Wo", m.Wo), ("gamma", m.gamma))}

out = {}
out["s90_write"] = R.storage_diag(model, st, facts, owner, chunks)[0]

R.train_read_task(model, st, tok, facts, train_keys, random.Random(1000 + 20261028 % 100))
out["s90_after_train"] = R.storage_diag(model, st, facts, owner, chunks)[0]
out["param_drift"] = {f"{li}_{nm}": float((t.detach() - snap[(li, nm)]).abs().max())
                      for li, m in st.mems.items()
                      for nm, t in (("Wq", m.Wq), ("Wo", m.Wo), ("gamma", m.gamma))}


def restore(which):
    """把 `which` 中的参数名还原到训练前；返回各层 max|Δ| 以备核对。"""
    for li, m in st.mems.items():
        for nm, t in (("Wq", m.Wq), ("Wo", m.Wo), ("gamma", m.gamma)):
            if nm in which:
                with __import__("torch").no_grad():
                    t.copy_(snap[(li, nm)])


def measure(tag):
    out[tag] = R.storage_diag(model, st, facts, owner, chunks)[0]


# 单独还原（其余保持训练后的值）
for nm in ("Wq", "Wo", "gamma"):
    kept = {k: v.detach().clone() for k, v in
            {(li, n2): t for li, m in st.mems.items()
             for n2, t in (("Wq", m.Wq), ("Wo", m.Wo), ("gamma", m.gamma))}.items()}
    restore((nm,))
    measure(f"s90_only_{nm}_restored")
    for li, m in st.mems.items():                # 放回训练后的值
        for n2, t in (("Wq", m.Wq), ("Wo", m.Wo), ("gamma", m.gamma)):
            with __import__("torch").no_grad():
                t.copy_(kept[(li, n2)])

restore(("Wq", "Wo", "gamma"))
measure("s90_restore_all")
st.remove()

p = pathlib.Path(__file__).resolve().parent / "out"
p.mkdir(exist_ok=True)
txt = json.dumps(out, ensure_ascii=False, indent=1)
(p / "probe_s90attrib.json").write_text(txt, encoding="utf-8")
print(txt)