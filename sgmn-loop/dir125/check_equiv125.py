#!/usr/bin/env python3
"""`DIR-125` 等价性闸门（`DEF-106-15` 同族）：驱动运行期间改过 `R1_task125.py`
⇒ 必须**逐位**证明**测量路径**未被改动。

做法：把某个**已完成**的格子在 scratch 目录**重跑一次**，与已落盘的那份
逐位比对 `acc_by_mult` 与 `mech_by_mult[*].pick_argmin`。一致 ⇒ 改动只在
`stage_formal` 一侧，归档读数可信；不一致 ⇒ 该格（以及它前后同版本的格）必须重跑。

用法：/opt/pytorch/bin/python dir125/check_equiv125.py [--dose D] [--seed S]
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dir119"))

import R1_task125 as R                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dose", type=int, default=None)
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()

cells_dir = HERE / "out" / "cells"
avail = sorted(cells_dir.glob("COLOCATED_*.json"))
if not avail:
    print("没有已完成的格，跳过等价性检查")
    sys.exit(0)
# 默认挑**最后完成**的那格（最接近当前文件版本）
def _mt(f):
    return f.stat().st_mtime


pick = None
if a.dose is not None and a.seed is not None:
    pick = cells_dir / ("COLOCATED_%d_%d.json" % (a.dose, a.seed))
else:
    pick = max(avail, key=_mt)

d = json.loads(pick.read_text(encoding="utf-8"))
dose, seed = int(d["dose"]), int(d["seed"])
print("等价性检查：重跑 %s（dose=%d, seed=%d）" % (pick.name, dose, seed))

scratch = HERE / "out" / "det" / ("equiv_%d_%d" % (dose, seed))
scratch.mkdir(parents=True, exist_ok=True)
tok, model = R.D90.model_and_tok()
facts, parts, csf, IT = R.P118.ground(tok, R.ARM)
tv, tr, ev = IT["tv"], IT["tr"], IT["ev"]
ci, cm, ct = R.build_c2(tok, facts, ev)
fresh = R.run_cell(seed, dose, tok, model, facts, tr, ev,
                   IT["items_ev"], IT["meta_ev"], tv, ci, cm, ct,
                   write_cell=False, scratch_out=scratch)

same_acc = (fresh["acc_by_mult"] == d["acc_by_mult"])
mults = sorted(fresh["mech_by_mult"], key=float)
diff = [k for k in mults
        if (fresh["mech_by_mult"].get(k) or {}).get("pick_argmin")
        != ((d["mech_by_mult"].get(k)) or {}).get("pick_argmin")]
same_pick = not diff
c2s = ("10000" in (fresh.get("c2") or {})) and \
      all(str(r) in fresh["c2"]["10000"] for r in R.C2_RNGS) == \
      all(str(r) in (d.get("c2") or {}).get("10000", {}) for r in R.C2_RNGS)

out = {"dose": dose, "seed": seed, "file": pick.name,
       "acc_by_mult_same": same_acc, "pick_argmin_same": same_pick,
       "differing_mults": diff, "c2_presence_same": c2s,
       "ok": bool(same_acc and same_pick)}
(R.OUT / "equiv125.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
print("  acc_by_mult 逐位相同 =", same_acc)
print("  9 个 mult 的 pick_argmin 逐位相同 =", same_pick, ("（不同：%r）" % diff) if diff else "")
print("  c2 出现情况一致 =", c2s)
print("  ⇒ 等价性 %s" % ("通过" if out["ok"] else "**未通过**：该格必须重跑"))
sys.exit(0 if out["ok"] else 1)
