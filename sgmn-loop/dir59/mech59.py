#!/opt/pytorch/bin/python
r"""DIR-59 · 机制读数汇总（`VERDICT` 的机制段直接引用本脚本输出）

对每格四臂打印：`L`、`|q|`、`|y|`、`m̄`、`Wmax`、`cos_q_k`、`dWq1`；`AA` 臂另打印 `rc`。
**只读 `dir59/out/DIR59r10_c{j}_{arm}.json`，不重算任何判据**（判据归 `adjudicate59.py`）。
"""
import json
import pathlib
import statistics as st

D = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
ARMS = ("AA", "AB", "BA", "BB")
print("j  arm      L      |q|    |y|    m̄     Wmax  cos_q_k  dWq1   dWq1̄_co  dWq1̄_mis")
for j in (1, 2, 3, 4, 5):
    recs = {}
    for arm in ARMS:
        p = D / ("DIR59r10_c%d_%s.json" % (j, arm))
        if p.exists():
            recs[arm] = json.loads(p.read_text(encoding="utf-8"))
    if len(recs) < 4:
        print("%d  （缺臂 %d/4）" % (j, len(recs)))
        continue
    co = st.fmean([recs[a]["dWq1_from_init"] for a in ("AA", "BB")])
    mis = st.fmean([recs[a]["dWq1_from_init"] for a in ("AB", "BA")])
    for arm in ARMS:
        r = recs[arm]
        print("%d  %-4s %7.4f %6.3f %6.3f %6.3f %6.3f %+7.3f %6.3f %8.3f %8.3f"
              % (j, arm, r["mean_L"], r["q_norm"], r["y_norm"], r["m_abs"], r["Wmax"],
                 r["cos_q_k"], r["dWq1_from_init"], co, mis))
    rc = recs["AA"].get("rc")
    if rc:
        print("   rc(c%d)=%.4f  ⇒ rc−AA=%.4f  |  L_mis=%.4f  (L_mis−L_co=%.4f)"
              % (j, rc["mean_L"], rc["mean_L"] - recs["AA"]["mean_L"],
                 st.fmean([recs["AB"]["mean_L"], recs["BA"]["mean_L"]]),
                 st.fmean([recs["AB"]["mean_L"], recs["BA"]["mean_L"]])
                 - st.fmean([recs["AA"]["mean_L"], recs["BB"]["mean_L"]])))
