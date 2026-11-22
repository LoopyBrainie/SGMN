"""DIR-84 正式轮：绝对口径吸引域半径 R_abs(γ) 的二分测量。

12 格 = γ ∈ {3,10,20,30} × δ̂ ∈ {7777,7778,7779}。
窗口 [0.02, 2.0]、rel = 3%、N = 2e5 —— 全部按卡 rev 1.1 §3/§4 冻结，此处不得改。
每格完成即写 partial JSON，便于跨 cycle 监视。
"""
import sys, os, json, time
for q in ("dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81", "dir82", "dir83", "dir84"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np
import dev83, dev84

PART = os.path.join(dev84.OUT84, "formal84_partial.json")
cells = {}
if os.path.exists(PART):                      # 断点续跑：已完成格不重算
    cells = json.load(open(PART))
    print("续跑：已有 %d 格" % len(cells))

t0 = time.time()
for g in dev84.GAM:
    eta = dev83.eta_mid(g)
    for sd in dev84.SEEDS:
        key = "g%g_s%d" % (g, sd)
        if key in cells:
            continue
        bs = dev84.bisect_radius(g, eta, seed=sd)
        cells[key] = {k: v for k, v in bs.items()}
        json.dump(cells, open(PART, "w"), indent=1)
        print("  %-10s R=%-10s bracket=%-28s mono=%-5s bound=%-4s evals=%d  [%.0fs]" %
              (key, ("%.6f" % bs["R"]) if bs["R"] else "None",
               str(bs["bracket"]), bs["mono_ok"], bs["bound_hit"], bs["evals"],
               time.time() - t0), flush=True)

# ---------------- 判据 ----------------
rows = {}
for g in dev84.GAM:
    per = [cells["g%g_s%d" % (g, sd)] for sd in dev84.SEEDS]
    Rs = [c["R"] for c in per]
    rows[g] = dict(
        g=g, eta=dev83.eta_mid(g), nv0=dev84.nv0_of(g),
        R_per_seed=Rs, R_mean=(float(np.mean(Rs)) if all(r is not None for r in Rs) else None),
        brackets=[c["bracket"] for c in per], mono=[c["mono_ok"] for c in per],
        bound=[c["bound_hit"] for c in per],
        dir_spread=(float((max(Rs) - min(Rs)) / np.mean(Rs)) if all(r is not None for r in Rs) else None),
        traces=[c["trace"] for c in per],
    )
    if rows[g]["R_mean"] is not None:
        rows[g]["R_rel"] = rows[g]["R_mean"] / rows[g]["nv0"]

means = [rows[g]["R_mean"] for g in dev84.GAM]
c1 = None
if all(m is not None for m in means):
    c1 = float((max(means) - min(means)) / np.mean(means))
c2 = all(rows[g]["mono"][i] for g in dev84.GAM for i in range(len(dev84.SEEDS)))
c3 = sum(1 for g in dev84.GAM for b in rows[g]["bound"] if b is not None)
anc = json.load(open(os.path.join(dev84.OUT84, "anchors84.json")))
an = [c for c in anc["checks"] if c["what"] == "absolute==relative"]
c4 = len(an) == 8 and all(c["same_finite"] and c["same_t_nan"] for c in an)

crit = {
    "c1_range_20pct": dict(value=c1, threshold=0.20, ok=(c1 is not None and c1 <= 0.20)),
    "c2_mono_all": dict(value=c2, ok=bool(c2)),
    "c3_no_bound_hit": dict(value=c3, ok=(c3 == 0)),
    "c4_anchors_8of8": dict(value=c4, ok=bool(c4)),
}

# 次级读数（非门禁）
sec = {}
if all("R_rel" in rows[g] for g in dev84.GAM):
    rr = [rows[g]["R_rel"] for g in dev84.GAM]
    sec["R_rel"] = dict(zip([str(g) for g in dev84.GAM], rr))
    sec["R_rel_ratio_3_over_30"] = rr[0] / rr[-1]
    sec["P3_predicted_7_08_in_3_15"] = bool(3.0 <= rr[0] / rr[-1] <= 15.0)
spread = [rows[g]["dir_spread"] for g in dev84.GAM if rows[g]["dir_spread"] is not None]
sec["dir_spread"] = dict(zip([str(g) for g in dev84.GAM], spread))
sec["P4_all_le_10pct"] = bool(all(x <= 0.10 for x in spread)) if spread else None

h84 = bool(crit["c1_range_20pct"]["ok"] and crit["c2_mono_all"]["ok"]
           and crit["c3_no_bound_hit"]["ok"] and crit["c4_anchors_8of8"]["ok"])
out = dict(rows={str(g): rows[g] for g in dev84.GAM}, crit=crit, sec=sec,
           h84_confirmed=h84, elapsed=time.time() - t0,
           card="revolve/DIR-84/DIRECTION_rev84_v11.md",
           card_md5="7e42399158d32451934b41665aa9fa8c")
json.dump(out, open(os.path.join(dev84.OUT84, "formal84.json"), "w"), indent=1)
print("\n=== 判据 ===")
for k, v in crit.items():
    print("  %-20s value=%s  ok=%s" % (k, v["value"], v["ok"]))
print("R_mean:", [rows[g]["R_mean"] for g in dev84.GAM])
print("R_rel :", sec.get("R_rel"), "ratio=", sec.get("R_rel_ratio_3_over_30"))
print("dir_spread:", sec.get("dir_spread"))
print("H84 confirmed =", h84, "| elapsed %.0f s" % out["elapsed"])