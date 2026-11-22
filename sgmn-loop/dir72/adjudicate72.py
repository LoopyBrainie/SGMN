"""DIR-72 裁决（判据字面照卡 rev 1.0 §2；不改阈值）。
① 五格 |ηmeas/ηpred_factor − 1| ≤ 0.30 全部
② 矩阵空间比值 ∈[0.5,2] 的格数 ≤ 1
③ C_pred(γ)=η_pred·γ² 峰值在 γ=10（C(10)>C(3) 且 C(10)>C(30)）
裁决：① PASS ⇒ SUCCEEDED；① FAIL 但 ② 或 ③ PASS ⇒ PARTIAL；全 FAIL ⇒ FALSIFIED。
"""
import json, glob, os, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
cells = []
for f in sorted(glob.glob("/home/ec2-user/sgmn-loop/dir72/out/grid/spec_*.json")):
    cells.append(json.load(open(f)))
cells.sort(key=lambda d: (d["mode"], d["gamma"]))
c1 = all(abs(c["ratio_factor"] - 1.0) <= 0.30 for c in cells) and len(cells) == 5
n_mat_in = sum(1 for c in cells if 0.5 <= c["ratio_matrix"] <= 2.0)
c2 = n_mat_in <= 1
tg = {c["gamma"]: c["eta_pred_factor"] * c["gamma"] ** 2 for c in cells if c["mode"] == "target"}
c3 = bool(tg and tg[10] > tg[3] and tg[10] > tg[30])
tg_m = {c["gamma"]: c["eta_meas"] * c["gamma"] ** 2 for c in cells if c["mode"] == "target"}
verdict = "SUCCEEDED" if c1 else ("PARTIAL" if (c2 or c3) else "FALSIFIED")
print("%-6s %-4s %11s %11s %8s %11s %8s %s" % ("mode", "γ", "η_meas", "η_pred_fac", "ratio", "η_pred_mat", "ratio", "verify"))
for c in cells:
    print("%-6s %-4g %11.6g %11.6g %8.4f %11.6g %8.4f %s"
          % (c["mode"], c["gamma"], c["eta_meas"], c["eta_pred_factor"], c["ratio_factor"],
             c["eta_pred_matrix"], c["ratio_matrix"], c["verify"]["ok"]))
print("\n① 五格全部 |ratio−1|≤0.30 : %s  (通过 %d/5)" % (c1, sum(1 for c in cells if abs(c["ratio_factor"] - 1) <= 0.30)))
print("② 矩阵空间 ∈[0.5,2] 格数 = %d ≤1 : %s" % (n_mat_in, c2))
print("③ C_pred 峰值在 γ=10 : %s   C_pred=%s   C_meas=%s" % (c3, {k: round(v, 4) for k, v in tg.items()}, {k: round(v, 4) for k, v in tg_m.items()}))
print("\nVERDICT", verdict)
print("\n绑定模（因子空间最小的 2|a|/|λ|²）:")
for c in cells:
    print("  %-6s γ=%-3g lmax_decay=%.4f  zeros=%d  top1=%s" % (c["mode"], c["gamma"], c["lmax_decay_factor"], c["zeros"], c["top5"][0] if c["top5"] else None))
print("\n因子/矩阵 预言比（因子空间的放大倍数）:", {("%s_%g" % (c["mode"], c["gamma"])): round(c["eta_pred_factor"] / c["eta_pred_matrix"], 4) for c in cells})
json.dump(dict(cells=cells, c1=c1, c2=c2, c3=c3, n_mat_in=n_mat_in, C_pred=tg, C_meas=tg_m, verdict=verdict),
          open("/home/ec2-user/sgmn-loop/dir72/out/verdict72.json", "w"), indent=1)
