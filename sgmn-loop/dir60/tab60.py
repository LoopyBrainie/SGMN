import glob, json
rows = []
for p in sorted(glob.glob("/home/ec2-user/sgmn-loop/dir60/out/PROBE60_c*_*.json")):
    d = json.load(open(p)); rows.append(d)
rows.sort(key=lambda d: (d["j"], d["arm"]))
hdr = f'{"格":>2} {"臂":7s} {"L":>8} {"sd":>6} {"cos":>7} {"dWq1":>6} {"m":>5} {"rank":>4} {"fro":>6} {"gram":>6} {"rows":>6} {"sha256_pv":>10} {"secs":>5}'
print(hdr)
for d in rows:
    print(f'{d["j"]:>2} {d["arm"]:7s} {d["mean_L"]:8.4f} {d["sd_L"]:6.3f} {d["cos_q_k"]:+7.3f} '
          f'{d["dWq1_from_init"]:6.2f} {d["m_abs"]:5.3f} {d["pv_rank"]:4d} {d["pv_fro"]:6.3f} '
          f'{d["pv_gram_offdiag_max"]:6.3f} {d["pv_row_norm_mean"]:6.3f} {d["sha256_pv"][:10]:>10} {d["secs"]:5.0f}')
print("pk1_seed:", {(d["j"], d["arm"]): d["pk1_seed"] for d in rows if d["arm"] == "RAND"})
print("pv_seed:", {(d["j"], d["arm"]): d["pv_seed"] for d in rows if d["arm"] == "RAND"})
print("protocol:", rows[0]["protocol"])
