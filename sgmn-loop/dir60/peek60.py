import glob, json
ks = None
for p in sorted(glob.glob("/home/ec2-user/sgmn-loop/dir60/out/PROBE60_c*_*.json")):
    d = json.load(open(p))
    if ks is None:
        ks = sorted(d.keys()); print("keys:", ks)
    g = d.get
    print(f'{d["j"]} {d["arm"]:7s} L={d["mean_L"]:8.4f} sd={d["sd_L"]:6.3f} '
          f'cos={g("cos_q_k", float("nan")):+.3f} dWq1={g("dWq1_from_init", float("nan")):.3f} '
          f'm={g("m_abs", float("nan")):.3f} rank={g("pv_rank")} fro={g("pv_fro", float("nan")):.3f} '
          f'gram={g("pv_gram_offdiag_max", float("nan")):.3f} rows={g("pv_row_norm_mean", float("nan")):.3f}')
