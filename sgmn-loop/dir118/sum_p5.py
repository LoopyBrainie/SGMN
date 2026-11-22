import json, pathlib
p = pathlib.Path(__file__).resolve().parent / "out" / "ruler_p118_5_formal.json"
d = json.load(open(p))
print("verdict:", d["verdict"], "| fails:", d["fails"], "| elapsed:", d["elapsed_s"], "s")
s = d["structure"]
print("E_random:", s["E_random"], "| ceiling:", s["ceiling"], "| mult_dist:", s["mult_dist"],
      "| n_facts:", s["n_facts"], "| distinct_vals:", s["distinct_vals"])
print("\n-- controls --")
for L in ("8", "12", "16"):
    x = d["controls"]["layers"][L]
    perms = [q["n_hit_v"] for q in x["perm"]]
    print(f"L{L}: oracle={x['oracle_n_hit_v']}/32  W0={x['W0_n_hit_v']}/32  "
          f"resid={x['oracle_resid']:.3e}  perm={perms}  perm_max={x['perm_max']}  "
          f"sigma1/sigmin={x['sigma1_over_sigma_min']:.1f}")
print("\n-- intact --")
for k in sorted(d["intact"]):
    for L in ("8", "12", "16"):
        x = d["intact"][k]["layers"][L]
        print(f"{k:20s} L{L:2s}: nhit={x['n_hit_v']:2d}/32 strict={x['n_hit_s']:2d} "
              f"L2={x['n_hit_v_L2']:2d} srank={x['stable_rank_dW']:.2f} "
              f"nu={x['nu']:.4f} s_white={x['s_white']:.4f} maxcos={x['maxcos_raw']:.4f}")
w = [d["intact"][k]["layers"][L]["n_hit_v"] for k in d["intact"] if k.startswith("white")
     for L in ("8", "12", "16")]
n = [d["intact"][k]["layers"][L]["n_hit_v"] for k in d["intact"] if k.startswith("native")
     for L in ("8", "12", "16")]
print(f"\nwhite: min={min(w)} max={max(w)} mean={sum(w)/len(w):.2f}  (n={len(w)})")
print(f"native: min={min(n)} max={max(n)} mean={sum(n)/len(n):.2f}  (n={len(n)})")
print("\n预注册分支判定:")
print(f"  white >= 24/32 ? {min(w) >= 24}  -> 去相关后 W 承载值内容" if min(w) >= 24
      else f"  white <= E+3 ? {max(w) <= d['structure']['E_random']+3}")
