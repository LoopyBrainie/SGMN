import sys, os, json, argparse
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir69")
import ref69 as S
ap = argparse.ArgumentParser(); ap.add_argument("--cells", required=True); ap.add_argument("--out", default="dir69/out/cells")
x = ap.parse_args(); os.makedirs(x.out, exist_ok=True)
C, N, A = 6.0, 200000, 0.1
for cell in x.cells.split(","):
    g_s, s_s = cell.split(":"); g, s = float(g_s), int(s_s)
    fn = os.path.join(x.out, "g%g_s%d.json" % (g, s))
    if os.path.exists(fn): print("skip", fn, flush=True); continue
    b = S.bisect(g, C, N, seed=s, init=A)
    o = S.run_stack(C, g, 0.2, N, seed=s, init=A); p = S.pred(g)
    rec = dict(gamma=g, seed=s, eta=b["eta"], ok=b["ok"], N=N, c=C, a=A,
               u1=o["u1"], u2=o["u2"], u1_pred=p["u1"], u2_pred=p["u2"], r_pred=p["r"])
    json.dump(rec, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
    print("done γ=%-4g seed=%d eta=%.5f u1=%.4f" % (g, s, b["eta"], o["u1"]), flush=True)
