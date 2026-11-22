import sys, os, json, argparse
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir69")
import ref69 as S
ap = argparse.ArgumentParser(); ap.add_argument("--g", type=float, required=True); ap.add_argument("--seeds", default="20261118")
ap.add_argument("--out", default="dir69/out/grid")
x = ap.parse_args(); os.makedirs(x.out, exist_ok=True)
C, N, A = 6.0, 200000, 0.1
GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
for s in [int(v) for v in x.seeds.split(",")]:
    fn = os.path.join(x.out, "g%g_s%d.json" % (x.g, s))
    rec = json.load(open(fn)) if os.path.exists(fn) else dict(gamma=x.g, seed=s, N=N, c=C, a=A, cells={})
    for eta in GRID:
        k = "%.0e" % eta
        if k in rec["cells"]: continue
        o = S.run_stack(C, x.g, eta, N, seed=s, init=A)
        rec["cells"][k] = dict(div=bool(o["div"]), steps=int(o["steps"]),
                               u1=(float(o["u1"]) if abs(o["u1"]) < 1e30 else None),
                               u2=(float(o["u2"]) if abs(o["u2"]) < 1e30 else None))
        json.dump(rec, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
        print("γ=%-4g seed=%d η=%-8.0e div=%s u1=%s" % (x.g, s, eta, rec["cells"][k]["div"], rec["cells"][k]["u1"]), flush=True)
