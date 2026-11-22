import sys, os, json, argparse, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir66")
import ref66 as R
p = argparse.ArgumentParser()
p.add_argument("--cells", required=True, help="a:seed,a:seed,...")
p.add_argument("--dir", default="dir67/out/cells")
x = p.parse_args(); os.makedirs(x.dir, exist_ok=True)
C, N = 6.0, 200000
for c in x.cells.split(","):
    a_s, s_s = c.split(":"); aa, s = float(a_s), int(s_s)
    fn = os.path.join(x.dir, "a%s_s%d.json" % ("%.0e" % aa, s))
    if os.path.exists(fn): print("skip", fn, flush=True); continue
    b = R.bisect_a(C, aa, N, seed=s)
    json.dump(dict(a=aa, seed=s, eta=b["eta"], lo=b["lo"], hi=b["hi"], ok=b["ok"], N=N, c=C),
              open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
    print("done a=%.0e seed=%d η*=%.5f" % (aa, s, b["eta"]), flush=True)
