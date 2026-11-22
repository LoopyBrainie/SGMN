import sys, os, json, argparse, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir68")
import ref68 as R
p = argparse.ArgumentParser(); p.add_argument("--cells", required=True); p.add_argument("--dir", default="dir68/out/cells")
x = p.parse_args(); os.makedirs(x.dir, exist_ok=True)
A, N, C = 0.1, 200000, 6.0
for c in x.cells.split(","):
    kind, s_s = c.split(":"); s = int(s_s)
    fn = os.path.join(x.dir, "%s_s%d.json" % (kind, s))
    if os.path.exists(fn): print("skip", fn, flush=True); continue
    if kind == "ORTH":   # 该纤维上 η* 不存在 ⇒ 单点核验 η=2.0 有界
        o = R.run_dir("ORTH", C, 2.0, A, N, s)
        rec = dict(kind=kind, a=A, seed=s, eta=None, ok=not o["div"], N=N, c=C,
                   maxW=o["maxW"], div=o["div"], eta_probe=2.0)
        print("done ORTH seed=%d maxW=%.2e div=%s" % (s, o["maxW"], o["div"]), flush=True)
    else:
        b = R.bisect_dir(kind, C, A, N, seed=s)
        rec = dict(kind=kind, a=A, seed=s, eta=b["eta"], ok=b["ok"], N=N, c=C)
        print("done %s seed=%d eta=%.5f" % (kind, s, b["eta"]), flush=True)
    json.dump(rec, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
