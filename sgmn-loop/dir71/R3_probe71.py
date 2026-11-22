import sys, os, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir71"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import ref70
OUT = "/home/ec2-user/sgmn-loop/dir71/out/grid"
MODE, GAMMA, ETA, N = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
os.makedirs(OUT, exist_ok=True)
o = ref70.run_stack(6.0, GAMMA, ETA, N, seed=20261118, init=0.1, mode=MODE)
fn = os.path.join(OUT, "probe_%s_g%g_eta%.4g_n%d.json" % (MODE, GAMMA, ETA, N))
json.dump(dict(mode=MODE, gamma=GAMMA, eta=ETA, n=N, div=bool(o["div"]), steps=int(o["steps"]),
               u1=(float(o["u1"]) if abs(o["u1"]) < 1e30 else None), finite=bool(o["steps"] > 0)), open(fn, "w"), indent=1)
print("DONE probe %s γ=%g η=%g N=%d div=%s steps=%d u1=%s" % (MODE, GAMMA, ETA, N, o["div"], o["steps"], o["u1"]), flush=True)
