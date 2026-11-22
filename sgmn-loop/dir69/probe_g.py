import sys, os, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir69")
import ref69 as S
g = float(sys.argv[1]); G = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
rec = dict(gamma=g, seed=20261118, N=200000, c=6.0, a=0.1, cells={}, note="补充探针（未预注册）")
for e in G:
    o = S.run_stack(6.0, g, e, 200000, seed=20261118, init=0.1)
    rec["cells"]["%.0e" % e] = dict(div=bool(o["div"]), steps=int(o["steps"]),
                                    u1=(float(o["u1"]) if abs(o["u1"]) < 1e30 else None))
    os.makedirs("/home/ec2-user/sgmn-loop/dir69/out/grid", exist_ok=True)
    json.dump(rec, open("/home/ec2-user/sgmn-loop/dir69/out/grid/g%g_sup.json" % g, "w"), indent=1)
    print("γ=%g η=%.0e div=%s u1=%s" % (g, e, o["div"], rec["cells"]["%.0e" % e]["u1"]), flush=True)
