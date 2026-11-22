"""DIR-74 补充诊断（不进判据）：线性放大曲线说"不放大"时，**非线性**轨迹在哪些 η/代表点上逃逸？
直接派生 `near73.run_from`（原样）+ 逐步记录 ‖W1‖,‖W2‖,‖U1‖,‖V1‖。"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72
mode, g = "keys", 10.0
el = near73.eta_lin(mode, g)
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
p = ref70.pred_keys(6.0, g)
pts = {"bal": tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2")),
       "warm": near73.freeze_fp(mode, g, 0.3 * el)[0]}
rows = []
for rep, xs in pts.items():
    x0, meas = near73.perturb(xs, 1e-3)
    for f in (0.5679, 0.8, 1.0, 1.2):
        eta = f * el
        U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in x0)
        td, tr = None, []
        for t in range(1, 200001):
            U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, mode, g)
            if t in (100, 1000, 10000, 100000) or (t % 25 == 0 and (abs(U1 @ V1.T).max() > 1e3 or abs(U2 @ V2.T).max() > 1e3)):
                tr.append((t, float(np.linalg.norm(U1 @ V1.T)), float(np.linalg.norm(U2 @ V2.T)),
                           float(np.linalg.norm(U1)), float(np.linalg.norm(V1))))
            if t % 25 == 0 or t == 200000:
                if ref70._div(U1 @ V1.T) or ref70._div(U2 @ V2.T): td = t; break
        rows.append(dict(rep=rep, f=f, eta=eta, div=td is not None, td=td, trace=tr))
        print("DIAG %-4s f=%.4f div=%s td=%s  ‖W1‖@100/1e3/1e4/1e5=%s" % (rep, f, td is not None, td,
              ["%.2e" % r[1] for r in tr if r[0] in (100, 1000, 10000, 100000)][:4]), flush=True)
json.dump(rows, open("/home/ec2-user/sgmn-loop/dir74/out/grid/diag74.json", "w"), indent=1)
print("DIAG done", flush=True)
