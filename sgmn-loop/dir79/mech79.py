"""DIR-79 机制探针：同一 warm 代表点处 J 的**非正规瞬态放大**是否随 gamma 增长。
同时把不动点因子存盘（gamma=30 需 6.8e6 步，不可每轮重算）。
量：A_T = max_{1<=t<=T} ||J^t||_2 / lam_max^t（纯谱衰减 ⇒ 1）。"""
import sys, json, os, time
for q in ("dir69","dir70","dir72","dir73","dir78","dir79"): sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np, warm78, jac79

OUT = "/home/ec2-user/sgmn-loop/dir79/out/mech79.json"
formal = json.load(open("/home/ec2-user/sgmn-loop/dir79/out/DIR79_formal.json"))
T = 5000
res = {}
if os.path.exists(OUT): res = json.load(open(OUT))
for g in [10.0, 20.0, 30.0]:
    key = str(g)
    if key in res: print("skip", key); continue
    t0 = time.time()
    eta_w = formal[key]["eta_warm"]
    x, info = warm78.warm_fp("target", g, eta_w)
    np.savez("/home/ec2-user/sgmn-loop/dir79/out/fp_g%g.npz" % g, U1=x[0], V1=x[1], U2=x[2], V2=x[3], eta=eta_w)
    J = jac79.jac(x, eta_w, "target", g)
    w = np.linalg.eigvals(J); lm = float(np.max(np.abs(w)))
    s2 = float(np.linalg.norm(J, 2))
    # ||J^t||_2 逐步
    M = np.eye(48); best = 1.0; best_t = 0; traj = []
    for t in range(1, T + 1):
        M = J @ M
        v = float(np.linalg.norm(M, 2)) / (lm ** t)
        if v > best: best, best_t = v, t
        if t in (10, 100, 1000, T): traj.append((t, v))
    res[key] = dict(g=g, eta_warm=eta_w, fp_res=info["res"], fp_steps=info["steps"], lmax=lm,
                    norm2=s2, ampl=s2 / lm, sup_ampl=best, sup_t=best_t, traj=traj, secs=time.time() - t0)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("γ=%-3g lmax=%.8f ||J||2=%.4f sup||J^t||/λ^t=%.4f @t=%d (%.0fs)" % (g, lm, s2, best, best_t, time.time()-t0), flush=True)
