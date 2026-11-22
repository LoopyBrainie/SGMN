"""DIR-73 Stage A · A1..A7（守卫只检验仪器）"""
import sys, os, json, math
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70
P = ref70.P
FAIL = []
def chk(name, ok, note):
    print("%-58s %s | %s" % (name, "PASS" if ok else "FAIL", note), flush=True)
    if not ok: FAIL.append(name)

# A4 逐位一致性（先做：装置其它断言都建在它上面）
rng = np.random.default_rng(near73.SD)
U1 = rng.normal(size=(6, 2)) * 0.1; V1 = rng.normal(size=(6, 2)) * 0.1
U2 = rng.normal(size=(6, 2)) * 0.1; V2 = rng.normal(size=(6, 2)) * 0.1
a, b, c_, d = U1.copy(), V1.copy(), U2.copy(), V2.copy()
import step72
for t in range(20000):
    a, b, c_, d = step72.step(a, b, c_, d, 0.05, "target", 0.0)
ref = ref70.run_stack(6.0, 0.0, 0.05, 20000, seed=near73.SD, init=0.1, mode="target")
chk("A4 逐位一致性 γ=0 2e4 步", float(np.max(np.abs(a @ b.T - ref["W1"]))) == 0.0,
    "max|ΔW1|=%.1e" % float(np.max(np.abs(a @ b.T - ref["W1"]))))

# A6 可达性：3 格在 10·η*_lin 必须发散
xs30, fp30 = near73.freeze_fp("target", 30.0, eta_warm=0.3 * near73.eta_lin("target", 30.0))
xs10k, fp10k = near73.freeze_fp("keys", 10.0, eta_warm=0.3 * near73.eta_lin("keys", 10.0))
xs3, fp3 = near73.freeze_fp("target", 3.0, eta_warm=0.3 * near73.eta_lin("target", 3.0))
x30, m30 = near73.perturb(xs30, 1e-3); x10k, m10k = near73.perturb(xs10k, 1e-3); x3, m3 = near73.perturb(xs3, 1e-3)
divs = {}
for tag, x0, mode, g in [("target3", x3, "target", 3.0), ("target30", x30, "target", 30.0), ("keys10", x10k, "keys", 10.0)]:
    divs[tag] = near73.run_from(x0, 10.0 * near73.eta_lin(mode, g), mode, g)["div"]
chk("A6 可达性 η=10η*_lin 必发散", all(divs.values()), str(divs))
# A5 扰动定义
chk("A5 扰动定义 |meas−δ|/δ ≤20%", all(abs(m - 1e-3) / 1e-3 <= 0.2 for m in m30),
    "meas=%s" % ["%.2e" % m for m in m30])
# A2 η→0 收敛
r2 = near73.run_from(x30, 1e-4 * near73.eta_lin("target", 30.0), "target", 30.0)
U1e, V1e, _, _ = (np.array(X, dtype=float, copy=True) for X in x30)
for t in range(200000):
    U1e, V1e, _, _ = step72.step(U1e, V1e, _, _, 1e-4 * near73.eta_lin("target", 30.0), "target", 30.0)
Ws = fp30["u1"] * P
res = float(np.linalg.norm(U1e @ V1e.T - Ws) / np.linalg.norm(Ws))
chk("A2 η→0 从 x0 收敛到 FP（<1e-10）", res < 1e-10, "rel=%.3e div=%s" % (res, r2["div"]))

# A1 δ 无关性（target 30：δ=1e-2 vs 1e-3）
el30 = near73.eta_lin("target", 30.0)
x2, _ = near73.perturb(xs30, 1e-2)
b2 = near73.bisect_near("target", 30.0, x2, 2e-4, 4e-3, verbose=False)
b1 = near73.bisect_near("target", 30.0, x30, 2e-4, 4e-3, verbose=False)
rel = abs(b2["eta"] - b1["eta"]) / b1["eta"] if (b2["ok"] and b1["ok"]) else float("nan")
chk("A1 δ 无关性 |Δ|/η ≤5%", (rel == rel) and rel <= 0.05, "δ=1e-2: %.6g / δ=1e-3: %.6g rel=%.4f" % (b2["eta"], b1["eta"], rel))

# A3 装置复现（旧协议 init=0.1，target 3 必须复现 0.0654901）
def bisect_old(mode, gamma, lo, hi, n=near73.N_DEF, rel=near73.REL):
    a_, b_ = math.log(lo), math.log(hi)
    fa = ref70.run_stack(6.0, gamma, lo, n, seed=near73.SD, init=0.1, mode=mode)["div"]
    fb = ref70.run_stack(6.0, gamma, hi, n, seed=near73.SD, init=0.1, mode=mode)["div"]
    if fa or not fb: return float("nan")
    for _ in range(80):
        if (b_ - a_) <= math.log(1 + rel): break
        m = 0.5 * (a_ + b_)
        d = ref70.run_stack(6.0, gamma, math.exp(m), n, seed=near73.SD, init=0.1, mode=mode)["div"]
        if d: b_ = m
        else: a_ = m
    return float(math.exp(0.5 * (a_ + b_)))
e_old = bisect_old("target", 3.0, 0.04, 0.12)
chk("A3 装置复现 target3 旧协议（±3% of 0.0654901）", abs(e_old - 0.0654901) / 0.0654901 <= 0.03, "η=%.7g" % e_old)

# A7 正控（target 3 近 FP 比值 ∈[0.90,1.10]）—— 同时落盘供正式轮复用
r3 = near73.bisect_near("target", 3.0, x3, 0.04, 0.12, verbose=False)
ratio3 = r3["eta"] / near73.eta_lin("target", 3.0) if r3["ok"] else float("nan")
chk("A7 正控 target3 比值 ∈[0.90,1.10]", (ratio3 == ratio3) and 0.90 <= ratio3 <= 1.10, "ratio=%.4f" % ratio3)
os.makedirs("/home/ec2-user/sgmn-loop/dir73/out/grid", exist_ok=True)
json.dump(dict(mode="target", gamma=3.0, delta=1e-3, eta_lin=near73.eta_lin("target", 3.0), eta_near=r3["eta"],
               ratio=ratio3, ok=r3["ok"], note=r3["note"], n_eval=r3["n_eval"], bracket=[r3["lo"], r3["hi"]],
               hist=r3["hist"], fp_res=fp3["res"], fp_steps=fp3["steps"], pert_meas=m3,
               eta_warm=0.3 * near73.eta_lin("target", 3.0), source="stageA73.A7"),
          open("/home/ec2-user/sgmn-loop/dir73/out/grid/near_target_g3_d0.001.json", "w"), indent=1)
json.dump(dict(A1=dict(eta_d2=b2["eta"], eta_d3=b1["eta"], rel=rel, ok2=b2["ok"], ok3=b1["ok"]),
               A3=e_old, A7=ratio3, A6=divs, A2=res, A5=m30, A4=1.0),
          open("/home/ec2-user/sgmn-loop/dir73/out/stageA73.json", "w"), indent=1)
print("\nSTAGE_A", "PASS" if not FAIL else "FAIL")
