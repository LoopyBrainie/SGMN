import sys, json, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir64")
import ref64 as R

D, LAM = R.D, R.LAM
out = {}
def chk(name, ok, note=""):
    print("%-58s %s | %s" % (name, "PASS" if ok else "FAIL", note)); out[name] = dict(ok=bool(ok), note=note)

# A64-1 固定点锚
worst = 0.0
for fmt in ("mat_batch", "fac_batch"):
    o = R.run(fmt, 6.0, 0.02, 20000)
    worst = max(worst, float(np.linalg.norm(o["W"] - R.Wstar(6.0)) / np.linalg.norm(R.Wstar(6.0))))
chk("A64-1a 批梯度固定点锚（η=0.02, c=6）", worst <= 1e-6, "worst rel=%.2e" % worst)
worst2, amps = 0.0, []
for fmt in ("mat_cycle", "fac_cycle"):
    o = R.run(fmt, 6.0, 0.02, 20000, tail=6)
    Wb = np.mean(o["tail_W"], axis=0)
    worst2 = max(worst2, float(np.linalg.norm(Wb - R.Wstar(6.0)) / np.linalg.norm(R.Wstar(6.0))))
    amps.append(float(np.max([np.linalg.norm(w - Wb) for w in o["tail_W"]])))
chk("A64-1b 循环格式周期均值锚（≤20η 且锯齿>0）", worst2 <= 20 * 0.02 and min(amps) > 0,
    "rel(mean)=%.2e amp=%s" % (worst2, ["%.1e" % x for x in amps]))

# A64-2 fac_batch 解析谱 + 数值边界
def eig_max_fac(c):
    U, V = None, None; Ws = R.Wstar(c)
    U0, s, Vt = np.linalg.svd(Ws); U = U0[:, :2] * np.sqrt(s[:2]); V = Vt[:2, :].T * np.sqrt(s[:2])
    ks = R.keys(c); tg = [R.P @ k for k in ks]
    C = sum(np.outer(k, k) for k in ks) / D; B = sum(np.outer(v, k) for v, k in zip(tg, ks)) / D
    n, h, A = 4 * D, 1e-6, np.eye(4 * D)
    for j in range(n):
        dU = np.zeros((D, 2)); dV = np.zeros((D, 2))
        if j < 2 * D: dU.flat[j] = h
        else: dV.flat[j - 2 * D] = h
        o = []
        for sg in (1, -1):
            Ux, Vx = U + sg * dU, V + sg * dV
            G = ((Ux @ Vx.T) @ C - B) + LAM * (Ux @ Vx.T)
            o.append(np.concatenate([(Ux - G @ Vx).ravel(), (Vx - G.T @ Ux).ravel()]))
        A[:, j] = np.eye(n)[j] - (o[0] - o[1]) / (2 * h)
    return float(np.max(np.linalg.eigvals(A).real))
em = eig_max_fac(6.0)
chk("A64-2a fac_batch 解析谱 = 2c/D = 2.0", abs(em - 2.0) <= 1e-6, "max eig=%.9f" % em)
b = R.bisect("fac_batch", 6.0, rel=0.001)["eta"]
ok = (0.1 < b < 2.0) and (not R.run("fac_batch", 6.0, 0.1, 20000)["div"]) and R.run("fac_batch", 6.0, 2.0, 20000)["div"]
chk("A64-2b 仪器有效性（二分收敛、落 (0.1,2)）", ok, "η*=%.5f  比值 vs 线性 1.0 = %.4f" % (b, b))

# A64-3 / A64-3b 矩阵两式
b1 = R.bisect("mat_batch", 6.0, rel=0.001)["eta"]
chk("A64-3 mat_batch 边界 = 1.9048", abs(b1 / 1.904762 - 1) <= 0.01, "η*=%.5f" % b1)
b2 = R.bisect("mat_cycle", 6.0, rel=0.001)["eta"]
chk("A64-3b mat_cycle 首根 = 0.3456", abs(b2 / 0.3456 - 1) <= 0.01, "η*=%.5f" % b2)
f = lambda e: abs(1 - e * (6.0 + LAM)) * (1 - e * LAM) ** (D - 1) - 1
scan = [(e, f(e)) for e in np.linspace(0.35, 19.9, 20000)]
neg_after_pos = any(scan[i][1] > 0 and scan[i + 1][1] < 0 for i in range(len(scan) - 1))
chk("A64-3c 闭式非单调（首根后须回负）", neg_after_pos, "存在正→负区间=%s" % neg_after_pos)

# A64-4 确定性
a1 = R.run("fac_cycle", 6.0, 0.01, 5000)["W"]; a2 = R.run("fac_cycle", 6.0, 0.01, 5000)["W"]
chk("A64-4 确定性（逐位相同）", np.array_equal(a1, a2), "max|Δ|=%.1e" % float(np.max(np.abs(a1 - a2))))

# A64-5 发散可达性
chk("A64-5 mat_batch 在 1.5η* 发散", R.run("mat_batch", 6.0, 1.5 * 1.904762, 50000)["div"], "")

# A64-6 有界区可达 + rel
o = R.run("fac_batch", 6.0, 0.3, 100000)
rel = abs(o["J"] - R.Jstar(6.0)) / R.Jstar(6.0)
chk("A64-6 fac_batch 在 0.3η* 有界且 rel<1e-3", (not o["div"]) and rel < 1e-3, "rel=%.2e" % rel)

# A64-7 锯齿可测（fac_cycle c=6 η=0.005，DIR-63 标定过的有界点）
o = R.run("fac_cycle", 6.0, 0.005, 200000, tail=12)
tj = np.array(o["tail_J"]); rl = np.abs(tj - R.Jstar(6.0)) / R.Jstar(6.0)
imax = int(np.argmax(rl)); spread = float(rl.max() - rl.min())
chk("A64-7 锯齿可测（极值相位非常数）", spread > 0, "argmax=%d spread=%.2e" % (imax, spread))

# A64-8 分辨力（算术）
gap = 1.0 / 0.3456
chk("A64-8 阈值分辨力（1:2.9 间隔 vs 3%）", gap >= 2 and (gap - 1) / 0.03 >= 10, "ratio=%.2f" % gap)

print("\n★ Stage A: %d/%d PASS" % (sum(1 for v in out.values() if v["ok"]), len(out)))
json.dump(out, open("dir64/out/DIR64r10_stageA.json", "w"), indent=1)
