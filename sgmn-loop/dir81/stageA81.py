"""DIR-81 Stage A：A1–A11 硬门。任一失败 ⇒ 本轮不得进正式轮。
昂贵项（A7 括号 8 格、A10 warm 8 格）用 4 进程并行。
守卫只检验仪器，不携带假设方向（纪律 158）：A2/A11 是反向断言（"量必须变"）。
"""
import sys, os, math, json, time
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np, ref70, step72, jac81

OUT = "/home/ec2-user/sgmn-loop/dir81/out"
N, SD, C, G = 200000, 20261118, 6.0, 10.0
INITS = [0.01, 0.03, 0.1, 0.3]
MODES = ["keys", "target"]
# 已知的 D-101 读数，只作为 warm 代表点的**输入**（不作断言 —— DEF-79-3 的教训）
# 已知的 D-101 读数，只作为 warm 代表点的**输入**（不作断言 —— DEF-79-3 的教训）。
# ★ DEF-81-2：初版误写 target=0.06497468 —— 那是 **γ=3** 的 η_th，γ=10 的是 0.007303405。
ETA_TH_REF = {("keys", 0.1): 0.027032409992084817, ("target", 0.1): 0.007303405}
RES = {}
T0 = time.time()


def rec(k, ok, **kw):
    RES[k] = dict(ok=bool(ok), **kw)
    print("  %-5s %s %s" % (k, "PASS" if ok else "**FAIL**",
                            " ".join("%s=%s" % (a, b) for a, b in kw.items() if a != "ok")), flush=True)


def run(mode, init, eta, n=N):
    return ref70.run_stack(C, G, eta, n, seed=SD, init=init, mode=mode)


# ================= A1 / A1b：step72.step ≡ ref70 内步 =================
def guard_step_equiv(tag, mode, init, n=300, eta=1e-3, g=G):
    """步进等价。η 必须**安全地**低于 η_th（否则 run_stack 提前 break、而手写循环跑到 NaN，
    两边不可比 —— DEF-81-1）；并显式要求两端有限（NaN 不算读数，纪律 159/167）。"""
    r = ref70.run_stack(C, g, eta, n, seed=SD, init=init, mode=mode)
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in jac81.draw_init(init, SD))
    for _ in range(n):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, mode, g, c=C)
    W1, W2 = U1 @ V1.T, U2 @ V2.T
    finite = bool(np.isfinite(W1).all() and np.isfinite(W2).all()
                  and np.isfinite(r["W1"]).all() and np.isfinite(r["W2"]).all())
    d = max(float(np.max(np.abs(W1 - r["W1"]))), float(np.max(np.abs(W2 - r["W2"])))) if finite else float("inf")
    rec(tag, finite and d == 0.0, max_abs_dW=("%.3e" % d) if finite else "non-finite", n=n, eta=eta)


guard_step_equiv("A1", "keys", 0.1)
guard_step_equiv("A1b", "target", 0.3)

# ================= A2：init 真的进装置（反向断言） =================
ra, rb = run("keys", 0.01, 0.02, n=200), run("keys", 0.3, 0.02, n=200)
d = float(np.max(np.abs(ra["W1"] - rb["W1"])))
rec("A2", d > 1e-3, max_abs_dW="%.3e" % d, thr=">1e-3")

# ================= A11：init 幅度真的改了初值（反向断言） =================
na = float(np.linalg.norm(jac81.draw_init(0.01, SD)[0] @ jac81.draw_init(0.01, SD)[1].T))
nb = float(np.linalg.norm(jac81.draw_init(0.3, SD)[0] @ jac81.draw_init(0.3, SD)[1].T))
ratio = nb / na
rec("A11", 899.0 <= ratio <= 901.0, ratio="%.2f" % ratio, band="[899,901]")

# ================= A8：二分终止判据真的触发（合成谓词，DEF-79-4 回归） =================
def synth_bisect(rel=0.005, nmax=60, thr=0.037):
    a, b, nev, cap = math.log(1e-6), math.log(1.0), 0, False
    while True:
        if (b - a) <= rel: break
        if nev >= nmax: cap = True; break
        m = 0.5 * (a + b); nev += 1
        if math.exp(m) > thr: b = m
        else: a = m
    return float(math.exp(0.5 * (a + b))), dict(nev=nev, cap_hit=cap, width=b - a)


eta_s, dg = synth_bisect()
err = abs(eta_s - 0.037) / 0.037
rec("A8", (not dg["cap_hit"]) and dg["width"] <= 0.005 and err <= 0.005,
    width="%.5f" % dg["width"], rel_err="%.4f" % err, cap_hit=dg["cap_hit"], nev=dg["nev"])

# ================= 一个真实不动点（keys, γ=10, init=0.1） =================
eta_w = 0.1 * ETA_TH_REF[("keys", 0.1)]
x, info = jac81.warm_from_init("keys", G, eta_w, 0.1, seed=SD)
rec("A10-keys0.1", info["res_fp"] <= 1e-13, res="%.2e" % info["res_fp"], steps=info["steps"])
v = jac81.flat(x)
J = jac81.jac(x, eta_w, "keys", G, c=C)

# ================= A3：差分步长收敛 =================
J2 = jac81.jac(x, eta_w, "keys", G, h=2e-6, c=C)
dh = float(np.max(np.abs(J - J2) / (np.abs(J) + 1e-12)))
rec("A3", dh <= 1e-3, max_rel="%.3e" % dh)

# ================= A4：方向导数一致性 =================
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(5):
    dd = rng.normal(size=v.shape); dd /= np.linalg.norm(dd); h = 1e-6
    g_fd = (jac81.step48(v + h * dd, eta_w, "keys", G, C) - jac81.step48(v - h * dd, eta_w, "keys", G, C)) / (2 * h)
    r = float(np.linalg.norm(J @ dd - g_fd) / np.linalg.norm(g_fd))
    worst = max(worst, r)
rec("A4", worst <= 1e-8, worst="%.3e" % worst)

# ================= A6：中性模计数 =================
nev = int(np.sum(np.abs(np.linalg.eigvals(J) - 1.0) <= 1e-8))
rec("A6", nev >= 8, count=nev, thr=">=8")

# ================= A5：γ=0 块三角 =================
x0, i0 = jac81.warm_from_init("keys", 0.0, 0.05, 0.1, seed=SD)
J0 = jac81.jac(x0, 0.05, "keys", 0.0, c=C)
blk = float(np.max(np.abs(J0[:24, 24:])))
rec("A5", blk == 0.0, max_abs="%.3e" % blk, res="%.2e" % i0["res_fp"])

# ================= A9：谱扫描自洽（DEF-80-1 的写法） =================
eta_c, dg = jac81.rho_scan(x, "keys", G, lo=1e-6, hi=1.0)
lm_at = float(jac81.lmax(jac81.jac(x, eta_c, "keys", G, c=C)))
lm_h = jac81.rho_nn(jac81.jac(x, 0.5 * eta_c, "keys", G, c=C))
lm_hi = jac81.rho_nn(jac81.jac(x, 1.1 * eta_c, "keys", G, c=C))
rec("A9", (1 + 1e-6 <= lm_at <= 1 + 1e-2) and lm_h <= 1 + 1e-6 and lm_hi > 1 + 1e-6,
    rho="%.6g" % eta_c, lmax_at="%.9f" % lm_at, rnn_half="%.9f" % lm_h, rnn_1p1="%.9f" % lm_hi,
    cap_hit=dg["cap_hit"])

# ================= A7 / A10：8 格括号与 warm（并行） =================
from concurrent.futures import ProcessPoolExecutor


def cell_gate(args):
    """A7 = 括号；A10 = warm 从 8 个起点都能收敛。
    A10 的 eta **故意用 0.1*eta_th_ref（γ=10 的真值）**（远低于任何可能的 eta_th），
    以免把 H81 的方向（大 init ⇒ eta_th 变小）写进守卫 —— 纪律 158。
    正式轮用真实的 0.1*eta_th 复核 A10，未收敛的格按卡记为「判据从未被计算」。"""
    mode, init = args
    lo = ref70.run_stack(C, G, 1e-6, N, seed=SD, init=init, mode=mode)
    hi = ref70.run_stack(C, G, 1.0, 4000, seed=SD, init=init, mode=mode)
    eta_a10 = 0.1 * ETA_TH_REF[(mode, 0.1)]
    xe, ie = jac81.warm_from_init(mode, G, eta_a10, init, seed=SD)
    return (mode, init), dict(lo_div=bool(lo["div"]), hi_div=bool(hi["div"]),
                              warm_res=ie["res_fp"], warm_steps=ie["steps"],
                              converged=ie["converged"], eta_a10=eta_a10)


with ProcessPoolExecutor(max_workers=4) as ex:
    got = list(ex.map(cell_gate, [(m, i) for m in MODES for i in INITS]))

a7_ok, a10_ok, det = True, True, {}
for (mode, init), d in got:
    k = "%s_%03d" % (mode, int(round(init * 100)))
    det[k] = dict(lo_div=d["lo_div"], hi_div=d["hi_div"], warm_res=d["warm_res"],
                  warm_steps=d["warm_steps"], eta_a10=d["eta_a10"])
    a7_ok = a7_ok and (not d["lo_div"]) and d["hi_div"]
    a10_ok = a10_ok and (d["warm_res"] <= 1e-13)
rec("A7", a7_ok, n_ok=sum(1 for d in det.values() if (not d["lo_div"]) and d["hi_div"]), n=8)
rec("A10", a10_ok, n_conv=sum(1 for d in det.values() if d["warm_res"] <= 1e-13), n=8)

# ================= 汇总 =================
allok = all(v["ok"] for v in RES.values())
doc = dict(stageA="PASS" if allok else "FAIL", secs=round(time.time() - T0, 1),
           guards=RES, cells=det, eta_w_ref=eta_w, g=G, n=N, seed=SD)
json.dump(doc, open(os.path.join(OUT, "stageA81.json"), "w"), ensure_ascii=False, indent=1)
print("\n===== Stage A %s（%d/%d） %.0fs =====" % (
    "PASS" if allok else "FAIL", sum(v["ok"] for v in RES.values()), len(RES), time.time() - T0))
sys.exit(0 if allok else 1)