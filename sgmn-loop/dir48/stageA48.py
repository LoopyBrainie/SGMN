#!/opt/pytorch/bin/python
r"""DIR-48 · Stage A（**硬门**）：卡 §7 的 `A48-1…8`。任一失败 ⇒ 正式轮不得启动。

用法：`stageA48.py --out dir48/out/DIR48r10_stageA.json`
"""
import argparse, glob, hashlib, json, math, os, sys
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task47 as T47      # noqa: E402
import R1_task46 as T46      # noqa: E402
import R1_task48 as T48      # noqa: E402
import workex48 as WX        # noqa: E402

CARD = os.path.join(ROOT, "revolve/DIR-48/DIRECTION_rev48_v11.md")
CARD_MD5 = "62022454dcefd936b000a347f709b18b"
FORMAL = list(range(27400, 28400))
SEEDS = [27400 + 25 * i for i in range(40)]      # ★ 40 个横跨正式段（A48-8 的发散率要有统计意义）
res, ok_all = {}, True
WANT_NW = {"quiet": 4, "decay": 35, "busy": 11}


def rep(tag, ok, detail):
    global ok_all
    res[tag] = {"ok": bool(ok), "detail": detail}
    ok_all = ok_all and bool(ok)
    print(("  ✓ " if ok else "  ✗ ") + tag + " " + detail, flush=True)


print("=== DIR-48 Stage A ===", flush=True)
rows = {(sd, a): T48.trace_one(sd, a) for sd in SEEDS for a in T48.ARMS}
ok_rows = {k: v for k, v in rows.items() if not v["undef"]}

# A48-1 写入数
nw_ok = all(v["n_writes"] == WANT_NW[k[1]] for k, v in rows.items())
rep("A48-1", nw_ok, "写入数逐行符合臂定义：" + ", ".join(
    f"{a}={sorted({v['n_writes'] for k, v in rows.items() if k[1]==a})}" for a in T48.ARMS))

# A48-2 工作例
bad, wc, ww = 0, 0.0, 0.0
for sd in SEEDS:
    for arm in T48.ARMS:
        cs, ws, L, cens = WX.indep_run(sd, arm)
        got = rows[(sd, arm)]
        if got["undef"]:
            continue
        if (L, cens) != (got["L"], got["censored"]):
            bad += 1
        for d in range(len(cs)):
            dc, dw = abs(cs[d] - got["c"][d]), abs(ws[d] - got["wnorm"][d])
            wc, ww = max(wc, dc), max(ww, dw)
            bad += 0 if (dc <= 1e-12 and dw <= 1e-12) else 1
rep("A48-2", bad == 0, f"工作例 {len(SEEDS)}×3 臂、{sum(T48.W_WIN[a] for a in T48.W_WIN)*len(SEEDS)} 单元；"
    f"max|Δc|={wc:.3e} max|Δ‖W‖|={ww:.3e}；不一致 {bad}")

# A48-3 三臂前缀逐位相同（d=0 的 c/‖W‖ + 前缀 K/V）
pref_bad, wpref = 0, 0.0
for sd in SEEDS:
    if any(rows[(sd, a)]["undef"] for a in T48.ARMS):
        continue
    c0s = {a: rows[(sd, a)]["c"][0] for a in T48.ARMS}
    w0s = {a: rows[(sd, a)]["wnorm"][0] for a in T48.ARMS}
    wpref = max(wpref, max(c0s.values()) - min(c0s.values()), max(w0s.values()) - min(w0s.values()))
    if len({*c0s.values(), *w0s.values()}) != 2:
        pref_bad += 1
# 同批 k/v：decay 与 busy 用同一 content 流 ⇒ 直接比 KV 数组
# ★ 修 `DEF-48-5`：原版拿两次**同样**的调用比（空转断言）。现改为跨窗口长度比同一前缀。
kv_bad = 0.0
for sd in SEEDS:
    _, Ka, Va = T48.build_KV(sd, T48.CTX + 16)     # quiet/busy 的长度
    _, Kb, Vb = T48.build_KV(sd, T48.CTX + 32)     # decay 的长度
    for t in range(T48.CTX + 16):
        kv_bad = max(kv_bad, float((Ka[t] - Kb[t]).abs().max()), float((Va[t] - Vb[t]).abs().max()))
# `DEF-48-4` 的复现证据：若按"各自长度"分别生成，前缀**不一致**（旧写法就是错的）
old_bug = 0.0
for sd in SEEDS:
    a16 = T46.content(sd, T48.CTX + 16, "perstep")
    a32 = T46.content(sd, T48.CTX + 32, "perstep")
    old_bug = max(old_bug, float((a16 - a32[:T48.CTX + 16]).abs().max()))
rep("A48-3", pref_bad == 0 and wpref == 0.0 and kv_bad == 0.0,
    f"三臂 d=0 的 c/‖W‖ 逐位相同={pref_bad==0}（max|Δ|={wpref:.3e}）；跨窗口长度 K/V 前缀逐位 max|Δ|={kv_bad:.3e}"
    f"（★ 对照：若按各自长度生成则前缀差 max|Δ|={old_bug:.3e} ⇒ `DEF-48-4` 为真、已用固定 `NMARK` 修掉）")

# A48-4 退化锚：quiet 读数不改状态
q_ok = all(all(x == v["c"][0] for x in v["c"]) and all(x == v["wnorm"][0] for x in v["wnorm"])
           for k, v in rows.items() if k[1] == "quiet" and not v["undef"])
rep("A48-4", q_ok, "quiet 臂 c(d) 与 ‖W_d‖ 全部与 d=0 逐位相同（读数不改状态）")

# A48-5 L 与曲线自洽
l_bad = 0
for k, v in ok_rows.items():
    c, W, L, cens = v["c"], v["W"], v["L"], v["censored"]
    thr = 0.5 * v["c0"]
    if cens:
        l_bad += 0 if (L == W and all(c[d] >= thr for d in range(1, W))) else 1
    else:
        first = next(d for d in range(1, W) if c[d] < thr)
        l_bad += 0 if (L == first and L < W) else 1
rep("A48-5", l_bad == 0, f"{len(ok_rows)} 行：L 与 c/ρ 自洽、censored⇔L=W、L 为首个跌破点（违例 {l_bad}）")

# A48-6 种子段隔离
used = set()
for f in glob.glob(os.path.join(ROOT, "dir*/out/*.jsonl")):
    try:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if isinstance(o, dict) and isinstance(o.get("seed"), int):
                    used.add(o["seed"])
    except OSError:
        pass
inter = sorted(set(FORMAL) & used)
rep("A48-6", not inter, f"正式段 {FORMAL[0]}..{FORMAL[-1]}（1000）vs 既往已落盘 seed {len(used)} 个；交集 {len(inter)} 个 {inter[:5]}")

# A48-7 卡 md5 + 常数 + τ_ref
md5 = hashlib.md5(open(CARD, "rb").read()).hexdigest()
eta, lam = M.ETA, M.LAM
tau = 1.0 / (eta * lam)
consts_ok = (T48.N_REP == 16 and abs(M.ALPHA - 1.0) < 1e-15 and abs(lam - 0.05) < 1e-15
             and abs(eta - 0.2) < 1e-15 and R.D_H == 8 and R.D_K == 16 and abs(tau - 100.0) < 1e-9)
rep("A48-7", md5 == CARD_MD5 and consts_ok,
    f"卡 md5={md5}；N_REP/ALPHA/LAM/ETA/D_H/D_K={T48.N_REP}/{M.ALPHA}/{lam}/{eta}/{R.D_H}/{R.D_K}；τ_ref={tau:.1f} 步")

# A48-8 发散率前置（busy）
nd = sum(1 for k, v in rows.items() if k[1] == "busy" and v["undef"])
div_busy = nd / len(SEEDS)
rep("A48-8", div_busy <= 0.20, f"Stage A 40 个横跨正式段的种子上 busy 发散 {nd}/{len(SEEDS)} = {div_busy:.3f} ≤ 0.20")

# A48-9 内容流长度不变量（DEF-48-4）
kv9 = 0.0
for sd in SEEDS[:8]:
    _, Ka, Va = T48.build_KV(sd, T48.CTX + 8)
    _, Kb, Vb = T48.build_KV(sd, T48.CTX + 32)
    for tt in range(T48.CTX + 8):
        kv9 = max(kv9, float((Ka[tt] - Kb[tt]).abs().max()), float((Va[tt] - Vb[tt]).abs().max()))
rep("A48-9", kv9 == 0.0 and old_bug > 0.0,
    f"跨窗口长度（CTX+8 vs CTX+32）K/V 前缀 max|Δ|={kv9:.3e}；★ 反例（按各自长度生成）max|Δ|={old_bug:.3e} > 0 ⇒ DEF-48-4 为真且已被 NMARK 修掉")

print(f"\n{'★ Stage A 全部通过' if ok_all else '✗ Stage A 失败 ⇒ 正式轮不得启动'}", flush=True)
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"card_md5": CARD_MD5, "checks": res, "all_pass": ok_all},
          open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
sys.exit(0 if ok_all else 1)
