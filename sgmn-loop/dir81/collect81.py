"""DIR-81 汇总器：算判据 ①②③、三态裁决、P1–P6 记分、O1 观测。
用法：python collect81.py
"""
import json, os, math, itertools

OUT = "/home/ec2-user/sgmn-loop/dir81/out"
INITS = [0.01, 0.03, 0.1, 0.3]
MODES = ["keys", "target"]


def ls_slope(xs, ys):
    n = len(xs); sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if den else float("nan")


def load(mode, init):
    p = os.path.join(OUT, "cell_%s_%03d.json" % (mode, int(round(init * 100))))
    if not os.path.exists(p): return None
    d = json.load(open(p))
    d["_path"] = p
    return d


cells = {}
for m in MODES:
    for i in INITS:
        cells[(m, i)] = load(m, i)

# ---------- 有效性（卡 §2-有效性） ----------
for k, d in cells.items():
    if d is None:
        cells[k] = dict(valid=False, note="missing")
        continue
    stage = d.get("stage")
    eta = d.get("eta_th")
    if eta is None:
        d["valid"] = False; d["note"] = "no_eta_th(%s)" % stage; continue
    if d.get("valid") is None:
        d["valid"] = False; d["note"] = "no_tdiv_probe(%s)" % stage; continue
    if not d["valid"]:
        d["note"] = d.get("valid_note", "invalid")
    else:
        d["note"] = "ok"
    if d.get("rho") is None and stage in ("rho_skipped",):
        d["note"] += "+rho_skipped:" + str(d.get("rho_note"))

# ---------- ① 斜率 + 单调性 ----------
def series(mode):
    pts = [(i, cells[(mode, i)]["eta_th"]) for i in INITS
           if cells[(mode, i)].get("valid") and cells[(mode, i)].get("eta_th")]
    return pts


pk = series("keys"); pt = series("target")
b_k = ls_slope([math.log(i) for i, _ in pk], [math.log(e) for _, e in pk]) if len(pk) >= 2 else float("nan")
b_t = ls_slope([math.log(i) for i, _ in pt], [math.log(e) for _, e in pt]) if len(pt) >= 2 else float("nan")
mono = all(pk[j + 1][1] <= 1.01 * pk[j][1] for j in range(len(pk) - 1)) if len(pk) >= 2 else None

C1_ok = (len(pk) >= 3) and (b_k <= -0.15) and bool(mono)
C1_note = ("n_valid=%d" % len(pk)) if len(pk) >= 3 else "n_valid=%d<3 ⇒ 从未被计算" % len(pk)

# ---------- ② rho 的幅度不变性 ----------
rk = [cells[("keys", i)]["rho"] for i in INITS if cells[("keys", i)].get("rho")]
if len(rk) >= 2:
    spread = (max(rk) - min(rk)) / (sum(rk) / len(rk))
    C2_ok = spread <= 0.02
else:
    spread = float("nan"); C2_ok = None
C2_note = "n_rho=%d" % len(rk)
if len(pk) < 3: C2_ok = None; C2_note += ";主判据未成立时②只作诊断"

# ---------- ③ target 空白对照 ----------
C3_ok = (len(pt) >= 3) and (abs(b_t) <= 0.05)
C3_note = "n_valid=%d" % len(pt)

# ---------- 三态裁决 ----------
if C1_ok is False and len(pk) < 3:
    verdict = "INCONCLUSIVE"
elif not C1_ok:
    verdict = "FALSIFIED"
elif C3_ok:
    verdict = "SUCCEEDED"
else:
    verdict = "PARTIAL"

# ---------- P1–P6 ----------
P = dict(
    P1=dict(claim="b_keys in [-0.45,-0.10]", ok=bool(-0.45 <= b_k <= -0.10) if b_k == b_k else None,
            read=round(b_k, 4)),
    P2=dict(claim="b_keys = -0.64 +-0.15", ok=bool(-0.79 <= b_k <= -0.49) if b_k == b_k else None,
            read=round(b_k, 4)),
    P3=dict(claim="|b_target| <= 0.05", ok=bool(abs(b_t) <= 0.05) if b_t == b_t else None,
            read=round(b_t, 4)),
    P4=dict(claim="eta_th monotone non-increasing, 4/4", ok=mono, read=mono),
    P5=dict(claim="keys rho spread < 2%", ok=bool(spread <= 0.02) if spread == spread else None,
            read=round(spread, 5) if spread == spread else None),
    P6=dict(claim="init=0.3 keys valid", ok=bool(cells[("keys", 0.3)].get("valid")),
            read=cells[("keys", 0.3)].get("note")),
)

# ---------- O1 ----------
def mu_est(d):
    td = d.get("t_div_probe"); 
    if not td: return None
    w0 = d.get("warm_info", {}).get("normW1_init")
    if not w0: return None
    return math.log(1e3 / w0) / td


rows = []
for (m, i), d in cells.items():
    rows.append(dict(mode=m, init=i, eta_th=d.get("eta_th"), rho=d.get("rho"),
                     gap=(d["rho"] / d["eta_th"] - 1) if (d.get("rho") and d.get("eta_th")) else None,
                     valid=d.get("valid"), note=d.get("note"),
                     t_div=d.get("t_div_probe"), warm_steps=d.get("fp_steps"),
                     sup=(d.get("sup") or {}).get("sup"),
                     mu_hat=mu_est(d) if d.get("valid") else None))

doc = dict(
    verdict=verdict,
    C1=dict(ok=C1_ok, b=b_k, band=-0.15, monotone=mono, note=C1_note,
            series=[(i, e) for i, e in pk]),
    C2=dict(ok=C2_ok, spread=None if spread != spread else spread, band=0.02, note=C2_note,
            rhos=[(i, cells[("keys", i)].get("rho")) for i in INITS]),
    C3=dict(ok=C3_ok, b=b_t, band=0.05, note=C3_note,
            series=[(i, e) for i, e in pt]),
    predictions=P, rows=rows,
)
json.dump(doc, open(os.path.join(OUT, "DIR81_formal.json"), "w"), ensure_ascii=False, indent=1)

print("=" * 66)
print("%-6s %-6s %-12s %-12s %-9s %-6s %-8s %s" % ("mode", "init", "eta_th", "rho", "gap", "valid", "t_div", "note"))
for r in rows:
    print("%-6s %-6g %-12s %-12s %-9s %-6s %-8s %s" % (
        r["mode"], r["init"],
        "%.7g" % r["eta_th"] if r["eta_th"] else "—",
        "%.7g" % r["rho"] if r["rho"] else "—",
        "%.5f" % r["gap"] if r["gap"] is not None else "—",
        r["valid"], r["t_div"] if r["t_div"] else "—", r["note"]))
print("=" * 66)
print("① b_keys=%.4f (≤-0.15) 单调=%s ⇒ %s  [%s]" % (b_k, mono, C1_ok, C1_note))
print("② rho 相对极差=%.4f%% (≤2%%=%s) n=%d" % ((spread * 100) if spread == spread else float("nan"), C2_ok, len(rk)))
print("③ b_target=%.4f (|·|≤0.05) ⇒ %s  [%s]" % (b_t, C3_ok, C3_note))
print("裁决:", verdict)
print("预言:", {k: (v["ok"], v["read"]) for k, v in P.items()})
print("=" * 66)