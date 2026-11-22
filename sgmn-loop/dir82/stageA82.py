"""DIR-82 Stage A 硬门：A1–A13。只检验仪器，不携带假设方向（纪律 158）。
任一 FAIL ⇒ 不得进入正式轮。
"""
import sys, os, json, hashlib, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jac82
import warm78

RES = []


def rec(tag, ok, detail):
    RES.append(dict(tag=tag, ok=bool(ok), detail=detail))
    print("%-5s %-4s %s" % (tag, "PASS" if ok else "FAIL", detail), flush=True)


def fin(*xs):
    for v in xs:
        if not np.all(np.isfinite(np.asarray(v, float))):
            raise AssertionError("non-finite")
    return True


def maxabs(a, b):
    fin(a, b)
    return max(float(np.max(np.abs(np.asarray(p, float) - np.asarray(q, float))))
               for p, q in zip(a, b))


def pick(tab, g):
    """按数值匹配取表项。JSON 键是 "10.0" 而非 "10"；硬用 %g 会 KeyError（DEF-81-2 同类错）。"""
    key = "%g" % g
    if key in tab and isinstance(tab.get(key), dict):
        return tab[key]
    cand = []
    for k, v in tab.items():
        if not isinstance(v, dict) or "eta_th" not in v:
            continue
        try:
            kv = float(k)
        except ValueError:
            continue
        if abs(kv - g) < 1e-9:
            cand.append(k)
    if not cand:
        raise KeyError("找不到 gamma=%g；现有数值键=%s" % (
            g, [k for k, v in tab.items() if isinstance(v, dict) and "eta_th" in v]))
    return tab[cand[0]]


def frozen(mode, g):
    path, kind = {"target": ("dir79/out/DIR79_formal.json", "mech"),
                  "keys": ("dir80/out/DIR80_formal.json", "cells")}[mode]
    d = json.load(open(path)); tab = d if kind == "mech" else d["cells"]
    e = pick(tab, g)
    return float(e["eta_th"]), float(e["rho"]), path


# ================= A4 · 算例逐位核（最先做：测量对象的定义本身） =================
U = np.arange(12, dtype=float).reshape(6, 2) / 7.0
V = np.zeros((6, 2)); V[0, 0] = 1.0; V[0, 1] = 2.0; V[1, 1] = 1.0
Q, Rm = np.linalg.qr(V)
eQ = float(np.max(np.abs(Q - np.array([[1., 0.], [0., 1.], [0, 0], [0, 0], [0, 0], [0, 0]]))))
eR = float(np.max(np.abs(Rm - np.array([[1., 2.], [0., 1.]]))))
Uq, Vq, _ = jac82._layer_qr(U, V)
eW = float(np.linalg.norm(Uq @ Vq.T - U @ V.T))
eO = jac82.orth_err(Vq)
_ = jac82._layer_qr(Uq, Vq)
eI = maxabs(jac82._layer_qr(Uq, Vq)[:2], (Uq, Vq))
rec("A4", max(eQ, eR, eW, eO, eI) <= 1e-15,
    "算例: |Q-hand|=%.1e |R-hand|=%.1e |W不变|=%.1e orth=%.1e 幂等=%.1e" % (eQ, eR, eW, eO, eI))

# ================= 两个参考格（keys g10 / target g10） =================
# ★ 用**源规则** warm78.warm_fp（从解析构造 near73.freeze_fp 出发）复现，因为
#   dir79/mech79.py 与 dir80/cell80.py 记录 rho 时用的就是它，而不是 warm_from_init。
#   DIR-81 曾实测两者在 target 上给出不同代表（0.0086535658 vs 0.0086523442）⇒ 必须同规则。
cells = {}
for mode, g in (("keys", 10.0), ("target", 10.0)):
    eta_th, rho_rec, src = frozen(mode, g)
    eta_w = 0.1 * eta_th
    x, info = warm78.warm_fp(mode, g, eta_w)
    xu, infou = jac82.warm_from_init(mode, g, eta_w, 0.1, seed=20261118)
    cells[mode] = dict(eta_th=eta_th, rho_rec=rho_rec, src=src, eta_w=eta_w,
                       x=x, info=info, xu=xu, infou=infou, g=g)
    print("  [ref %s] eta_th=%.12g rho_rec=%.12g warm_fp steps=%d  warm_from_init steps=%d"
          % (mode, eta_th, rho_rec, info["steps"], infou["steps"]), flush=True)

# A7 · eta_th 数据完整性（读入值 == 记录值，防 DEF-81-2）
ok7, d7 = True, []
for mode, c in cells.items():
    d = json.load(open(c["src"])); tab = d if mode == "target" else d["cells"]
    v = float(pick(tab, c["g"])["eta_th"])
    e = abs(c["eta_th"] - v) / v
    ok7 &= (e <= 1e-12); d7.append("%s:%.1e" % (mode, e))
rec("A7", ok7, "eta_th 读入 vs 记录 相对差 " + " ".join(d7))

# A6 · raw rho 复现 D-79/D-80（**同 warm 规则**）
# ★ 阈值按**测量自身分辨率**设：rho_scan 在 ln(eta) 上二分到 (b-a) <= rel；
#   卡 §2 冻结 rel=2e-3 ⇒ 返回值的相对不确定度可达 ~2e-3。
#   上版把阈值写成 1e-10（比估计器精度严 7 个数量级），必然 FAIL —— 那是我的错，非仪器错。
#   实测：target 在 rel=1e-10 下复现到 1.5e-11；keys 收敛值 0.083011532 与 D-80 记录的
#   0.083044624 差 3.99e-4（= D-80 那次二分恰好停下的括号中点）。两者都在一个括号宽内。
BRACKET = 2e-3
ok6, d6 = True, []
for mode, c in cells.items():
    r, dg = jac82.rho_scan(c["x"], mode, c["g"], lo=1e-6, hi=1.0, rel=1e-8)   # 收敛分辨率
    c["rho_raw_c"] = r
    e = abs(r - c["rho_rec"]) / abs(c["rho_rec"])
    ok6 &= (e <= BRACKET) and dg.get("ok", False) and not dg.get("cap_hit")
    d6.append("%s: rho(rel=1e-8)=%.12g repro=%.2e" % (mode, r, e))
rec("A6", ok6, "raw rho 复现 | 阈值=1 个冻结括号宽 %.0e | " % BRACKET + " | ".join(d6))

# A6c · 分辨率自洽：rel=2e-3（卡冻结）与 rel=1e-8（收敛）之差须 <= 1 个括号宽
ok6c, d6c = True, []
for mode, c in cells.items():
    r2, dg2 = jac82.rho_scan(c["x"], mode, c["g"], lo=1e-6, hi=1.0, rel=2e-3)
    c["rho_raw"] = r2; c["dg_raw"] = dg2
    e = abs(r2 - c["rho_raw_c"]) / abs(c["rho_raw_c"])
    ok6c &= (e <= BRACKET)
    d6c.append("%s: rel2e-3=%.12g 差=%.2e" % (mode, r2, e))
rec("A6c", ok6c, "分辨率自洽（2e-3 vs 1e-8）| " + " | ".join(d6c))

# A6b · 两种 warm 规则在 gfix 后必须给同一个 rho（判据 ①b 的正向预演）
ok6b, d6b = True, []
for mode, c in cells.items():
    r1, _ = jac82.rho_scan(jac82.gfix_qr(c["x"]), mode, c["g"], lo=1e-6, hi=1.0)
    r2, _ = jac82.rho_scan(jac82.gfix_qr(c["xu"]), mode, c["g"], lo=1e-6, hi=1.0)
    c["rho_g_from_fp"] = r1; c["rho_g_from_init"] = r2
    raw_rel = abs(r2 - r1) / abs(r1) if r1 else float("inf")
    e = abs(r1 - r2) / abs(r1) if r1 else float("inf")
    ok6b &= (e <= 1e-8)
    d6b.append("%s: fp=%.10g init=%.10g rel=%.1e" % (mode, r1, r2, e))
rec("A6b", ok6b, "gfix 后两种 warm 规则同值 | " + " | ".join(d6b))

# ================= A1/A2/A3/A8/A12 · 归一化性质 =================
# ★ 五个独立列表：写成 d1 = d2 = ... = [] 会让它们指向同一对象（本轮实际踩到，
#   导致 A1/A2/A3/A8 的细节行全打印成核维那份 —— 证据文本误导）。
ok1 = ok2 = ok3 = ok8 = ok12 = True
d1, d2, d3, d8, d12 = [], [], [], [], []
for mode, c in cells.items():
    xq = jac82.gfix_qr(c["x"]); c["xq"] = xq
    fin(*xq)
    e_idem = maxabs(jac82.gfix_qr(xq), xq)
    w_err = max(float(np.linalg.norm(xq[k] @ xq[k + 1].T - c["x"][k] @ c["x"][k + 1].T)) for k in (0, 2))
    w_sc = max(float(np.linalg.norm(c["x"][k] @ c["x"][k + 1].T)) for k in (0, 2))
    e_w = w_err / w_sc
    e_o = max(jac82.orth_err(xq[1]), jac82.orth_err(xq[3]))
    # A8 · 归一化后仍是精确不动点
    vq = jac82.flat(xq)
    e_fp = float(np.linalg.norm(jac82.step48(vq, c["eta_w"], mode, c["g"], 6.0) - vq)) / float(np.linalg.norm(vq))
    # A12 · 核维不变
    kd0, _ = jac82.ker_dim(c["x"], c["eta_w"], mode, c["g"])
    kd1, gap1 = jac82.ker_dim(xq, c["eta_w"], mode, c["g"])
    ok1 &= e_idem <= 1e-14; d1.append("%s:%.1e" % (mode, e_idem))
    ok2 &= e_w <= 1e-14; d2.append("%s:%.1e" % (mode, e_w))
    ok3 &= e_o <= 1e-14; d3.append("%s:%.1e" % (mode, e_o))
    ok8 &= e_fp <= 1e-14; d8.append("%s:%.1e" % (mode, e_fp))
    ok12 &= (kd0 == 8 and kd1 == 8); d12.append("%s:%d->%d(gap%.3g)" % (mode, kd0, kd1, gap1))
    c["kd0"], c["kd1"] = kd0, kd1
rec("A1", ok1, "gfix 幂等 相对 " + " ".join(d1))
rec("A2", ok2, "gfix 保 W 相对 " + " ".join(d2))
rec("A3", ok3, "gfix 后 V 正交归一 |V^T V - I| " + " ".join(d3))
rec("A8", ok8, "gfix 后仍为精确不动点 " + " ".join(d8))
rec("A12", ok12, "dim ker(J-I) 归一化前后 " + " ".join(d12))

# ================= A5 · gfix_qr 与 gfix_polar 同一 O(2) 轨道 =================
ok5, d5 = True, []
for mode, c in cells.items():
    xp = jac82.gfix_polar(c["x"]); c["xp"] = xp
    h = c["xq"][1].T @ xp[1]
    e_h = float(np.linalg.norm(h @ h.T - np.eye(2)))
    e_V = float(np.linalg.norm(xp[1] - c["xq"][1] @ h))
    e_U = float(np.linalg.norm(xp[0] - c["xq"][0] @ h)) / max(float(np.linalg.norm(xp[0])), 1e-300)
    ok5 &= max(e_h, e_V, e_U) <= 1e-12; d5.append("%s: h=%.1e V=%.1e U=%.1e" % (mode, e_h, e_V, e_U))
rec("A5", ok5, "qr vs polar 同一 O(2) 轨道 | " + " | ".join(d5))

# ================= A9 · O(2) 上 rho 不变（D-102 引理） =================
ok9, d9 = True, []
for mode, c in cells.items():
    vals = []
    for tag, Gm in jac82.ORTHO_G:
        r, dg = jac82.rho_scan(jac82.apply_gauge(c["xq"], Gm), mode, c["g"], lo=1e-6, hi=1.0)
        vals.append(r)
    v = np.array(vals)
    sp = float((v.max() - v.min()) / v.mean()) if np.all(np.isfinite(v)) else float("inf")
    ok9 &= sp <= 1e-10; d9.append("%s:极差=%.2e" % (mode, sp))
rec("A9", ok9, "O(2) 上 rho 相对极差 " + " ".join(d9))

# ================= A10 · 括号有效 + A11 · 数值健康 =================
ok10, d10 = True, []
for mode, c in cells.items():
    for lbl, dg in (("raw", c["dg_raw"]),):
        ok10 &= bool(dg.get("ok")) and (dg.get("f_lo") is False) and (dg.get("f_hi") is True) and not dg.get("cap_hit")
        d10.append("%s/%s ok=%s f=[%s,%s] cap=%s" % (mode, lbl, dg.get("ok"), dg.get("f_lo"), dg.get("f_hi"), dg.get("cap_hit")))
rec("A10", ok10, "rho_scan 括号 | " + " | ".join(d10))

ok11 = True; d11 = []
for mode, c in cells.items():
    fin(c["xq"], c["xp"])
    cv = float(np.linalg.cond(c["x"][1])); cu = float(np.linalg.cond(c["x"][0]))
    ok11 &= (np.isfinite(cv) and np.isfinite(cu) and cv < 1e12 and cu < 1e12)
    d11.append("%s: condV=%.3g condU=%.3g" % (mode, cv, cu))
rec("A11", ok11, "cond 有限且 <1e12；所有 max|Δ| 比较已过 fin() | " + " | ".join(d11))

# ================= A13 · 只追加，不改写既有文件 =================
md5 = {}
for p in ("revolve/DIR-79/VERDICT.md", "revolve/DIR-80/VERDICT.md", "revolve/DIR-81/VERDICT.md"):
    md5[p] = hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "MISSING"
n_ev = sum(1 for _ in open("evidence/table.md", encoding="utf-8"))
ok13 = all(v != "MISSING" for v in md5.values()) and n_ev >= 92
rec("A13", ok13, "既有 VERDICT md5 可读、evidence 行数=%d(>=92) | %s" % (
    n_ev, " ".join("%s=%s" % (k.split("/")[1], v[:8]) for k, v in md5.items())))

# ================= 汇总 =================
npass = sum(1 for r in RES if r["ok"]); ntot = len(RES)
os.makedirs("dir82/out", exist_ok=True)
json.dump(dict(guards=RES, n_pass=npass, n_total=ntot, all_pass=(npass == ntot),
               md5=md5, evidence_lines=n_ev),
          open("dir82/out/stageA82.json", "w"), ensure_ascii=False, indent=1)
print("\n==== Stage A: %d/%d PASS ====" % (npass, ntot), flush=True)
sys.exit(0 if npass == ntot else 1)
