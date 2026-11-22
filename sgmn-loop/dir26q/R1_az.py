"""DIR-26-Q（卡片 rev 1.0）—— `perm` 流（只打乱条目内方位角的轮序）与 κ 判据。

派生纪律（卡片 §5）：**不复制** `dir26o/R1_ar1.py` 的任何代码，而是 `import` 后
只替换 `build_stream_ar1` 这一处（monkeypatch），因此 `run_unit` 的**测量闭包逐位不变**
⇒ A26Q-12 由构造保证。

唯一的新东西：
  build_stream_perm —— 先按参照脚本原样算键，再从键**反解**出 $Uc$（纯代数：$K/\\sqrt{S}=c\\,e+s\\,Uc$），
  用独立生成器 `seed+13` 做**条目内轮序置换**，最后按原构造式重建键。
  ⇒ ‖K‖=√S、cos(K,E)=cosθ、条目内方位角**多重集**逐位不变；只有轮序被改变。
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "dir26o"))
import R1_ar1 as RA  # noqa: E402

DTYPE = RA.DTYPE
S_KEY = RA.S_KEY
N_KV = RA.N_KV
SEEDS = RA.SEEDS
PHI_MAIN = 0.9
PERM_SALT = 13
# ★ 必须在 import 时抓住**原函数**引用：`run_perm` 会把 RA.build_stream_ar1 换成
#   `build_stream_perm` 本身，若这里再走 `RA.build_stream_ar1` 就是无限递归（DEF-26Q-1）。
_RA_BUILD = RA.build_stream_ar1
OUT = _HERE / "out"

# ★ 归档参照（D-26-O 正式轮 30 种子均值，**历史读数**，非本轮数据）
ARCH = {"A_E0_ret1_phi09": 0.7944, "B_E0_ret1_phi00": 0.8067,
        "A_E0_mu_phi09": 0.0546, "B_E0_mu_phi00": 0.0563}
KAP_LO, KAP_HI = 0.25, 0.75          # 卡片 §2.2（冻结）
RES_FRAC = 0.35                       # 分辨率要求 3se ≤ 0.35·(B−A)
TOL_GEO = 1e-6                        # 卡片 W2
TOL_RHO = 0.05                        # 卡片 Q-β


# ══════════════════════════════════════ perm 流
def _e(i, d):
    v = torch.zeros(d, dtype=DTYPE)
    v[i] = 1.0
    return v


def _uc_of(K, n2, r_pres, c, rk):
    """从键**反解** $Uc$：$K/\\sqrt S=c\\,e+s\\,Uc$ ⇒ $Uc=\\mathrm{unit}(K/\\sqrt S-c\\,e)$。
    ★ 只用于诊断（ρ_u、Λ）；**判据用的键置换走索引**，不经此反解（避免 fp 漂移）。"""
    Uc = torch.zeros(n2, r_pres, K.shape[-1], dtype=DTYPE)
    for rho in range(r_pres):
        for i in range(n2):
            u = K[rho, i] / rk - c * _e(i, K.shape[-1])
            Uc[i, rho] = u / float(u.norm())
    return Uc


def build_stream_perm(b, seed, d, n, r_pres, theta, phi):
    """`R1_ar1.build_stream_ar1` 的 perm 分支（φ=0 拒绝运行，A26Q-3）。"""
    if float(phi) == 0.0:
        raise ValueError("A26Q-3：perm 流在 φ=0 未定义（防止用 perm 假装 iid）")
    D = _RA_BUILD(b, seed, d, n, r_pres, theta, phi)
    K = D["K"]
    c, s = math.cos(theta), math.sin(theta)
    rk = math.sqrt(S_KEY)
    Uc = _uc_of(K, n2=n, r_pres=r_pres, c=c, rk=rk)          # 仅诊断用（fp 反解）
    # ── 置换：**K 的行索引置换**（逐位精确；K 只是 Uc 的函数 ⇒ 置换 Uc ≡ 置换 K 行）
    gp = torch.Generator().manual_seed(int(seed) + PERM_SALT)
    perms = []
    for i in range(n):
        perms.append(torch.randperm(r_pres, generator=gp))
    K2 = torch.zeros_like(K)
    for i in range(n):
        K2[:, i, :] = K[:, i, :][perms[i]]
    Ucp = torch.zeros_like(Uc)
    for i in range(n):
        Ucp[i] = Uc[i][perms[i]]
    Uc = Ucp
    dev_g = dev_c = 0.0
    for rho in range(r_pres):
        for i in range(n):
            e = _e(i, d)
            dev_g = max(dev_g, abs(float((K2[rho, i] * K2[rho, i]).sum().sqrt()) - rk))
            dev_c = max(dev_c, abs(float((K2[rho, i] / rk) @ e) - c))
    # ── 诊断：ρ_u（置换后）、ρ_k、Λ（条目内集合集中度）、置换非恒等比例
    rho_u = rho_k = 0.0
    cnt = 0
    kn = torch.sqrt((K2 * K2).sum(-1))
    for i in range(n):
        for rho in range(r_pres - 1):
            rho_u += float(Uc[i, rho] @ Uc[i, rho + 1])
            rho_k += float(K2[rho, i] @ K2[rho + 1, i]) / float(kn[rho, i] * kn[rho + 1, i])
            cnt += 1
    lam = torch.sqrt((Uc.sum(1) ** 2).sum(-1)) / r_pres        # ★ dim=1：对**轮**求和（DEF-26Q-2）
    nontrivial = float(np.mean([int((pm != torch.arange(r_pres)).any()) for pm in perms]))
    D["K"] = K2
    D["diag"] = {"rho_u": rho_u / max(cnt, 1), "rho_k": rho_k / max(cnt, 1),
                 "g2": D["diag"]["g2"], "lam_mean": float(lam.mean()),
                 "lam_p10": float(np.percentile(lam.numpy(), 10)),
                 "perm_nontrivial_frac": nontrivial,
                 "geo_dev_norm": dev_g, "geo_dev_cos": dev_c}
    D["km"] = None            # ★ perm 臂不重算 km（不进任何判据；防误读）
    return D


def _nat_diag(seed, phi, b=N_KV):
    return _RA_BUILD(b, seed, RA.D_MODEL, int(N_KV), RA.R_PRES, RA.theta_of(0.9), phi)["diag"]


def run_perm(seed, b=N_KV):
    """一条 (seed, φ=0.9) 的 perm 单元 —— 经**未修改**的 `RA.run_unit` 跑（测量闭包不变）。"""
    save = RA.build_stream_ar1
    RA.build_stream_ar1 = build_stream_perm
    try:
        return RA.run_unit(seed, PHI_MAIN, RA.cal_at(N_KV)["c_tau"], b=b)
    finally:
        RA.build_stream_ar1 = save


def run_nat(seed, phi, b=N_KV):
    return RA.run_unit(seed, phi, RA.cal_at(N_KV)["c_tau"], b=b)


# ══════════════════════════════════════ 判据
ANCHOR_FIELDS = ("issued", "ret1", "mu", "clean", "wsig", "ho4_mean", "ho8_mean", "P_norm", "cosP")


def crit_perm(perm_rows, nat_rows, base_B):
    """κ = (C − A)/(B − A)；A = **同种子** nat@0.9 均值（配对），B 由位级锚点复核的归档值给出。"""
    C = float(np.mean([r["arms"]["E0"]["ret1"] for r in perm_rows]))
    Cmu = float(np.mean([r["arms"]["E0"]["mu"] for r in perm_rows]))
    base_A = float(np.mean([r["arms"]["E0"]["ret1"] for r in nat_rows]))
    A = base_A
    d = np.array([pr["arms"]["E0"]["ret1"] - nr["arms"]["E0"]["ret1"]
                  for pr, nr in zip(perm_rows, nat_rows)])
    k = 3 * float(d.std(ddof=1)) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    denom = base_B - base_A
    kap = (C - base_A) / denom if denom != 0 else float("nan")
    res_ok = bool(k <= RES_FRAC * denom)
    rho_u = float(np.mean([r["diag"]["rho_u"] for r in perm_rows]))
    lam = float(np.mean([r["diag"]["lam_mean"] for r in perm_rows]))
    if not res_ok:
        ch = "POWER-LIMITED"
    elif not math.isnan(kap) and kap <= KAP_LO:
        ch = "SPREAD-CARRIER"
    elif not math.isnan(kap) and kap >= KAP_HI:
        ch = "ORDER-ONLY"
    else:
        ch = "MIXED"
    return {"C": C, "C_mu": Cmu, "A": A, "B": base_B, "denom": denom, "kappa": kap,
            "pair_mean": float(d.mean()), "pair_3se": k, "res_ok": res_ok, "channel": ch,
            "rho_u_perm": rho_u, "lam_perm": lam, "n_seeds": len(perm_rows),
            "ok": ch in ("SPREAD-CARRIER", "ORDER-ONLY", "MIXED")}


# ══════════════════════════════════════ 可达性分析（rev 1.1 的核心判据）
def _perm_rho(Uc, pm):
    """给定条目内置换 pm，返回 (ρ_u, Λ)；Uc: (n, R, d)。"""
    n, R, _ = Uc.shape
    U = Uc[:, list(pm), :]
    ru = float(np.mean([float((U[:, k, :] * U[:, k + 1, :]).sum(-1).mean()) for k in range(R - 1)]))
    lam = float(torch.sqrt(((U.sum(1)) ** 2).sum(-1)).mean()) / R   # ★ 不走 np.mean（torch 的 __array_function__ 拒 axis）
    return ru, lam


def analyze(phis=(0.3, 0.6, 0.9), seeds=None, b=N_KV):
    """穷举**全部** $R!$ 个轮序置换 ⇒ ρ_u 的可达集合是否覆盖 ≤0.05？"""
    import itertools
    seeds = list(seeds or SEEDS[:3])
    R = RA.R_PRES
    out = {"perm_space": int(math.factorial(R)), "rows": [], "by_phi": {}}
    for phi in phis:
        per_seed = []
        for s in seeds:
            D = _RA_BUILD(b, s, RA.D_MODEL, int(N_KV), R, RA.theta_of(0.9), phi)
            Uc = _uc_of(D["K"], n2=int(N_KV), r_pres=R,
                        c=math.cos(RA.theta_of(0.9)), rk=math.sqrt(S_KEY))
            cand = [_perm_rho(Uc, pm) for pm in itertools.permutations(range(R))]
            rus = np.array([c[0] for c in cand])
            # 自然（未置换）轮序 = identity
            ident = rus[list(itertools.permutations(range(R))).index(tuple(range(R)))]
            per_seed.append({"seed": int(s), "rho_identity": float(ident),
                             "rho_min": float(rus.min()), "rho_max": float(rus.max()),
                             "rho_mean": float(rus.mean()),
                             "lam_set": float(cand[0][1]),           # 置换不改集合 ⇒ Λ 恒定
                             "lam_spread": float(max(c[1] for c in cand) - min(c[1] for c in cand))})
        rho_min = float(np.mean([x["rho_min"] for x in per_seed]))
        rho_id = float(np.mean([x["rho_identity"] for x in per_seed]))
        lam = float(np.mean([x["lam_set"] for x in per_seed]))
        # 解析下界：链式可交换性下最短 Hamiltonian 路径的边权集 {2,3,2} ⇒ φ²(2+φ)/3
        bound = phi * phi * (2.0 + phi) / 3.0
        out["by_phi"][f"{phi:.1f}"] = {
            "rho_identity": rho_id, "rho_min": rho_min, "rho_max": float(np.mean([x["rho_max"] for x in per_seed])),
            "rho_mean": float(np.mean([x["rho_mean"] for x in per_seed])),
            "analytic_bound": bound, "bound_dev": rho_min - bound,
            "gain_max": rho_id - rho_min,            # 「重排」能拿到的最大去相关量
            "lam_set": lam, "lam_spread_max": float(max(x["lam_spread"] for x in per_seed)),
            "reach_005": bool(rho_min <= TOL_RHO),
        }
        out["rows"].extend([{**x, "phi": phi} for x in per_seed])
    # rev 1.1 判据
    c9 = out["by_phi"].get("0.9", {})
    out["verdict"] = {
        "bound_ok": all(abs(v["bound_dev"]) <= 0.05 for v in out["by_phi"].values()),
        "min_at_09": c9.get("rho_min"), "reach_005_at_09": c9.get("reach_005"),
        "channel": ("INFEASIBLE-STRUCTURAL"
                    if c9.get("rho_min", 1.0) > TOL_RHO else "FEASIBLE"),
    }
    return out


# ══════════════════════════════════════ Stage A（rev 1.1）
def stageA11(seeds=(20261028,)):
    """rev 1.1 的 Stage A：全部为**廉价**断言（无新增装置驱动；锚点沿用 rev 1.0 的 72/72）。"""
    import itertools
    ck = []

    def chk(cid, ok, msg):
        ck.append({"id": cid, "ok": bool(ok), "msg": msg})

    chk("A26Q'-1", _md5(_ROOT / "dir22" / "sgmn_module_hebb_v7.py") == "1a242d02c530307f6c76646ca13469ef",
        "装置 md5 零改动")
    chk("A26Q'-2", _md5(_ROOT / "dir26o" / "R1_ar1.py") == "a3c0981bbf8e09a92b2b5411742d76f1",
        "参照脚本 md5")
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    defdefs = [ln.strip() for ln in src.splitlines() if ln.lstrip().startswith("def ")]
    chk("A26Q'-2b", not any(d.startswith(("def run_unit", "def build_stream_ar1")) for d in defdefs),
        "不重定义参照函数（按行首 def 判）")

    perms = list(itertools.permutations(range(RA.R_PRES)))
    chk("A26Q'-3", len(perms) == math.factorial(RA.R_PRES) == len(set(perms)) == 24,
        f"穷举 $R!$：{len(perms)} 个置换、互不相同（非抽样）")

    # W1（解析下界与一阶近似的同一公式）
    phi = 0.9
    w1 = phi * phi * (2.0 + phi) / 3.0
    chk("A26Q'-4", abs(w1 - (2 * phi ** 2 + phi ** 3) / 3.0) < 1e-12 and abs(w1 - 0.783) < 1e-3,
        f"W1：φ=0.9 的下界 = {w1:.4f}（手算 0.783）")

    # W2（置换不改 Λ） + A26Q'-8（反解正确）
    devunit = devorth = 0.0
    lam_spread = 0.0
    rho_min = []
    for sg in seeds:
        D = _RA_BUILD(RA.N_KV, sg, RA.D_MODEL, int(N_KV), RA.R_PRES, RA.theta_of(0.9), phi)
        c = math.cos(RA.theta_of(0.9))
        Uc = _uc_of(D["K"], n2=int(N_KV), r_pres=RA.R_PRES, c=c, rk=math.sqrt(S_KEY))
        for i in (0, 17, 59):
            v = Uc[i, 0]
            devunit = max(devunit, abs(float(v.norm()) - 1.0))
            devorth = max(devorth, abs(float(v @ _e(i, D["K"].shape[-1]))))
        cand = [_perm_rho(Uc, pm) for pm in perms]
        lam_spread = max(lam_spread, max(x[1] for x in cand) - min(x[1] for x in cand))
        rho_min.append(min(x[0] for x in cand))
    chk("A26Q'-5", lam_spread <= 1e-6, f"W2：24 置换的 Λ 极差 {lam_spread:.2e} ≤1e-6（集合逐位不变）")
    chk("A26Q'-8", devunit <= 1e-5 and devorth <= 1e-5,
        f"反解：‖Uc‖−1 偏差 {devunit:.1e}、Uc·e_i {devorth:.1e}（float32 ≤1e-5）")

    # W3 位级锚点：沿用 rev 1.0 Stage A 的 72/72（脚本与装置 md5 均未变 ⇒ 无需重跑）
    f10 = OUT / "DIR26Qr10_stageA.json"
    ok_anchor, msg_anchor = False, "缺少 rev 1.0 stageA.json"
    if f10.exists():
        j = json.load(open(f10, encoding="utf-8"))
        a = j.get("anchor", {})
        ok_anchor = (a.get("n_bad") == 0 and a.get("n_cmp") == 72)
        msg_anchor = f"W3：沿用 rev 1.0 的位级锚点 {a.get('n_cmp', 0) - a.get('n_bad', 0)}/{a.get('n_cmp', 0)} 逐位相等（md5 未变 ⇒ 不重跑）"
    chk("A26Q'-6", ok_anchor, msg_anchor)

    # 确定性
    r1 = analyze(phis=(0.9,), seeds=seeds[:1])
    r2 = analyze(phis=(0.9,), seeds=seeds[:1])
    chk("A26Q'-7", r1["by_phi"]["0.9"]["rho_min"] == r2["by_phi"]["0.9"]["rho_min"],
        "同 (φ,seed) 两次 min_π ρ_u 逐位相同")

    # A26Q'-10 可行性前置（3 种子 vs 解析下界）
    dev3 = abs(float(np.mean(rho_min)) - w1)
    chk("A26Q'-10", dev3 <= 0.05,
        f"可行性前置：3 种子 min_π ρ_u={float(np.mean(rho_min)):.3f} vs 解析下界 {w1:.3f}、偏差 {dev3:.3f} ≤0.05")

    chk("A26Q'-9", len(ck) > 0, f"断言集非空（{len(ck)} 条）")
    return {"checks": ck, "passed": all(c["ok"] for c in ck)}


# ══════════════════════════════════════ Stage A
def _md5(p):
    import hashlib
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def stageA(seeds_anchor=(20261028,)):
    ck = []

    def chk(cid, ok, msg):
        ck.append({"id": cid, "ok": bool(ok), "msg": msg})

    chk("A26Q-1", _md5(_ROOT / "dir22" / "sgmn_module_hebb_v7.py") == "1a242d02c530307f6c76646ca13469ef",
        "装置 md5 零改动")
    chk("A26Q-2", _md5(_ROOT / "dir26o" / "R1_ar1.py") == "a3c0981bbf8e09a92b2b5411742d76f1",
        "参照脚本 md5")
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    defdefs = [ln.strip() for ln in src.splitlines() if ln.lstrip().startswith("def ")]
    chk("A26Q-2b", not any(d.startswith(("def run_unit", "def build_stream_ar1")) for d in defdefs),
        "本文件不重定义 run_unit / build_stream_ar1（只 monkeypatch build_stream_ar1）")

    # A26Q-3
    ok3 = False
    try:
        build_stream_perm(RA.N_KV, 1, RA.D_MODEL, int(N_KV), RA.R_PRES, RA.theta_of(0.9), 0.0)
    except ValueError:
        ok3 = True
    chk("A26Q-3", ok3, "perm 在 φ=0 被拒绝")

    # A26Q-4  工作例 W1（手算复现）
    a = torch.tensor([1.0, 0.0] + [0.0] * 6, dtype=DTYPE)
    bb = torch.tensor([0.0, 1.0] + [0.0] * 6, dtype=DTYPE)
    seq1 = [a, bb, a]
    seq2 = [a, a, bb]

    def ru(seq):
        return float(sum(seq[i] @ seq[i + 1] for i in range(len(seq) - 1)) / (len(seq) - 1))

    def lam(seq):
        return float((torch.stack(seq).sum(0) / len(seq)).norm())

    tolw = 1e-5 if DTYPE == torch.float32 else 1e-12        # ★ DEF-26Q-3：float32 手算容差
    chk("A26Q-4", abs(ru(seq1) - 0.0) < tolw and abs(ru(seq2) - 0.5) < tolw
        and abs(lam(seq1) - lam(seq2)) < tolw and abs(lam(seq1) - math.sqrt(5) / 3) < tolw,
        f"W1：ρ_u 0 / 0.5、Λ 同值 {lam(seq1):.4f}（=√5/3）")

    # A26Q-5 / A26Q-6（3 种子）
    gs = list(seeds_anchor)[:3]
    devg = devc = 0.0
    mult_ok = True
    for s in gs:
        Dnat = _RA_BUILD(RA.N_KV, s, RA.D_MODEL, int(N_KV), RA.R_PRES, RA.theta_of(0.9), PHI_MAIN)
        Dper = build_stream_perm(RA.N_KV, s, RA.D_MODEL, int(N_KV), RA.R_PRES, RA.theta_of(0.9), PHI_MAIN)
        devg = max(devg, Dper["diag"]["geo_dev_norm"])
        devc = max(devc, Dper["diag"]["geo_dev_cos"])
        # 多重集逐位：用同一置换还原后应**逐位相同**
        gp = torch.Generator().manual_seed(int(s) + PERM_SALT)
        for i in range(int(N_KV)):
            pm = torch.randperm(RA.R_PRES, generator=gp)
            # perm 臂的第 k 行必须**逐位等于** nat 的第 pm[k] 行 ⇒ 多重集不变、只有轮序变
            if not torch.equal(Dper["K"][:, i, :], Dnat["K"][:, i, :][pm]):
                mult_ok = False
    chk("A26Q-5", (devg <= TOL_GEO) and (devc <= TOL_GEO) and mult_ok,
        f"W2：几何偏差 ‖K‖ {devg:.2e} / cos {devc:.2e}（≤{TOL_GEO:g}）、多重集逐位还原 {mult_ok}")

    ru_nat9 = float(np.mean([_nat_diag(s, 0.9)["rho_u"] for s in gs]))
    ru_nat0 = float(np.mean([_nat_diag(s, 0.0)["rho_u"] for s in gs]))
    ru_per = float(np.mean([build_stream_perm(RA.N_KV, s, RA.D_MODEL, int(N_KV), RA.R_PRES,
                                              RA.theta_of(0.9), PHI_MAIN)["diag"]["rho_u"] for s in gs]))
    lam_nat9 = float(np.mean([build_stream_perm(RA.N_KV, s, RA.D_MODEL, int(N_KV), RA.R_PRES,
                                                RA.theta_of(0.9), PHI_MAIN)["diag"]["lam_mean"] for s in gs]))
    chk("A26Q-6", (abs(ru_per) <= TOL_RHO) and (ru_nat9 >= 0.80) and (abs(ru_nat0) <= TOL_RHO),
        f"Q-β：ρ_u nat@0.9={ru_nat9:.3f} ≥0.80、perm@0.9={ru_per:+.3f} ≤{TOL_RHO}、"
        f"nat@0={ru_nat0:+.3f}；Λ(perm)={lam_nat9:.3f}")

    # A26Q-7  位级锚点（1 种子 × 4 档 × 2 臂 × 9 读数）
    arch = json.load(open(_ROOT / "dir26o" / "out" / "DIR26Or11_formal.json", encoding="utf-8"))["rows"]
    n_bad = n_cmp = 0
    for s in seeds_anchor:
        for p in RA.PHIS:
            ref = next((r for r in arch if r["seed"] == s and abs(r["phi"] - p) < 1e-9), None)
            if ref is None:
                n_bad += 1
                continue
            got = run_nat(s, p)
            for arm in ("E0", "tau"):
                for f in ANCHOR_FIELDS:
                    n_cmp += 1
                    if got["arms"][arm][f] != ref["arms"][arm][f]:
                        n_bad += 1
    chk("A26Q-7", n_bad == 0, f"W3：位级锚点 {n_cmp - n_bad}/{n_cmp} 逐位相等（坏 {n_bad}）")

    # A26Q-8  确定性（1 种子跑两次）
    r1 = build_stream_perm(RA.N_KV, seeds_anchor[0], RA.D_MODEL, int(N_KV), RA.R_PRES, RA.theta_of(0.9), PHI_MAIN)
    r2 = build_stream_perm(RA.N_KV, seeds_anchor[0], RA.D_MODEL, int(N_KV), RA.R_PRES, RA.theta_of(0.9), PHI_MAIN)
    chk("A26Q-8", torch.equal(r1["K"], r2["K"]) and r1["diag"]["rho_u"] == r2["diag"]["rho_u"],
        "同种子两次重建键与诊断逐位相同（置换只依赖 seed）")

    # A26Q-9 口径 O（一条真实 perm 单元）
    u = run_perm(seeds_anchor[0])
    chk("A26Q-9", bool(u["arms"]["E0"]["mu_ret1_consistent"] and u["arms"]["tau"]["mu_ret1_consistent"]),
        f"mu_ret1_consistent（seed {seeds_anchor[0]}、perm@0.9）")
    chk("A26Q-9b", bool(u["arms"]["E0"]["W_unchanged"]), "测量不扰动 W")

    # A26Q-12 测量闭包（字段集与参照单元逐位同构）
    ref_u = run_nat(seeds_anchor[0], PHI_MAIN)
    same_keys = set(u["arms"]["E0"]) == set(ref_u["arms"]["E0"]) and set(u["arms"]["tau"]) == set(ref_u["arms"]["tau"])
    chk("A26Q-12", same_keys, f"测量闭包字段集同构（{len(u['arms']['E0'])} 个字段）")

    # A26Q-11 非空
    chk("A26Q-11", len(ck) > 0, f"断言集非空（{len(ck)} 条）")
    passed = all(c["ok"] for c in ck)
    return {"checks": ck, "passed": passed, "rho_u": {"nat09": ru_nat9, "perm09": ru_per, "nat00": ru_nat0},
            "lam_perm": lam_nat9, "anchor": {"n_cmp": n_cmp, "n_bad": n_bad}}


# ══════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("stageA", "stageA11", "analyze", "stage0", "smoke", "formal"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if args.stage == "stageA":
        v = stageA()
        json.dump(v, open(OUT / "DIR26Qr10_stageA.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for c in v["checks"]:
            print(("  ✓ " if c["ok"] else "  ✗ ") + c["id"] + " " + c["msg"])
        print(f"=== Stage A {'PASS' if v['passed'] else 'FAIL'}"
              f"（{sum(c['ok'] for c in v['checks'])}/{len(v['checks'])}） {time.time() - t0:.1f}s ===")
        sys.exit(0 if v["passed"] else 1)

    if args.stage == "stageA11":
        v = stageA11()
        json.dump(v, open(OUT / "DIR26Qr11_stageA.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for c in v["checks"]:
            print(("  ✓ " if c["ok"] else "  ✗ ") + c["id"] + " " + c["msg"])
        print(f"=== Stage A(rev 1.1) {'PASS' if v['passed'] else 'FAIL'}"
              f"（{sum(c['ok'] for c in v['checks'])}/{len(v['checks'])}） {time.time() - t0:.1f}s ===")
        sys.exit(0 if v["passed"] else 1)

    if args.stage == "analyze":
        v = analyze(seeds=args.seeds)
        json.dump(v, open(OUT / "DIR26Qr11_analyze.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("=== 轮序置换的可达性（穷举 R! = %d）===" % v["perm_space"])
        print("| φ | ρ_u(自然) | ρ_u 最小 | ρ_u 最大 | 最大去相关 | Λ(集合) | 解析下界 φ²(2+φ)/3 | 偏差 | 可达≤0.05 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k, d in v["by_phi"].items():
            print(f"| {k} | {d['rho_identity']:+.3f} | {d['rho_min']:+.3f} | {d['rho_max']:+.3f} | "
                  f"{d['gain_max']:+.3f} | {d['lam_set']:.3f} (±{d['lam_spread_max']:.1e}) | "
                  f"{d['analytic_bound']:.3f} | {d['bound_dev']:+.3f} | {d['reach_005']} |")
        print(f"★ 解析下界成立（三档偏差 ≤0.05）: {v['verdict']['bound_ok']}")
        print(f"★ φ=0.9 可达最小 ρ_u = {v['verdict']['min_at_09']:+.3f} ⇒ 通道 = {v['verdict']['channel']}")
        sys.exit(0)

    seeds = args.seeds or (list(SEEDS[:3]) if args.stage == "smoke" else list(SEEDS))
    if args.stage == "stage0":
        seeds = args.seeds or list(SEEDS[:10])
    rows, nats = [], []
    print(f"=== {args.stage}（perm@φ=0.9 vs nat@φ=0.9 同种子配对 × {len(seeds)} 种子）===")
    for s in seeds:
        tt = time.time()
        u = run_perm(s)
        v0 = run_nat(s, PHI_MAIN)
        rows.append(u); nats.append(v0)
        e0, r0 = u["arms"]["E0"], v0["arms"]["E0"]
        print(f"  seed={s:3d} | perm {e0['ret1']:.5f} nat {r0['ret1']:.5f} Δ{e0['ret1'] - r0['ret1']:+.5f} "
              f"| mu {e0['mu']:+.5f}/{r0['mu']:+.5f} | ρ_u {u['diag']['rho_u']:+.3f} Λ {u['diag']['lam_mean']:.3f} "
              f"| {time.time() - tt:.0f}s", flush=True)

    B = float(ARCH["B_E0_ret1_phi00"])          # 由 A26Q-7 位级锚点复核（卡片 §6-3）
    v = crit_perm(rows, nats, B)
    res = {"stage": args.stage, "rows": rows, "nat_rows": nats, "verdict": v,
           "meta": {"card": "DIRECTION_rev26q_v10.md", "phi": PHI_MAIN,
                    "KAP_LO": KAP_LO, "KAP_HI": KAP_HI, "RES_FRAC": RES_FRAC,
                    "ARCH_ref": ARCH, "B_source": "ARCH[0.8067]（A26Q-7 位级锚点复核）"}}
    p = OUT / f"DIR26Qr10_{args.stage}{args.tag}.json"
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n===== 读数 =====")
    print(f"  A(同种子 nat@0.9, {len(nats)})={v['A']:.5f}  B(归档 nat@0, 锚点复核)={B:.5f}  Δ={B - v['A']:+.5f}")
    print(f"  C(perm@0.9)={v['C']:.5f}  κ={(v['kappa']):+.3f}  配对 3se={v['pair_3se']:.5f}"
          f"（要求 ≤{RES_FRAC:.2f}Δ={(RES_FRAC * (B - v['A'])):.5f}）res_ok={v['res_ok']}")
    print(f"  ρ_u(perm)={v['rho_u_perm']:+.3f}  Λ(perm)={v['lam_perm']:.3f}  n={v['n_seeds']}")
    ch = v["channel"]
    if ch == "POWER-LIMITED":
        den = RES_FRAC * (B - v["A"])
        need = int(math.ceil(len(rows) * (v["pair_3se"] / den) ** 2)) if den else 0
        print(f"  ★ 通道 = POWER-LIMITED ⇒ 需要 ≥{need} 种子（卡片 §2.3 预登记规则）")
    else:
        print(f"  ★ 通道 = {ch}")
    print(f"  → {p}  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
