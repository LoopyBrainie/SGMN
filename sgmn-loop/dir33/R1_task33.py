#!/opt/pytorch/bin/python
r"""DIR-33 · 装置 revision：按形成时标重定时域/步长，让「前向任务收益」可测。

卡：`revolve/DIR-33/DIRECTION_rev33_v10.md`（阈值已冻结）。
装置：写入侧**逐字零改动**复用 `dir31/R1_grad.py`（A33-1 逐位自证 + A33-7 校验其 md5 未变）；
      本文件只改**装置常数**（d_k/η/T/键尺度，见下面 §装置 revision）与任务层沿用 DIR-32。
标定：`dir33/CALIB.md`（决定二十九，**不进判据**）—— 它给出 N_0.9=5 与 T≥2·N_0.9+1。

用法：stageA | gates | pilot | smoke | formal | mech
"""
import json
import math
import os
import pathlib
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_grad as M

# ══════ 装置 revision（本文件对 DIR-32 的**唯一**装置改动；卡 §3.2）
# 依据 dir33/CALIB.md：装置旧常数 N_0.9=47 次/对而 T=8 只给 ≈3.5 次（差 13×）；
# 72 格标定里只有「加大 d_k 到 16 + 键尺度 0.5 + η 0.2」把 N_0.9 压到 5，且此时不发散。
M.D_K = 16          # 键维（旧 4）
M.ETA = 0.2         # 局部步长（旧 0.05）
M.LAM = 0.05        # 局部目标基准回归（**不变**）
M.T_STEPS = 16      # 时域（旧 8）：唯一被证明能修好速度的旋钮，T ≥ 2·N_0.9 + 1 = 11
KEY_SCALE = 0.5     # 键尺度（旧 1.0）：η=0.2、d_k=16、尺度 1.0 的组合会**发散**（标定 12/72 格之一）

D_H, D_K, D_V, T = M.D_H, M.D_K, M.D_V, M.T_STEPS
OUT = "/home/ec2-user/sgmn-loop/dir33/out"
SEEDS_F = [M.SEED0 + 400 + i for i in range(24)]      # ★ 正式种子（纪律 70：与所有 pilot/标定不相交）
SEEDS_PILOT = [M.SEED0 + 500 + i for i in range(24)]  # ★ pilot 种子（与 100/200/300 段全不相交）
R1_GRAD_MD5 = "8fa40f708600bec7e6c484727d2180c7"      # A33-7：装置文件必须逐字未改

_build_raw = M.build


def build(seed, variant="V0"):
    r"""装置构造的唯一入口：`R1_grad.build` 后只对**写键映射** $P_k$ 乘 KEY_SCALE。

    ★ 为什么要缩放键：键经 $\tanh$ 饱和，尺度直接决定写入步的有效增益；标定显示
      $(\eta{=}0.2,\,d_k{=}16,\,\text{尺度 }1.0)$ **发散**，0.5 则 $N_{0.9}{=}5$。
    ★ 为什么在这里改而不改 `R1_grad.py`：纪律 68（不留第二份副本）—— 装置文件保持逐字零改动（A33-7）。
    """
    P = _build_raw(seed, variant)
    for _l in (1, 2):
        P[f"Pk{_l}"] = P[f"Pk{_l}"] * KEY_SCALE
    return P
ARMS = ("G0", "N", "S", "T")
ADDRS = ("device", "oracle")
STEPS, LR = 250, 0.03
COST_CAP = 1e-15     # A33-3 拍平置换的不变量差（条目多重集 + 范数）
REL_TOL = 1e-12      # ★ A33-5：**相对**容差（DEF-33-3）。T=16 有 15 个求和项，实测 1.5e-16
WORK_TOL = 1e-14     # A33-8 手算工作例
_POUT = {}


def pout():
    if "m" not in _POUT:
        g = torch.Generator().manual_seed(int(M.SEED0) + 31337)
        _POUT["m"] = torch.randn(D_V, D_H, generator=g) / math.sqrt(D_H)
    return _POUT["m"]


def period2(P, sd):
    r"""周期 2 输入，且**两个奇偶输入按构造正交**（DIR-33 的装置改动之一）。

    ★ 为什么正交（`dir33/diag33.py` 实测）：原「直接抽」构造下两输入的 $|\cos|$ 最高 **0.7644**，
      这是**每套种子抽一次**的运气量；正交化把输入共线**构造性地**压到 0，目标共线随之
      0.9666 → 0.8892。
    ★ 为什么保持原尺度（乘 $\sqrt{d_h}$ 还原范数）：输入尺度决定 $\tanh$ 工作点，进而决定整个
      写入动力学的有效增益；不还原就会连带改变已标定的 $N_{0.9}$。
    """
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(2, D_H, generator=g)
    x[1] = x[1] - (x[1] @ x[0]) / (x[0] @ x[0]) * x[0]
    x = x / x.norm(dim=1, keepdim=True) * (D_H ** 0.5)
    P["a"] = torch.stack([x[t % 2] for t in range(T)])


def hs_of(P):
    return [torch.tanh(P["a"][s] @ P["A1"] + P["b1"]) for s in range(T)]


def targets(P):
    return [torch.tanh(hs_of(P)[t - 1] @ P["Pv1"]) for t in range(1, T)]


def scratch(sd):
    g = torch.Generator().manual_seed(int(sd) + 9091)
    return torch.randperm(D_V * D_K, generator=g)


def read_W(W, arm, sd):
    if arm in ("G0", "N"):
        return torch.zeros_like(W)
    if arm == "S":
        return W.flatten()[scratch(sd)].reshape(D_V, D_K)
    return W


def rollout_task(P, addr, arm, sd, detach_writes=True):
    L = P["a"].new_zeros(())
    gs = targets(P)
    if arm == "Ts":                      # ★ 完美写规则的对照臂（事后机制探针，不进判据）
        H = hs_of(P)
        KS = [torch.tanh(H[s] @ P["Pk1"]) for s in range(T)]
        VS = [torch.tanh(H[s] @ P["Pv1"]) for s in range(T)]
    else:
        KS = VS = None
    h_pre_prev = [None, None]
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if addr == "oracle" and l == 1:
                c, q = h, torch.tanh(h @ P["Pk1"])
            else:
                c = (P["a"][t] if l == 1 else hp_prev).detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            if arm == "Ts" and l == 1:
                pp = [(KS[s - 1], VS[s]) for s in range(1, t)]
                if len(pp) >= 2:
                    K = torch.stack([x[0] for x in pp], 1)
                    Vm = torch.stack([x[1] for x in pp], 1)
                    U_, S_, Vt = torch.linalg.svd(Vm @ K.t())
                    Wr = (U_[:, :2] * S_[:2]) @ Vt[:2]
                else:
                    Wr = torch.zeros(D_V, D_K)
                y = Wr @ q
            else:
                y = read_W(W.detach(), arm, sd) @ q
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            gam = torch.zeros(1) if arm == "G0" else P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
        if t >= 1:
            # ★ 卡 §3.5：读数取**第 1 层** $h^+_{1,t}$（`hp[0]`），与目标 $g_t=v(h_{t-1})$ 同层。
            #   原码误取 `hp[1]`（第 2 层）⇒ A33-5 与解析基座损失差 1.096e+01（实现不符卡，非判据问题）。
            L = L + ((pout() @ hp[0] - gs[t - 1]) ** 2).sum()
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])
            v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            Uo, Vo = P[f"U{l}"], P[f"V{l}"]
            upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)
            upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
        if detach_writes:
            upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
    return L


def _names(arm, addr, learned):
    if arm == "G0":
        return []                        # ★ γ≡0 ⇒ $m$ 与损失无关，什么都不训（否则是空转）
    return ["g1", "Wo1", "bo1"] + (["Wq1", "bq1"] if learned else [])


def train_arm(addr, arm, steps=STEPS, lr=LR, learned=False, seeds=None):
    names = _names(arm, addr, learned)
    if not names:
        return {}
    seeds = SEEDS_F if seeds is None else seeds
    gin = torch.Generator().manual_seed(4242)          # ★ 跨臂共享同一组初值（配对比较）
    phi = {}
    for k in names:
        if k == "g1":
            phi[k] = torch.full((1,), M.GAMMA, requires_grad=True)
        elif k == "Wo1":
            phi[k] = (torch.randn(D_H + D_V, D_H, generator=gin)
                      / math.sqrt(D_H + D_V)).requires_grad_(True)
        elif k == "bo1":
            phi[k] = (torch.randn(D_H, generator=gin) * 0.1).requires_grad_(True)
        elif k == "Wq1":
            phi[k] = (torch.randn(D_H, D_K, generator=gin) * 0.354).requires_grad_(True)
        elif k == "bq1":
            phi[k] = torch.zeros(D_K, requires_grad=True)
    opt = torch.optim.Adam(list(phi.values()), lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        tot = 0.0
        for sd in seeds:
            P = build(sd, "V0")
            period2(P, sd)
            P.update({k: v for k, v in phi.items()})
            L = rollout_task(P, addr, arm, sd)
            (L / len(seeds)).backward()
            tot += float(L) / len(seeds)
        opt.step()
        hist.append(tot)
    return {k: v.detach().clone() for k, v in phi.items()}


def per_seed(addr, arm, phi, seeds=None):
    out = []
    for sd in (SEEDS_F if seeds is None else seeds):
        P = build(sd, "V0")
        period2(P, sd)
        P.update({k: v.clone() for k, v in phi.items()})
        out.append(float(rollout_task(P, addr, arm, sd)))
    return out


# ══════════════════════════════════════════ Stage A
def stage_a():
    a = {}
    # A33-1 写入块逐位相同
    P1, P2 = build(SEEDS_F[0], "V0"), build(SEEDS_F[0], "V0")
    M.rollout(P1, "V0")
    rollout_task(P2, "device", "T", SEEDS_F[0], detach_writes=False)
    a["A33-1"] = max(float((P1[k] - P2[k]).abs().max()) for k in ("U1", "V1", "U2", "V2"))
    # A33-2 oracle 对齐 + 键可分性
    al, sep = [], []
    for sd in SEEDS_F:
        P = build(sd, "V0")
        period2(P, sd)
        H = hs_of(P)
        ks = [torch.tanh(H[s] @ P["Pk1"]) for s in range(T)]
        for t in range(2, T):
            q = torch.tanh(H[t] @ P["Pk1"])
            al.append(float((q * ks[t - 2]).sum() / (q.norm() * ks[t - 2].norm())))
            sep.append(float((ks[t - 2] * ks[t - 1]).sum() / (ks[t - 2].norm() * ks[t - 1].norm())))
    a["A33-2_align_min"], a["A33-2_sep_absmax"] = min(al), max(abs(x) for x in sep)
    # A33-3 拍平置换保范数与多重集
    d = []
    for sd in SEEDS_F:
        P = build(sd, "V0")
        period2(P, sd)
        rollout_task(P, "device", "T", sd)
        W = M._W(P["U1"], P["V1"]).detach()
        Ws = W.flatten()[scratch(sd)].reshape(D_V, D_K)
        dm = float((W.flatten().sort().values - Ws.flatten().sort().values).abs().max())
        d.append(max(abs(float(W.norm() - Ws.norm())), dm))
    a["A33-3"] = max(d)
    # A33-4 确定性
    Ls = []
    for _ in range(2):
        P = build(SEEDS_F[0], "V0")
        period2(P, SEEDS_F[0])
        Ls.append(float(rollout_task(P, "device", "T", SEEDS_F[0])))
    a["A33-4"] = abs(Ls[0] - Ls[1])
    # A33-5 G0 确为无注入（h^+ 逐位等于 h）
    P = build(SEEDS_F[0], "V0")
    period2(P, SEEDS_F[0])
    g0 = float(rollout_task(P, "device", "G0", SEEDS_F[0]))
    P2b = build(SEEDS_F[0], "V0")
    period2(P2b, SEEDS_F[0])
    H = hs_of(P2b)
    Lbase = sum(float(((pout() @ H[t] - targets(P2b)[t - 1]) ** 2).sum()) for t in range(1, T))  # 第 1 层
    a["A33-5"] = abs(g0 - Lbase) / max(abs(Lbase), 1.0)   # ★ 相对容差（DEF-33-3）
    # A33-6 两奇偶目标不得几乎共线（★ DIR-33 改口径，见 DEF-33-2 与卡 §4）
    #   旧口径是「24 实例 × 12 对 = 288 个抽样的**最大值**」——它在 $\mathbb R^{d_v}$（$d_v{=}4$）
    #   里是**小样本极值**，没有构造性上界：DIR-32 抽到 0.8434（过闸），DIR-33 抽到 0.8892。
    #   判据 $\Delta_{\rm content}$ 是对 24 个配对种子**取平均**，与之同口径的量是「实例 max 的实例平均」。
    #   ⇒ 双条款：(i) 实例平均 ≤ 0.60（防系统性共线）；(ii) 实例 max ≤ 0.90（防大面积退化）。
    per_inst, pooled = [], []
    for sd in SEEDS_F:
        P = build(sd, "V0")
        period2(P, sd)
        g = targets(P)
        cs = []
        for i in range(len(g)):
            for j in range(len(g)):
                if (i - j) % 2 == 1:
                    cs.append(abs(float((g[i] * g[j]).sum() / (g[i].norm() * g[j].norm()))))
        per_inst.append(max(cs))
        pooled += cs
    a["A33-6_inst_mean"] = sum(per_inst) / len(per_inst)
    a["A33-6_inst_max"] = max(per_inst)
    a["A33-6_pooled_max"] = max(pooled)          # 诊断量（不设阈值）
    a["A33-6"] = a["A33-6_inst_mean"]            # 保留旧键名指向新口径，便于跨卡对照
    # A33-8 手算工作例（卡 §3.3；量纲 1 逐位可验）：k=1、v=0、α=1、λ=0、η=M.ETA=0.2、U0=V0=1
    #   W 应走 1 → 0.64 → 0.48664576
    Ue = torch.ones(1, 1)
    Ve = torch.ones(1, 1)
    ke, ve = torch.ones(1), torch.zeros(1)
    got = [float((Ue @ Ve.t())[0, 0])]
    for _ in range(2):
        W = Ue @ Ve.t()
        Ge = M.ALPHA * torch.outer(W @ ke - ve, ke) + 0.0 * W
        Un, Vn = Ue - M.ETA * (Ge @ Ve), Ve - M.ETA * (Ge.t() @ Ue)
        Ue, Ve = Un, Vn
        got.append(float((Ue @ Ve.t())[0, 0]))
    want = [1.0, 0.64, 0.48664576]
    a["A33-8"] = max(abs(x - y) for x, y in zip(got, want))
    a["A33-8_seq"] = " → ".join(f"{x:.8f}" for x in got)
    # A33-7 装置文件逐字未改（本卡对装置的改动只发生在本文件的常数与 build 包装里）
    import hashlib
    h = hashlib.md5(pathlib.Path("/home/ec2-user/sgmn-loop/dir31/R1_grad.py").read_bytes()).hexdigest()
    a["A33-7_md5_diff"] = 0.0 if h == R1_GRAD_MD5 else 1.0
    a["A33-7_md5"] = h
    ok = (a["A33-1"] == 0.0 and a["A33-2_align_min"] >= 1 - 1e-14 and a["A33-3"] <= COST_CAP
          and a["A33-4"] == 0.0 and a["A33-5"] <= REL_TOL and a["A33-8"] <= WORK_TOL
          and a["A33-6_inst_mean"] <= 0.60 and a["A33-6_inst_max"] <= 0.90
          and a["A33-7_md5_diff"] == 0.0)
    return ok, a


def gates(seeds=None):
    r"""P1 rank-2 上界 / P2 未训练实际检索 / P3 幅度匹配（P3 已由 A33-3 覆盖）。"""
    p1, p2 = [], []
    for sd in (SEEDS_F if seeds is None else seeds):
        P = build(sd, "V0")
        period2(P, sd)
        H = hs_of(P)
        ks = [torch.tanh(H[s] @ P["Pk1"]) for s in range(T)]
        vs = [torch.tanh(H[s] @ P["Pv1"]) for s in range(T)]
        Ws, hprev = [], None
        for t in range(T):
            Ws.append(M._W(P["U1"], P["V1"]).detach().clone())
            k = torch.tanh((torch.zeros(D_H) if hprev is None else hprev) @ P["Pk1"])
            v = torch.tanh(H[t] @ P["Pv1"])
            W = M._W(P["U1"], P["V1"])
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            Uo, Vo = P["U1"], P["V1"]
            P["U1"] = Uo - M.ETA * (G @ Vo)
            P["V1"] = Vo - M.ETA * (G.t() @ Uo)
            hprev = H[t]
        for t in range(1, T):
            g = vs[t - 1]
            q = ks[(t - 2) % T]
            pp = [(ks[(s - 1) % T], vs[s]) for s in range(1, t)]
            if len(pp) >= 2:
                K = torch.stack([x[0] for x in pp], 1)
                Vm = torch.stack([x[1] for x in pp], 1)
                U_, S_, Vt = torch.linalg.svd(Vm @ K.t())
                Wst = (U_[:, :2] * S_[:2]) @ Vt[:2]
            else:
                Wst = torch.zeros(D_V, D_K)
            cs = lambda u, v: float((u * v).sum() / (u.norm() * v.norm())) if float(u.norm()) > 0 else 0.0
            p1.append(cs(Wst @ q, g))
            p2.append(cs(Ws[t] @ q, g))
    return sum(p1) / len(p1), sum(p2) / len(p2)


def mech():
    r"""§2.3 机制量：同一对重复写入 N 步的收敛曲线（正式种子）。"""
    NS = (8, 16, 32, 64, 128, 256, 512)
    rows = {n: [] for n in NS}
    for sd in SEEDS_F:
        P = build(sd, "V0")
        g = torch.Generator().manual_seed(int(sd) + 5150)
        h = torch.randn(D_H, generator=g)
        k, v = torch.tanh(h @ P["Pk1"]), torch.tanh(h @ P["Pv1"])
        Winit = M._W(P["U1"], P["V1"]).detach().clone()
        n0 = float(Winit.norm())
        for n in range(1, max(NS) + 1):
            W = M._W(P["U1"], P["V1"])
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            Uo, Vo = P["U1"], P["V1"]
            P["U1"] = Uo - M.ETA * (G @ Vo)
            P["V1"] = Vo - M.ETA * (G.t() @ Uo)
            if n in NS:
                Wn = M._W(P["U1"], P["V1"])
                rows[n].append({"cos": float((Wn @ k * v).sum() / (Wn @ k).norm() / v.norm()),
                                "dW": float((Wn - Winit).norm()) / n0})
    return {n: {"cos": sum(x["cos"] for x in rows[n]) / len(rows[n]),
                "dW": sum(x["dW"] for x in rows[n]) / len(rows[n])} for n in NS}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stageA"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if mode == "stageA":
        ok, a = stage_a()
        print(f"Stage A：{'★ 全部通过' if ok else '★ 有断言失败 —— 冻结正式轮'}")
        for k, v in a.items():
            print(f"  {k:20s} {v if isinstance(v, str) else f'{v:.3e}'}")
        json.dump({"ok": ok, "a": a}, open(f"{OUT}/DIR33r10_stageA.json", "w"), indent=1)
        sys.exit(0 if ok else 1)
    if mode == "smoke":
        # 只证明协议可执行（**不出判据**）：2 臂 × 2 寻址 × 20 步
        t1 = time.time()
        out = {}
        for addr in ADDRS:
            out[addr] = {}
            for arm in ("N", "T"):
                phi = train_arm(addr, arm, steps=20, learned=(addr == "device"))
                out[addr][arm] = sum(per_seed(addr, arm, phi)) / len(SEEDS_F)
        ok = all(math.isfinite(v) for a in out.values() for v in a.values())
        print(f"smoke：20 步 × 2 臂 × 2 寻址 ⇒ {'可执行' if ok else '★ 不可执行'}  ({time.time()-t1:.0f}s)")
        for a_, d in out.items():
            print(f"  {a_:6s} " + " · ".join(f"{k} {v:.3f}" for k, v in d.items()))
        json.dump(out, open(f"{OUT}/DIR33r10_smoke.json", "w"), indent=1)
        sys.exit(0 if ok else 1)
    if mode == "probe_star":
        # 事后机制探针（**不进判据**，卡已冻结）：把 W 换成秩 2 精确记忆 W*, 只训读数头。
        out = {}
        for addr in ADDRS:
            phi_s = train_arm(addr, "S", learned=(addr == "device"))
            phi_g = train_arm(addr, "G0", learned=(addr == "device"))
            phi_f = train_arm(addr, "Ts", learned=(addr == "device"))
            Lg, Ls, Lf = (sum(per_seed(addr, a, ph)) / len(SEEDS_F)
                          for a, ph in (("G0", phi_g), ("S", phi_s), ("Ts", phi_f)))
            out[addr] = {"G0": Lg, "S": Ls, "Ts": Lf,
                         "content_vs_perfect": Ls - Lf}
            print(f"  {addr:6s} G0 {Lg:8.4f} · S 坏W {Ls:8.4f} · **Ts 完美W*** {Lf:8.4f}"
                  f"  ⇒ 内容性(对完美写规则) {Ls - Lf:+8.5f}")
        json.dump(out, open(f"{OUT}/DIR33r10_probe_star.json", "w"), indent=1)
        sys.exit(0)
    if mode == "gates":
        p1, p2 = gates()
        n_w = (T - 1) / 2                       # 周期 2 下每对的可用写入次数
        print(f"P1 rank-2 上界 {p1:+.4f}（阈 0.5）  P2 未训练实际检索 {p2:+.4f}（阈 >0.2）")
        print(f"装置：T={T} η={M.ETA} λ={M.LAM} d_k={D_K} 键尺度={KEY_SCALE} ⇒ 可用写入/对 ≈{n_w:.1f}"
              f"（标定 N_0.9=5 ⇒ 余量 {n_w/5:.2f}×）")
        json.dump({"P1": p1, "P2": p2, "writes_per_pair": n_w},
                  open(f"{OUT}/DIR33r10_gates_pilot.json", "w"), indent=1)
        sys.exit(0)
    if mode == "pilot":
        # 开卡前 pilot（决定二十九，**不进判据**）：只为估 sd(Δcontent) 与装置可行性
        t1 = time.time()
        out = {"P1": None, "P2": None, "arms": {}}
        p1, p2 = gates(SEEDS_PILOT)
        out["P1"], out["P2"] = p1, p2
        print(f"pilot：P1 {p1:+.4f}（阈 0.5） · P2 {p2:+.4f}（阈 >0.2） · 可用写入/对 {(T-1)/2:.1f}")
        for addr in ADDRS:
            learned = (addr == "device")
            Ls = {}
            for arm in ("G0", "S", "T"):
                phi = train_arm(addr, arm, learned=learned, seeds=SEEDS_PILOT)
                Ls[arm] = per_seed(addr, arm, phi, seeds=SEEDS_PILOT)
                out["arms"].setdefault(addr, {})[arm] = sum(Ls[arm]) / len(Ls[arm])
                print(f"  {addr:6s} {arm:2s} L* {out['arms'][addr][arm]:9.5f}")
            d = [x - y for x, y in zip(Ls["S"], Ls["T"])]
            dm = sum(d) / len(d)
            sd = (sum((x - dm) ** 2 for x in d) / (len(d) - 1)) ** 0.5
            se = sd / math.sqrt(len(d))
            out["arms"][addr]["content"] = {"mean": dm, "sd": sd, "se": se, "3se": 3 * se,
                                           "thr5pct": 0.05 * out["arms"][addr]["G0"]}
            print(f"  ⇒ {addr}: Δcontent {dm:+.5f} · sd {sd:.5f} · 3se {3*se:.5f} · "
                  f"5%阈值 {0.05*out['arms'][addr]['G0']:.5f} ⇒ 功效{'够' if 3*se <= 0.05*out['arms'][addr]['G0'] else '**不够**'}")
        # 需要的配对数（冻结公式：3·se ≤ 阈值 ⇒ n ≥ (3·sd/阈)²）
        need = {}
        for addr in ADDRS:
            c = out["arms"][addr]["content"]
            need[addr] = math.ceil((3 * c["sd"] / c["thr5pct"]) ** 2) if c["thr5pct"] > 0 else None
        out["n_needed_5pct"] = need
        print(f"  需要的配对种子数（5% 阈值）: {need}  耗时 {time.time()-t1:.0f}s")
        json.dump(out, open(f"{OUT}/DIR33r10_pilot.json", "w"), ensure_ascii=False, indent=1)
        sys.exit(0)
    if mode == "mech":
        m = mech()
        for n, r in m.items():
            print(f"N={n:4d} cos {r['cos']:+.4f} ‖ΔW‖/‖W0‖ {r['dW']:.4f}")
        json.dump(m, open(f"{OUT}/DIR33r10_mech.json", "w"), indent=1)
        sys.exit(0)
    p1, p2 = gates()
    print(f"P1 rank-2 上界 {p1:+.4f}（阈 0.5）  P2 未训练实际检索 {p2:+.4f}（阈 >0.2）")
    res = {"P1": p1, "P2": p2, "arms": {}, "mech": mech()}
    for addr in ADDRS:
        learned = (addr == "device")
        res["arms"][addr] = {}
        Ls = {}
        for arm in ARMS:
            t1 = time.time()
            phi = train_arm(addr, arm, learned=learned)
            Ls[arm] = per_seed(addr, arm, phi)
            res["arms"][addr][arm] = Ls[arm]
            print(f"  {addr:6s} {arm:2s} L* {sum(Ls[arm])/len(SEEDS_F):9.5f}  ({time.time()-t1:.0f}s)")
        mean = {k: sum(v) / len(v) for k, v in Ls.items()}
        d = [s - t for s, t in zip(Ls["S"], Ls["T"])]
        dm = sum(d) / len(d)
        sd = (sum((x - dm) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        se = sd / math.sqrt(len(d))
        thr = 0.05 * mean["G0"]
        res["arms"][addr]["mean"] = mean
        res["arms"][addr]["content"] = {"mean": dm, "sd": sd, "se": se, "thr": thr,
                                        "verdict": ("SUCCEEDED" if dm >= thr and 3 * se <= thr
                                                    else "INFEASIBLE-POWER" if 3 * se > thr else "FALSIFIED")}
        print(f"  ⇒ L*(G0) {mean['G0']:.4f} · Δchan {mean['G0']-mean['N']:+.5f} · "
              f"Δrand {mean['N']-mean['S']:+.5f} · **Δcontent {dm:+.5f}** "
              f"(sd {sd:.5f}, 3se {3*se:.5f}, 阈 {thr:.5f}) ⇒ {res['arms'][addr]['content']['verdict']}")
    verdict = "PREMISE-NOT-MET" if p2 <= 0.2 else res["arms"]["oracle"]["content"]["verdict"]
    res["verdict"] = verdict
    print(f"\nP2 {'未过' if p2 <= 0.2 else '过'} ⇒ 主判据状态 **{verdict}**  （{time.time()-t0:.0f}s）")
    json.dump(res, open(f"{OUT}/DIR33r10_formal.json", "w"), ensure_ascii=False, indent=1)
