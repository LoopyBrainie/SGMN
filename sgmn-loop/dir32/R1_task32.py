#!/opt/pytorch/bin/python
r"""DIR-32 · 集成接口验收（二）：**前向任务收益**的「内容性 vs 通用注入」拆解。

卡：`revolve/DIR-32/DIRECTION_rev32_v10.md`（阈值已冻结）。
装置：写入侧**零改动**复用 `dir31/R1_grad.py`（A32-1 逐位自证）；本文件只加任务层。

用法：stageA | smoke | formal | mech
"""
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_grad as M

D_H, D_K, D_V, T = M.D_H, M.D_K, M.D_V, M.T_STEPS
OUT = "/home/ec2-user/sgmn-loop/dir32/out"
SEEDS_F = [M.SEED0 + 200 + i for i in range(24)]      # ★ 正式种子（纪律 70：与 pilot 不相交）
SEEDS_PILOT = [M.SEED0 + 100 + i for i in range(24)]
ARMS = ("G0", "N", "S", "T")
ADDRS = ("device", "oracle")
STEPS, LR = 250, 0.03
COST_CAP = 1e-15
_POUT = {}


def pout():
    if "m" not in _POUT:
        g = torch.Generator().manual_seed(int(M.SEED0) + 31337)
        _POUT["m"] = torch.randn(D_V, D_H, generator=g) / math.sqrt(D_H)
    return _POUT["m"]


def period2(P, sd):
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(2, D_H, generator=g)
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
            #   原码误取 `hp[1]`（第 2 层）⇒ A32-5 与解析基座损失差 1.096e+01（实现不符卡，非判据问题）。
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


def train_arm(addr, arm, steps=STEPS, lr=LR, learned=False):
    names = _names(arm, addr, learned)
    if not names:
        return {}
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
        for sd in SEEDS_F:
            P = M.build(sd, "V0")
            period2(P, sd)
            P.update({k: v for k, v in phi.items()})
            L = rollout_task(P, addr, arm, sd)
            (L / len(SEEDS_F)).backward()
            tot += float(L) / len(SEEDS_F)
        opt.step()
        hist.append(tot)
    return {k: v.detach().clone() for k, v in phi.items()}


def per_seed(addr, arm, phi):
    out = []
    for sd in SEEDS_F:
        P = M.build(sd, "V0")
        period2(P, sd)
        P.update({k: v.clone() for k, v in phi.items()})
        out.append(float(rollout_task(P, addr, arm, sd)))
    return out


# ══════════════════════════════════════════ Stage A
def stage_a():
    a = {}
    # A32-1 写入块逐位相同
    P1, P2 = M.build(SEEDS_F[0], "V0"), M.build(SEEDS_F[0], "V0")
    M.rollout(P1, "V0")
    rollout_task(P2, "device", "T", SEEDS_F[0], detach_writes=False)
    a["A32-1"] = max(float((P1[k] - P2[k]).abs().max()) for k in ("U1", "V1", "U2", "V2"))
    # A32-2 oracle 对齐 + 键可分性
    al, sep = [], []
    for sd in SEEDS_F:
        P = M.build(sd, "V0")
        period2(P, sd)
        H = hs_of(P)
        ks = [torch.tanh(H[s] @ P["Pk1"]) for s in range(T)]
        for t in range(2, T):
            q = torch.tanh(H[t] @ P["Pk1"])
            al.append(float((q * ks[t - 2]).sum() / (q.norm() * ks[t - 2].norm())))
            sep.append(float((ks[t - 2] * ks[t - 1]).sum() / (ks[t - 2].norm() * ks[t - 1].norm())))
    a["A32-2_align_min"], a["A32-2_sep_absmax"] = min(al), max(abs(x) for x in sep)
    # A32-3 拍平置换保范数与多重集
    d = []
    for sd in SEEDS_F:
        P = M.build(sd, "V0")
        period2(P, sd)
        rollout_task(P, "device", "T", sd)
        W = M._W(P["U1"], P["V1"]).detach()
        Ws = W.flatten()[scratch(sd)].reshape(D_V, D_K)
        dm = float((W.flatten().sort().values - Ws.flatten().sort().values).abs().max())
        d.append(max(abs(float(W.norm() - Ws.norm())), dm))
    a["A32-3"] = max(d)
    # A32-4 确定性
    Ls = []
    for _ in range(2):
        P = M.build(SEEDS_F[0], "V0")
        period2(P, SEEDS_F[0])
        Ls.append(float(rollout_task(P, "device", "T", SEEDS_F[0])))
    a["A32-4"] = abs(Ls[0] - Ls[1])
    # A32-5 G0 确为无注入（h^+ 逐位等于 h）
    P = M.build(SEEDS_F[0], "V0")
    period2(P, SEEDS_F[0])
    g0 = float(rollout_task(P, "device", "G0", SEEDS_F[0]))
    P2b = M.build(SEEDS_F[0], "V0")
    period2(P2b, SEEDS_F[0])
    H = hs_of(P2b)
    Lbase = sum(float(((pout() @ H[t] - targets(P2b)[t - 1]) ** 2).sum()) for t in range(1, T))  # 第 1 层
    a["A32-5"] = abs(g0 - Lbase)
    # A32-6 两奇偶目标不得几乎共线
    c = []
    for sd in SEEDS_F:
        P = M.build(sd, "V0")
        period2(P, sd)
        g = targets(P)
        # $g_t$ 的指标从 $t=1$ 起：$g[0]=g_1$（奇）、$g[1]=g_2$（偶）。**只比异奇偶**：
        # 周期 2 下同奇偶的目标是**同一个量**，比它必然 $\cos=1$（那是定义的必然，不是泄漏）。
        for i in range(len(g)):
            for j in range(len(g)):
                if (i - j) % 2 == 1:
                    c.append(abs(float((g[i] * g[j]).sum() / (g[i].norm() * g[j].norm()))))
    a["A32-6"] = max(c)
    ok = (a["A32-1"] == 0.0 and a["A32-2_align_min"] >= 1 - 1e-14 and a["A32-3"] <= COST_CAP
          and a["A32-4"] == 0.0 and a["A32-5"] <= COST_CAP and a["A32-6"] <= 0.9)
    return ok, a


def gates():
    r"""P1 rank-2 上界 / P2 未训练实际检索 / P3 幅度匹配（P3 已由 A32-3 覆盖）。"""
    p1, p2 = [], []
    for sd in SEEDS_F:
        P = M.build(sd, "V0")
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
        P = M.build(sd, "V0")
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
            print(f"  {k:20s} {v:.3e}")
        json.dump({"ok": ok, "a": a}, open(f"{OUT}/DIR32r10_stageA.json", "w"), indent=1)
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
        json.dump(out, open(f"{OUT}/DIR32r10_smoke.json", "w"), indent=1)
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
        json.dump(out, open(f"{OUT}/DIR32r10_probe_star.json", "w"), indent=1)
        sys.exit(0)
    if mode == "mech":
        m = mech()
        for n, r in m.items():
            print(f"N={n:4d} cos {r['cos']:+.4f} ‖ΔW‖/‖W0‖ {r['dW']:.4f}")
        json.dump(m, open(f"{OUT}/DIR32r10_mech.json", "w"), indent=1)
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
    json.dump(res, open(f"{OUT}/DIR32r10_formal.json", "w"), ensure_ascii=False, indent=1)
