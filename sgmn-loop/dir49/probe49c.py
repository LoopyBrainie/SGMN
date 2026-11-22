#!/opt/pytorch/bin/python
r"""DIR-49 · 开卡前诊断 C：**正交锚**（`ov_k → 0` 的干扰写）与**长期性**（损伤是否持续）。

★ 要回答两件事：
  (1) **正交锚可行性**：构造一个 `ov_k ≈ 0` 的干扰对 ⇒ 若"正交写几乎不伤、地址重合写摧毁目标"，
      干扰就是**键选择性**的；若两者一样伤，干扰是**均匀**的（与键无关）。
      构造法：`a_4` 的 8 维自由参数 + 梯度下降最小化 `|cos(K_5, K_3)|`（`K_5 = tanh(tanh(a_4 A1+b1) Pk1)`）。
      ★ 只动 `a_4` ⇒ 目标的 `(k,v)` 不变、干扰的**值** `v_5` 不变。
  (2) **长期性**：干扰写之后再写 7 个自然对，读 `c(8)` ⇒ 损伤是**持续**的（长期干扰）还是**瞬时**的（会自愈）。

用法：`probe49c.py --n 250 --out dir49/out/DIR49_probe_c.json`
"""
import argparse, json, math, os, sys
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_task48 as T48      # noqa: E402

DOSES = (0.0, 0.5, 1.0)
N_AFTER = 7                  # 干扰写之后的自然对数


def cosv(a, b):
    den = float(a.norm() * b.norm())
    return float(a @ b) / den if den > 0 else float("nan")


def norm_rows(x):
    return x / x.norm(dim=1, keepdim=True) * math.sqrt(x.shape[1])


def rebuild(P, a):
    """由内容流 `a` 重建 (K, V)（与 `R1_task48.build_KV` 逐字同构）。"""
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(a.shape[0])]
    zero = torch.zeros(T48.D_H)
    K = [torch.tanh((zero if t == 0 else H[t - 1]) @ P["Pk1"]) for t in range(a.shape[0])]
    V = [torch.tanh(H[t] @ P["Pv1"]) for t in range(a.shape[0])]
    return K, V


def setup(sd):
    P, K, V = T48.build_KV(sd, T48.NMARK)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    for t in range(T48.CTX + 1):
        U, Vm = T48.step_pair(U, Vm, K[t], V[t], 1.0)
    kt, vt = K[T48.CTX], V[T48.CTX]
    c0, _ = T48.read_target(U, Vm, kt, vt)
    return P, K, V, U, Vm, kt, vt, c0


def ortho_a4(P, a, target_k, iters=300):
    r"""找 `a_4`（8 维）使 `|cos(K_5, K_3)|` 最小（梯度下降 + 随机重启）。"""
    A1, b1, Pk1 = P["A1"], P["b1"], P["Pk1"]
    base = a[T48.CTX + 1].clone()
    best_x, best_ov = None, 9.9

    def ov_of(x):
        h = torch.tanh(x @ A1 + b1)
        k5 = torch.tanh(h @ Pk1)
        # ★ 可微版本：|cos| 的平方（避免 abs 的梯度尖点）
        return (k5 @ target_k) ** 2 / (float(k5.norm() * target_k.norm()) ** 2 + 1e-30), k5

    g = torch.Generator().manual_seed(int(P["U1"].sum().item() * 0) + 12345)
    for restart in range(4):
        if restart == 0:
            x = base.clone()
        else:
            x = base + 0.5 * torch.randn(T48.D_H, generator=g)
        x = x.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.05)
        for _ in range(iters):
            loss, _ = ov_of(x)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                x.clamp_(-3 * math.sqrt(T48.D_H), 3 * math.sqrt(T48.D_H))
        with torch.no_grad():
            sq, _ = ov_of(x)
            ov = math.sqrt(max(float(sq), 0.0))
            if ov < best_ov:
                best_ov, best_x = ov, x.detach().clone()
    return best_x, best_ov


def run_arm(sd, a4_override):
    r"""ctx+目标写好（用**自然**内容）⇒ 干扰写（index 5，键由 `a4_override` 决定）⇒ 读 c1；
       再写 `N_AFTER` 个自然对（索引 6..12）⇒ 读 c(1+N_AFTER)。"""
    P, K, V, U, Vm, kt, vt, c0 = setup(sd)
    if not (math.isfinite(c0) and c0 > 0):
        return None
    a = T48.content(sd, T48.NMARK)
    a = a.clone()
    if a4_override is not None:
        a[T48.CTX + 1] = a4_override
    Kk, Vv = rebuild(P, a)
    n_int = T48.CTX + 2                      # 干扰对索引 = 5
    kn, vn = Kk[n_int], Vv[n_int]
    ovk = abs(cosv(kt, kn)); ovv = abs(cosv(vt, vn))
    U2, V2 = T48.step_pair(U, Vm, kn, vn, 1.0)
    c1, _ = T48.read_target(U2, V2, kt, vt)
    if not math.isfinite(c1):
        return {"undef": True, "ov_k": ovk, "ov_v": ovv}
    for t in range(T48.CTX + 3, T48.CTX + 3 + N_AFTER):
        U2, V2 = T48.step_pair(U2, V2, Kk[t], Vv[t], 1.0)
    cN, _ = T48.read_target(U2, V2, kt, vt)
    return {"undef": False, "ov_k": ovk, "ov_v": ovv,
            "rel1": c1 / c0, "relN": (cN / c0) if math.isfinite(cN) else None, "c0": c0}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def sd3(v):
    v = [x for x in v if x is not None]
    n = len(v)
    if n < 2:
        return None
    m = sum(v) / n
    return 3 * ((sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5) / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed0", type=int, default=29200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    acc = {k: [] for k in ("dose0.0", "dose0.5", "dose1.0", "ortho")}
    ovk_ortho, ovv_ortho = [], []
    nbad = 0
    for i in range(a.n):
        sd = a.seed0 + i
        P, K, V, U, Vm, kt, vt, c0 = setup(sd)
        if not (math.isfinite(c0) and c0 > 0):
            nbad += 1
            continue
        a_full = T48.content(sd, T48.NMARK)
        base4 = a_full[T48.CTX + 1].clone()
        tgt2 = a_full[T48.CTX - 1].clone()
        x_ortho, ov_ortho = ortho_a4(P, a_full, kt)
        for w in DOSES:
            a4 = norm_rows(((1 - w) * base4.unsqueeze(0) + w * tgt2.unsqueeze(0)))[0]
            r = run_arm(sd, a4)
            if r and not r.get("undef"):
                acc["dose%.1f" % w].append(r)
        r = run_arm(sd, x_ortho)
        if r and not r.get("undef"):
            acc["ortho"].append(r)
            ovk_ortho.append(r["ov_k"]); ovv_ortho.append(r["ov_v"])
    out = {"n": a.n, "seed0": a.seed0, "n_bad": nbad, "arms": {}}
    for k, rs in acc.items():
        out["arms"][k] = {"n": len(rs),
                          "ov_k": mean([r["ov_k"] for r in rs]),
                          "ov_v": mean([r["ov_v"] for r in rs]),
                          "rel1": mean([r["rel1"] for r in rs]), "rel1_3se": sd3([r["rel1"] for r in rs]),
                          "relN": mean([r["relN"] for r in rs]), "relN_3se": sd3([r["relN"] for r in rs])}
        print("[%s] n=%d ov_k=%.3f ov_v=%.3f rel1=%.4f(±%.4f) relN=%.4f(±%.4f)" % (
            k, out["arms"][k]["n"], out["arms"][k]["ov_k"], out["arms"][k]["ov_v"],
            out["arms"][k]["rel1"] or -1, out["arms"][k]["rel1_3se"] or -1,
            out["arms"][k]["relN"] or -1, out["arms"][k]["relN_3se"] or -1), flush=True)
    out["ortho_ovk_max"] = max(ovk_ortho) if ovk_ortho else None
    out["ortho_ovk_median"] = sorted(ovk_ortho)[len(ovk_ortho) // 2] if ovk_ortho else None
    print("ortho 达到的 ov_k：中位 %.4f，最大 %.4f" % (out["ortho_ovk_median"] or -1, out["ortho_ovk_max"] or -1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
