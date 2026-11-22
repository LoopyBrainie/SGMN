#!/opt/pytorch/bin/python
r"""DIR-49 · 实现：**长期干扰的结构** —— 单次干扰写的伤害 vs 键重叠度（四臂配对剂量-反应）。

卡：`revolve/DIR-49/DIRECTION_rev49_v10.md`（rev 1.0，md5 见上，**阈值已冻结**）。
★ **逐字派生**：装置（`dir31/R1_grad.py` + `dir33/R1_task33.py`）、内容流与 `NMARK=36` 固定长度（`dir48/R1_task48`，
   含 `DEF-48-4` 的修法）、写循环 `step_pair`、读数 `read_target` —— **一字不改**。
★ **唯一自变量 = 干扰对的键 `k_5`**：只改 `a_4`（⇒ 只改 `k_5`），目标 `(k_3,v_3)`、干扰值 `v_5`、追加写的 `(k,v)` 全部不变。

用法：`R1_task49.py --arm {ortho,d0.0,d0.5,d1.0} --n 1000 --seed0 29400 --out PATH [--tag r10]`
"""
import argparse, json, math, os, sys, time
import torch
torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402

# ══════ 冻结常数（卡 §2；**开卡后一字不改**）
CTX = T48.CTX                 # 3
N_INT = CTX + 2               # 5：干扰对索引
N_AFTER = 7                   # 追加的自然对数（index 6..12）
NMARK = T48.NMARK             # 36
ARMS = ("ortho", "d0.0", "d0.5", "d1.0")
DOSES = {"d0.0": 0.0, "d0.5": 0.5, "d1.0": 1.0}
N_WRITES = (CTX + 1) + 1 + N_AFTER      # = 12


def norm_rows(x):
    return x / x.norm(dim=1, keepdim=True) * math.sqrt(x.shape[1])


def rebuild(P, a):
    r"""由内容流 `a` 重建 `(K,V)`（与 `R1_task48.build_KV` **逐字同构**）。"""
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(a.shape[0])]
    zero = torch.zeros(T48.D_H)
    K = [torch.tanh((zero if t == 0 else H[t - 1]) @ P["Pk1"]) for t in range(a.shape[0])]
    V = [torch.tanh(H[t] @ P["Pv1"]) for t in range(a.shape[0])]
    return K, V


def cosv(a, b):
    den = float(a.norm() * b.norm())
    return float(a @ b) / den if den > 0 else float("nan")


def ortho_a4(P, a, target_k, iters=300, restarts=4, seed=12345):
    r"""找 `a_4`（8 维）使 `|cos(k_5, k_3)|` 最小（Adam，最小化 `cos²`）。"""
    A1, b1, Pk1 = P["A1"], P["b1"], P["Pk1"]
    base = a[CTX + 1].clone()
    best_x, best_ov = None, 9.9
    g = torch.Generator().manual_seed(seed)
    for rs in range(restarts):
        x = (base.clone() if rs == 0 else base + 0.5 * torch.randn(T48.D_H, generator=g)).requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.05)
        for _ in range(iters):
            k5 = torch.tanh(torch.tanh(x @ A1 + b1) @ Pk1)
            sq = (k5 @ target_k) ** 2 / (float(k5.norm() * target_k.norm()) ** 2 + 1e-30)
            opt.zero_grad(); sq.backward(); opt.step()
            with torch.no_grad():
                x.clamp_(-3 * math.sqrt(T48.D_H), 3 * math.sqrt(T48.D_H))
        with torch.no_grad():
            k5 = torch.tanh(torch.tanh(x @ A1 + b1) @ Pk1)
            ov = abs(float(k5 @ target_k) / (float(k5.norm() * target_k.norm()) + 1e-300))
            if ov < best_ov:
                best_ov, best_x = ov, x.detach().clone()
    return best_x, best_ov


def trace_one(sd, arm):
    r"""单实例单臂。返回逐量读数（含 `c0/c1/cN` 原始余弦，供裁决器重算）。"""
    assert arm in ARMS
    P, K, V = T48.build_KV(sd, NMARK)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    n_writes = 0
    for t in range(CTX + 1):                     # ctx + 目标（四臂**逐位相同**）
        U, Vm = T48.step_pair(U, Vm, K[t], V[t], 1.0)
        n_writes += 1
    kt, vt = K[CTX], V[CTX]
    c0, w0 = T48.read_target(U, Vm, kt, vt)

    a = T48.content(sd, NMARK).clone()
    if arm == "ortho":
        a4, ov_ortho = ortho_a4(P, a, kt)
        a[CTX + 1] = a4
    else:
        w = DOSES[arm]
        base4 = a[CTX + 1].clone(); tgt2 = a[CTX - 1].clone()
        a[CTX + 1] = norm_rows(((1 - w) * base4.unsqueeze(0) + w * tgt2.unsqueeze(0)))[0]
    Kk, Vv = rebuild(P, a)

    kn, vn = Kk[N_INT], Vv[N_INT]
    ov_k = abs(cosv(kt, kn)); ov_v = abs(cosv(vt, vn))
    Wm = M._W(U, Vm)
    res = float((Wm @ kn - vn).norm())
    U2, V2 = T48.step_pair(U, Vm, kn, vn, 1.0)     # 干扰写
    n_writes += 1
    c1, w1 = T48.read_target(U2, V2, kt, vt)
    for t in range(CTX + 3, CTX + 3 + N_AFTER):    # 追加的自然对
        U2, V2 = T48.step_pair(U2, V2, Kk[t], Vv[t], 1.0)
        n_writes += 1
    cN, wN = T48.read_target(U2, V2, kt, vt)

    row = {"seed": sd, "arm": arm, "n_writes": n_writes,
           "c0": c0, "c1": c1, "cN": cN, "wnorm": [w0, w1, wN],
           "ov_k": ov_k, "ov_v": ov_v, "res": res,
           "k_tgt": [float(x) for x in kt], "v_tgt": [float(x) for x in vt],
           "k_int": [float(x) for x in kn], "v_int": [float(x) for x in vn],
           "finite": all(math.isfinite(x) for x in (c0, c1, cN)),
           "finite_c0c1": (math.isfinite(c0) and math.isfinite(c1)),
           "finite_cN": math.isfinite(cN)}
    # ★ `DEF-49-4`：`rel1` **只**依赖 `(c0,c1)`，`relN` **只**依赖 `(c0,cN)` ⇒ 后 7 次自然写发散（`cN=NaN`）
    #   **不得**连带作废判据量 `rel1`（卡 §5 规则 2 的发散条件只提 `c0`；红线②明说 `relN` 不入判据）。
    ok0 = math.isfinite(c0) and c0 > 0
    rel1 = (c1 / c0) if (ok0 and math.isfinite(c1)) else None
    relN = (cN / c0) if (ok0 and math.isfinite(cN)) else None
    row["rel1"] = rel1; row["relN"] = relN
    row["bad_criterion"] = (rel1 is None)      # ★ 判据量不可算（`c0`/`c1` 非有限或 `c0≤0`）
    row["bad_report"] = (relN is None)          # ★ 报告量不可算（`cN` 非有限）—— **不影响**判据
    # ★ A49-1 写入数
    assert n_writes == N_WRITES, (sd, arm, n_writes)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=29400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="r10")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.n):
            row = trace_one(a.seed0 + i, a.arm)
            row["tag"] = a.tag
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"[{a.arm}] {i+1}/{a.n}  {time.time()-t0:.1f}s", flush=True)
    print(f"DONE {a.arm} n={a.n} seed0={a.seed0} → {a.out}  墙钟 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
