#!/opt/pytorch/bin/python
r"""DIR-51（人格漂移代理量）· §0 定向探针：四类写入对「人格 trait」读出的效应 + 可标定性检查。

装置线 = `dir48/R1_task48.py`（**逐字**沿用 `build_KV` / `step_pair` / `read_target`）。
★ 只回答一件事：装置能否给出「人格漂移」的**可测代理量**，并**预判**：
   (i) trait 在**自然背景 `α=1`** 下能否撑过 >3 步；否则任何「漂移」读数都饱和。
   (ii) 同键换值**非人格**写入（`random`）对 trait 的伤害 vs 同键换值**人格**写入（`same-k-new-v`）。
   (iii) 同键同值（`replay`）是修复还是维持。
   (iv) 结构扰动（`ortho` 的 `k'⟂k`）的自然背景伤害有多大。
"""
import argparse, json, math, os, sys, time
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M
import R1_task33 as R
import R1_task46 as T46
import R1_task47 as T47
import R1_task48 as T48

D_H = M.D_H
D_K = M.D_K
D_V = M.D_V
N_REP = T47.N_REP
KEY_SCALE = R.KEY_SCALE

def gen_vec(g, D):
    v = torch.randn(D, generator=g); return v / v.norm()

def make_ctx_trait(seed, n_persona, n_others, arm):
    """`n_persona` 个 persona 写入（同键`k_P`不同值）+ `n_others` 个非人格自然写入（同装置 `build_KV`）。
    返回 dict：逐字节存 P,k,v 以便独立重算。"""
    g = torch.Generator().manual_seed(seed)
    P = R.build(seed, "V0")
    kP = KEY_SCALE * gen_vec(g, D_K)
    vs = [gen_vec(g, D_V) for _ in range(n_persona)]
    a = T48.content(seed + 7919, n_others)     # perstep（**逐字** dir48 内容生成器）；独立种子段避免与 persona 撞
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(n_others)]
    z = torch.zeros(D_H)
    K = [torch.tanh((z if t == 0 else H[t - 1]) @ P["Pk1"]) for t in range(n_others)]
    V = [torch.tanh(H[t] @ P["Pv1"]) for t in range(n_others)]
    return {"P": {kk: vv.clone() for kk, vv in P.items()}, "kP": kP, "vs": [v.clone() for v in vs],
            "K": K, "V": V, "g": torch.Generator().manual_seed(seed + 12345)}

def run(seed, arm, n_persona, n_others):
    d = make_ctx_trait(seed, n_persona, n_others, arm)
    P, kP, vs, K, V, g = d["P"], d["kP"], d["vs"], d["K"], d["V"], d["g"]
    U, Vm = P["U1"].clone(), P["V1"].clone()
    # 写入 persona
    for v in vs:
        U, Vm = T48.step_pair(U, Vm, kP, v, 1.0)
    # 记录 persona trait 读数曲线 c(t)：每次 persona 写入后读
    def read_P():
        Wm = M._W(U, Vm); return float(Wm @ kP @ vs[0])   # ★ cosine 口径：cos(Wk_P, v_P)（vs[0] 为单位向量）
    cs = []
    t_persona = 0
    # 自然写入（非人格），期间读 persona trait
    c_before = read_P()
    for j in range(n_others):
        k_j, v_j = K[j], V[j]
        if arm == "random":          # 同键换值（非人格值）
            kv, vv = kP, gen_vec(g, D_V)
        elif arm == "same-k-new-v":  # 同键换值（**人格**值 = 换一个 persona 值）
            kv, vv = kP, vs[(j + 1) % n_persona]
        elif arm == "replay":        # 同键同值
            kv, vv = kP, vs[0]
        elif arm == "ortho":         # 正交键自然写入
            base = gen_vec(g, D_K); perp = base - float(base @ kP) * kP; perp = perp / perp.norm()
            kv, vv = KEY_SCALE * perp, gen_vec(g, D_V)
        else:
            raise ValueError(arm)
        U, Vm = T48.step_pair(U, Vm, kv, vv, 1.0)
        cs.append(read_P())
    return {"c_before": c_before, "cs": cs, "arm": arm, "seed": seed,
            "ratio": [c / c_before if c_before > 0 else float("nan") for c in cs]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=41000)
    ap.add_argument("--out", default=os.path.join(ROOT, "dir51/out/DIR51_probe_a.json"))
    ap.add_argument("--n-persona", type=int, default=1)
    ap.add_argument("--n-others", type=int, default=20)
    A = ap.parse_args()
    arms = ("none", "random", "same-k-new-v", "replay", "ortho")
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    res = {a: [] for a in arms}
    t0 = time.time()
    for a in arms:
        for i in range(A.n):
            sd = A.seed0 + i
            if a == "none":
                d = make_ctx_trait(sd, A.n_persona, 0, a)
                U, Vm = d["P"]["U1"].clone(), d["P"]["V1"].clone()
                for v in d["vs"]:
                    U, Vm = T48.step_pair(U, Vm, d["kP"], v, 1.0)
                Wm = M._W(U, Vm); c0 = float(Wm @ d["kP"] @ d["vs"][0])
                res[a].append({"c_before": c0, "cs": [c0] * A.n_others, "ratio": [1.0] * A.n_others, "seed": sd, "arm": a})
            else:
                r = run(sd, a, A.n_persona, A.n_others)
                res[a].append(r)
        ratios = [r["ratio"] for r in res[a]]
        # 汇总：每步的中位 ratio 与末步
        import statistics as st
        per_step = [st.median([rr[t] for rr in ratios]) for t in range(A.n_others)]
        last = [rr[-1] for rr in ratios]
        res[a + "_summary"] = {"median_ratio_per_step": per_step,
                               "last_step_median": st.median(last),
                               "last_step_min": min(last), "last_step_max": max(last), "n": A.n}
        print(f"[{a:12s}] n={A.n}  persona读出中位 ratio@step{ {5:per_step[4] if len(per_step)>4 else None} }...  "
              f"末步中位 {st.median(last):.4f}  [min {min(last):.4f}, max {max(last):.4f}]  ({time.time()-t0:.1f}s)")
    json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("WROTE", A.out)

if __name__ == "__main__":
    main()
