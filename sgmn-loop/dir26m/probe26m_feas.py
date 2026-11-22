r"""DIR-26-M · probe26m_feas.py —— 在**冻结卡片之前**测两件"能不能判"的事（不出判决、不定阈值）

背景：D-43/D-26-L 证明剂量 $G$ 随写世界规模**次线性**增长（$G\propto n_{\rm kv}^{0.43}$，到 120 未封顶），
但那根轴上 ① 条目数、② 总突触预算 $k_B$、③ **世界中不同值模式的数目** 三者共线 ⇒
"剂量究竟跟着谁走"没被分离。D-44 §2-M2 的说法是"剂量是写入世界**多样性**的函数"，尚未受检。

本探针只做两件事（**不进判据、不定阈值**，纪律 (i)）：
  ① **构造可行性**：把 $n_{\rm kv}{=}60$ **固定**，只把 40 个**非读条目**的值模式映射到
     $b\in\{40,20,4\}$ 个不同模式（$b{=}40$ 即原世界，逐位相同），检查
     读集（条目 0…19 的 $K/\mathrm{proto}/v^a/v^b$）是否**逐位不受影响**、
     $k_B$、$T$、条目数、键互异性是否不变；
  ② **功效核算**：三档的 $G$、$\bar q$、$\nu_{\rm cap}$、日程距离，以及**同种子配对差** $\Delta$ 的
     sd ⇒ 据此算"要判出 $\Delta$ 需要多少种子"，避免重演 D-44 §1.1-4 的"跑完才发现带宽是 $\pm21\%$"。

★ 探针种子 = D-43 的**前 5 个共享种子**（`SHARED_SEEDS[:5]`），不新增种子表、不进正式轮判决。
★ 只在进程内把 `R1_gating12.build_stream` 换成包装（`run_unit` 按模块全局名解析）：
  装置 md5、`dir26k/R1_gating12.py`、`dir26l/rerun_missing.py` 全部零改动。
"""
import json
import sys
import time

import numpy as np
import torch

for _p in ("dir26m", "dir26l", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g", "dir26f",
           "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating12 as K
from R1_gating import D_MODEL, build_stream as ORIG_BUILD
from R1_gating5 import theta_of
from R1_gating7 import md5
from R1_gating12 import DEV_MD5, N_READ
from R1_gating import M, S_KEY
from rerun_missing import measure_hit_bounded      # DEC-38：只换假断言，数值同源

N_KV = 60
R_PRES = 4
BS = (40, 20, 4)                                    # 非读条目的**不同值模式数**
SEEDS = K.SHARED_SEEDS[:5]
D43 = "/home/ec2-user/sgmn-loop/dir26k/out/DIR26Kr10_formal.json"
OUT = "/home/ec2-user/sgmn-loop/dir26m/out/P26M_feas.json"


def alias_stream(b):
    r"""把 40 个非读条目的**值模式**折叠到 $b$ 个：$\mathrm{pat}(i)=N_{\rm read}+((i-N_{\rm read})\bmod b)$。
    读集（$i<N_{\rm read}$）恒有 $\mathrm{pat}(i)=i$ ⇒ 读侧 $K/\mathrm{proto}/v^a/v^b$ 逐位不变。"""
    def f(seed, d, n, r_pres, theta):
        D = ORIG_BUILD(seed, d, n, r_pres, theta)
        if int(n) == N_KV and b <= n - N_READ:          # b=40 ⇒ 折叠是恒等（仍走包装，用来证同）
            idx = torch.arange(n)
            pat = torch.where(idx < N_READ, idx, N_READ + (idx - N_READ) % b)
            D["va"] = D["va"][pat].clone()
            D["vb"] = D["vb"][pat].clone()
            D["pat"] = pat
        return D
    return f


def structure(b):
    r"""纯构造检查（不跑 SNN）：读集逐位不变、键互异、写数不变。"""
    th = theta_of(0.9)
    D0 = ORIG_BUILD(SEEDS[0], D_MODEL, N_KV, R_PRES, th)
    K.build_stream = alias_stream(b)
    D1 = K.build_stream(SEEDS[0], D_MODEL, N_KV, R_PRES, th)
    K.build_stream = ORIG_BUILD
    out = {"b": b}
    out["read_K_same"] = bool(torch.equal(D0["K"][:, :N_READ], D1["K"][:, :N_READ]))
    out["read_proto_same"] = bool(torch.equal(D0["proto"][:N_READ], D1["proto"][:N_READ]))
    out["read_va_same"] = bool(torch.equal(D0["va"][:N_READ], D1["va"][:N_READ]))
    out["read_vb_same"] = bool(torch.equal(D0["vb"][:N_READ], D1["vb"][:N_READ]))
    out["keys_distinct"] = bool(len({tuple(np.round(r.numpy(), 6)) for r in D1["K"].reshape(-1, D_MODEL)}) == N_KV * R_PRES)
    out["n_val_va"] = int(len(torch.unique(torch.round(D1["va"] * 1e6), dim=0)))
    out["n_val_vb"] = int(len(torch.unique(torch.round(D1["vb"] * 1e6), dim=0)))
    out["T"] = int(R_PRES * N_KV * 2 * M)
    out["identity_b40"] = None
    if b >= N_KV - N_READ:          # b=40：折叠是恒等 ⇒ 必须与未包装逐位相同
        out["identity_b40"] = bool(torch.equal(D0["va"], D1["va"]) and torch.equal(D0["vb"], D1["vb"])
                                   and torch.equal(D0["K"], D1["K"]))
    return out


def main():
    t0 = time.time()
    print("=== ① 构造可行性（不跑 SNN） ===")
    print(f"  装置 md5 == {DEV_MD5} ? {md5('/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py') == DEV_MD5}")
    structs = [structure(b) for b in BS]
    for s in structs:
        print(f"  b={s['b']:>2} | 读集 K/proto/va/vb 逐位不变 "
              f"{s['read_K_same']}/{s['read_proto_same']}/{s['read_va_same']}/{s['read_vb_same']} | "
              f"键互异 {s['keys_distinct']} | 不同 va/vb 模式 {s['n_val_va']}/{s['n_val_vb']} | "
              f"T {s['T']} | 恒等(b=40) {s['identity_b40']}")

    print("\n=== ② 标定（冻结 tau：三档共用 $\bar q$ 的外生定标） ===")
    c = K.cal_at(N_KV)
    tau = c["c_tau"]
    print(f"  c_tau(60)={tau:.4f}  skip={c['skip_rate']:.3f}  gap={c['gap']:+.3f}  "
          f"rep={c['skip_frac_repeat']:.3f}  gate_ok={c['gate_ok']}")
    print(f"  复位 N_DIST==35 ? {K.G5.N_DIST == 35}")

    print(f"\n=== ③ {len(BS)} 档 × {len(SEEDS)} 种子（tau 冻结 = {tau:.4f}） ===")
    rows = []
    K.measure_hit = measure_hit_bounded                 # DEC-38
    for b in BS:
        K.build_stream = alias_stream(b)
        for seed in SEEDS:
            t1 = time.time()
            g = K.run_unit(N_KV, seed, tau)
            g["b"] = b
            g["wall"] = round(time.time() - t1, 1)
            rows.append(g)
            print(f"  b={b:>2} seed={seed} | G_O {g['G_O']:+.5f} G_X {g['G_X']:+.5f} "
                  f"U0 {g['U0_O']:+.5f} | F {g['arms']['Fu']['mu']:+.5f} K {g['arms']['Ku']['mu']:+.5f} | "
                  f"k_B {g['k']['B']} q̄ {g['qbar']:.3f} ν_cap {g['nu_cap']:.3f} "
                  f"dist {g['sched_dist']:.3f} | {g['wall']}s", flush=True)
        K.build_stream = ORIG_BUILD

    print("\n=== ④ 与 D-43 存档的逐位对照（b=40 即原世界：同种子必须逐位相同） ===")
    arch = {(r["n_kv"], r["seed"]): r for r in json.load(open(D43))["rows"]}
    same, n_cmp = [], 0
    for g in rows:
        if g["b"] != 40:
            continue
        a = arch.get((N_KV, g["seed"]))
        if not a:
            continue
        keys = ("G_O", "U0_O", "G_X", "qbar", "nu_cap", "sched_dist")
        bad = [k for k in keys if float(a[k]) != float(g[k])]
        n_cmp += 1
        same.append({"seed": g["seed"], "bad": bad})
    print(f"  比对 {n_cmp} 个单元（键 = G_O/U0_O/G_X/q̄/ν_cap/日程距离）："
          f"{'全部逐位相同' if all(not s['bad'] for s in same) else '有不符'}")
    for s in same:
        if s["bad"]:
            print(f"    seed={s['seed']} 不符字段 {s['bad']}")

    print("\n=== ⑤ 功效核算（只作预算，不定阈值） ===")
    by = {}
    for g in rows:
        by.setdefault(g["b"], []).append(g)
    stats = {}
    for b in BS:
        G = np.array([g["G_O"] for g in by[b]])
        q = np.array([g["qbar"] for g in by[b]])
        nu = np.array([g["nu_cap"] for g in by[b]])
        ds = np.array([g["sched_dist"] for g in by[b]])
        GX = np.array([g["G_X"] for g in by[b]])
        stats[b] = {"G_mean": float(G.mean()), "G_sd": float(G.std(ddof=1)) if len(G) > 1 else float("nan"),
                    "qbar_mean": float(q.mean()), "nu_mean": float(nu.mean()),
                    "dist_mean": float(ds.mean()), "GX_mean": float(GX.mean())}
        print(f"  b={b:>2} : G {G.mean():+.5f} (sd {stats[b]['G_sd']:.5f})  q̄ {q.mean():.3f}  "
              f"ν_cap {nu.mean():.3f}  日程距离 {ds.mean():.3f}  G_X {GX.mean():+.5f}")
    base = {g["seed"]: g for g in by[40]}
    for b in (20, 4):
        d = np.array([g["G_O"] - base[g["seed"]]["G_O"] for g in by[b]])
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        se = sd / np.sqrt(len(d))
        print(f"  配对 Δ(40→{b}) = {d.mean():+.5f}   sd {sd:.5f}   3se@{len(d)}种子 {3*se:.5f}   "
              f"符号 {int((d > 0).sum())}/{len(d)}")
        for nseed in (20, 40, 60, 100):
            print(f"      若 sd 不变：n={nseed} 种子 ⇒ 3se ≈ {3*sd/np.sqrt(nseed):.5f}"
                  f"{'  ← 仍大于 |Δ|/2' if 3*sd/np.sqrt(nseed) > abs(d.mean())/2 else '  ✔ 可达'}")
    print(f"\n  单元耗时（n_kv=60）：{np.mean([g['wall'] for g in rows]):.1f} s ± "
          f"{np.std([g['wall'] for g in rows], ddof=1):.1f} s；"
          f"总墙钟 {time.time()-t0:.1f} s")
    print("  ★ 探针读数不进判据、不定阈值（纪律 (i)）；正式轮种子表另行冻结。")

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"structs": structs, "rows": rows, "stats": stats,
               "calib": {"c_tau": tau, "skip": c["skip_rate"], "gap": c["gap"], "rep": c["skip_frac_repeat"]},
               "seeds": SEEDS, "bs": list(BS), "n_kv": N_KV,
               "wall_seconds": round(time.time() - t0, 1)},
              open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"  [落盘] {OUT}")


if __name__ == "__main__":
    main()
