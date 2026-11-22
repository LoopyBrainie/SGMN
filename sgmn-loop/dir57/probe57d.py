#!/opt/pytorch/bin/python
r"""DIR-57 · **前提探针 D**：补上 57a 缺的**决定性对照**——**未写过的键**读出多少？

★ 为什么必须补（`PROBE57A` 的读数逼出来的）：
   57a 里 `cos(Wk_a, v_P)=0.9852`、`cos(Wk_b, v_P)=0.9927`，但 `cos(r_a, r_b)=0.9906`、`σ=[1.967, 0.442, 0.000]`
   ⇒ `W` 在写完两笔后已近似 **rank-1**：`W ≈ σ u v̂ᵀ` ⇒ **任何**键 `k` 都给出 `Wk ∥ v̂`。
   那么"两键都指向 `v_P`"就**不是绑定**，而是**全局值塌缩**（每把钥匙都开同一扇门）。
   ⇒ 唯一能区分二者的量 = **未写过的随机键**在同一末态下的余弦 `c_rand`：
      若 `c_rand ≈ c_a ≈ c_b` ⇒ 无键选择性（①**不可判定/无意义**）；若 `c_rand` 低而 `c_a,c_b` 高 ⇒ 真绑定。
★ 三臂（同一实例、同一起跑状态）：
   `one` ：只写 `(k_a, v_P)` ⇒ `c_a`、`c_rand`
   `two` ：再写 `(k_b, v_P)` ⇒ `c_a`、`c_b`、`c_rand`（★ 该键**从未写过**，取 5 把的中位）
   `two_distinct`：`(k_a, v_P)` + `(k_b, v_Q ⊥)` ⇒ `c_a`、`c_rand`（异值是否也塌缩）
★ 探针纪律：不设阈值、不写判据；`c_rand` 只作**对照**，不与 `c_a` 竞争"判据"。
用法：`/opt/pytorch/bin/python dir57/probe57d.py` → `dir57/out/PROBE57D.json`
"""
import json
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402
import R1_task51 as T51      # noqa: E402

OUT = os.path.join(ROOT, "dir57", "out")
SEED0 = 20261117
SEEDS = [SEED0 + 58200 + i for i in range(64)]
OFF_K1, OFF_K2, OFF_VAL, OFF_RAND, OFF_PERP = 57301, 57302, 57303, 57304, 57305
N_RAND = 5


def write_pair(U, Vm, k, v):
    return T48.step_pair(U, Vm, k, v, 1.0)


def key_like(g, ref=None, max_cos=0.2):
    while True:
        k = R.KEY_SCALE * T51.gen_vec(g, M.D_K)
        if ref is None:
            return k
        if abs(float((k @ ref) / (k.norm() * ref.norm()))) <= max_cos:
            return k


def value_perp(vP, g):
    while True:
        b = T51.gen_vec(g, M.D_V)
        b = b - (b @ vP) * vP
        if float(b.norm()) > 0.5:
            return b / b.norm()


def read_c(U, Vm, k, v):
    return T51.read_P(U, Vm, k, v)[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sd in SEEDS:
        P = R.build(sd, "V0")
        U0, V0 = P["U1"].clone(), P["V1"].clone()
        g1 = torch.Generator().manual_seed(sd + OFF_K1)
        g2 = torch.Generator().manual_seed(sd + OFF_K2)
        gv = torch.Generator().manual_seed(sd + OFF_VAL)
        gr = torch.Generator().manual_seed(sd + OFF_RAND)
        kA = key_like(g1)
        kB = key_like(g2, ref=kA)
        vP = T51.gen_vec(gv, M.D_V)
        vQ = value_perp(vP, gv)
        kRand = [key_like(gr) for _ in range(N_RAND)]      # ★ 从未写过的键

        # one
        U1, V1 = write_pair(U0, V0, kA, vP)
        one = {"ca": read_c(U1, V1, kA, vP),
               "crand": statistics.median([read_c(U1, V1, k, vP) for k in kRand])}
        # two (same value)
        U2, V2 = write_pair(U1, V1, kB, vP)
        two = {"ca": read_c(U2, V2, kA, vP), "cb": read_c(U2, V2, kB, vP),
               "crand": statistics.median([read_c(U2, V2, k, vP) for k in kRand])}
        # two_distinct
        Ud, Vd = write_pair(U0, V0, kA, vP)
        Ud, Vd = write_pair(Ud, Vd, kB, vQ)
        dd = {"ca": read_c(Ud, Vd, kA, vP), "cbQ": read_c(Ud, Vd, kB, vQ),
              "crand_P": statistics.median([read_c(Ud, Vd, k, vP) for k in kRand]),
              "crand_Q": statistics.median([read_c(Ud, Vd, k, vQ) for k in kRand])}
        s1, s2 = [float(x) for x in torch.linalg.svdvals(M._W(U2, V2).detach())[:2]]

        rows.append({"seed": sd, "one": one, "two": two, "distinct": dd,
                     "sigma": [s1, s2],
                     "sel_one": one["ca"] - one["crand"],
                     "sel_two_a": two["ca"] - two["crand"],
                     "sel_two_b": two["cb"] - two["crand"],
                     "sel_distinct_a": dd["ca"] - dd["crand_P"]})

    def med(path):
        out = {}
        for x in rows:
            d = x
            for p in path.split("."):
                d = d[p]
            out.setdefault("v", []).append(d)
        return statistics.median(out["v"])

    def frac(pred):
        return sum(1 for x in rows if pred(x)) / len(rows)

    summ = {
        "n": len(rows),
        "one": {"ca": med("one.ca"), "crand": med("one.crand")},
        "two": {"ca": med("two.ca"), "cb": med("two.cb"), "crand": med("two.crand")},
        "distinct": {"ca": med("distinct.ca"), "cbQ": med("distinct.cbQ"),
                     "crand_P": med("distinct.crand_P"), "crand_Q": med("distinct.crand_Q")},
        "selectivity": {"one": med("sel_one"), "two_a": med("sel_two_a"), "two_b": med("sel_two_b"),
                        "distinct_a": med("sel_distinct_a")},
        "frac_no_selectivity_two": frac(lambda x: abs(x["sel_two_a"]) < 0.05 and abs(x["sel_two_b"]) < 0.05),
        "sigma_median": [med("sigma")[0] if False else statistics.median([x["sigma"][0] for x in rows]),
                         statistics.median([x["sigma"][1] for x in rows])],
    }
    res = {"probe": "57d", "device": "DIR-51 (write=dir48.step_pair)",
           "seeds": [SEEDS[0], SEEDS[-1]], "n_instances": len(SEEDS), "rows": rows, "summary": summ}
    with open(os.path.join(OUT, "PROBE57D.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    s = summ
    print("── one  ：c_a %.4f · c_rand(未写过) %.4f ⇒ 选择性 %.4f"
          % (s["one"]["ca"], s["one"]["crand"], s["selectivity"]["one"]))
    print("── two  ：c_a %.4f · c_b %.4f · c_rand %.4f ⇒ 选择性 %.4f / %.4f"
          % (s["two"]["ca"], s["two"]["cb"], s["two"]["crand"],
             s["selectivity"]["two_a"], s["selectivity"]["two_b"]))
    print("── distinct：c_a(v_P) %.4f · c_b(v_Q) %.4f · c_rand→v_P %.4f → v_Q %.4f ⇒ 选择性 %.4f"
          % (s["distinct"]["ca"], s["distinct"]["cbQ"], s["distinct"]["crand_P"],
             s["distinct"]["crand_Q"], s["selectivity"]["distinct_a"]))
    print("── 无键选择性的实例（|sel_a|<0.05 且 |sel_b|<0.05）：%d/%d；σ2/σ1 中位 %.3f"
          % (round(s["frac_no_selectivity_two"] * len(rows)), len(rows),
             s["sigma_median"][1] / s["sigma_median"][0]))


if __name__ == "__main__":
    main()