#!/opt/pytorch/bin/python
r"""DIR-57 · 实现：**同一身份值跨多键绑定**的跨键一致性（卡 `revolve/DIR-57/DIRECTION_rev57_v10.md`，rev 1.0 冻结）。

★ 逐字派生：装置（`dir31`/`dir33`/`dir46`/`dir47`/`dir48`）与写入内层 `R1_task48.step_pair` **一字不改**；
  **唯一自变量** = 写入协议（`m1` / `m2` / `m4` / `m4_int8`），测量层按卡 §2.3 冻结。
★ 唯一读出量（卡 §0.2 / §2.3）：`E_i = ‖W k_i − v_P‖`、`E_null = median_{未写键} ‖W k' − v_P‖`、`S_i = E_null − E_i`。
  ★ **不得**用 `cos` 作判据量（rank-1 值塌缩下余弦对寻址是盲的，`PROBE57E` / 纪律候选 140）。
★ 生成式冻结（与 `dir57/probe57f.py` 逐字同式）：`OFF_K=57501, OFF_V=57502, OFF_R=57503, OFF_INT=57504`；
  已写键两两 `|cos|≤0.30`；未写键 5 把（独立 generator）；干扰值 `|v·v_P|≤0.30`。

用法：`R1_task57.py --arm {m1,m2,m4,m4_int8} --n 96 --seed0 59000 --out PATH`
"""
import argparse
import hashlib
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

ARMS = ("m1", "m2", "m4", "m4_int8")
OFF_K, OFF_V, OFF_R, OFF_INT = 57501, 57502, 57503, 57504
N_UNWRITTEN = 5
N_INT = 8
M_MAX = 4


def write_pair(U, Vm, k, v):
    r"""唯一写规则（逐字调用 `dir48.step_pair`，α=1）。"""
    return T48.step_pair(U, Vm, k, v, 1.0)


def key_batch(g, m, max_cos=0.30):
    ks = []
    while len(ks) < m:
        k = R.KEY_SCALE * T51.gen_vec(g, M.D_K)
        if all(abs(float((k @ q) / (k.norm() * q.norm()))) <= max_cos for q in ks):
            ks.append(k)
    return ks


def val_near_orth(vP, g, max_cos=0.30):
    while True:
        v = T51.gen_vec(g, M.D_V)
        if abs(float(v @ vP)) <= max_cos:
            return v


def read_errors(U, Vm, ks, vP):
    r"""★ 卡 §2.3 的唯一读出量：`‖W k − v_P‖`（**不含任何余弦/归一化**）。"""
    W = M._W(U, Vm)
    return [float((W @ k - vP).norm()) for k in ks]


def pair_cos_max(ks):
    m = 0.0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            m = max(m, abs(float((ks[i] @ ks[j]) / (ks[i].norm() * ks[j].norm()))))
    return m


def run_one(seed, arm):
    assert arm in ARMS, arm
    P = R.build(seed, "V0")
    U0, V0 = P["U1"].clone(), P["V1"].clone()
    init_fp = hashlib.sha256(torch.cat([U0.flatten(), V0.flatten()]).detach().numpy().tobytes()).hexdigest()[:16]
    gk = torch.Generator().manual_seed(seed + OFF_K)
    gv = torch.Generator().manual_seed(seed + OFF_V)
    gr = torch.Generator().manual_seed(seed + OFF_R)
    gi = torch.Generator().manual_seed(seed + OFF_INT)
    ks = key_batch(gk, M_MAX)
    vP = T51.gen_vec(gv, M.D_V)
    kUn = key_batch(gr, N_UNWRITTEN)
    kInt = key_batch(gi, N_INT)
    vInt = [val_near_orth(vP, gi) for _ in range(N_INT)]

    m = {"m1": 1, "m2": 2, "m4": 4, "m4_int8": 4}[arm]
    U, Vm = U0.clone(), V0.clone()
    for i in range(m):
        U, Vm = write_pair(U, Vm, ks[i], vP)
    E = read_errors(U, Vm, ks[:m], vP)
    E_null = statistics.median(read_errors(U, Vm, kUn, vP))
    S = [E_null - e for e in E]

    out = {"seed": seed, "arm": arm, "init_fp": init_fp,
           "E": E, "E_null": E_null, "S": S,
           "dispE": (max(E) - min(E)) if len(E) > 1 else 0.0,
           "key_cos_max": pair_cos_max(ks[:m]) if m > 1 else 0.0,
           "finite": all(map(lambda x: x == x and abs(x) != float("inf"), [E_null] + E + S)),
           "sigma": [float(x) for x in torch.linalg.svdvals(M._W(U, Vm).detach())[:3]]}

    if arm == "m4_int8":
        Sin, Ein = [], []
        # ★ 干扰臂复用同一个末态：先算 m4 读数，再追加 8 条异值
        for j in range(N_INT):
            U, Vm = write_pair(U, Vm, kInt[j], vInt[j])
        Ein = read_errors(U, Vm, ks, vP)
        E_null_in = statistics.median(read_errors(U, Vm, kUn, vP))
        Sin = [E_null_in - e for e in Ein]
        out.update({"E_int8": Ein, "E_null_int8": E_null_in, "S_int8": Sin,
                    "dispE_int8": max(Ein) - min(Ein),
                    "dS": [S[i] - Sin[i] for i in range(m)],
                    "n_int": N_INT,
                    "int_val_cos_max": max(abs(float(v @ vP)) for v in vInt),
                    "int_key_unique": len({tuple(round(float(x), 12) for x in k) for k in kInt}) == N_INT,
                    "finite_int8": all(map(lambda x: x == x and abs(x) != float("inf"), [E_null_in] + Ein + Sin))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed0", type=int, default=59000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [a.seed0 + i for i in range(a.n)]
    rows = [run_one(sd, a.arm) for sd in seeds]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = {"arm": a.arm, "n": a.n, "seed0": a.seed0, "seeds": [seeds[0], seeds[-1]],
           "n_unwritten": N_UNWRITTEN, "n_int": N_INT, "m_max": M_MAX,
           "rows": rows}
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    # 摘要（仅诊断，不参与判定）
    Sn = [statistics.median([x["S"][i] for x in rows]) for i in range(len(rows[0]["S"]))]
    print("[%s] n=%d  seeds %d..%d" % (a.arm, a.n, seeds[0], seeds[-1]))
    print("  写后 E_null 中位 %.4f · E 中位 %.4f · S 逐位置 %s"
          % (statistics.median([x["E_null"] for x in rows]),
             statistics.median([statistics.fmean(x["E"]) for x in rows]),
             ["%.4f" % v for v in Sn]))
    if a.arm == "m4_int8":
        S8 = [statistics.median([x["S_int8"][i] for x in rows]) for i in range(len(rows[0]["S_int8"]))]
        print("  干扰后 E_null 中位 %.4f · E 中位 %.4f · S 逐位置 %s"
              % (statistics.median([x["E_null_int8"] for x in rows]),
                 statistics.median([statistics.fmean(x["E_int8"]) for x in rows]),
                 ["%.4f" % v for v in S8]))


if __name__ == "__main__":
    main()