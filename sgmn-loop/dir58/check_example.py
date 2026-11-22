#!/opt/pytorch/bin/python
r"""DIR-58 卡 §2.3 的工作例 + `A58-8` 独立重实现核对（只跑 seed 59000，不是实验轮）。"""
import json
import os
import statistics as st
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
import probe58b as P58       # noqa: E402

SD = 59000


def indep_alpha0(U, Vm, n_steps):
    r"""独立重实现：每步重算 W，V 用**更新后**的 U（与 `step_pair(alpha=0)` 同式）。"""
    U, Vm = U.clone(), Vm.clone()
    for _ in range(n_steps):
        W = M._W(U, Vm)
        G = M.LAM * W
        U = U - M.ETA * (G @ Vm)
        Vm = Vm - M.ETA * (G.t() @ U)
    return U, Vm


def main():
    P = R.build(SD, "V0")
    gk = torch.Generator().manual_seed(SD + P58.OFF_K)
    gv = torch.Generator().manual_seed(SD + P58.OFF_V)
    gr = torch.Generator().manual_seed(SD + P58.OFF_R)
    ks = P58.key_batch(gk, 4)
    vP = T51.gen_vec(gv, M.D_V)
    kUn = P58.key_batch(gr, 5)
    U1, V1 = T48.step_pair(P["U1"].clone(), P["V1"].clone(), ks[0], vP, 1.0)
    out = {"seed": SD, "n_rep": T48.N_REP, "eta": M.ETA, "lam": M.LAM}
    W = M._W(U1, V1)
    out["q0"] = {"E": float((W @ ks[0] - vP).norm()),
                 "E_null": st.median([float((W @ k - vP).norm()) for k in kUn]),
                 "Wk_norm": float((W @ ks[0]).norm()),
                 "W_norm": float(W.norm()),
                 "cos_Wk_v": float((W @ ks[0]) @ vP / ((W @ ks[0]).norm() * vP.norm()))}
    out["q0"]["S"] = out["q0"]["E_null"] - out["q0"]["E"]
    Ua, Va = T48.step_pair(U1.clone(), V1.clone(), ks[0], vP, 0.0)
    Ub, Vb = indep_alpha0(U1, V1, T48.N_REP)
    Wa, Wb = M._W(Ua, Va), M._W(Ub, Vb)
    out["A58-8"] = {"max_dW": float((Wa - Wb).abs().max()),
                    "dE": abs(float((Wa @ ks[0] - vP).norm() - (Wb @ ks[0] - vP).norm())),
                    "dU": float((Ua - Ub).abs().max()), "dV": float((Va - Vb).abs().max())}
    out["q1"] = {"E": float((Wa @ ks[0] - vP).norm()),
                 "E_null": st.median([float((Wa @ k - vP).norm()) for k in kUn]),
                 "Wk_norm": float((Wa @ ks[0]).norm()),
                 "W_norm": float(Wa.norm())}
    out["q1"]["S"] = out["q1"]["E_null"] - out["q1"]["E"]
    out["ratio_W_q1_over_q0"] = out["q1"]["W_norm"] / out["q0"]["W_norm"]
    p = os.path.join(ROOT, "dir58", "out", "WORKED58.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()