"""DIR-87 诊断：rehearse 协议是否真的改变了 W？（CORR-16：判据 FAIL 先定位装置 vs 被测对象）"""
import pathlib, sys, torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dir22"))
sys.path.insert(0, str(HERE))
from R1_task87 import build_core, make_data, decode_hits, SEEDS, E_REH, N_SLOTS

for d in (8, 64):
    for N in (1, 2, 4, 16):
        K, V = make_data(d, SEEDS[0])
        kk, vv = K[:N], V[:N]
        Ws = {}
        for ep in (1, 2, 8, 32):
            c = build_core("hebbian", d, SEEDS[0])
            for _ in range(ep):
                for i in range(N):
                    c.write(kk[i], vv[i], sig=None)
            with torch.no_grad():
                Y = c.read(kk)
            hit, _, _ = decode_hits(Y, vv, torch.arange(N))
            Ws[ep] = (c._W().detach().clone(), hit, c.data_norm, c.reg_norm)
        r = {ep: (float((Ws[ep][0]-Ws[1][0]).norm()), Ws[ep][1]) for ep in Ws}
        print(f"d={d:3d} N={N:3d} | ||W_ep-W_1||: " +
              " ".join(f"{ep}:{r[ep][0]:.4f}" for ep in (2,8,32)) +
              f" || hit: 1:{r[1][1]:.3f} 2:{r[2][1]:.3f} 8:{r[8][1]:.3f} 32:{r[32][1]:.3f}"
              f" || data_norm={Ws[32][2]:.2e} reg_norm={Ws[32][3]:.2e}", flush=True)
