"""DIR-87b 诊断：① N=2 闭式的正确形式 ② 三种 hebbian 配置的差异（D-116 用的是哪一种）"""
import pathlib, sys, torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dir22")); sys.path.insert(0, str(HERE))
from R1_task87b import build_core, make_data, ALPHA, LAM, ETA, SEEDS

d, sd = 8, SEEDS[0]
K, V = make_data(d, sd)
k0, k1, v0, v1 = K[0], K[1], V[0], V[1]
nk, nv0, nv1 = float(k0.norm()), float(v0.norm()), float(v1.norm())
s, t = float(k0@k1), float(v0@v1)
r, delta = ETA*ALPHA, ETA*LAM
print(f"nk={nk:.8f} nv0={nv0:.8f} nv1={nv1:.8f} s={s:.8e} t={t:.8e}")

c = build_core("hebbian", d, sd, "empty")
c.write(k0, v0); c.write(k1, v1)
W2 = c.W.detach().clone()
A = (1-delta)*r*nk*nk - ETA*r*s*s
B = ETA*s
W2a = (1-delta)*r*torch.outer(v0,k0) - ETA*r*s*torch.outer(v0,k1) + ETA*torch.outer(v1,k1)
print(f"W2 vs 解析(0.0995·n_k²?? no: (1-δ)r·v0k0ᵀ - ηr·s·v0k1ᵀ + η·v1k1ᵀ): 相对误差 "
      f"{float((W2-W2a).norm()/W2a.norm()):.3e}")
with torch.no_grad():
    y0 = c.read(k0.unsqueeze(0))[0]
cosm = float(torch.nn.functional.cosine_similarity(y0, v0, dim=0))
y0a = A*v0 + B*v1
print(f"y0 vs A·v0+B·v1: 相对误差 {float((y0-y0a).norm()/y0a.norm()):.3e}")

def cosf(num, den): return num/den
cand = {
 "naive (A+Bt)/sqrt(A²+B²+2ABt)": (A+B*t)/((A*A+B*B+2*A*B*t)**0.5),
 "with nv0/nv1 (正确候选)": (A*nv0**2+B*t)/(nv0*((A*A*nv0**2+B*B*nv1**2+2*A*B*t)**0.5)),
 "fully scale-free: A,B 换成 a=A, b=B 归一后": (A*nv0**2+B*t)/(nv0*((A*nv0)**2+(B*nv1)**2+2*A*B*t)**0.5),
}
print(f"\ncos 实测 = {cosm:.12f}")
for k,v in cand.items(): print(f"  {k}: {v:.12f}  |差|={abs(cosm-v):.3e}")

print("\n=== 三种 hebbian 配置 ===")
for name, mk in [("empty(full,zero)", lambda: build_core("hebbian",d,sd,"empty")),
                 ("D-116的hebbian(full,random)", None),
                 ("rand(factorized)", lambda: build_core("hebbian",d,sd,"rand"))]:
    if mk is None:
        from sgmn_module_hebb_v7 import HebbianCore
        g = torch.Generator().manual_seed(sd)
        cc = HebbianCore(d, d, ALPHA, LAM, ETA, g, "cpu", factorized=False, n_entries=1024, d_model_sig=0)
    else:
        cc = mk()
    print(f"  {name:32s} factorized={cc.factorized!s:5s} ||W0||={float(cc.W0.norm()):.6f} "
          f"||_W()||={float(cc._W().norm()):.6f} kind={cc.kind}")
