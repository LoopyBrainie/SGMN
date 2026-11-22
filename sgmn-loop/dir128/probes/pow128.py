import json, math
import numpy as np
T=json.load(open("dir128/out/tensor128.json")); D=T["doses"]
DOSES=sorted(int(k) for k in D); LOG4=math.log(4.0)
def lg(p): return math.log(p/(1-p))
rng=np.random.default_rng(20261123)
print("="*112)
print("★★★ MC 功效（二项重采样，10000×，seed=20261123）：把 e1,e2 放大 k 倍后，log(ratio) 符号判对的概率")
print("="*112)
print("%-6s %-7s %-7s %-9s %-8s %-9s %-10s %-10s %s"%("dose","p1","p2","G-log4","现e2","k(80%)","e2*(80%)","e2*(95%)","当前符号"))
rows=[]
for d in DOSES:
    av=tuple(sorted(D[str(d)]["avail"])); per=D[str(d)]["per_rng"]; nc=T["cells_per_dose"][str(d)]
    n1=sum(per[r]["n1"] for r in av); n2=sum(per[r]["n2"] for r in av)
    e1=sum(sum(per[r]["e1"]) for r in av); e2=sum(sum(per[r]["e2"]) for r in av)
    N1,N2=nc*n1,nc*n2; p1,p2=e1/N1,e2/N2
    G=lg(p1)-lg(p2); delta=G-LOG4
    def power(k,reps=10000):
        E1=rng.binomial(int(round(k*N1)),p1,size=reps); E2=rng.binomial(int(round(k*N2)),p2,size=reps)
        ok=(E1>0)&(E2>0)&(E1<int(round(k*N1)))&(E2<int(round(k*N2)))
        g=np.full(reps,np.nan)
        with np.errstate(divide='ignore',invalid='ignore'):
            pp1=E1/np.maximum(np.round(k*N1),1); pp2=E2/np.maximum(np.round(k*N2),1)
            g=np.log(pp1/(1-pp1))-np.log(pp2/(1-pp2))
        v=g[ok]; s=np.sign(v-LOG4)
        return float(np.mean(s==np.sign(delta)))
    def findk(target):
        lo,hi=1.0,4000.0
        if power(hi)<target: return None
        for _ in range(26):
            mid=(lo*hi)**0.5
            if power(mid)>=target: hi=mid
            else: lo=mid
        return hi
    k80=findk(0.80); k95=findk(0.95)
    rows.append((d,p1,p2,delta,e2,k80,k95,0 if delta<0 else 1))
    print("%-6d %-7.4f %-7.4f %-9.4f %-8d %-9s %-10s %-10s %s"%(
        d,p1,p2,delta,e2,("%.1f"%k80 if k80 else ">4000"),
        ("%.0f"%(k80*e2) if k80 else "—"),("%.0f"%(k95*e2) if k95 else "—"),
        ("ratio<1" if delta<0 else "ratio>1")))
print()
print("★ 读法：k=把该剂量的 e1/e2 同时放大的倍数；e2*(80%)= 达到 80% 功效所需的 e2 计数。")
print("★ 全部 8 个可判剂量的「当前符号」= ", [("d=%d"%r[0], "ratio<1" if r[7]==0 else "ratio>1") for r in rows])
