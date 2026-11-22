import itertools, json, math
T=json.load(open("dir128/out/tensor128.json")); D=T["doses"]
DOSES=sorted(int(k) for k in D); LOG4=math.log(4.0)
def lg(p): return math.log(p/(1-p)) if 0<p<1 else None
def ratio_of(p1,p2):
    if None in (p1,p2) or not(0<p1<1 and 0<p2<1): return None
    return (p1*(1-p2))/(4.0*p2*(1-p1))
def mean(xs):
    xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
def rules(d,S):
    per=D[str(d)]["per_rng"]; CL=list(range(T["cells_per_dose"][str(d)]))
    n1=sum(per[r]["n1"] for r in S); n2=sum(per[r]["n2"] for r in S); nc=len(CL)
    e1c={c:sum(per[r]["e1"][c] for r in S) for c in CL}
    e2c={c:sum(per[r]["e2"][c] for r in S) for c in CL}
    pr=[(per[r]["e1"][c]/per[r]["n1"], per[r]["e2"][c]/per[r]["n2"]) for r in S for c in CL]
    P1r=[sum(per[r]["e1"])/(nc*per[r]["n1"]) for r in S]
    P2r=[sum(per[r]["e2"])/(nc*per[r]["n2"]) for r in S]
    o={}
    o["R0 N-weighted pool"]=ratio_of(sum(e1c.values())/(nc*n1), sum(e2c.values())/(nc*n2))
    o["R1 rng-equal"]      =ratio_of(mean(P1r), mean(P2r))
    o["R2 pair-equal"]     =ratio_of(mean(x for x,_ in pr), mean(y for _,y in pr))
    o["R3 cell-equal mean of rates"]=ratio_of(mean([e1c[c]/n1 for c in CL]), mean([e2c[c]/n2 for c in CL]))
    rr=[ratio_of(a,b) for a,b in zip(P1r,P2r)]; rr=[x for x in rr if x]
    o["R4 geo-mean of per-rng ratio"]=math.exp(mean([math.log(x) for x in rr])) if rr else None
    return o
print("="*104)
print("★★★ 规则敏感性：在**规范全子集**（4 个 rng 全用）上，换聚合规则值变多少？")
print("="*104)
print("%-6s %s"%("dose"," ".join("%-11s"%k.split(" ",1)[0] for k in rules(25,tuple(D["25"]["avail"])))))
for d in DOSES:
    av=tuple(sorted(D[str(d)]["avail"])); r=rules(d,av)
    vs=[v for v in r.values() if v]
    print("%-6d %s   ⇒ 规则间 max/min = %.4f×"%(d," ".join("%-11.4f"%v if v else "%-11s"%"—" for v in r.values()), max(vs)/min(vs)))
print()
print("="*104)
print("★★★ 功效：分辨「log(ratio) 的符号」所需的**最小 e2 计数**（用实测 p1,p2；δ 法 SE）")
print("="*104)
print("%-6s %-8s %-8s %-9s %-10s %-11s %-11s %s"%("dose","p1","p2","G-log4","N1/N2","现 e2","所需 e2(N2=N1·ρ)","倍数"))
for d in DOSES:
    av=tuple(sorted(D[str(d)]["avail"])); per=D[str(d)]["per_rng"]; nc=T["cells_per_dose"][str(d)]
    n1=sum(per[r]["n1"] for r in av); n2=sum(per[r]["n2"] for r in av)
    e1=sum(sum(per[r]["e1"]) for r in av); e2=sum(sum(per[r]["e2"]) for r in av)
    N1,P1=nc*n1,nc*n2; p1,p2=e1/N1,e2/N2
    g=lg(p1)-lg(p2); delta=abs(g-LOG4)
    if delta<=0: print("%-6d 恰在临界上，不可判"%d); continue
    # 需要 1.96*SE < delta，SE^2 = 1/(N1 p1(1-p1)) + 1/(N2 p2(1-p2))
    c1=1.0/(p1*(1-p1)); c2=1.0/(p2*(1-p2)); rho=N2/N1
    need=(1.96/delta)**2*(c1+c2/rho)/N1   # N1 must be >= need
    print("%-6d %-8.4f %-8.4f %-9.4f %-10.2f %-11d %-11.0f %.2f×"%(
        d,p1,p2,delta,rho,e2,need*rho*P1/N1, (need/N1) if N1 else 0))
