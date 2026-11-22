import itertools, json, math
T=json.load(open("dir128/out/tensor128.json")); D=T["doses"]
DOSES=sorted(int(k) for k in D)
def ratio_of(p1,p2):
    if None in (p1,p2) or not(0<p1<1 and 0<p2<1): return None
    return (p1*(1-p2))/(4.0*p2*(1-p1))
def mean(xs):
    xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
def hmean(xs):
    xs=[x for x in xs if x is not None and x>0]; return len(xs)/sum(1.0/x for x in xs) if xs else None
def med(xs):
    xs=sorted(x for x in xs if x is not None)
    if not xs: return None
    n=len(xs); return xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2])
def trim(xs,f=0.2):
    xs=sorted(x for x in xs if x is not None)
    if len(xs)<5: return mean(xs)
    k=int(len(xs)*f); return mean(xs[k:len(xs)-k]) if len(xs)-2*k>0 else mean(xs)
def jeff(xs,n):
    return [(e+0.5)/(nn+1.0) for e,nn in zip(xs,n)]

def rules(d,S):
    per=D[str(d)]["per_rng"]; CL=list(range(T["cells_per_dose"][str(d)])); nc=len(CL)
    n1=sum(per[r]["n1"] for r in S); n2=sum(per[r]["n2"] for r in S)
    e1c={c:sum(per[r]["e1"][c] for r in S) for c in CL}
    e2c={c:sum(per[r]["e2"][c] for r in S) for c in CL}
    pr=[(per[r]["e1"][c]/per[r]["n1"], per[r]["e2"][c]/per[r]["n2"]) for r in S for c in CL]
    P1r=[sum(per[r]["e1"])/(nc*per[r]["n1"]) for r in S]
    P2r=[sum(per[r]["e2"])/(nc*per[r]["n2"]) for r in S]
    A1=[x for x,_ in pr]; A2=[y for _,y in pr]
    o={}
    o["R0 N-w pool"]      = ratio_of(sum(e1c.values())/(nc*n1), sum(e2c.values())/(nc*n2))
    o["R1 rng-equal"]     = ratio_of(mean(P1r), mean(P2r))
    o["R2 pair-equal"]    = ratio_of(mean(A1), mean(A2))
    o["R3 cell-equal"]    = ratio_of(mean([e1c[c]/n1 for c in CL]), mean([e2c[c]/n2 for c in CL]))
    rr=[ratio_of(a,b) for a,b in zip(P1r,P2r)]; rr=[x for x in rr if x]
    o["R4 geo per-rng"]   = math.exp(mean([math.log(x) for x in rr])) if rr else None
    o["R5 median pair"]   = ratio_of(med(A1), med(A2))
    o["R6 harm pair"]     = ratio_of(hmean(A1), hmean(A2))
    o["R7 trim20 pair"]   = ratio_of(trim(A1), trim(A2))
    # R8: Jeffreys 平滑（同样的 pair-equal，但每个 pair 各 +0.5）
    pair_counts=[(per[r]["e1"][c], per[r]["n1"]) for r in S for c in CL]
    o["R8 Jeffreys pair"] = ratio_of(mean(jeff([x for x,_ in pair_counts],[y for _,y in pair_counts])),
                                    ratio_of.__self__ if False else mean(
        [(per[r]["e2"][c]+0.5)/(per[r]["n2"]+1.0) for r in S for c in CL]))
    # R9: N2 权重同时用于 p1/p2（消除 n1 vs n2 权重错配）
    o["R9 N2-weight both"] = ratio_of(
        sum(per[r]["n2"]*P1r[i] for i,r in enumerate(S))/sum(per[r]["n2"] for r in S),
        sum(per[r]["n2"]*P2r[i] for i,r in enumerate(S))/sum(per[r]["n2"] for r in S))
    return o

K=rules(450,tuple(sorted(D["450"]["avail"]))); NAMES=list(K)
print("规则族（10 个）：")
for i,n in enumerate(NAMES): print("   %-20s"%n, end="\n" if i%2 else "")
print()
print("="*100)
print("★★★ 规则轴不变性 —— 对**每个** (dose, |S|>=2 子集) 算「规则间 max/min」")
print("="*100)
worst=(0,None)
per_dose={}
for d in DOSES:
    av=D[str(d)]["avail"]; vals=[]
    for k in range(2,len(av)+1):
        for S in itertools.combinations(sorted(av),k):
            o=rules(d,S); v=[x for x in o.values() if x]
            if len(v)>=2:
                s=max(v)/min(v); vals.append(s)
                if s>worst[0]: worst=(s,(d,S))
    if vals:
        per_dose[d]=(max(vals),len(vals))
        print("  dose=%-4d 结构数=%-3d  规则间 max/min = %.4f×   %s"%(
            d,len(vals),max(vals),"✓ ≤1.10×" if max(vals)<=1.10 else "✗"))
print()
print("★ 全表最差 = %.4f×  @ %s"%(worst[0],worst[1]))
print("★ 判据 ≤1.10× ⇒", "PASS 全部通过" if worst[0]<=1.10 else "FAIL")
