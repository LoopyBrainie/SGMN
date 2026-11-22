import itertools, json, math
T=json.load(open("dir128/out/tensor128.json")); D=T["doses"]
DOSES=sorted(int(k) for k in D)
def R(p1,p2):
    if None in (p1,p2) or not(0<p1<1 and 0<p2<1): return None
    return (p1*(1-p2))/(4.0*p2*(1-p1))
def mean(xs):
    xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
def agg_rules(d,S):
    """**计数聚合族**：先把计数按某规则加总/平均，**再**形成率。分母 >= 9*min(n)。"""
    per=D[str(d)]["per_rng"]; CL=list(range(T["cells_per_dose"][str(d)])); nc=len(CL)
    n1=sum(per[r]["n1"] for r in S); n2=sum(per[r]["n2"] for r in S)
    e1c={c:sum(per[r]["e1"][c] for r in S) for c in CL}
    e2c={c:sum(per[r]["e2"][c] for r in S) for c in CL}
    P1r=[sum(per[r]["e1"])/(nc*per[r]["n1"]) for r in S]
    P2r=[sum(per[r]["e2"])/(nc*per[r]["n2"]) for r in S]
    o={}
    o["A0 Nw pool"]      = R(sum(e1c.values())/(nc*n1), sum(e2c.values())/(nc*n2))
    o["A1 rng-eq"]       = R(mean(P1r), mean(P2r))
    o["A2 cell-eq"]      = R(mean([e1c[c]/n1 for c in CL]), mean([e2c[c]/n2 for c in CL]))
    w2=[per[r]["n2"] for r in S]
    o["A3 N2w both"]     = R(sum(a*b for a,b in zip(w2,P1r))/sum(w2), sum(a*b for a,b in zip(w2,P2r))/sum(w2))
    w1=[per[r]["n1"] for r in S]
    o["A4 N1w both"]     = R(sum(a*b for a,b in zip(w1,P1r))/sum(w1), sum(a*b for a,b in zip(w1,P2r))/sum(w1))
    o["A5 cell-eq Jeff"] = R(mean([(e1c[c]+.5)/(n1+1.) for c in CL]), mean([(e2c[c]+.5)/(n2+1.) for c in CL]))
    o["A6 rng-eq Jeff"]  = R(mean([(sum(per[r]["e1"])+.5)/(nc*per[r]["n1"]+1.) for r in S]),
                             mean([(sum(per[r]["e2"])+.5)/(nc*per[r]["n2"]+1.) for r in S]))
    o["A7 pool Jeff"]    = R((sum(e1c.values())+.5)/(nc*n1+1.), (sum(e2c.values())+.5)/(nc*n2+1.))
    return o
print("="*100)
print("★★★ 计数聚合族（8 条规则）—— 对每个 (dose, |S|>=2)，规则间 max/min")
print("="*100)
worst=(0,None); alld=[]
for d in DOSES:
    av=sorted(D[str(d)]["avail"]); vals=[]
    for k in range(2,len(av)+1):
        for S in itertools.combinations(av,k):
            o=agg_rules(d,S); v=[x for x in o.values() if x]
            if len(v)>=2:
                s=max(v)/min(v); vals.append(s); alld.append((s,d,S))
                if s>worst[0]: worst=(s,(d,S))
    if not vals:
        print("  dose=%-4d 结构=0    ——（无 |S|>=2 结构）"%d); continue
    print("  dose=%-4d 结构=%-3d  max/min = %.4f×  %s"%(d,len(vals),max(vals),"OK" if max(vals)<=1.10 else "FAIL"))
print()
print("★ 全表最差 = %.4f× @ %s   ⇒ 判据 ≤1.10× : %s"%(worst[0],worst[1],"PASS" if worst[0]<=1.10 else "FAIL"))
alld.sort(reverse=True)
print("★ 最差 5 个结构：")
for s,d,S in alld[:5]: print("     %.4f×  dose=%-4d S=%s"%(s,d,S))

print()
print("="*100)
print("★★★★ 规范读数（全子集 4 个 rng 全用）—— 8 条计数聚合规则的终值")
print("="*100)
NAMES=["A0 Nw pool","A1 rng-eq","A2 cell-eq","A3 N2w both","A4 N1w both",
       "A5 cell-eq Jeff","A6 rng-eq Jeff","A7 pool Jeff"]
print("%-6s %s  %s"%("dose"," ".join("%-9s"%n.split()[0]+n.split()[1][:3] for n in NAMES),"max/min"))
allsp=[]
for d in DOSES:
    o=agg_rules(d,tuple(sorted(D[str(d)]["avail"])))
    v=[o[n] for n in NAMES]
    vv=[x for x in v if x]
    sp=max(vv)/min(vv); allsp.append((sp,d))
    print("%-6d %s   %.4f×  %s"%(d," ".join(("%-11.4f"%x) if x else "%-11s"%"—" for x in v),sp,"OK" if sp<=1.10 else "FAIL"))
print()
print("★ 规范读数上规则间最差 = %.4f× @ dose=%d  ⇒ 判据 ≤1.10× : %s"%(
    max(allsp)[0],max(allsp)[1],"PASS 9/9" if max(allsp)[0]<=1.10 else "FAIL"))
