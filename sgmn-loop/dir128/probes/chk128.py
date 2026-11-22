import itertools, json, math
T=json.load(open("dir128/out/tensor128.json")); D=T["doses"]
DOSES=sorted(int(k) for k in D)
def ratio_of(p1,p2):
    if None in (p1,p2) or not(0<p1<1 and 0<p2<1): return None
    return (p1*(1-p2))/(4.0*p2*(1-p1))
def subsets(d):
    av=D[str(d)]["avail"]; o=[]
    for k in range(1,len(av)+1): o.extend(tuple(sorted(c)) for c in itertools.combinations(av,k))
    return o
def pool(d,S):
    per=D[str(d)]["per_rng"]; nc=T["cells_per_dose"][str(d)]
    n1=sum(per[r]["n1"] for r in S); n2=sum(per[r]["n2"] for r in S)
    e1=sum(sum(per[r]["e1"]) for r in S); e2=sum(sum(per[r]["e2"]) for r in S)
    return ratio_of(e1/(nc*n1), e2/(nc*n2))

print("="*100)
print("★ 关键验证：极值是否由 |S|=1 取得？（若是 ⇒ 任何权重口径都给出同一 max/min）")
print("="*100)
print("%-6s %-9s %-24s %-24s %-9s %s"%("dose","max/min","argmax","argmin","maxmin","|S| of 极值"))
for d in DOSES:
    Ss=subsets(d)
    if len(Ss)<2: print("%-6d —（只 1 个子集，无内变异）"%d); continue
    vs={S:pool(d,S) for S in Ss}
    vs={S:v for S,v in vs.items() if v}
    mx=max(vs,key=vs.get); mn=min(vs,key=vs.get)
    print("%-6d %-9.3f× %-24s %-24s %-9.4f %s"%(
        d, vs[mx]/vs[mn], str(mx)+" =%.4f"%vs[mx], str(mn)+" =%.4f"%vs[mn], vs[mn], (len(mx),len(mn))))

print()
print("="*100)
print("★★ 只保留 |S|>=2 的「真合并」结构：内变异还剩多少？")
print("="*100)
for d in DOSES:
    Ss=[S for S in subsets(d) if len(S)>=2]
    vs=[pool(d,S) for S in Ss]; vs=[v for v in vs if v]
    if len(vs)>=2:
        print("  dose=%-4d n=%-2d  |S|>=2 内 max/min = %.3f×   全子集(含单) = %.3f×"%(
            d,len(vs),max(vs)/min(vs), (lambda a:[max(a)/min(a)][0])([v for v in (pool(d,S) for S in subsets(d)) if v])))
    else:
        print("  dose=%-4d 无 |S|>=2 结构"%d)
