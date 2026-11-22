#!/usr/bin/env python3
"""★ 核心判别：**哪些剂量在子集轴上会翻符号**？（advisor 指出的、本卡最强的正面结果）

若某剂量在 15 个子集上 `ratio` 同时出现 `<1` 与 `>1` ⇒ 该剂量的**符号不是状态属性**，
而是 (状态, 题集) 的联合性质。
"""
import itertools, json, math
T=json.load(open("dir128/out/tensor128.json")); D=T["doses"]
DOSES=sorted(int(k) for k in D)
def ratio_of(p1,p2):
    if None in (p1,p2) or not(0<p1<1 and 0<p2<1): return None
    return (p1*(1-p2))/(4.0*p2*(1-p1))
print("="*92)
print("★ 子集轴上的符号翻覆（口径 = A0 现定义 / 全聚合）")
print("="*92)
print("%-6s %-5s %-10s %-10s %-9s %-9s %s"%("dose","nsub","min ratio","max ratio","全部<1","全部>1","翻符号"))
flip=[]
for d in DOSES:
    av=sorted(D[str(d)]["avail"]); per=D[str(d)]["per_rng"]; nc=T["cells_per_dose"][str(d)]
    vs=[]
    for k in range(1,len(av)+1):
        for S in itertools.combinations(av,k):
            n1=sum(per[r]["n1"] for r in S); n2=sum(per[r]["n2"] for r in S)
            e1=sum(sum(per[r]["e1"]) for r in S); e2=sum(sum(per[r]["e2"]) for r in S)
            v=ratio_of(e1/(nc*n1), e2/(nc*n2))
            if v: vs.append((v,S))
    lo=min(vs); hi=max(vs)
    alllo=all(v<1 for v,_ in vs); allhi=all(v>1 for v,_ in vs)
    f=(not alllo) and (not allhi)
    if f: flip.append(d)
    print("%-6d %-5d %-10.4f %-10.4f %-9s %-9s %-6s   argmin=%s argmax=%s"%(
        d,len(vs),lo[0],hi[0],"是的" if alllo else "no","是的" if allhi else "no",
        "★★★ 翻!" if f else "—", lo[1], hi[1]))
print()
print("★ 翻符号的剂量 =", flip if flip else "无")
print("★ 结论：", ("只有 dose=%s 一个剂量在子集轴上翻符号 ⇒ 该剂量处 `ratio` 的符号**不是状态属性**"%flip) if len(flip)==1
      else ("%d 个剂量翻符号"%len(flip)))
