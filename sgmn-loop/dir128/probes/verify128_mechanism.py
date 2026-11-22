#!/usr/bin/env python3
"""★ ① 规范读数上「符号对聚合规则稳健」是否 9/9；② `dose=450` 异常是否可归因 `rng=557`。"""
import itertools, json, math
T=json.load(open("dir128/out/tensor128.json")); D=T["doses"]
DOSES=sorted(int(k) for k in D)
def R(p1,p2):
    if None in (p1,p2) or not(0<p1<1 and 0<p2<1): return None
    return (p1*(1-p2))/(4.0*p2*(1-p1))
def mean(xs):
    xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
def agg(d,S):
    """Axis-1 族（**未平滑 plug-in 加权**）A0..A4 —— advisor 建议的闭环口径。"""
    per=D[str(d)]["per_rng"]; CL=list(range(T["cells_per_dose"][str(d)])); nc=len(CL)
    n1=sum(per[r]["n1"] for r in S); n2=sum(per[r]["n2"] for r in S)
    e1c={c:sum(per[r]["e1"][c] for r in S) for c in CL}
    e2c={c:sum(per[r]["e2"][c] for r in S) for c in CL}
    P1r=[sum(per[r]["e1"])/(nc*per[r]["n1"]) for r in S]
    P2r=[sum(per[r]["e2"])/(nc*per[r]["n2"]) for r in S]
    w2=[per[r]["n2"] for r in S]; w1=[per[r]["n1"] for r in S]
    return {
      "A0 Nw":   R(sum(e1c.values())/(nc*n1), sum(e2c.values())/(nc*n2)),
      "A1 req":  R(mean(P1r), mean(P2r)),
      "A2 ceq":  R(mean([e1c[c]/n1 for c in CL]), mean([e2c[c]/n2 for c in CL])),
      "A3 N2w":  R(sum(a*b for a,b in zip(w2,P1r))/sum(w2), sum(a*b for a,b in zip(w2,P2r))/sum(w2)),
      "A4 N1w":  R(sum(a*b for a,b in zip(w1,P1r))/sum(w1), sum(a*b for a,b in zip(w1,P2r))/sum(w1)),
    }
NAMES=["A0 Nw","A1 req","A2 ceq","A3 N2w","A4 N1w"]
print("="*100); print("① 规范读数（全 rng）：Axis-1 族 5 条规则的符号是否一致？"); print("="*100)
print("%-6s %s  %-8s %s"%("dose"," ".join("%-10s"%n for n in NAMES),"max/min","符号一致?"))
bad=[]
for d in DOSES:
    o=agg(d,tuple(sorted(D[str(d)]["avail"]))); v=[o[n] for n in NAMES]; vv=[x for x in v if x]
    sp=max(vv)/min(vv); sg={("+" if x>1 else "-") for x in vv}
    if len(sg)>1: bad.append(d)
    print("%-6d %s  %-8.4f %s"%(d," ".join("%-10.4f"%x for x in v),sp,"OK(全 %s)"%sg.pop() if len(sg)==1 else "★★ 不一致!"))
print("\n★ 符号不一致的剂量 =", bad if bad else "无 ⇒ **符号对规则稳健 9/9**")
print("★ Axis-1 值不变性：dose!=450 时 max/min <= %.4f×；dose=450 时 = %.4f×"%(
    max(max([x for x in agg(d,tuple(sorted(D[str(d)]['avail']))).values() if x])/min([x for x in agg(d,tuple(sorted(D[str(d)]['avail']))).values() if x]) for d in DOSES if d!=450),
    (lambda v:max(v)/min(v))([x for x in agg(450,('555','556','557','558')).values() if x])))

print()
print("="*100); print("② 机制：`dose=450` 的异常能否归因 `rng=557`？（n1=9,n2=29 极端失衡 + e2 极小）"); print("="*100)
print("%-4s %-4s %-5s %-5s %-6s %-6s %-9s %-9s"%("dose","rng","n1","n2","Σe1","Σe2","ratio(单)","与全集中位比"))
for d in (400,450,600):
    per=D[str(d)]["per_rng"]; nc=T["cells_per_dose"][str(d)]
    full=agg(d,tuple(sorted(per))); ref=R(max(full.values()),1e-9)
    med=sorted(full.values())[len(full)//2]
    for r in sorted(per):
        e1=sum(per[r]["e1"]); e2=sum(per[r]["e2"])
        v=R(e1/(nc*per[r]["n1"]), e2/(nc*per[r]["n2"]))
        print("%-4d %-4s %-5d %-5d %-6d %-6d %-9s %-9s"%(d,r,per[r]["n1"],per[r]["n2"],e1,e2,
            ("%.4f"%v) if v else "—", ("%.3f×"%(v/med)) if v else "—"))
    print()
print("★ 读法：若 557 的单题集读数偏离同剂量其它题集最远 ⇒ dose=450 的翻符号可归因它。")
