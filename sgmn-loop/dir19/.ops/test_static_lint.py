#!/usr/bin/env python3
"""DIR-19 · 派生脚本的**静态 lint**（不依赖 torch / GPU，秒级可跑）。

覆盖本轮实际踩到的三类**同源**缺陷：
  ① 包装层 ↔ 实现层**签名不一致**（D80）⇒ ⇒ 调用处 NameError；
  ② **全局名遮蔽**：判据块里把全局读秩 `R`（数字）当成旋钮臂名用（D65 / D82）⇒ ⇒ 判据**从未被计算**；
  ③ `rec()` 的**判据名重复**（DIR-18 D57 同类）⇒ ⇒ 两条判据互相覆盖。

⇒ ⇒ 纪律 36 的应用：派生代码的接口层与判据层必须有**独立于装置**的最小单元测试。
"""
import ast, pathlib, re, sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "R1_protect.py"
text = SRC.read_text(encoding="utf-8")
tree = ast.parse(text)
fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
ok = True


def kw(f):
    a = f.args
    return {x.arg for x in a.args} | {x.arg for x in a.kwonlyargs}


# —— ① 签名一致性 ——
for w, i in [("train_arm", "_train_arm_impl")]:
    if w in fns and i in fns:
        miss, extra = kw(fns[w]) - kw(fns[i]), kw(fns[i]) - kw(fns[w])
        if miss or extra:
            print(f"[FAIL] ① {w} ↔ {i}：实现层缺失={sorted(miss)} 多出={sorted(extra)}")
            ok = False
        else:
            print(f"[PASS] ① {w} ↔ {i}：参数集合一致（{len(kw(fns[w]))} 个）")
    else:
        print(f"[FAIL] ① 找不到 {w} / {i}")
        ok = False

# —— ② 全局名遮蔽：判据块（"汇总与判据" 之后）里不得出现裸 `R` 作容器下标 / 格式化 ——
lines = text.split("\n")
start = next((i for i, l in enumerate(lines) if "汇总与判据" in l), 0)
crit = "\n".join(lines[start:])
bad = []
for i, l in enumerate(crit.split("\n"), start + 1):
    code = l.split("#")[0]
    if re.search(r"\bR\s+not\s+in\b", code) or re.search(r"\{R[:}]", code) or re.search(r"\bR\s+in\s+agg\b", code):
        bad.append(i)
if bad:
    print(f"[FAIL] ② 判据块里出现裸 `R`（全局读秩）当作容器键的行：{bad}")
    ok = False
else:
    print("[PASS] ② 判据块无裸 `R` 误用")

# —— ③ `rec()` 判据名不得重复（**互斥的 if/else 兜底分支允许同名**）——
calls = []
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "rec" and n.args:
        a0 = n.args[0]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
            calls.append((a0.value.split("〔")[0].strip(), n))


def branch_sig(call):
    """返回该 call 所在的「互斥分支路径」：每一层是 ('if'/'else', if_node_id)。"""
    path = []
    cur = call
    while cur is not None:
        par = getattr(cur, "parent", None)
        if par is None:
            break
        if isinstance(par, ast.If) and cur in (par.body + par.orelse):
            path.append(("if" if cur in par.body else "else", id(par)))
        cur = par
    return tuple(path)


# 建立 parent 指针
for parent in ast.walk(tree):
    for child in ast.iter_child_nodes(parent):
        child.parent = parent

by_name = {}
for nm, c in calls:
    by_name.setdefault(nm, []).append(c)
dups = {}
for nm, cs in by_name.items():
    if len(cs) < 2:
        continue
    sigs = [branch_sig(c) for c in cs]
    if len(set(sigs)) == len(sigs):       # 互斥分支 ⇒ 合法
        continue
    dups[nm] = [c.lineno for c in cs]
if dups:
    print(f"[FAIL] ③ `rec()` 判据名在**同一执行路径**上重复：{dups}")
    ok = False
else:
    print(f"[PASS] ③ {len(calls)} 条 `rec()` 判据名无同路径重复"
          f"（互斥 if/else 兜底同名 {sum(1 for cs in by_name.values() if len(cs) > 1)} 组，已豁免）")

# —— 负向对照：证明本测试**有区分力**（纪律 ⑬）——
fake = ast.parse("def w(a, b):\n    pass\ndef i(a):\n    pass\n")
fk = {n.name: n for n in ast.walk(fake) if isinstance(n, ast.FunctionDef)}
if not (kw(fk['w']) - kw(fk['i'])):
    print("[FAIL] 负向对照失效 ⇒ 测试无区分力")
    ok = False
else:
    print("[PASS] 负向对照：签名缺失能被检出（测试有区分力）")

sys.exit(0 if ok else 1)
