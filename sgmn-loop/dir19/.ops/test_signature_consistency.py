#!/usr/bin/env python3
"""DIR-19 · 独立单元测试：派生脚本的「包装层 ↔ 实现层」签名必须一致。

★★ 缺陷 D80 的直接来源：给 `train_arm` 包装层加了参数 `ref_logits`，
   却**忘了**给 `_train_arm_impl` 实现层加 ⇒ ⇒ 调用处抛
   `NameError: name 'ref_logits' is not defined` ⇒ ⇒ **首臂即崩**（Stage A 虽 30/30 PASS）。

⇒ ⇒ 本测试**独立于装置**，只做 AST 静态检查（不需要 torch / GPU）⇒ ⇒ 可以在服务器上秒级跑。
⇒ ⇒ 纪律 36 的应用：**派生代码的接口层必须单独测试**。
"""
import ast, pathlib, sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "R1_protect.py"
tree = ast.parse(SRC.read_text(encoding="utf-8"))
fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def kw(f):
    a = f.args
    return {x.arg for x in a.args} | {x.arg for x in a.kwonlyargs}


PAIRS = [("train_arm", "_train_arm_impl")]
ok = True
for w, i in PAIRS:
    if w not in fns or i not in fns:
        print(f"[FAIL] 找不到 {w} / {i}")
        ok = False
        continue
    kw_w, kw_i = kw(fns[w]), kw(fns[i])
    missing = kw_w - kw_i          # 实现层缺的（会 NameError）
    extra = kw_i - kw_w
    if missing or extra:
        print(f"[FAIL] {w} ↔ {i}：实现层缺失={sorted(missing)} 多出={sorted(extra)}")
        ok = False
    else:
        print(f"[PASS] {w} ↔ {i}：参数集合一致（{len(kw_w)} 个）")

# —— 负向对照：证明本测试**有区分力**（纪律 ⑬）——
fake = ast.parse("def w(a, b):\n    pass\ndef i(a):\n    pass\n")
fk = {n.name: n for n in ast.walk(fake) if isinstance(n, ast.FunctionDef)}
miss = kw(fk["w"]) - kw(fk["i"])
print(f"[负向对照] 缺 {sorted(miss)} ⇒ 断言应触发：{'OK（测试有区分力）' if miss else '★ 无区分力！'}")
if not miss:
    ok = False

sys.exit(0 if ok else 1)
