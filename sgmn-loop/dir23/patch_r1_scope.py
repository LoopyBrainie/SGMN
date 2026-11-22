#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-23 · `dir23/` 的**装置复制与派生核验器**（可审计、可重放）。

与 DIR-22 时代的 `patch_*.py`（动态改写上游脚本文本）不同，本轮的实施口径是 **DEC-09**：
装置**零改动**、`S` 全部在测量层 ⇒ 本文件的职责从"打补丁"变成"**证明没有打补丁**"：

  V01 `dir23/sgmn_module_hebb_v7.py` ← `dir22/` 同名文件的**逐字节复制**（md5 相等，否则拒跑）。
  V02 `R1_scope.py` 从 `dir22/R1_rate.py` **逐字复制**的函数（`_norm_cdf`/`sign_test_p`/`paired`）
      —— 用 AST 抽出源码文本逐字比对；不一致即拒跑（防止"抄错一个字符"）。
  V03 生成 `dir23/MANIFEST.json`：三份 md5 + 行数 + 复制清单 ⇒ 供打包/审计复核。

用法：`/opt/pytorch/bin/python dir23/patch_r1_scope.py`
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
UP = HERE.parent / "dir22"
DEV_NAME = "sgmn_module_hebb_v7.py"
UPSTREAM_SCRIPT = "R1_rate.py"
COPIED_FUNCS = ["_norm_cdf", "sign_test_p", "paired"]


def md5(p: pathlib.Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def func_src(path: pathlib.Path, name: str) -> str:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node).strip()
    raise KeyError(f"{name} 不在 {path.name} 的顶层定义中")


def main() -> int:
    fail = []

    # ── V01 装置逐字节复制
    src_dev, dst_dev = UP / DEV_NAME, HERE / DEV_NAME
    if not src_dev.exists():
        print(f"[FAIL] 上游装置不存在：{src_dev}")
        return 1
    if not dst_dev.exists() or md5(dst_dev) != md5(src_dev):
        dst_dev.write_bytes(src_dev.read_bytes())
        print(f"[copy] {DEV_NAME} ← dir22/（md5 {md5(dst_dev)[:12]}）")
    same = md5(dst_dev) == md5(src_dev)
    print(f"[{'PASS' if same else 'FAIL'}] V01 装置 md5 相等：{md5(dst_dev)}")
    if not same:
        fail.append("V01")

    # ── V02 逐字复制的函数
    up = UP / UPSTREAM_SCRIPT
    cur = HERE / "R1_scope.py"
    for fn in COPIED_FUNCS:
        a, b = func_src(up, fn), func_src(cur, fn)
        ok = a == b
        print(f"[{'PASS' if ok else 'FAIL'}] V02 {fn} 与 dir22/{UPSTREAM_SCRIPT} 逐字一致"
              f"（{len(a)} 字符）")
        if not ok:
            fail.append(f"V02:{fn}")

    # ── V03 MANIFEST
    man = {
        "dir": "DIR23r11",
        "device": {"file": DEV_NAME, "md5": md5(dst_dev),
                   "copied_from": f"dir22/{DEV_NAME}", "copy_is_byte_identical": same},
        "script": {"file": "R1_scope.py", "md5": md5(cur),
                   "lines": len(cur.read_text(encoding="utf-8").splitlines()),
                   "verbatim_funcs_from": f"dir22/{UPSTREAM_SCRIPT}",
                   "verbatim_funcs": COPIED_FUNCS},
        "decisions": ["DEC-09（装置零改动；S 全在测量层）"],
        "checks": {"V01": same, "V02": [f for f in fail if f.startswith("V02")] == []},
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"[写] MANIFEST.json")
    print(f"\n{'★ 全部通过' if not fail else '★ FAIL: ' + ','.join(fail)}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
