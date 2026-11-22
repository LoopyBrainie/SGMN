#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-24 · `dir24/` 的**装置复制与派生核验器**（可审计、可重放）。

同 DIR-23 的 DEC-09 口径：装置**零改动**，秩约束只在初值支撑上施加 ⇒ 本文件证明"没有打补丁"：

  V01 `dir24/sgmn_module_hebb_v7.py` ← `dir22/` 同名文件的**逐字节复制**（md5 相等，否则拒跑）。
  V02 `R1_rank.py` 从 `dir23/R1_scope.py` **逐字复制**的函数（`_norm_cdf`/`sign_test_p`/`unit`/`item_z`/`best_q`）
      —— AST 抽出源码文本逐字比对；不一致即拒跑。
  V03 生成 `dir24/MANIFEST.json`：md5 + 行数 + 复制清单 + 派生说明。

用法：`/opt/pytorch/bin/python dir24/patch_r1_rank.py`
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
UPSTREAM_SCRIPT = HERE.parent / "dir23" / "R1_scope.py"
COPIED_FUNCS = ["_norm_cdf", "sign_test_p", "unit", "item_z", "best_q"]


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
    up_dev, dev = UP / DEV_NAME, HERE / DEV_NAME
    if md5(up_dev) != md5(dev):
        print(f"V01 FAIL: 装置 md5 不等 {md5(up_dev)} vs {md5(dev)}"); return 2
    print(f"V01 OK   装置逐字节相同 md5={md5(dev)}")

    ours = HERE / "R1_rank.py"
    bad = []
    for name in COPIED_FUNCS:
        if func_src(UPSTREAM_SCRIPT, name) != func_src(ours, name):
            bad.append(name)
    if bad:
        print(f"V02 FAIL: 逐字不符 {bad}"); return 2
    print(f"V02 OK   {len(COPIED_FUNCS)} 个函数与 dir23/R1_scope.py 逐字相同：{COPIED_FUNCS}")

    man = {
        "dir": "DIR24r10",
        "device": {"file": DEV_NAME, "md5": md5(dev),
                   "copied_from": "dir22/sgmn_module_hebb_v7.py",
                   "copy_is_byte_identical": True},
        "script": {"file": "R1_rank.py", "md5": md5(ours),
                   "lines": len(ours.read_text(encoding="utf-8").splitlines()),
                   "verbatim_funcs_from": "dir23/R1_scope.py",
                   "verbatim_funcs": COPIED_FUNCS,
                   "derived_changes": [
                       "build_A：去掉 S 字段 u；d 与 N_pair 从常量改为参数（卡片 §6 允许）",
                       "make_core → make_core(d, seed, r0)：首 r0 列支撑施加秩约束（DEC-15，装置零改动）",
                       "ARMS（六臂）→ R0_LEVELS（7 个秩水平）+ D_LEVELS（2 个维度轴）",
                       "新增 rank_W / proj_resid_max / rank_snaps 三个机制读数"]},
        "decisions": ["DEC-09（装置零改动）", "DEC-15（零改动施秩）", "DEC-16（可辨识性）",
                      "DEC-17（草案订正）", "DEC-18（d=64 的水平裁剪）"],
        "checks": {"V01": True, "V02": True},
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print("V03 OK   MANIFEST.json 已生成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
