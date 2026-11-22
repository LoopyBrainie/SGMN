#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-20 · 装置 v5 → v6 的**逐字派生**（唯一改动：`W` 缺省时的形状校验）。

★ 缺陷 D91（卡片 rev 1.5 §0.9）：v5 把 `export_state` 的 `W` 改成可缺省（`None`），
  但 `import_state` 里**形状校验那一句**（在 v4 中位于 `copy_` 之前）仍是无条件的
  ⇒ ⇒ `d["W"].shape` 在 `d["W"] is None` 时 ⇒
  ⇒ `AttributeError: 'NoneType' object has no attribute 'shape'` ⇒ `X6_verbatim` 臂仍崩。

修法：把形状校验并入「`W` 存在」的分支（与 v5 的复制逻辑对齐）。
写规则 / 读规则 / 梯度隔离 / 诊断量**一字未改**；v5（md5 `aec73445…`）留档不改。
"""
import hashlib
import pathlib

SRC = pathlib.Path("/home/ec2-user/sgmn-loop/dir20/sgmn_module_hebb_v5.py")
DST = pathlib.Path("/home/ec2-user/sgmn-loop/dir20/sgmn_module_hebb_v6.py")

OLD = '''                if d["W"].shape != m.core.W.shape:
                    raise ValueError(f"层 {li}：W 形状不符 {tuple(d['W'].shape)} vs {tuple(m.core.W.shape)}")
                # ★★【DIR-20 · 装置 v5 · 缺陷 D89】`W` 可缺省（`explicit_kv` 核没有 W）。
                if d.get("W") is not None:
                    if not hasattr(m.core, "W"):
                        raise ValueError(f"层 {li}：当前核没有 W，无法导入")
                    if d["W"].shape != m.core.W.shape:
                        raise ValueError(f"层 {li}：W 形状不符 {tuple(d['W'].shape)} vs {tuple(m.core.W.shape)}")
                    m.core.W.copy_(d["W"].to(m.core.W.device))
'''

NEW = '''                # ★★【DIR-20 · 装置 v6 · 缺陷 D91】`W` 可缺省（`explicit_kv` 核没有 W）：
                #   ⇒ 形状校验**必须**与复制逻辑同在「`W` 存在」的分支内
                #   ⇒ （v5 把校验漏在了分支外 ⇒ `d["W"].shape` 对 None 崩）。
                if d.get("W") is not None:
                    if not hasattr(m.core, "W"):
                        raise ValueError(f"层 {li}：当前核没有 W，无法导入")
                    if d["W"].shape != m.core.W.shape:
                        raise ValueError(f"层 {li}：W 形状不符 {tuple(d['W'].shape)} vs {tuple(m.core.W.shape)}")
                    m.core.W.copy_(d["W"].to(m.core.W.device))
'''


def main():
    t = SRC.read_text(encoding="utf-8")
    if t.count(OLD) != 1:
        raise SystemExit(f"[派生失败] 目标段命中 {t.count(OLD)} 次（期望 1）")
    t = t.replace(OLD, NEW, 1)
    t = t.replace(
        "# ================================================================ DIR-20 新增",
        "# ★★★ 装置 v6（DIR-20 · 2026-11-07）：v5 的**逐字派生**，**只改 import 的 W 形状校验归属**\n"
        "#     （缺陷 D91：`W=None` 时 `d[\"W\"].shape` 崩）。写规则、读规则**一字未改**。\n"
        "# ================================================================ DIR-20 新增", 1)
    DST.write_text(t, encoding="utf-8")
    print("写出", DST)
    print("v5 md5 =", hashlib.md5(SRC.read_bytes()).hexdigest())
    print("v6 md5 =", hashlib.md5(t.encode("utf-8")).hexdigest())
    print("行数 =", t.count("\n") + 1)


if __name__ == "__main__":
    main()
