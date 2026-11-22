#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-20 · 装置 v4 → v5 的**逐字派生**（唯一改动：状态搬运对「无 W 的核」可用）。

★ 缺陷 D89（卡片 rev 1.4 §0.7）：`export_state` 无条件访问 `core.W`
  ⇒ 而 `explicit_kv` 核（**X6_verbatim 臂**用）**没有 W**（它的内容在地址表的 `vals` 里）
  ⇒ ⇒ `AttributeError: 'ExplicitKVCore' object has no attribute 'W'` ⇒ 主循环在 X6 臂崩溃。

修法（**只改搬运层，不改任何写/读规则**）：
  ① `export_state`：核心没有 `W` 时写 `None`；地址表的 `keys/vals/sigs` 逐个按 `hasattr` 取；
  ② `import_state`：`W is None` ⇒ 跳过；`keys/vals/sigs` 逐个按 `hasattr` + 形状校验处理；
     `count` 只在状态里存在时设置（`include_addr=False` ⇒ 不设置 ⇒ 保持 0，与 A20-5 一致）。
  ③ `state_fingerprint`：`None` 走「非张量」分支（哈希 `W=None`）⇒ 往返指纹仍逐位可比。

纪律：装置改动 ⇒ **新版本 + md5 + 卡片 revision**（rev 1.4）⇒ 旧版 v4 留档不改。
"""
import hashlib
import pathlib

SRC = pathlib.Path("/home/ec2-user/sgmn-loop/dir20/sgmn_module_hebb_v4.py")
DST = pathlib.Path("/home/ec2-user/sgmn-loop/dir20/sgmn_module_hebb_v5.py")

OLD_EXPORT = '''            d = {
                "mem_kind": m.mem_kind, "key_mode": m.key_mode,
                "query_mode": m.query_mode, "addr_mode": m.addr_mode,
                "read_beta": float(m.read_beta),
                "W": core.W.detach().cpu().clone(),
            }
            if include_addr:
                d["keys"] = core.keys.detach().cpu().clone()
                d["vals"] = core.vals.detach().cpu().clone()
                d["sigs"] = core.sigs.detach().cpu().clone()
                d["count"] = int(core.count)
'''

NEW_EXPORT = '''            d = {
                "mem_kind": m.mem_kind, "key_mode": m.key_mode,
                "query_mode": m.query_mode, "addr_mode": m.addr_mode,
                "read_beta": float(m.read_beta),
                # ★★【DIR-20 · 装置 v5 · 缺陷 D89】核心**可能没有 W**：
                #   `explicit_kv` 核（X6_verbatim 臂）的内容在地址表的 `vals` 里 ⇒ 没有 W。
                #   ⇒ ⇒ 没有 W 时写 `None`（导入时跳过）⇒ 状态搬运对**所有核**都可用。
                "W": (core.W.detach().cpu().clone() if hasattr(core, "W") else None),
            }
            if include_addr:
                for _k in ("keys", "vals", "sigs"):
                    if hasattr(core, _k):
                        d[_k] = getattr(core, _k).detach().cpu().clone()
                d["count"] = int(core.count)
'''

OLD_IMPORT = '''                m.core.W.copy_(d["W"].to(m.core.W.device))
                if "keys" in d:
                    if d["keys"].shape != m.core.keys.shape:
                        raise ValueError(f"层 {li}：keys 形状不符")
                    m.core.keys.copy_(d["keys"].to(m.core.keys.device))
                    m.core.vals.copy_(d["vals"].to(m.core.vals.device))
                    m.core.sigs.copy_(d["sigs"].to(m.core.sigs.device))
                    m.core.count = int(d["count"])
'''

NEW_IMPORT = '''                # ★★【DIR-20 · 装置 v5 · 缺陷 D89】`W` 可缺省（`explicit_kv` 核没有 W）。
                if d.get("W") is not None:
                    if not hasattr(m.core, "W"):
                        raise ValueError(f"层 {li}：当前核没有 W，无法导入")
                    if d["W"].shape != m.core.W.shape:
                        raise ValueError(f"层 {li}：W 形状不符 {tuple(d['W'].shape)} vs {tuple(m.core.W.shape)}")
                    m.core.W.copy_(d["W"].to(m.core.W.device))
                for _k in ("keys", "vals", "sigs"):
                    if _k in d:
                        if not hasattr(m.core, _k):
                            raise ValueError(f"层 {li}：当前核没有 {_k}，无法导入")
                        _t = getattr(m.core, _k)
                        if d[_k].shape != _t.shape:
                            raise ValueError(f"层 {li}：{_k} 形状不符")
                        _t.copy_(d[_k].to(_t.device))
                if "count" in d:
                    m.core.count = int(d["count"])
'''


def main():
    t = SRC.read_text(encoding="utf-8")
    for tag, old, new in (("export", OLD_EXPORT, NEW_EXPORT),
                           ("import", OLD_IMPORT, NEW_IMPORT)):
        if t.count(old) != 1:
            raise SystemExit(f"[派生失败] {tag} 段命中 {t.count(old)} 次（期望 1）")
        t = t.replace(old, new, 1)
    # 文件头：加 v5 派生说明
    head_marker = "# ================================================================ DIR-20 新增"
    if t.count(head_marker) != 1:
        raise SystemExit("[派生失败] 找不到 DIR-20 段标记")
    t = t.replace(
        head_marker,
        "# ★★★ 装置 v5（DIR-20 · 2026-11-07）：v4 的**逐字派生**，**只改状态搬运层**\n"
        "#     （缺陷 D89：`explicit_kv` 核没有 W ⇒ `export_state` 崩 ⇒ 改为 W 可缺省）。\n"
        "#     写规则、读规则、梯度隔离、诊断量**一字未改**。\n"
        + head_marker, 1)
    DST.write_text(t, encoding="utf-8")
    print("写出", DST)
    print("v4 md5 =", hashlib.md5(SRC.read_bytes()).hexdigest())
    print("v5 md5 =", hashlib.md5(t.encode("utf-8")).hexdigest())
    print("行数 =", t.count("\n") + 1)


if __name__ == "__main__":
    main()
