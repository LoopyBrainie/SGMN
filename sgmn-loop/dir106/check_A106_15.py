#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`A106-15`：跨代码版本的**读数可比性**闸门（`DEF-106-12`）。

背景：Cycle 198 在**驱动运行期间**改过 `R1_task106.py`（~12:24）——
前 ~29 格的 `card` 是 rev 1.2，其后是 rev 1.3。改动只在
`CARD` / `verdict_main` / `EXPECT_*` / `se_band` / 输出路径，
**`run_cell` 测量路径一字未改** ⇒ 读数**应当**可比。

★ 但「我看过 diff」不是证据。本检查**重跑一个已存的早期格**并与之**逐位比对**：
  一致 ⇒ 改动确实没触及测量路径 ⇒ 合并合法；
  不一致 ⇒ 本次正式轮**跨版本混装**，必须**整轮重跑**，不得 merge。

★ 必须在**驱动结束之后**跑：并发跑会争 GPU，可能扰动正要测量的东西。

被检格：`main_16_0_20261028`（第一格，确定跑在 rev 1.2 文件下）。

用法：`/opt/pytorch/bin/python check_A106_15.py`
"""
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TARGET = "main_16_0_20261028"
SRC = HERE / "out" / "cells" / f"{TARGET}.json"
TOL_KEYS = ("acc_train", "acc_held", "W_drift", "key_maxcos", "violations", "n_held",
            "acc_item_train", "acc_item_held", "margin_train", "margin_held")


def load(p):
    d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    cs = d.get("cells", [])
    assert len(cs) == 1, f"{p}: 期望 1 格，实得 {len(cs)}"
    return cs[0], d.get("card")


def main():
    if not SRC.exists():
        print(f"✗ 缺少基准格 {SRC}（驱动尚未跑到或已被删）⇒ 本闸门无法判定，不得 merge")
        return 2

    old, card_old = load(SRC)
    print(f"基准格 {TARGET}（card={card_old}）")
    print(f"  acc_train={old['acc_train']}  acc_held={old['acc_held']}  W_drift={old.get('W_drift')}")

    # ---- ★ 重跑会写 `out/cell_<arm>_<n>_<sp>_<sd>.json`（`_dump` 的命名）。
    #   `_load_cells` glob 的是 `out/cells/*.json` ⇒ 这不会污染 merge；
    #   但本脚本**跑完仍会自删**，让它**不可能**污染（不靠默认参数碰巧正确）。
    #   先清掉任何既存同名产物，确保下方「取最新」无法被旧文件欺骗。
    for f in (HERE / "out").glob("cell_*.json"):
        f.unlink()
    cmd = [sys.executable, "R1_task106.py", "--stage", "formal", "--arm", "main",
           "--n-train", str(old["n_train"]), "--split", str(old["split"]),
           "--seed", str(old["seed"])]
    print(f"重跑：{' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0:
        print("✗ 重跑失败：")
        print((r.stdout or "")[-1500:])
        print((r.stderr or "")[-1500:])
        return 1

    cand = sorted((HERE / "out").glob("cell_*.json"), key=lambda p: p.stat().st_mtime)
    if not cand:
        print("✗ 重跑未产生 cell_*.json")
        return 1
    src = cand[-1]
    new, card_new = load(src)
    # 断言重跑格确实是基准格的同一坐标（防止取到别的产物）
    for k in ("arm", "n_train", "split", "seed"):
        assert new.get(k) == old.get(k), f"重跑坐标不符：{k} {new.get(k)} != {old.get(k)}"
    print(f"重跑格（card={card_new}）")
    print(f"  acc_train={new['acc_train']}  acc_held={new['acc_held']}  W_drift={new.get('W_drift')}")

    bad = []
    for k in TOL_KEYS:
        a, b = old.get(k), new.get(k)
        if a != b:                      # 逐位，不放容差
            bad.append((k, a, b))

    print()
    if bad:
        print("✗ A106-15 FAIL：跨版本读数**不一致** ⇒ 本正式轮跨版本混装，必须整轮重跑，不得 merge")
        for k, a, b in bad:
            print(f"    {k}: 基准={a!r}  重跑={b!r}")
        verdict = "FAIL"
    else:
        print(f"✓ A106-15 PASS：{len(TOL_KEYS)} 个读数键**逐位一致** ⇒ "
              f"rev 1.2→1.3 的改动未触及测量路径 ⇒ 合并合法")
        verdict = "PASS"

    src.unlink()                      # ★ 自删重跑产物，绝不留在 out/ 根目录
    (HERE / "out" / "A106_15.json").write_text(json.dumps(
        {"A106-15": verdict, "target": TARGET, "card_old": card_old, "card_new": card_new,
         "old": {k: old.get(k) for k in TOL_KEYS}, "new": {k: new.get(k) for k in TOL_KEYS},
         "mismatches": bad}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {HERE / 'out' / 'A106_15.json'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())