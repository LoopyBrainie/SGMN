#!/opt/pytorch/bin/python
r"""DIR-43 · **离线自检 ②**：冻结守卫（未冻结 ⇒ 拒绝；卡被改过 ⇒ 拒绝；守卫挂在 `formal_arm`/`merge` 上）。

★ **只读源码 / 不训练**。用法：python3 dir43/verify_freeze43.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path("/home/ec2-user/sgmn-loop")
sys.path.insert(0, str(ROOT / "dir43"))
import R1_task43 as T43


def main():
    fails = []
    src = (ROOT / "dir43/R1_task43.py").read_text(encoding="utf-8")

    # ── ① 占位符状态：未冻结 ⇒ 守卫必须**抛错**
    saved = T43.CARD_MD5_EXPECTED
    try:
        T43.CARD_MD5_EXPECTED = "__CARD_MD5__"
        try:
            T43._card_guard("selftest")
            fails.append("未冻结时守卫未拒绝")
            print("[1] ★ FAIL：未冻结时守卫未拒绝")
        except SystemExit:
            print("[1] OK：未冻结 ⇒ 拒绝执行")
    finally:
        T43.CARD_MD5_EXPECTED = saved

    # ── ② 卡被改过（md5 不符）⇒ 必须拒绝
    try:
        T43.CARD_MD5_EXPECTED = "0" * 32
        try:
            T43._card_guard("selftest")
            fails.append("卡被改过时守卫未拒绝")
            print("[2] ★ FAIL：卡 md5 不符时守卫未拒绝")
        except SystemExit:
            print("[2] OK：卡 md5 不符 ⇒ 拒绝执行")
    finally:
        T43.CARD_MD5_EXPECTED = saved

    # ── ③ 守卫必须挂在 `formal_arm` 与 `merge` 的**最前面**
    for fn in ("formal_arm", "merge"):
        m = re.search(rf"def {fn}\(.*?\n(.*?)(?=\ndef |\nif __name__)", src, re.S)
        body = m.group(1) if m else ""
        ok = "_card_guard" in body and body.index("_card_guard") < 400
        print(f"[3] {fn}：守卫在入口 ⇒ {ok}")
        if not ok:
            fails.append(f"{fn} 未在入口挂冻结守卫")

    # ── ④ 冻结后：真实卡 md5 必须与常量一致
    real = T43._md5(T43.CARD_PATH)
    ok = (saved != "__CARD_MD5__" and real == saved)
    print(f"[4] 卡 md5 实测 {real[:8]}… vs 常量 {str(saved)[:8]}… ⇒ {ok}")
    if not ok:
        fails.append("卡 md5 与常量不一致（未冻结或被改过）")

    print("=" * 60)
    if fails:
        print("★ verify_freeze43 FAIL：")
        for f in fails:
            print("   -", f)
        return 1
    print("★ verify_freeze43 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
