#!/opt/pytorch/bin/python
r"""DIR-43 · **离线自检 ③**：卡 ↔ 代码一致性（**阈值漂移探测器**）：卡 §3.2 的每个阈值
必须与代码常量**逐字**一致；卡 §6 的种子段必须与代码一致；卡 §7 的断言表必须与代码里的断言一致。

★ **只读源码 / 不训练**。用法：python3 dir43/verify_card43.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path("/home/ec2-user/sgmn-loop")
sys.path.insert(0, str(ROOT / "dir43"))
import R1_task43 as T43

CARD = ROOT / "revolve/DIR-43/DIRECTION_rev43_v10.md"


def main():
    fails = []
    card = CARD.read_text(encoding="utf-8")
    src = (ROOT / "dir43/R1_task43.py").read_text(encoding="utf-8")

    # ── ① 阈值：卡 §3.2 的常量 ↔ 代码常量
    thresholds = [("SIG_K", T43.SIG_K), ("N_INST", T43.N_INST), ("N_CTRL", T43.N_CTRL),
                  ("STEPS", T43.STEPS), ("LR", T43.LR), ("N_REP", T43.N_REP),
                  ("COS_DIRECT", T43.COS_DIRECT)]
    for name, val in thresholds:
        row = [ln for ln in card.split("\n") if ln.startswith(f"| `{name}`")]
        if not row:
            fails.append(f"卡 §3.2 缺 {name}")
            print(f"[1] ★ FAIL：卡 §3.2 缺 {name}")
            continue
        m = re.search(r"\*\*([0-9.]+)\*\*", row[0])
        cv = float(m.group(1)) if m else None
        ok = (cv is not None and abs(cv - float(val)) < 1e-12)
        print(f"[1] {name:12s} 卡={cv} 代码={val} {'OK' if ok else '★ FAIL'}")
        if not ok:
            fails.append(f"{name} 阈值漂移：卡 {cv} ≠ 代码 {val}")

    # ── ② 窗口
    for name, val in (("TRAIN_WIN", list(T43.TRAIN_WIN)), ("EVAL_WIN", list(T43.EVAL_WIN))):
        ok = f"`[{val[0]},{val[1]}]`" in card
        print(f"[2] {name:12s} 卡窗口 {val} ⇒ {ok}")
        if not ok:
            fails.append(f"{name} 与卡不一致")

    # ── ③ 种子段
    segs = {"INST_OFF": (T43.INST_OFF[0], T43.INST_OFF[-1]),
            "CTRL_OFF": (T43.CTRL_OFF[0], T43.CTRL_OFF[-1]),
            "SMOKE_INST_OFF": (T43.SMOKE_INST_OFF[0], T43.SMOKE_INST_OFF[-1]),
            "PROBE_INST_OFF": (T43.PROBE_INST_OFF[0], T43.PROBE_INST_OFF[-1])}
    for name, (a, b) in segs.items():
        ok = f"**{a}..{b}**" in card
        print(f"[3] {name:16s} {a}..{b} ⇒ 卡含此段：{ok}")
        if not ok:
            fails.append(f"{name} 段 {a}..{b} 与卡不一致")
    ok = (T43.READER_SEED_OFFSET == 10_000_000 and T43.RAND_KEY_SEED_OFFSET == 55_555_555
          and "55,555,555" in card and "10,000,000" in card)
    print(f"[3] 两条种子流偏移在卡中登记 ⇒ {ok}")
    if not ok:
        fails.append("种子流偏移与卡不一致")

    # ── ④ 断言表：卡 §7 列出的 A43-n ⇒ 代码里都存在
    card_ids = sorted(set(re.findall(r"A43-(\d+)", card)))
    code_ids = sorted(set(re.findall(r"A43-(\d+)", src)))
    miss = [f"A43-{i}" for i in card_ids if i not in code_ids]
    extra = [f"A43-{i}" for i in code_ids if i not in card_ids]
    print(f"[4] 卡断言 {len(card_ids)} 条 / 代码断言 {len(code_ids)} 条；"
          f"卡有代码无={miss}；代码有卡无={extra}")
    if miss or extra:
        fails.append(f"断言表不一致：{miss} / {extra}")

    # ── ⑤ 符号约定：卡 §0.1 必须写明 `g_desc = −g_task`
    ok = "g_desc" in card and "−" in card and "cos(g_desc" in card.replace(" ", "")
    print(f"[5] 卡 §0.1 含下降方向符号约定 ⇒ {ok}")
    if not ok:
        fails.append("卡 §0.1 符号约定缺失")

    # ── ⑥ 判决逻辑只有一份：`merge` 必须调用 `decide()`
    ok = "decide(" in re.search(r"def merge\(\):(.*?)\nif __name__", src, re.S).group(1)
    print(f"[6] merge() 调用 decide() ⇒ {ok}")
    if not ok:
        fails.append("merge 未走 decide()（判定逻辑可能分叉）")

    print("=" * 60)
    if fails:
        print("★ verify_card43 FAIL：")
        for f in fails:
            print("   -", f)
        return 1
    print("★ verify_card43 PASS（阈值/窗口/种子段/断言表/符号约定/单一判定逻辑 全一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
