#!/usr/bin/env python3
"""`P118-3` + `P118-6` 的自测。

`P118-3`（`DEF-118-11`）：`audit_counts.derive_count` 能否**正确表达撤回**，且**仍保留**原 `max()` 语义。
`P118-6`：分母（方向总数）能否**从归档机械数出**，且**分歧不会被静默抹平**。

★ 纪律 ㊲（`DEF-118-17`）：自测必须能**断言红** —— 不许只证明"能跑通"。
   故本文件另配 `--inject N` 开关，故意破坏第 N 条断言，用来**当场验证这份测试本身会红**；
   且**判红时同时打印「退出码 + [FAIL] 行数」**，因为崩溃与断言失败都以退出码 1 收场，
   两者只能靠 `[FAIL]` 行数区分（`DEF-118-17` 的直接教训）。

★★ `P118-6` 的设计要点：`candidate_denominators()` 把**所有**口径一次算出（分歧现形），
   `derive_denominator()` **没有默认口径**（默认值会让分歧重新隐身）。
   本题把这两条都当成**被测性质**，而不是实现细节。

用法：
    python3 test_audit_counts.py            # 正常跑，应全部 PASS、exit 0
    python3 test_audit_counts.py --inject 3 # 注入第 3 条，应 1 条 FAIL、exit 1
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import audit_counts as AC   # noqa: E402

FAILS = []


def chk(tag, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: 实测={got!r} 期望={want!r}")
    if not ok:
        FAILS.append(tag)


def chk_raises(tag, fn, want_in_msg):
    """断言 `fn()` 抛 `CountError`，且消息里含指定子串（防"抛错但抛错了原因"）。"""
    try:
        got = fn()
    except AC.CountError as e:
        ok = want_in_msg in str(e)
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: 抛出 CountError（含 {want_in_msg!r}={ok}）"
              f"\n          消息：{e}")
        if not ok:
            FAILS.append(tag)
        return
    except Exception as e:                                    # noqa: BLE001
        print(f"  [FAIL] {tag}: 抛了**非** CountError 的异常 {type(e).__name__}: {e}")
        FAILS.append(tag)
        return
    print(f"  [FAIL] {tag}: 本应抛 CountError，却正常返回 {got!r}")
    FAILS.append(tag)


# ---- 合成证据表：只保留与计数有关的几行，模拟真实表的形状 -------------------------
HISTORY = """\
| 轮次 | 计数 |
| 某年某月 | 当前计数：SUCCEEDED = 49 / 49 |
| 某年某月 | 当前计数：SUCCEEDED = 53 / 55 |
| 某年某月 | 当前计数：SUCCEEDED = 54 / 55 |
| 某年某月 | 当前计数：SUCCEEDED = 55 / 56 |
"""

# ---- P118-6 合成归档：4 个方向目录，故意让五个口径给出**三个不同**的数 -------------
ENTRIES = [
    {"name": "DIR-01", "has_direction": True,  "has_verdict": True},
    {"name": "DIR-02", "has_direction": True,  "has_verdict": True},
    {"name": "DIR-03", "has_direction": False, "has_verdict": True},
    {"name": "DIR-04", "has_direction": True,  "has_verdict": False},
]
REG = """\
# 登记表
| ID | 方向 |
|---|---|
| DIR-01 | 甲 |
| DIR-02 | 乙 |
正文里又提到 DIR-07（没有表格行、也没有目录）
"""


# ---- P118-7 合成证据表：两条 +1（其中一条取不到卡片）+ 一条「不增」+ 一条早期自由文本 ----
CODOMAIN_TB = """\
| **D-101（`DIR-101`）** | 某个问题 |
| **`SUCCEEDED`** | **`SUCCEEDED = 51 / 55`（+1）** —— 证实 |
| **D-102（`DIR-102`）** | 另一个问题 |
| **`SUCCEEDED`** | **`SUCCEEDED = 51 / 55`（不增）** —— 否证 |
| **D-103（`DIR-103`）** | 又一个问题 |
| **`SUCCEEDED`** | **`SUCCEEDED = 52 / 55`（+1）** —— 证实 |
| **D-104（DIR-104）** | ★ 早期行：只有自由文本，没有标准字段 |
| **判定** | **CONFIRMED**（旧格式） |
"""

#: 共域测试用「哪些方向有冻结卡片」
CODOMAIN_CARDS = {"101": True, "102": True, "103": False, "104": True}

# ---- P119-1 合成证据表：*真实表的形状* —— 被撤回的行在**上方**，撤回登记在**下方** -------------
# `D-149` 提 55 → `CORR-31` 撤回（登记区块在表格下方）→ `D-150` **又重新挣回** 55；
# 最后还有一个「共域」区引用了计数器起源 `SUCCEEDED = 32 / 32`（历史数字，不是状态转移）。
# 旧算法（`max−每条撤回`）在这里只能得 54；直接赋值的重放会得 32。两者都是错的。
ANCHORED_TB = """\
| **D-148（`DIR-117`）** | 甲 |
| **`SUCCEEDED`** | **`SUCCEEDED = 54 / 55`（+1）** |
| **D-149（`DIR-118`）** | 乙 |
| **`SUCCEEDED`** | **`SUCCEEDED = 55 / 56`（+1）** |
| **`CORR-31`（勘误 · 针对** `D-149`） | 本卡不计入成功计数（真实计数 54 / 56） |
| **D-150（`DIR-119`）** | 丙 |
| **`SUCCEEDED`** | **`SUCCEEDED = 55 / 56`（+1）** |

## 撤回登记（withdrawal register）

WITHDRAWN: D-149 55 -> 54

## 共域（co-domain）状态

1. `DIRECTION-REGISTER.md:5` 写明计数器起源 **`SUCCEEDED = 32 / 32`**。
"""

#: (tag, callable, 期望值)
TESTS = [
    # ── P118-3：撤回 ────────────────────────────────────────────────────────
    ("① 无撤回标记 ⇒ 仍取 max = 55（★ 证明原语义未被改动）",
     lambda: AC.derive_count(HISTORY), 55),

    ("② 任务指定的合成例：先 55 再撤回 54 ⇒ 推导值 = 54",
     lambda: AC.derive_count(HISTORY + "WITHDRAWN: D-149 55 -> 54\n"), 54),

    ("③ 撤回标记写成行内（不在行首）⇒ 不算标记，仍 55",
     lambda: AC.derive_count(HISTORY + "| 说明 | WITHDRAWN: D-149 55 -> 54 |\n"), 55),

    ("④ 多处撤回各自 −1 ⇒ 55 → 54 → 53",
     lambda: AC.derive_count(HISTORY
                             + "WITHDRAWN: D-149 55 -> 54\n"
                             + "WITHDRAWN: D-150 54 -> 53\n"), 53),

    ("⑤ 空表 / 无 SUCCEEDED 提及 ⇒ None（不返回 0）",
     lambda: AC.derive_count(""), None),

    ("⑥ 无关分数（如 2/8 seed）不得被当成计数",
     lambda: AC.derive_count("比例 2/8 seed 而已，没有计数提及\n"), None),

    # ── P118-6：分母 ────────────────────────────────────────────────────────
    ("⑦ 五个口径**一次全算出**，分歧现形（4 目录 / 3 有卡 / 3 有裁决 / 2 表格行 / 3 基号）",
     lambda: AC.candidate_denominators(REG, ENTRIES),
     {"revolve_dirs": 4, "revolve_direction": 3, "revolve_verdict": 3,
      "register_table": 2, "register_basenums": 3}),

    ("⑧ 具名口径推导：revolve_dirs ⇒ 4",
     lambda: AC.derive_denominator(REG, ENTRIES, "revolve_dirs"), 4),

    ("⑨ 具名口径推导：register_basenums ⇒ 3（DIR-07 只在正文出现也算登记）",
     lambda: AC.derive_denominator(REG, ENTRIES, "register_basenums"), 3),

    ("⑩ 无 `DENOM:` 声明 ⇒ None（未声明状态必须可表达，不得假装已声明）",
     lambda: AC.parse_denom_declaration(HISTORY), None),

    ("⑪ 有声明 ⇒ (口径, 数值)",
     lambda: AC.parse_denom_declaration(HISTORY + "DENOM: source=revolve_dirs 4\n"),
     ("revolve_dirs", 4)),

    ("⑫ ★ 声明值与口径重算不符 ⇒ 判红（这正是 audit ④c 的极性；此处用真实世界的 56 vs 4）",
     lambda: AC.parse_denom_declaration("DENOM: source=revolve_dirs 56\n")[1]
             != AC.derive_denominator(REG, ENTRIES, "revolve_dirs"), True),

    # ── P118-7：共域（co-domain）────────────────────────────────────────────
    ("⑯ `evidence_rows` 切出 4 行、基号正确（★ 含早期无标准字段的行）",
     lambda: [n for _d, n, _b in AC.evidence_rows(CODOMAIN_TB)],
     ["101", "102", "103", "104"]),

    ("⑰ `counted_directions` 只认 `（+1）`：判 +1 = D-101/D-103，判不出 2 行",
     lambda: ([d for d, _n in AC.counted_directions(CODOMAIN_TB)[0]],
              len(AC.counted_directions(CODOMAIN_TB)[1])), (["D-101", "D-103"], 2)),

    ("⑱ `codomain_violations` 抓到「声明 +1 但无卡片」的 D-103（★ 这就是分子∉分母）",
     lambda: AC.codomain_violations(CODOMAIN_TB, CODOMAIN_CARDS.get), [("D-103", "103")]),

    ("⑲ 无违规时返回空表（★ 但空**不等于**共域成立：判不出的行仍须报告）",
     lambda: AC.codomain_violations(CODOMAIN_TB, lambda n: True), []),

    ("⑳ `PARALLEL_REPORT_KEYS` 两个键存在且互不相同（★ 防「把键表掏空」绕过 ④d）",
     lambda: (len(AC.PARALLEL_REPORT_KEYS), len(set(AC.PARALLEL_REPORT_KEYS))), (2, 2)),

    # ── P119-1：撤回后**重新挣回同一数值**（DEF-119-06）───────────────────
    ("㉑ 撤回后又挣回同一个数（55→撤回→54→再提 55）⇒ 55"
     "（★ 旧算法给 54；这条是 DEF-119-06 的直接反例）",
     lambda: AC.derive_count(HISTORY + "WITHDRAWN: D-149 55 -> 54\n"
                             + "| 后来 | 当前计数：SUCCEEDED = 55 / 56 |\n"), 55),

    ("㉒ ★ 撤回锚定到**它所撤回的行**（登记区块在表格下方、`D-150` 在上方又挣回）⇒ 55"
     "（★ 按物理位置重放会给 54；按「赋值」重放会被末尾 32/32 引用打成 32）",
     lambda: AC.derive_count(ANCHORED_TB), 55),

    ("㉓ 引用一个**旧的**计数（55 之后又出现 `SUCCEEDED = 32 / 32`）⇒ 仍 55"
     "（★ 否则引旧数就能绕过 `WITHDRAWN` 记账）",
     lambda: AC.derive_count(HISTORY + "| 引用 | 起源 SUCCEEDED = 32 / 32 |\n"), 55),
]


#: (tag, callable, 期望的错误消息子串)
NEG = [
    ("⑬ `from` 接不上账（标记称 57 撤回，而推导值是 55）⇒ 必须抛错",
     lambda: AC.derive_count(HISTORY + "WITHDRAWN: D-149 57 -> 56\n"), "合不上账"),

    ("⑭ 步长不是 −1（55 -> 53）⇒ 必须抛错",
     lambda: AC.derive_count(HISTORY + "WITHDRAWN: D-149 55 -> 53\n"), "恰好 −1"),

    ("⑮ 未知分母口径名 ⇒ 必须抛错（★ 证明**没有**默认口径：默认值会让分歧重新隐身）",
     lambda: AC.derive_denominator(REG, ENTRIES, "nope"), "未知的分母口径"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", type=int, default=0,
                    help="故意破坏第 N 条断言（1-based，跨正向与负向），用于证明本测试能红")
    a = ap.parse_args()

    # ★ 注入：把指定条目的**期望值**改坏 —— 若测试真在工作，就必须报 FAIL。
    global TESTS, NEG
    if a.inject:
        n = a.inject
        total = len(TESTS) + len(NEG)
        if not (1 <= n <= total):
            print(f"✗ --inject {n} 超出范围 1..{total}")
            return 2
        if n <= len(TESTS):
            k = n - 1
            tag, fn, _w = TESTS[k]
            TESTS[k] = (tag + " ［已注入故障］", fn, "__INJECTED__")
        else:
            k = n - len(TESTS) - 1
            tag, fn, _w = NEG[k]
            NEG[k] = (tag + " ［已注入故障：期望的错误子串改为不存在串］", fn,
                      "絶対に存在しない文字列")

    print("=" * 80)
    print("audit_counts 自测（P118-3 撤回 · P118-6 分母 · P118-7 共域 · P119-1 重放）")
    print("=" * 80)
    for tag, fn, want in TESTS:
        try:
            chk(tag, fn(), want)
        except Exception as e:                                # noqa: BLE001
            print(f"  [FAIL] {tag}: 意外异常 {type(e).__name__}: {e}")
            FAILS.append(tag)
    for tag, fn, want in NEG:
        chk_raises(tag, fn, want)

    print("-" * 80)
    # ★ DEF-118-18（Cycle 225 抓到）：这行**不得**以 `[FAIL]` 开头。原写法 `print(f"[FAIL] 行数 = …")`
    #   会让 `grep -c "\[FAIL\]"` 在**全绿**时也数到 1 ⇒ 「断言红 vs 崩溃红」的判别器**自我污染**
    #   （正是 `DEF-118-17` 的教训在**新代码里复发**）。改用一个不会被误计的标签。
    print(f"断言失败条数 = {len(FAILS)}")
    print("★ 全部 PASS" if not FAILS else f"★ {len(FAILS)} 项未通过：" + str(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())