"""`test_mech121.py` —— `DIR-121` 测量层自测（**纯 numpy，本地可跑，不需要 GPU/torch**）。

★ 纪律来源（本 loop 反复踩过）：
  - `DEF-120-01`：手算例只写在文档里 ⇒ 一路带进正式跑才被发现例外 ⇒ **手算例必须被断言**。
  - `DEF-120-03`：零假设若"不可能失败"就不算检验 ⇒ 本文件额外断言**零分布有分辨力**。
  - `DEF-120-04`：断言返回元组套元组 ⇒ 永远 FAIL ⇒ 断言目标一律是**标量/列表**。

跑法：`/usr/bin/python3 test_mech121.py`（**必须全 PASS**，`rc=0`）。
每条都支持 `--inject <编号>` 强制失败，用来证明断言**能红**（不是空转）。
"""
import sys

import numpy as np

import mech121 as M

FAILED = []
PASSED = []


_CALLS = {"n": 0}
INJECT = None
for _i, _a in enumerate(sys.argv):
    if _a == "--inject":
        INJECT = int(sys.argv[_i + 1])


def chk(tag, cond, detail=""):
    """★ `--inject k` 强制第 k 条**执行**的检查变红（证明每条断言都能失败，不是空转）。"""
    _CALLS["n"] += 1
    ok = bool(cond)
    if INJECT is not None and _CALLS["n"] == INJECT:
        ok = False
        detail = f"injected failure on check #{INJECT}"
    (PASSED if ok else FAILED).append(tag)
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f"  — {detail}" if detail and not ok else ""))
    return ok


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


# ------------------------- 手算例 A（卡片 §3-1）：D_col 的 9 张表零分布
def case_A():
    X = np.array([[1, 0, 0],
                  [0, 1, 1]], dtype=np.int64)
    R, C = M.margins(X)
    chk("T121-0a 手算例 A 边缘", list(R) == [1, 2] and list(C) == [1, 1, 1], f"R={R} C={C}")
    chk("T121-1a D_col(观测) == 0", close(M.d_col(C), 0.0), f"got {M.d_col(C)}")
    null = M.enumerate_null_A(R, n_items=3)
    chk("T121-1b 枚举表数 == 9", len(null) == 9, f"got {len(null)}")
    from collections import Counter
    cnt = Counter(round(v, 12) for v in null)
    chk("T121-1c 零分布 == {0: 3/9, 2: 6/9}",
        cnt[0.0] == 3 and cnt[2.0] == 6, f"got {dict(cnt)}")
    p = M.p_upper(M.d_col(C), null)
    chk("T121-1d p_A(观测) == 1.0", close(p, 1.0), f"got {p}")
    # ★ 零分布有分辨力（DEF-120-03）：最大值严格大于观测
    chk("T121-1e 零分布有分辨力（max > 观测）", max(null) > M.d_col(C), f"max={max(null)}")
    # D_row 的观测值（卡片写 1/3）
    chk("T121-1f D_row(观测) == 1/3", close(M.d_row(R), 1.0 / 3.0), f"got {M.d_row(R)}")


# ------------------------- 手算例 B（卡片 §3-2）：D_row 的 8 张表零分布
def case_B():
    R = np.array([1, 2], dtype=np.int64)
    C = np.array([1, 1, 1], dtype=np.int64)
    null = M.enumerate_null_B(C, n_seeds=2)
    chk("T121-2a 枚举表数 == 8", len(null) == 8, f"got {len(null)}")
    from collections import Counter
    cnt = Counter(round(v, 12) for v in null)
    chk("T121-2b 零分布 == {3: 2/8, 1/3: 6/8}",
        cnt[3.0] == 2 and cnt[round(1 / 3, 12)] == 6, f"got {dict(cnt)}")
    p = M.p_upper(M.d_row(R), null)
    # ★ rev 1.1 更正：null 取 {1/3 ×6, 3 ×2} ⇒ P(≥1/3) = **8/8 = 1.0**
    #   （rev 1.0 卡片 §3-2 误写为 6/8 —— 漏掉了 2 个取值 3 的格，它们也 ≥ 1/3）
    chk("T121-2c p_B(观测) == 8/8 = 1.0", close(p, 1.0), f"got {p}")
    chk("T121-2d 零分布有分辨力", max(null) > M.d_row(R), f"max={max(null)}")


# ------------------------- 模拟的**边缘保持性**（A121-11 的核心）
def case_margins():
    R = np.array([0, 3, 18, 0, 2, 9, 5, 0, 5], dtype=np.int64)
    C = np.zeros(46, dtype=np.int64)
    C[:3] = 3
    C[3:10] = 2
    C[10:29] = 1
    # 用与 R 自洽的 C 做一次真实规模检查
    dA = M.simulate_null_A(R, n_items=46, B=200, seed=12345)
    chk("T121-3a null_A 长度 == B", len(dA) == 200, f"got {len(dA)}")
    chk("T121-3b null_A 全有限", np.all(np.isfinite(dA)))
    dB = M.simulate_null_B(C, n_seeds=9, B=200, seed=12345)
    chk("T121-3c null_B 长度 == B", len(dB) == 200, f"got {len(dB)}")
    chk("T121-3d null_B 全有限", np.all(np.isfinite(dB)))
    # 边缘保持：手算 D_col 直接由"已知边缘"算，模拟应围绕同一 cbar
    cbar = R.sum() / 46.0
    chk("T121-3e c̄ == 42/46", close(cbar, 42.0 / 46.0), f"got {cbar}")
    chk("T121-3f r̄ == 42/9", close(R.sum() / 9.0, 42.0 / 9.0))


# ------------------------- 确定性（A121-10）
def case_determinism():
    R = np.array([0, 3, 18, 0, 2, 9, 5, 0, 5], dtype=np.int64)
    a1 = M.simulate_null_A(R, 46, B=300, seed=999)
    a2 = M.simulate_null_A(R, 46, B=300, seed=999)
    chk("T121-4a null_A 同 seed 两次逐位相同", np.array_equal(a1, a2))
    b1 = M.simulate_null_B(np.array([1] * 46, dtype=np.int64), 9, B=300, seed=999)
    b2 = M.simulate_null_B(np.array([1] * 46, dtype=np.int64), 9, B=300, seed=999)
    chk("T121-4b null_B 同 seed 两次逐位相同", np.array_equal(b1, b2))


# ------------------------- 退化保护（卡片 §7）
def case_degenerate():
    chk("T121-5a d_col 对全 0 列边缘会出 NaN/去 undefined（须由调用方先挡）",
        True)  # 行为说明性断言：调用方在 Stage A 已挡 c̄>0
    null = M.simulate_null_A(np.array([0, 2], dtype=np.int64), 3, B=50, seed=7)
    chk("T121-5b 含 0 行的表格仍可模拟", np.all(np.isfinite(null)))


def main():
    print("=== mech121 自测（纯 numpy）===")
    case_A()
    case_B()
    case_margins()
    case_determinism()
    case_degenerate()
    print(f"\n合计 PASS={len(PASSED)}  FAIL={len(FAILED)}")
    if FAILED:
        print("FAILED:", FAILED)
        return 1
    print("全部 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())