#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""`P118-5` 的自测：`n_hit` / `S_i` / `E_random` / oracle 的语义是否如卡片 §4 手算所写。

★★ 纪律 ㊲（`DEF-118-17`）：自测必须能**断言红**，不许只证明"能跑通"。
   故配 `--inject N`，故意破坏第 N 条断言，**当场验证这份测试本身会红**。
★★ `DEF-118-18`：收尾行**不得**以 `[FAIL]` 开头 —— 否则 `grep -c '[FAIL]'` 在全绿时也会数到 1，
   把"断言红 vs 崩溃红"的判别器**自我污染**。判红时同时打印「退出码 + `[FAIL]` 行数」。

★★ 本文件的两条自我更正（`RULER-P118-5-REV1.1.md`）：
   `DEF-118-20`：卡片 §4 的手算 `E_random` 写错了。`Σ_v m_v²` 是**按不同值**求和，
      本 toy 的值重数只有 `m_{v0}=2, m_{v1}=1` ⇒ `Σm² = 4+1 = 5`、`E_random = 5/3 ≈ 1.667`，
      **不是**我原先写的 `(2²+1²+2²)/3 = 3.0`（那是把**每项的 |S|** 当成**每值的重数**了）。
      等价式 `Σ_i |S_i| = Σ_v m_v²` 给出同一个 5 ⇒ 装置口径 `64/32 = 2.0` **不受影响**。
   `DEF-118-21`：原 toy 取 `d_k = 2 < n = 3` ⇒ `pinv(Kᵀ)Kᵀ ≠ I`，`W_or Kᵀ = Vᵀ` **原理上不成立**，
      实测残差 `0.3125`（不是数值噪声）⇒ oracle 阳性对照在这种维度下**无意义**。
      ⇒ **`G5-3`（残差 <1e-6）同时是一条结构前提检查：`d_k ≥ n`。** 本 toy 改取 `d_k = n = 3`；
      真装置 `d_k = 896 ≥ 32` ✓。
   `DEF-118-22`：原自测把 toy 张量放在 CPU，而 `n_hit` 内部硬编码 `device=DEV`(cuda) ⇒ 5 条假"意外异常"。
      修法：toy 一律建在 `D98.DEV` 上（★ 崩溃与断言失败都退出码 1，靠这道修才看清真正的错因）。

用法（在 trainer 上）：
    /opt/pytorch/bin/python test_ruler118.py            # 应全 PASS、rc=0
    /opt/pytorch/bin/python test_ruler118.py --inject 3 # 第 3 条应 FAIL、rc=1
"""
import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
for _d in ("dir118", "dir98", "dir97", "dir91", "dir90"):
    sys.path.insert(0, str(_ROOT / _d))

import torch                       # noqa: E402
import R1_task98 as D98            # noqa: E402   ← 与主脚本复用**同一个** n_hit

n_hit = D98.n_hit
FAILS = []
DEV = D98.DEV                      # ★ DEF-118-22：n_hit 内部用 DEV，toy 必须同设备
DT = torch.float64                 # toy 用 float64 ⇒ 残差可判到 1e-9，不受 float32 噪声干扰

# ---------------------------------------------------------------- 卡片 §4 的手算例（rev 1.1）
# 值：v0, v1, v0 ⇒ 重数 m_{v0}=2, m_{v1}=1 ⇒ Σm² = 4+1 = 5、E_random = 5/3
# ★ d_k = n = 3（`DEF-118-21`：d_k < n 时 oracle 原理上不可精确）
V = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]], device=DEV, dtype=DT)
K = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0.707, 0.707, 1.]], device=DEV, dtype=DT)
vals = [tuple(round(x, 6) for x in row) for row in V.tolist()]
SS = [[j for j in range(3) if vals[j] == vals[i]] for i in range(3)]      # [[0,2],[1],[0,2]]
W_I = torch.eye(3, device=DEV, dtype=DT)
W_0 = torch.zeros(3, 3, device=DEV, dtype=DT)


def erandom(ss):
    """`E_random = Σ_v m_v² / N`。恒等式 `Σ_i |S_i| = Σ_v m_v·m_v` ⇒ 可用 S 大小求和实现。"""
    return sum(len(s) for s in ss) / len(ss)


def oracle_resid(Vm, Km):
    Wo = Vm.T @ torch.linalg.pinv(Km.T)
    return float((Wo @ Km.T - Vm.T).norm() / Vm.T.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", type=int, default=0, help="故意破坏第 N 条断言（1-based）")
    a = ap.parse_args()

    cases = [
        ("① S 集合构造：v0 出现两次 ⇒ S = [[0,2],[1],[0,2]]", lambda: SS, [[0, 2], [1], [0, 2]]),
        ("② E_random = Σm²/N = (2²+1²)/3 = 5/3（★ 按不同值求和，不是按项）",
         lambda: round(erandom(SS), 9), round(5 / 3, 9)),
        ("③ 恒等式 Σ_i|S_i| == Σ_v m_v²（5 == 4+1）",
         lambda: sum(len(s) for s in SS), 5),
        ("④ W=I ⇒ 值一致命中 3/3",
         lambda: n_hit(W_I, K, V, SS)[0], 3),
        ("⑤ W=I ⇒ 严格下标只有 2/3（i=2 的 argmax 是 0 不是 2）",
         lambda: n_hit(W_I, K, V, SS)[1], 2),
        ("⑥ W=0 ⇒ cos 全并列、argmax 恒 0 ⇒ 值一致只得 2/3（常数 argmax 天花板）",
         lambda: n_hit(W_0, K, V, SS)[0], 2),
        ("⑦ W=0 ⇒ 严格下标 1/3（i=0 恰好对 ⇒ 证明 ⑥ 的 2 分是位置巧合、不是能力）",
         lambda: n_hit(W_0, K, V, SS)[1], 1),
        ("⑧ oracle = V K⁺ ⇒ 值一致 3/3",
         lambda: n_hit(V.T @ torch.linalg.pinv(K.T), K, V, SS)[0], 3),
        ("⑨ d_k = n ⇒ oracle 残差 <1e-9（G5-3 的结构前提成立）",
         lambda: oracle_resid(V, K) < 1e-9, True),
        ("⑩ d_k < n ⇒ oracle 残差**必然**很大（证明 G5-3 是结构前提检查，非纯数值容差）",
         lambda: oracle_resid(V[:, :2].contiguous(), K[:, :2].contiguous()) > 0.1, True),
    ]

    if a.inject:
        n = a.inject
        if not (1 <= n <= len(cases)):
            print(f"✗ --inject {n} 超出范围 1..{len(cases)}")
            return 2
        tag, fn, _w = cases[n - 1]
        cases[n - 1] = (tag + " ［已注入故障］", fn, "__INJECTED__")

    print("=" * 84)
    print("P118-5 自测：n_hit / S 集合 / E_random / oracle 结构前提（卡片 rev 1.1）")
    print("=" * 84)
    for tag, fn, want in cases:
        try:
            chk_got = fn()
            ok = chk_got == want
            print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: 实测={chk_got!r} 期望={want!r}")
            if not ok:
                FAILS.append(tag)
        except Exception as e:                                # noqa: BLE001
            print(f"  [FAIL] {tag}: 意外异常 {type(e).__name__}: {e}")
            FAILS.append(tag)

    print("-" * 84)
    # ★ 不得以 `[FAIL]` 开头（DEF-118-18）
    print(f"断言失败条数 = {len(FAILS)}")
    print("★ 全部 PASS" if not FAILS else f"★ {len(FAILS)} 项未通过：" + str(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())