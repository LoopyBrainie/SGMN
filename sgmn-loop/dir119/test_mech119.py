"""`test_mech119.py` —— `DIR-119` 测量层的**本地自测**（只用 numpy，无 torch/GPU）。

★ 存在理由（两条纪律的合并）：
  1. round standard 要求「测量对象须对一个**具体手算例**验证」——手算例写在卡片 §3；
  2. handoff 纪律「**手算例必须写成可断言的自测 —— 手算例本身就是被测对象**」。
★ 本文件把卡片 §3 的 8 个冻结读数逐位断言，外加 **两个构造性判据守卫**
  （`A119-10` M2 门会跳也会落、`A119-11` M1 门会跳也会落、`A119-6` 恒等式）。
★ `--inject N` 故意破坏第 N 条断言 ⇒ 用于证明**本测试真的会红**（`DEF-118-05` 教训）。
用法：
    /usr/bin/python3 dir119/test_mech119.py            # 全跑
    /usr/bin/python3 dir119/test_mech119.py --inject 3 # 第 3 条必须变红
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np            # noqa: E402
import mech119 as M           # noqa: E402

FAILS = []


def chk(tag, got, want, tol=1e-12):
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = (got == want)
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: 实测={got} 期望={want}")
    if not ok:
        FAILS.append(tag)


def case():
    """卡片 §3 的手算例：`tv = {x,z,q,r}`，2 项，`N_CAND=4`。"""
    tv = {"x", "z", "q", "r"}
    S = np.array([[1.5, 0.9, 2.0, 0.4],
                  [0.1, 3.0, 2.5, 1.8]], dtype=np.float64)
    meta = [{"cand_val": ["x", "y", "z", "w"]},
            {"cand_val": ["p", "q", "r", "s"]}]
    tgt = np.array([1, 0], dtype=np.int64)
    return S, tgt, meta, tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", type=int, default=0)
    a = ap.parse_args()
    inject = a.inject
    if inject and not (1 <= inject <= 17):
        # ★ 注入越界必须报错，不能"什么都不坏"⇒ 静默全绿（DEF-118-18 的同类陷阱）
        print(f"✗ --inject {inject} 超出范围 1..17")
        return 2

    def maybe_inject(n, want):
        """第 n 条被注入时，把期望值改坏 ⇒ 必须报 FAIL。"""
        return "__INJECTED__" if inject == n else want

    print("=" * 78)
    print("mech119 自测（DIR-119 §3 手算例 + 构造性判据守卫）")
    print("=" * 78)

    S, tgt, meta, tv = case()
    g = M.score_and_mech(S, tgt, meta, tv)

    # ── 卡片 §3 的 8 个冻结读数（逐位）────────────────────────────────────
    chk("① acc_argmin = 1/2", g["acc_argmin"], maybe_inject(1, 0.5))
    chk("② acc_argmax = 0/2", g["acc_argmax"], maybe_inject(2, 0.0))
    chk("③ rank_hist = [1,1,0,0]", g["rank_hist"], maybe_inject(3, [1, 1, 0, 0]))
    chk("④ n_err = 1", g["n_err"], maybe_inject(4, 1))
    chk("⑤ err_in_train_frac = 0/1", g["err_in_train_frac"],
        maybe_inject(5, 0.0))
    chk("⑥ err_in_train_null = (2/3)/1", g["err_in_train_null"],
        maybe_inject(6, 2.0 / 3.0))
    chk("⑦ n_seen_distr_mean = 2.0", g["n_seen_distr_mean"], maybe_inject(7, 2.0))
    gap = g["err_in_train_frac"] - g["err_in_train_null"]
    chk("⑧ GAP = −2/3", gap, maybe_inject(8, -2.0 / 3.0))

    # ── A119-6 恒等式：acc_argmax ≡ rank_hist[3]/n（随机 300 例）──────────
    rng = np.random.default_rng(4242)
    bad = 0
    n = len(tgt)
    for _ in range(300):
        Sr = rng.normal(size=(n, M.N_CAND))
        gr = M.score_and_mech(Sr, tgt, meta, tv)
        if abs(gr["acc_argmax"] - gr["rank_hist"][3] / float(n)) > 1e-15:
            bad += 1
    chk("⑨ 恒等式 acc_argmax ≡ rank_hist[3]/n（随机 300 例）", bad,
        maybe_inject(9, 0))

    # ── A119-10 M2 门：正确项 rank1 ⇒ 0.0（不触发）；rank4 ⇒ 1.0（触发）───
    rows = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (8, 1))
    t_lo = np.zeros(8, dtype=np.int64)          # 正确 = argmin ⇒ rank 1
    t_hi = np.full(8, 3, dtype=np.int64)        # 正确 = argmax ⇒ rank 4
    lo = M.score_and_mech(rows, t_lo, meta[:2] * 4, tv)
    hi = M.score_and_mech(rows, t_hi, meta[:2] * 4, tv)
    chk("⑩ M2 门：rank1 ⇒ acc_argmax=0.0（不触发）", lo["acc_argmax"],
        maybe_inject(10, 0.0))
    chk("⑪ M2 门：rank4 ⇒ acc_argmax=1.0（触发）", hi["acc_argmax"],
        maybe_inject(11, 1.0))

    # ── A119-11 M1 门：全选见过 ⇒ GAP=1/3（触发）；2/3 落见过 ⇒ GAP=0（不触发）
    tv_s = {"x", "z"}
    meta_s = [{"cand_val": ["x", "y", "z", "w"]} for _ in range(12)]
    t_s = np.ones(12, dtype=np.int64)            # 正确 = "y"（未见值）⇒ 题题都错
    S_hit = np.tile(np.array([0.0, 1.0, 9.0, 2.0]), (12, 1))   # argmin="x"（见过）
    gh = M.score_and_mech(S_hit, t_s, meta_s, tv_s)
    chk("⑫ M1 门：全落见过 ⇒ err_in_train_frac=1.0",
        gh["err_in_train_frac"], maybe_inject(12, 1.0))
    chk("⑬ M1 门：零假设 = 2/3", gh["err_in_train_null"],
        maybe_inject(13, 2.0 / 3.0))
    gap_hit = gh["err_in_train_frac"] - gh["err_in_train_null"]
    chk("⑭ M1 门：GAP=1/3 ≥ 阈值 0.20 ⇒ 触发", gap_hit >= M1_OK,
        maybe_inject(14, True))
    S_mix = np.empty((12, 4))
    for i in range(12):
        S_mix[i] = (np.array([0.0, 1.0, 9.0, 2.0]) if i % 3 != 2
                    else np.array([5.0, 3.0, 4.0, 1.0]))        # argmin="w"（未见）
    gm = M.score_and_mech(S_mix, t_s, meta_s, tv_s)
    gap_mix = gm["err_in_train_frac"] - gm["err_in_train_null"]
    chk("⑮ M1 门：2/3 落见过 ⇒ GAP=0 < 0.20 ⇒ 不触发", gap_mix < M1_OK,
        maybe_inject(15, True))

    # ── 边界：全错且零假设为 0 ⇒ frac 可为 0、null 为 0、GAP 为 0 ─────────
    meta0 = [{"cand_val": ["x", "y", "p", "r"]} for _ in range(4)]   # 干扰项全未见
    g0 = M.score_and_mech(np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (4, 1)),
                          np.ones(4, dtype=np.int64), meta0, {"z"})
    chk("⑯ 零假设=0 时 null=0.0 且 frac=0.0", (g0["err_in_train_null"],
                                               g0["err_in_train_frac"]),
        maybe_inject(16, (0.0, 0.0)))

    # ── 非有限值：必须 finite=False 且不产生读数 ──────────────────────────
    Sbad = np.array([[0.0, np.nan, 1.0, 2.0]], dtype=np.float64)
    gbad = M.score_and_mech(Sbad, np.array([1]), meta[:1], tv)
    chk("⑰ NaN ⇒ finite=False（删失，不得记 PASS/FAIL）",
        (gbad["finite"], "acc_argmin" in gbad), maybe_inject(17, (False, False)))

    print("-" * 78)
    print(f"断言失败条数 = {len(FAILS)}")
    print("★ 全部 PASS" if not FAILS else f"★ {len(FAILS)} 项未通过：{FAILS}")
    return 0 if not FAILS else 1


#: 与卡片冻结的 `M1_GAP` 一致性由 `R1_task119` 侧的常量核对；此处只做门限行为自测
M1_OK = 0.20

if __name__ == "__main__":
    sys.exit(main())