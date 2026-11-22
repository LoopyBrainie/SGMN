#!/opt/pytorch/bin/python
r"""DIR-52 · **定论流水线端到端演练**（`rehearse84.py`，CPU-free、秒级）。

动机（Cycle 84）：`VERDICT-SCAFFOLD.md §8` 列了 7 步（就绪门 → 五支自测 → 裁决器 → 机制工具），
但这些步骤**从未按真实顺序串起来跑过** —— 每一步都单独验过（Cycle 74/79/80/81/82），
而"串起来"才会暴露接口错配：产物路径不一致、`cwd` 假设不同、后一步读前一步的哪个键、
某个自测悄悄改了 `cwd` 下的临时文件……这类错在**定论当天**才第一次出现最贵。

做法：在**临时根目录 T** 造 `T/dir52/out/DIR52r12_shard{0..3}.jsonl`（4×50 = 冻结种子段
`62000..62199` 全 200 个，与真实落盘同构且**自洽**），然后：
  ① `ready52.py`（cwd=T）—— 必须打印 **READY**（无存活进程、无损坏、无缺失）
  ② 五支自测（cwd=仓库根）—— 必须全 0 退出
  ③ `adjudicate52.py` —— 模板 §3 的 18 条路径必须全部可解析
  ④ `mechanism52.py` —— §5 需要的键必须齐，且 `|mean(SHAPE) − (shape_F + shape_K)| ≤ 1e-12`
★ 全程 `REHEARSAL-ONLY`：**绝不读写真实 `dir52/out/`**（真实目录照原样保留）。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # 仓库根
LO, HI = 62000, 62199

SELFTESTS = [
    ("selftest_adjudicate52.py", "六支判定表 6/6"),
    ("selftest_missing_rows52.py", "缺失行 5/5 + 完整性 5/5"),
    ("selftest_paths52.py", "§3 路径表 12/12"),
    ("selftest_ready52.py", "就绪门六态 6/6"),
    ("selftest_fields52.py", "§3 字段覆盖 18/18"),
]
PATHS = ["n_rows", "n_expected", "n_eff", "indeterminate_frac",
         "n_missing_vs_expected", "indet_vs_expected", "invalid_reasons",
         "SHAPE.mean", "SHAPE.sd", "SHAPE.se", "SHAPE.threshold",
         "SHAPE.n_pos", "SHAPE.n_neg", "EARLY.mean", "LATE.mean", "TOTAL.mean",
         "D67_cross_check", "decision.outcome"]
MECH_KEYS = ["n_rows", "n_used", "dF2", "dF4", "dK2", "dK4",
             "shape_F", "shape_K", "shape", "early", "late"]


def synth_row(seed, e2, e4):
    base = 5.0
    ph = {}
    for phi, k2, k4 in (("0.0", base, base), ("0.9", base + e2, base + e4)):
        ph[phi] = {"keep_n": {"K": {"2": 40, "4": 60}, "F": {"2": 40, "4": 60}},
                   "mu": {"K": {"0": base, "2": k2, "4": k4},
                          "F": {"0": base, "2": base, "4": base}},
                   "wsig": {"K": {"2": .01, "4": .01}, "F": {"2": .01, "4": .01}},
                   "wnorm": {"K": {"2": 1., "4": 1.}, "F": {"2": 1., "4": 1.}}}
    return {"seed": seed, "tag": "rehearsal", "phi": ph,
            "dG": {"0": 0.0, "2": e2, "4": e4},
            "EARLY": e2, "TOTAL": e4, "LATE": e4 - e2,
            "SHAPE": e2 - (e4 - e2), "identity_err": 0.0, "finite": True}


def _resolve(d, path):
    cur = d
    try:
        for k in path.split("."):
            cur = cur[k]
        return True
    except (KeyError, TypeError):
        return False


def run(step, cmd, cwd, expect_rc=0, must_print=None):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == expect_rc
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    if ok and must_print and must_print not in r.stdout:
        ok, tail = False, f"未打印 {must_print!r}；末行={tail}"
    print(f"  {'OK  ' if ok else 'FAIL'}  {step:34s} {dt:5.1f}s  {tail[:78]}")
    if not ok:
        for ln in (r.stdout[-600:] + r.stderr[-600:]).splitlines()[-8:]:
            print(f"        | {ln[:150]}")
    return ok


def main():
    root = tempfile.mkdtemp(prefix="rehearse84_")
    print("★ REHEARSAL-ONLY（合成数据，非证据；真实 dir52/out/ 未读写）\n")
    fails = []
    try:
        out = os.path.join(root, "dir52", "out")
        os.makedirs(out, exist_ok=True)
        seeds = list(range(LO, HI + 1))
        per = len(seeds) // 4
        for i in range(4):
            p = os.path.join(out, f"DIR52r12_shard{i}.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for s in seeds[i * per:(i + 1) * per if i < 3 else None]:
                    j = (s - LO) % 5
                    f.write(json.dumps(synth_row(s, 0.020 + .001 * j,
                                                 0.050 + .001 * j)) + "\n")
        shards = sorted(os.path.join(out, os.listdir(out)[k]) for k in range(4))

        # ① 就绪门（cwd = 临时根，故 SHARD_GLOB 指向合成分片）
        #   ★ `live_procs()` 是**全机扫描**（`ps -eo args` 里的 `R1_task52.py`），
        #     故在正式轮仍在跑时它必然报 NOT-READY —— 那是**保守的正确行为**，不是缺陷。
        #     本演练用其**文档化的测试缝** `READY52_PROCS=0` 隔离掉"机器上别处还有进程"，
        #     从而真正走一遍 **READY 成功路径**（否则该路径只有在无人跑实验时才可测）。
        env = dict(os.environ, READY52_PROCS="0")
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, "ready52.py")],
                           cwd=root, capture_output=True, text=True, env=env)
        ok = r.returncode == 0 and "READY" in r.stdout and "NOT-READY" not in r.stdout
        print(f"  {'OK  ' if ok else 'FAIL'}  {'① ready52.py（进程数注入 0）':34s} "
              f"{time.time()-t0:5.1f}s  {(r.stdout.strip().splitlines() or [''])[-1][:70]}")
        if not ok:
            fails.append("ready52")

        # ② 五支自测
        for fn, desc in SELFTESTS:
            if not run(f"② {desc}", [sys.executable, os.path.join(HERE, fn)], ROOT):
                fails.append(fn)

        # ③ 裁决器
        jout = os.path.join(root, "adj.json")
        cmd = [sys.executable, os.path.join(HERE, "adjudicate52.py"),
               "--shards", *shards, "--delta-star", "0.000430",
               "--power-gate", "0.000861", "--n-expected", "200",
               "--max-indet-frac", "0.20", "--min-n-eff", "160",
               "--out", jout, "--seed-segment", str(LO), str(HI)]
        if not run("③ adjudicate52.py", cmd, ROOT):
            fails.append("adjudicate52")
        else:
            d = json.load(open(jout, encoding="utf-8"))
            miss = [p for p in PATHS if not _resolve(d, p)]
            ok = not miss
            print(f"  {'OK  ' if ok else 'FAIL'}  {'③ §3 十八路径':34s}      "
                  f"{'18/18' if ok else f'缺 {miss}'}")
            if not ok:
                fails.append("paths18")

        # ④ 机制工具
        mout = os.path.join(root, "mech.json")
        if not run("④ mechanism52.py", [sys.executable, os.path.join(HERE,
                "mechanism52.py"), "--shards", *shards, "--out", mout], ROOT):
            fails.append("mechanism52")
        else:
            m = json.load(open(mout, encoding="utf-8"))
            miss = [k for k in MECH_KEYS if k not in m]
            # ★ `shape`/`shape_F`/`shape_K` 是 **agg 字典**（mean/sd/se/t），不是标量；
            #   残差**由工具自己给**：`decomp_err_max`（逐种子 max|F+K−S|）—— 不得自己把字典相加。
            err = m.get("decomp_err_max")
            ok = (not miss) and err is not None and err <= 1e-12
            print(f"  {'OK  ' if ok else 'FAIL'}  {'④ §5 键 + decomp_err_max':34s}      "
                  f"缺={miss or '无'} decomp_err_max={err:.2e}")
            if not ok:
                fails.append("mech_keys")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{'★ 端到端全过 —— 定论当天可按 §8 顺序一次跑通' if not fails else '★ 断点：' + str(fails)}")
    print("  （排练结局不作任何判据结论；真实数据仍未裁决）")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())