#!/opt/pytorch/bin/python
r"""DIR-52 · **就绪门**（`ready52.py`）—— 一条命令回答「正式轮跑完了没有，能不能裁决」。

动机（Cycle 81）：定论当天我是在「4 个分片仍在写」的时刻去跑裁决器的。原来的判据是
**肉眼数 `wc -l`**，它答不出三个真问题：
  ① 是不是有一行**正写到一半**（末行半截 ⇒ 该行会被裁决器跳过、静默变成一条"缺失"）；
  ② 四个分片合起来是不是**恰好覆盖冻结种子段**、有无重复或越界；
  ③ 进程是否**已经死了**（死了就不会再涨，继续等是白等）。

本脚本**只读**，不改任何文件、不产生任何判据读数。输出**单行结论**：
  `READY`             —— 200 行全部可解析、种子段完整无重复无越界、无存活进程
  `NOT-READY:<原因>`  —— 其余一切情形（原因用 `;` 分隔，逐条可读）
  `DONE-BUT-SHORT`    —— 无存活进程，但行数 < 200（进程已死；缺失**如实披露**，不是阻塞）

★ 退出码：0 = READY 或 DONE-BUT-SHORT（可以裁决）；1 = NOT-READY（继续等）。
"""
import glob
import json
import os
import subprocess
import sys

SEG_LO, SEG_HI = 62000, 62199          # 冻结种子段（卡 rev 1.0 起未改）
N_EXPECT = SEG_HI - SEG_LO + 1
SHARD_GLOB = "dir52/out/DIR52r12_shard*.jsonl"
PROC_PAT = "R1_task52.py"


def live_procs():
    """返回仍在跑的 `R1_task52.py` 进程数（只数解释器本体，避开 grep/sed 子壳）。

    ★ 测试缝：环境变量 `READY52_PROCS` 可注入进程数 —— 否则「READY」状态**永远无法自测**
      （真实机器上总有进程在跑时只能测 NOT-READY），而一个只被测过失败路径的门
      **不能**用来判断「可以裁决了吗」。该变量**只读**、不改变任何真实探测逻辑。
    """
    fake = os.environ.get("READY52_PROCS")
    if fake is not None:
        return int(fake)
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    except OSError:
        return -1
    return sum(1 for ln in out.splitlines()
               if PROC_PAT in ln and "python" in ln and "grep" not in ln and "sed" not in ln)


def main():
    reasons = []
    paths = sorted(glob.glob(SHARD_GLOB))
    nsh = len(paths)
    if nsh != 4:
        reasons.append(f"分片文件 {nsh} 个（期望 4）")

    seeds, bad, torn = [], [], []
    rows_total = 0
    for p in paths:
        base = os.path.basename(p)
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                rows_total += 1
                try:
                    r = json.loads(s)
                    sd = int(r["seed"])
                except Exception as e:                     # noqa: BLE001 —— 故意兜住一切解析失败
                    bad.append(f"{base}:{ln}:{type(e).__name__}")
                    continue
                seeds.append(sd)
        # 末行是否**没有换行结尾** ⇒ 极可能正在写（半截）
        if os.path.getsize(p) > 0:
            with open(p, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    torn.append(base)

    n_live = live_procs()
    seen = set(seeds)
    oob = sorted({s for s in seeds if not (SEG_LO <= s <= SEG_HI)})
    miss = sorted(set(range(SEG_LO, SEG_HI + 1)) - seen)
    dup = len(seeds) - len(seen)

    print(f"分片 {nsh} · 行 {rows_total}/{N_EXPECT} · 唯一种子 {len(seen)} · "
          f"存活进程 {n_live} · 解析失败 {len(bad)} · 末行未收尾 {len(torn)}")
    if oob:
        print(f"  越界种子 {len(oob)}：{oob[:5]}{'…' if len(oob) > 5 else ''}")
    if bad:
        print(f"  解析失败行：{bad[:5]}{'…' if len(bad) > 5 else ''}")
    if torn:
        print(f"  末行未收尾（正写？）：{torn}")

    # ---- 判定 ----
    if n_live < 0:
        reasons.append("无法枚举进程")
    if torn:
        reasons.append("末行未收尾（正在写）")
    if bad:
        reasons.append(f"{len(bad)} 行解析失败（裁决器会跳过并计为缺失）")
    if dup:
        reasons.append(f"{dup} 个重复种子")
    if oob:
        reasons.append(f"{len(oob)} 个越界种子")

    if n_live > 0:
        reasons.append(f"{n_live} 个进程仍存活")

    # ★★ 判定顺序（Cycle 81 修正，原顺序有缺陷）：
    #   **损坏**（末行半截 / 重复 / 越界 / 解析失败）⇒ 一律 **NOT-READY**，与进程死没死无关。
    #   原实现把「进程已死 + 有损坏」也归到 DONE-BUT-SHORT（退出 0 = 可裁决）⇒ 会把一次
    #   **数据完整性故障**洗成「预注册的缺失行披露」。二者性质完全不同：前者是装置坏了、
    #   样本可能被**静默丢弃**，后者是装置好、样本**如实缺席**。只有后者可以裁决。
    if reasons:
        print("NOT-READY:" + "; ".join(reasons))
        return 1
    if len(seen) < N_EXPECT:
        # 进程已全部退出且**无任何损坏** ⇒ 再等不会涨；缺失**如实披露**即可裁决
        # （预注册口径：n_eff ≥ 160 仍可判，且 `n_missing_vs_expected` 会写进 VERDICT §3）
        print(f"DONE-BUT-SHORT:缺 {len(miss)} 个种子（首个 {miss[:3]}）；"
              f"n_eff 上限={len(seen)}（门槛 160）")
        return 0
    print("READY: 200/200 完整、种子段无重复无越界、无存活进程 ⇒ 可裁决")
    return 0


if __name__ == "__main__":
    sys.exit(main())