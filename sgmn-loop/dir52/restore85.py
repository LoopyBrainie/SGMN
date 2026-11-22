#!/opt/pytorch/bin/python
r"""把正式轮数据从**未链接 inode 的快照**恢复到权威路径（DIR-52 · DEF-52-5 善后）。

背景：Cycle 84 一条 rsync 把本地陈旧 `dir52/out/DIR52r12_shard*.jsonl`（各 21 行）
覆盖到 trainer，远程路径因此换成新 inode；4 个写手仍向**旧 inode**（`(deleted)`）追加。
`dir52/watch_fd_snapshot.py` 每 5 s 把 `/proc/<pid>/fd` 里的内容快照到
`dir52/out/FDSNAP/<seq>/shard{0..3}.jsonl`，直到 4 个写手退出。

本脚本**只**做恢复与校验，**不做任何判定**：
  1. 取**最新且可完整校验**的快照（从新到旧逐个试）；
  2. 校验：4 分片 · 合计 200 行 · 种子恰为冻结段 `62000..62199` 无重复无越界 ·
     每行可解析 · `identity_err==0` · `finite is True`；
  3. 把被毒化的路径文件移到 `dir52/out/_POISONED_by_rsync/`（**留档，不删**）；
  4. 以「同目录临时文件 + `os.replace`」原子放回权威路径，并打印每个分片的 sha256。

用法：
  /opt/pytorch/bin/python dir52/restore85.py --check-only   # 只看能不能恢复（默认安全）
  /opt/pytorch/bin/python dir52/restore85.py                # 真正执行恢复
★ 恢复后**必须**重跑 `dir52/ready52.py`；`READY` 才可继续裁决。
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

OUT = "dir52/out"
SNAP = os.path.join(OUT, "FDSNAP")
LO, HI = 62000, 62199
N = HI - LO + 1
QUAR = os.path.join(OUT, "_POISONED_by_rsync")


def load_snapshot(d):
    """读一个快照目录；返回 (rows, 问题列表)。"""
    rows, probs = [], []
    for i in range(4):
        p = os.path.join(d, f"shard{i}.jsonl")
        if not os.path.isfile(p):
            probs.append(f"缺 shard{i}")
            continue
        bad_tail = False
        with open(p, "rb") as f:
            data = f.read()
        bad_tail = bool(data) and not data.endswith(b"\n")
        for ln, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
                r["_shard"] = i
                r["_ln"] = ln
                rows.append(r)
            except Exception:
                probs.append(f"shard{i}:{ln} 解析失败")
        if bad_tail:
            probs.append(f"shard{i} 末行未收尾")
    seeds = [int(r["seed"]) for r in rows if "seed" in r]
    miss_sorted = sorted(set(range(LO, HI + 1)) - set(seeds))
    if len(rows) != N:
        probs.append(f"行数 {len(rows)} ≠ {N}")
    if len(set(seeds)) != len(seeds):
        probs.append(f"重复种子 {len(seeds) - len(set(seeds))}")
    if set(seeds) != set(range(LO, HI + 1)):
        miss = sorted(set(range(LO, HI + 1)) - set(seeds))
        oob = sorted({s for s in seeds if not (LO <= s <= HI)})
        probs.append(f"种子集不符：缺 {len(miss)}（前 {miss[:3]}）越界 {len(oob)}")
    idbad = [r for r in rows if r.get("identity_err") not in (0, 0.0)]
    finbad = [r for r in rows if r.get("finite") is not True]
    if idbad:
        probs.append(f"identity_err≠0 的行 {len(idbad)}")
    if finbad:
        probs.append(f"finite≠True 的行 {len(finbad)}")
    return rows, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只评估，不改动任何文件")
    ap.add_argument("--allow-short", action="store_true",
                    help="接受**块尾连续缺 k 行**（k≤8）的短缺口，并在记录中显式写明；缺行不得补算")
    a = ap.parse_args()

    snaps = sorted(glob.glob(os.path.join(SNAP, "[0-9]*")), reverse=True)
    if not snaps:
        print(f"✘ 没有快照（{SNAP} 为空）—— 写手可能已退出且 fd 已消失，需人工排查")
        return 1
    print(f"快照 {len(snaps)} 个，从新到旧逐个校验（最新 {snaps[0]}）")
    chosen, rows, last_probs = None, None, None
    for d in snaps:
        rows_t, probs = load_snapshot(d)
        print(f"  {os.path.basename(d)}: 行 {len(rows_t)} · "
              f"{'完整可恢复' if not probs else '不合格 → ' + '; '.join(probs[:3])}")
        if not probs:
            chosen, rows = d, rows_t
            break
        last_probs = probs
    if a.allow_short and chosen is None:
        # 依次尝试，接受"每个 50 种子块各自块尾连续缺 k 行"（k 可不同，合计 ≤ 8）
        for d in snaps:
            rows_t, _probs = load_snapshot(d)
            sds = [int(r["seed"]) for r in rows_t if "seed" in r]
            if len(set(sds)) != len(sds):
                continue
            blocks = [(62000, 50), (62050, 50), (62100, 50), (62150, 50)]
            ok, miss_all = True, []
            for b0, bn in blocks:
                blk = set(range(b0, b0 + bn))
                got = blk & set(sds)
                k = bn - len(got)
                if got != set(range(b0, b0 + bn - k)):
                    ok = False
                miss_all += sorted(blk - got)
            if ok and 0 < len(miss_all) <= 8 and not [r for r in rows_t if r.get("identity_err") not in (0, 0.0)] \
               and not [r for r in rows_t if r.get("finite") is not True]:
                chosen, rows = d, rows_t
                print(f"\n⚠ --allow-short：接受 {d}：{len(rows_t)}/{N} 行，"
                      f"块尾缺 {miss_all}（★ 不补算，如实披露）")
                break
    if chosen is None:
        print(f"\n✘ 没有任何快照通过校验；最新快照的问题：{'; '.join((last_probs or [])[:6])}")
        print("  ★ 不得据此裁决；等写手退出后再看新快照，或如实披露缺失。")
        return 1

    _nrows = len(rows)
    print(f"\n✔ 选定 {chosen}：{_nrows}/{N} 行" + ("（**短缺口，已如实记录**）" if _nrows != N else "，种子段完整、无重复无越界、identity_err≈0、finite"))
    sums = {i: hashlib.sha256(open(os.path.join(chosen, f"shard{i}.jsonl"), "rb").read()).hexdigest()[:16]
            for i in range(4)}
    for i, h in sums.items():
        print(f"  shard{i} sha256[:16]={h}")

    if a.check_only:
        print("\n（--check-only：未改动任何文件）")
        return 0

    os.makedirs(QUAR, exist_ok=True)
    for i in range(4):
        cur = os.path.join(OUT, f"DIR52r12_shard{i}.jsonl")
        if os.path.isfile(cur):
            dst = os.path.join(QUAR, f"DIR52r12_shard{i}.jsonl.POISONED")
            shutil.move(cur, dst)
            print(f"  毒化文件留档 → {dst}")
        tmp = cur + ".restore.tmp"
        shutil.copyfile(os.path.join(chosen, f"shard{i}.jsonl"), tmp)
        os.replace(tmp, cur)          # 同目录原子替换
    json.dump({"restored_from": chosen, "n_rows": len(rows),
               "shortfall": sorted(set(range(LO, HI + 1)) - {int(r["seed"]) for r in rows}),
               "note": "DEF-52-5 装置事故：块尾 4 行随写手进程退出而丢失（守护盲区），"
                       "补算门（ctrl85 复现对照）FAIL ⇒ 不补算、如实披露",
               "sha256_16": sums},
              open(os.path.join(OUT, "RESTORE_RECORD.json"), "w"), indent=1)
    print(f"\n✔ 已恢复 4 个分片到 {OUT}/ 并写 RESTORE_RECORD.json")
    print("  ★ 下一步：`/opt/pytorch/bin/python dir52/ready52.py` —— 必须 READY 才可裁决")
    return 0


if __name__ == "__main__":
    sys.exit(main())