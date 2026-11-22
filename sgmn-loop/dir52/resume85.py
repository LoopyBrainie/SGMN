#!/opt/pytorch/bin/python
r"""`DEF-52-5` 的**补算**：把 fd 快照里缺失的**最后若干种子**用同一装置补算，然后原子归位。

为什么允许补算（与"不得重跑补齐"不冲突）：
  · 卡片冻结的是**种子集** `62000..62199`，`n = 200`。补算**不改变**这个设计；
  · 缺的 4 个种子是**装置事故**（rsync 覆盖 ⇒ 写手 fd 变 `(deleted)` ⇒ 末行落在 5 s 快照之后、
    随进程退出而消失）造成的，**不是**实验结果。
  · 补算用的是**同一份 `R1_task52.py`、同一 seed 段、同一 `--tag`**，且 `--resume` 语义会**跳过所有已有种子**，
    只算缺的那一个 ⇒ 不会覆盖、不会重算已有行。
  · ★ 本脚本**不读任何判据量**（只核 `seed`/`identity_err`/`finite`/行数），补算前后都保持盲态。

拒绝条件（出现即停，如实披露，不猜）：
  · 缺失种子数 > 8，或任一分片缺 > 4 ⇒ 不是"末行窗口"型损失，需人工判断；
  · 补算后行数/种子集/唯一性/`identity_err`/`finite` 任一不合格。

用法：
  /opt/pytorch/bin/python dir52/resume85.py --check-only   # 只报告缺口与可行性
  /opt/pytorch/bin/python dir52/resume85.py                # 补算 + 归位（★ 约 13–15 min）
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

OUT = "dir52/out"
SNAP = os.path.join(OUT, "FDSNAP")
WORK = os.path.join(OUT, "RESUME")
QUAR = os.path.join(OUT, "_POISONED_by_rsync")
BLOCKS = [(62000, 50), (62050, 50), (62100, 50), (62150, 50)]
MAX_MISSING, MAX_PER_SHARD = 8, 4
PY = "/opt/pytorch/bin/python"


def rows_of(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    data = open(path, "rb").read()
    for line in data.decode("utf-8", "replace").splitlines():
        s = line.strip()
        if s:
            try:
                rows.append(json.loads(s))
            except Exception:
                pass
    return rows


def pick_snapshot():
    """取总行数最多的快照（同数取序号最大）。"""
    best = None
    for d in glob.glob(os.path.join(SNAP, "[0-9]*")):
        n = sum(len(rows_of(os.path.join(d, f"shard{i}.jsonl"))) for i in range(4))
        seq = int(os.path.basename(d))
        if best is None or (n, seq) >= (best[0], best[2]):
            best = (n, d, seq)
    return best


def survey(snap):
    plan = []
    for i, (b0, bn) in enumerate(BLOCKS):
        have = {int(r["seed"]) for r in rows_of(os.path.join(snap, f"shard{i}.jsonl")) if "seed" in r}
        block = set(range(b0, b0 + bn))
        miss = sorted(block - have)
        extra = sorted(have - block)
        plan.append({"shard": i, "have": len(have & block), "miss": miss, "extra": extra})
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    picked = pick_snapshot()
    if not picked:
        print(f"✘ {SNAP} 无快照"); return 1
    n, snap, _seq = picked
    print(f"最富快照 {snap}：合计 {n} 行")
    plan = survey(snap)
    tot_miss = 0
    for p in plan:
        tot_miss += len(p["miss"])
        print(f"  shard{p['shard']}: 有 {p['have']}/50 · 缺 {p['miss']} · 越界 {p['extra'][:3]}")
    if tot_miss == 0:
        print("✔ 无缺口 —— 直接走 restore85.py 即可"); return 0
    # ★ 只接受"块尾连续缺 k 个"：中段任意缺失不具备"结果无关性"，不得补算
    suffix_ok = True
    for p in plan:
        b0, bn = BLOCKS[p["shard"]]
        k = len(p["miss"])
        if k and p["miss"] != list(range(b0 + bn - k, b0 + bn)):
            suffix_ok = False
            print(f"  ✘ shard{p['shard']} 缺口不在块尾（{p['miss']}）⇒ 不接受")
    bad = (not suffix_ok) or tot_miss > MAX_MISSING or any(len(p["miss"]) > MAX_PER_SHARD for p in plan)
    if bad:
        print(f"✘ 缺口 {tot_miss} 不可补算（须为块尾连续缺、≤{MAX_MISSING}、单片≤{MAX_PER_SHARD}）⇒ 停止，如实披露")
        return 1
    if a.check_only:
        print(f"\n（--check-only）可补算 {tot_miss} 个种子：{[s for p in plan for s in p['miss']]}")
        return 0

    os.makedirs(WORK, exist_ok=True)
    procs, seeds_done = [], []
    for p in plan:
        i = p["shard"]
        stage = os.path.join(WORK, f"final_shard{i}.jsonl")
        shutil.copyfile(os.path.join(snap, f"shard{i}.jsonl"), stage)
        b0, bn = BLOCKS[i]
        lg = open(os.path.join(WORK, f"log_resume{i}.txt"), "w")
        # ★ 同一份装置、同一 seed 段、--resume 只补缺
        procs.append(subprocess.Popen(
            [PY, "dir52/R1_task52.py", "--seed0", str(b0), "--n", str(bn), "--out", stage, "--resume"],
            stdout=lg, stderr=subprocess.STDOUT))
        seeds_done.append((i, stage, p["miss"]))
    print(f"已并行启动 4 个补算进程（每个只算缺的种子，其余由 --resume 跳过）…")
    rc = [(i, pr.wait()) for i, pr in enumerate(procs)]
    print("退出码:", rc)
    if any(c != 0 for _, c in rc):
        print("✘ 补算子进程非零退出 ⇒ 不归位"); return 1

    # —— 校验 ——
    probs, sums = [], {}
    for i, stage, miss in seeds_done:
        rows = rows_of(stage)
        s = sorted(int(r["seed"]) for r in rows)
        b0, bn = BLOCKS[i]
        if len(rows) != bn: probs.append(f"shard{i} 行数 {len(rows)}≠{bn}")
        if s != list(range(b0, b0 + bn)): probs.append(f"shard{i} 种子集不符")
        if [r for r in rows if r.get("identity_err") not in (0, 0.0)]: probs.append(f"shard{i} identity_err≠0")
        if [r for r in rows if r.get("finite") is not True]: probs.append(f"shard{i} finite≠True")
        tags = {r.get("tag") for r in rows}
        if len(tags) != 1: probs.append(f"shard{i} tag 不一致 {tags}")
        else: tag_seen = tags.pop()
        sums[i] = hashlib.sha256(open(stage, "rb").read()).hexdigest()[:16]
        rec = sorted({int(r["seed"]) for r in rows} & set(miss))
        print(f"  shard{i}: {len(rows)} 行 · 补算 {rec} · sha256[:16]={sums[i]}")
    if probs:
        print("✘ 校验不合格：", "; ".join(probs[:6])); print("  ★ 不得归位、不得裁决"); return 1

    os.makedirs(QUAR, exist_ok=True)
    for i, stage, miss in seeds_done:
        cur = os.path.join(OUT, f"DIR52r12_shard{i}.jsonl")
        if os.path.isfile(cur):
            shutil.move(cur, os.path.join(QUAR, f"DIR52r12_shard{i}.jsonl.POISONED"))
        tmp = cur + ".resume.tmp"
        shutil.copyfile(stage, tmp)
        os.replace(tmp, cur)
    rec = {"mode": "fd-snapshot + tail-resume (DEF-52-5)", "snapshot": snap,
           "tags": sorted({r.get("tag") for _, st, _m in seeds_done for r in rows_of(st)}),
           "n_rows": sum(BLOCKS[i][1] for i in range(4)),
           "recomputed": {i: miss for i, _, miss in seeds_done},
           "same_device": "dir52/R1_task52.py --resume --tag r10 (default)",
           "sha256_16": sums}
    json.dump(rec, open(os.path.join(OUT, "RESUME_RECORD.json"), "w"), indent=1)
    json.dump(rec, open(os.path.join(OUT, "RESTORE_RECORD.json"), "w"), indent=1)
    print(f"\n✔ 已归位 200 行并写 RESUME_RECORD.json / RESTORE_RECORD.json")
    print("  ★ 下一步：ready52.py 必须 READY 才可裁决")
    return 0


if __name__ == "__main__":
    sys.exit(main())