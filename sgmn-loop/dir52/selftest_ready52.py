#!/opt/pytorch/bin/python
r"""DIR-52 · `ready52.py` 的**状态自测**（六态，全 CPU-free、秒级）。

★ 为什么必须测「READY」：一个只被验证过**失败路径**的门，不能用来回答「可以裁决了吗」——
若 READY 分支有 bug（例如种子段比对写错），它会在真正跑完时**永远说 NOT-READY**，
于是我会**无限期等待一个已经完成的轮次**。故六态都必须实测。

六态（`READY52_PROCS` 注入存活进程数；分片写在临时目录，**不碰真实 out/**）：
  ① READY              200 行满段、0 进程、末行收尾        ⇒ 期望 READY，退出 0
  ② 进程存活            200 行满段、4 进程                ⇒ 期望 NOT-READY，退出 1
  ③ 末行半截            199 行 + 半截 JSON、0 进程         ⇒ 期望 NOT-READY，退出 1
  ④ 重复种子            200 行但一个种子重复、缺一个        ⇒ 期望 NOT-READY，退出 1
  ⑤ 解析失败行          200 行但一行非 JSON、0 进程        ⇒ 期望 NOT-READY，退出 1
  ⑥ DONE-BUT-SHORT     190 行、0 进程                     ⇒ 期望 DONE-BUT-SHORT，退出 0（可裁决）
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
READY = os.path.join(HERE, "ready52.py")
LO, HI = 62000, 62199
OUT = os.path.join("dir52", "out")


def row(seed):
    return json.dumps({"seed": seed, "EARLY": 0.0, "LATE": 0.0, "TOTAL": 0.0,
                       "SHAPE": 0.0, "finite": True})


def write_shards(root, seeds, tail=b"\n"):
    """把 seeds 均分成 4 个分片写入 `<root>/dir52/out/DIR52r12_shard{i}.jsonl`。"""
    os.makedirs(os.path.join(root, OUT), exist_ok=True)
    q = (len(seeds) + 3) // 4
    for i in range(4):
        chunk = seeds[i * q:(i + 1) * q]
        p = os.path.join(root, OUT, f"DIR52r12_shard{i}.jsonl")
        with open(p, "wb") as f:
            for s in chunk:
                f.write((row(s) + "\n").encode())
            if chunk and tail != b"\n":
                f.write(tail)


def run(root, procs):
    env = dict(os.environ, READY52_PROCS=str(procs))
    return subprocess.run([sys.executable, READY, ], cwd=root, env=env,
                          capture_output=True, text=True)


def main():
    full = list(range(LO, HI + 1))
    cases = []

    # ① READY
    cases.append(("① READY", full, 0, b"\n", 0, "READY"))
    # ② 进程存活
    cases.append(("② 进程存活", full, 4, b"\n", 1, "NOT-READY:4 个进程仍存活"))
    # ③ 末行半截（199 完整 + 半截）
    cases.append(("③ 末行半截", full[:199], 0, b'{"seed": 62199, "EA', 1, "NOT-READY:"))
    # ④ 重复种子（末位换成倒数第二个）
    dupd = full[:199] + [full[198]]
    cases.append(("④ 重复种子", dupd, 0, b"\n", 1, "NOT-READY:"))
    # ⑤ 解析失败行 + 0 进程
    cases.append(("⑤ 解析失败", None, 0, b"\n", 1, "NOT-READY:"))

    ok = 0
    for label, seeds, procs, tail, want_rc, want_sub in cases:
        root = tempfile.mkdtemp()
        try:
            if label.startswith("⑤"):
                write_shards(root, full)
                p = os.path.join(root, OUT, "DIR52r12_shard3.jsonl")
                lines = open(p, encoding="utf-8").read().splitlines()
                lines[10] = "not-json-at-all"
                open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
            else:
                write_shards(root, seeds, tail)
            r = run(root, procs)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        good = r.returncode == want_rc and want_sub in r.stdout
        ok += good
        print(f"  {'OK  ' if good else 'FAIL'}  {label:14s} rc={r.returncode}(期望{want_rc})  {last[:96]}")

    # ⑥ DONE-BUT-SHORT（190 行、0 进程）
    root = tempfile.mkdtemp()
    try:
        write_shards(root, full[:190])
        r = run(root, 0)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    good = r.returncode == 0 and last.startswith("DONE-BUT-SHORT")
    ok += good
    print(f"  {'OK  ' if good else 'FAIL'}  {'⑥ DONE-BUT-SHORT':14s} rc={r.returncode}(期望0)  {last[:96]}")

    print(f"\n{ok}/6 通过")
    print("★ 全部通过 —— 六态齐备，含 **READY 成功路径**" if ok == 6 else "★ 有失败")
    return 0 if ok == 6 else 1


if __name__ == "__main__":
    sys.exit(main())