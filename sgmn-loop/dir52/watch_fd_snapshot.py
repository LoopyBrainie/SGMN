#!/opt/pytorch/bin/python
"""守护式 fd 快照：正式轮写手的 shard 文件已被 rsync 覆盖(unlink)，内容仍在未链接 inode 里。
每 60s 把 /proc/<pid>/fd 的内容快照到 dir52/out/FDSNAP/<seq>/，直到 4 个 pid 全部退出。
只做 I/O 拷贝，无计算、不碰判据。"""
import glob, json, os, time
PIDS = (96706, 96708, 96710, 96712)
DST = "dir52/out/FDSNAP"
os.makedirs(DST, exist_ok=True)
seq, size0 = 0, {}
while True:
    alive = [p for p in PIDS if os.path.exists(f"/proc/{p}")]
    got, snap = {}, {}
    for p in alive:
        for fd in glob.glob(f"/proc/{p}/fd/*"):
            try: t = os.readlink(fd)
            except OSError: continue
            if "DIR52r12_shard" in t:
                i = t.split("shard")[1][0]
                d = open(fd, "rb").read()
                got[i] = d
    if got and any(size0.get(i) != len(d) for i, d in got.items()):
        seq += 1
        sd = f"{DST}/{seq:03d}"
        os.makedirs(sd, exist_ok=True)
        for i, d in got.items():
            open(f"{sd}/shard{i}.jsonl", "wb").write(d)
            size0[i] = len(d)
        json.dump({"seq": seq, "ts": time.time(), "alive": alive,
                   "bytes": {i: len(d) for i, d in got.items()}},
                  open(f"{sd}/meta.json", "w"))
    if not alive:
        json.dump({"seq": seq, "done": True}, open(f"{DST}/DONE.json", "w"))
        break
    time.sleep(5)
