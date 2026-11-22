"""DIR-82 正式轮驱动器：16 格，4 路并行，逐格落盘，断点续跑。"""
import subprocess, os, sys, json, itertools, time

CELLS = []
for mode in ("keys", "target"):
    for g in (3.0, 10.0, 20.0, 30.0):
        CELLS.append((mode, g, 0.1, "fp", "G_%s_g%g" % (mode, g)))
for mode in ("keys", "target"):
    for i in (0.01, 0.03, 0.1, 0.3):
        CELLS.append((mode, 10.0, i, "init", "I_%s_i%g" % (mode, i)))

PAR = 4
todo = []
for c in CELLS:
    out = "dir82/out/cell_%s.json" % c[4]
    if os.path.exists(out):
        try:
            d = json.load(open(out))
            if d.get("stage") == "done":
                print("skip(done) %s" % c[4], flush=True); continue
        except Exception:
            pass
    todo.append(c)
print("=== 待跑 %d / 共 %d 格 ===" % (len(todo), len(CELLS)), flush=True)

os.makedirs("dir82/logs", exist_ok=True)
running = []
T0 = time.time()
while todo or running:
    while todo and len(running) < PAR:
        c = todo.pop(0)
        cmd = ["/opt/pytorch/bin/python", "dir82/cell82.py", c[0], repr(c[1]), repr(c[2]), c[3], c[4]]
        lf = open("dir82/logs/%s.log" % c[4], "w")
        env = dict(os.environ); env["OMP_NUM_THREADS"] = "1"
        running.append((c[4], subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
        print("start %-18s (并行 %d)" % (c[4], len(running)), flush=True)
    time.sleep(2)
    for item in list(running):
        lbl, p, lf = item
        if p.poll() is not None:
            lf.close(); running.remove(item)
            print("end   %-18s rc=%d  t=%.0fs" % (lbl, p.returncode, time.time() - T0), flush=True)
print("=== 全部完成 t=%.0fs ===" % (time.time() - T0), flush=True)
