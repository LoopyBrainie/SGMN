import sys, os, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir71"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import R2_bisect71 as B
# 预注册网格（卡片 §6 + SUPPLEMENT-1 §A）。tag 只选网格，gamma 永远是物理 γ（DEF-71-2 的教训）。
GRIDS = {("keys", 30, "base"): [1e-3, 1.4e-3, 1.9e-3, 2.6e-3, 3.5e-3],
         ("keys", 100, "base"): [4e-5, 8e-5, 1.2e-4, 1.7e-4, 2.4e-4, 3.4e-4],
         ("keys", 30, "ext"): [5e-3, 7e-3, 1e-2, 1.4e-2, 2e-2],
         ("keys", 100, "ext"): [5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3]}
OUT = "/home/ec2-user/sgmn-loop/dir71/out/grid"
def main(mode, gamma, tag):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, "%s_g%g%s.json" % (mode, gamma, "" if tag == "base" else "_" + tag))
    rec = json.load(open(fn)) if os.path.exists(fn) else dict(mode=mode, gamma=gamma, tag=tag, seed=20261118, N=200000, cells={})
    for eta in GRIDS[(mode, int(gamma), tag)]:
        k = "%.0e" % eta
        if k in rec["cells"]: continue
        rec["cells"].update(B.grid(mode, gamma, [eta]))
        json.dump(rec, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
    print("DONE %s γ=%g tag=%s" % (mode, gamma, tag), flush=True)
if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "base")
