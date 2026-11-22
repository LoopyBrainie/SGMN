import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir65")
import ref65 as R
print("① 长时程 (c=6)：‖W‖ 与不平衡比 t_imb")
for eta in (0.79, 0.80, 0.85):
    o = R.run(6.0, eta, 1000000, trace_every=100000)
    cells = []
    for t, imb, prod in o["trace"]:
        cells.append("%.0e:t=%.2f/W=%.2f" % (t, imb, np.sqrt(prod)))
    print("   η=%.2f  div@%s  " % (eta, o["td"]) + "  ".join(cells[::2]))
print("\n② η=0.85 失败前末 8 个记录（每 1e4 步）")
o = R.run(6.0, 0.85, 1000000, trace_every=10000)
print("   td=%s  记录数=%d" % (o["td"], len(o["trace"])))
for t, imb, prod in o["trace"][-8:]:
    print("   t=%-8d ‖W‖=%.3e  ‖U‖‖V‖=%.3e  t_imb=%.3f" % (t, np.sqrt(prod), prod, imb))
