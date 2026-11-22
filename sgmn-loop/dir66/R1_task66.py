import sys, json, argparse, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir66")
import ref66 as R
p = argparse.ArgumentParser(); p.add_argument("--out", default="dir66/out/DIR66r10_formal.json"); a = p.parse_args()
C, N = 6.0, 200000
AS = [1e-4, 1e-3, 1e-2, 3e-2, 5e-2, 0.1, 0.2, 0.3, 0.5]
res = {"eta_star": {}, "esc": {}, "td": {}}
for aa in AS:
    b = R.bisect_a(C, aa, N)
    res["eta_star"]["%.0e" % aa] = b["eta"]
    print("① a=%-9.0e η*=%.5f  (lo=%.5f hi=%.5f)" % (aa, b["eta"], b["lo"], b["hi"]))
for aa in (5e-2, 0.2, 0.3):
    eta = 1.05 * res["eta_star"]["%.0e" % aa]
    o = R.run_a(C, eta, aa, N)
    res["esc"]["%.0e" % aa] = dict(eta=eta, t_esc=o["t_esc"], td=o["td"], step=o["t_esc_step"])
    print("② a=%-9.0e η=%.4f t_esc=%.4f @%s T_div=%s" % (aa, eta, o["t_esc"], o["t_esc_step"], o["td"]))
json.dump(res, open(a.out, "w"), indent=1, default=float)
print("→", a.out)
