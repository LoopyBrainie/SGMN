import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26i")
import R1_gating10 as R
taus = {n: R.cal_at(n)["c_tau"] for n in (20, 35)}
print("c_tau", {k: round(v, 4) for k, v in taus.items()})
for cell in ({"n_kv": 20, "n_read": 12}, {"n_kv": 35, "n_read": 12}, {"n_kv": 20, "n_read": 16}):
    g = R.run_unit(cell, 20261028, taus[cell["n_kv"]])
    a = g["arms"]
    print(f"({cell['n_kv']}|{cell['n_read']}) k_B {g['k']['B']} q̄ {g['qbar']:.2f} qmax {g['qmax']}")
    for t in R.ARMS6:
        print(f"   {t:>3} ho8 {a[t]['ho8_mean']:.4f} ho4 {a[t]['ho4_mean']:.4f} mu {a[t]['mu']:+.4f}")
    print(f"   D_abs(F-K) {a['Fu']['mu']-a['Ku']['mu']:+.4f}  mu_U0 {a['Uu']['mu']:+.4f}  "
          f"R_mu {(a['Fu']['mu']-a['Ku']['mu'])/a['Uu']['mu']:+.4f}")
