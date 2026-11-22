#!/opt/pytorch/bin/python
r"""DIR-57 · **Stage A 硬门**（`A57-1..12`；任一失败 ⇒ 禁止正式轮）。卡：`revolve/DIR-57/DIRECTION_rev57_v10.md` §7。

用法：`/opt/pytorch/bin/python dir57/stageA57.py` → `dir57/out/DIR57r10_stageA.json`
"""
import hashlib
import inspect
import json
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402
import R1_task51 as T51      # noqa: E402
import R1_task57 as R57      # noqa: E402

OUT = os.path.join(ROOT, "dir57", "out")
SEED0, N = 59000, 96
DEVICE_MD5 = {
    "dir31/R1_grad.py": "8fa40f708600bec7e6c484727d2180c7",
    "dir33/R1_task33.py": "96ef89e87ec61556b52e0e464850087e",
    "dir46/R1_task46.py": "d66c849692d93bc5b24a91f9a445e0ef",
    "dir47/R1_task47.py": "fd03c70c6490cfd726489714b9c2c80f",
    "dir48/R1_task48.py": "bb6154687a48d4ea33f0ebd9bc0cb393",
}
HIST_SEGMENTS = [(55000, 55023), (56000, 56095), (57000, 57095), (58000, 58463)]
WORKED_E = 0.421325009952163
WORKED_E_NULL = 1.106142361815501


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    # A57-1 装置 md5
    got = {k: md5(os.path.join(ROOT, k)) for k in DEVICE_MD5}
    res["A57-1"] = {"pass": got == DEVICE_MD5, "md5": got, "frozen": DEVICE_MD5}
    # A57-2 build 确定性
    P1 = R.build(SEED0, "V0"); P2 = R.build(SEED0, "V0")
    d = max(float((P1[k] - P2[k]).abs().max()) for k in ("U1", "V1"))
    res["A57-2"] = {"pass": d == 0.0, "max_abs_diff": d}
    # 几何（全部 96 种子）：键对角度 / 干扰值角度 / 未写键与已写键不相交
    kmax_all, vmax_all, kmax_un_all, same_key = 0.0, 0.0, 0.0, 0
    for sd in range(SEED0, SEED0 + N):
        gk = torch.Generator().manual_seed(sd + R57.OFF_K)
        gv = torch.Generator().manual_seed(sd + R57.OFF_V)
        gr = torch.Generator().manual_seed(sd + R57.OFF_R)
        gi = torch.Generator().manual_seed(sd + R57.OFF_INT)
        ks = R57.key_batch(gk, R57.M_MAX)
        vP = T51.gen_vec(gv, M.D_V)
        kUn = R57.key_batch(gr, R57.N_UNWRITTEN)
        kInt = R57.key_batch(gi, R57.N_INT)
        vInt = [R57.val_near_orth(vP, gi) for _ in range(R57.N_INT)]
        kmax_all = max(kmax_all, R57.pair_cos_max(ks))
        vmax_all = max(vmax_all, max(abs(float(v @ vP)) for v in vInt))
        for a in ks:
            for b in kUn:
                c = abs(float((a @ b) / (a.norm() * b.norm())))
                kmax_un_all = max(kmax_un_all, c)
                if c > 1 - 1e-12:
                    same_key += 1
    res["A57-3"] = {"pass": kmax_all <= 0.30 + 1e-12, "max_pair_cos": kmax_all}
    res["A57-4"] = {"pass": vmax_all <= 0.30 + 1e-12, "max_int_val_cos": vmax_all}
    res["A57-5"] = {"pass": same_key == 0 and R57.OFF_K != R57.OFF_R and R57.OFF_K != R57.OFF_INT,
                    "max_cos_written_vs_unwritten": kmax_un_all, "n_identical_pairs": same_key,
                    "generators": {"K": R57.OFF_K, "UNWRITTEN": R57.OFF_R, "INT": R57.OFF_INT}}
    # A57-6 读出量守卫：`read_errors` 不得含 cos；E_null 必须是 5 把**未写键**的中位
    src = inspect.getsource(R57.read_errors)
    null_src = inspect.getsource(R57.run_one)
    uses_median_on_kUn = "statistics.median(read_errors(U, Vm, kUn, vP))" in null_src
    res["A57-6"] = {"pass": ("cos" not in src) and uses_median_on_kUn,
                    "read_errors_src_has_cos": "cos" in src,
                    "E_null_is_median_over_unwritten": uses_median_on_kUn,
                    "n_unwritten": R57.N_UNWRITTEN}
    # A57-7 S 恒等式 + dispE ≡ dispS
    row = R57.run_one(SEED0, "m4")
    ident = max(abs(row["S"][i] - (row["E_null"] - row["E"][i])) for i in range(len(row["E"])))
    disp_eq = abs(row["dispE"] - (max(row["S"]) - min(row["S"])))
    res["A57-7"] = {"pass": ident == 0.0 and disp_eq == 0.0, "max_identity_err": ident,
                    "disp_e_vs_s": disp_eq}
    # A57-8 工作例：独立重实现 step_pair
    P = R.build(SEED0, "V0")
    gk = torch.Generator().manual_seed(SEED0 + R57.OFF_K)
    gv = torch.Generator().manual_seed(SEED0 + R57.OFF_V)
    k = R57.key_batch(gk, R57.M_MAX)[0]
    vP = T51.gen_vec(gv, M.D_V)
    U1, V1 = R57.write_pair(P["U1"].clone(), P["V1"].clone(), k, vP)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    for _ in range(T48.N_REP):
        Wm = U @ Vm.t()
        G = torch.outer(Wm @ k - vP, k) + M.LAM * Wm
        U = U - M.ETA * (G @ Vm)
        Vm = Vm - M.ETA * (G.t() @ U)
    Wh = U @ Vm.t(); W = M._W(U1, V1)
    dW = float((Wh - W).abs().max())
    E_hand = float((Wh @ k - vP).norm()); E_drv = float((W @ k - vP).norm())
    res["A57-8"] = {"pass": dW < 1e-12 and abs(E_hand - E_drv) < 1e-12 and abs(E_drv - WORKED_E) < 1e-12,
                    "max_abs_dW": dW, "E_hand": E_hand, "E_driver": E_drv,
                    "frozen_E": WORKED_E, "delta_vs_frozen": abs(E_drv - WORKED_E)}
    # A57-9 四臂初始状态逐位相同
    fps = {a: R57.run_one(SEED0, a)["init_fp"] for a in R57.ARMS}
    res["A57-9"] = {"pass": len(set(fps.values())) == 1, "init_fp": fps}
    # A57-10 干扰臂结构
    r8 = R57.run_one(SEED0, "m4_int8")
    res["A57-10"] = {"pass": r8.get("n_int") == 8 and r8.get("int_key_unique") is True,
                     "n_int": r8.get("n_int"), "int_key_unique": r8.get("int_key_unique"),
                     "int_val_cos_max": r8.get("int_val_cos_max")}
    # A57-11 有限性（四臂 × 4 种子）
    bad = []
    for a in R57.ARMS:
        for sd in range(SEED0, SEED0 + 4):
            r = R57.run_one(sd, a)
            if not r["finite"] or (a == "m4_int8" and not r.get("finite_int8")):
                bad.append((a, sd))
    res["A57-11"] = {"pass": not bad, "bad": bad}
    # A57-12 种子段不相交
    seg = set(range(SEED0, SEED0 + N))
    hist = set()
    for lo, hi in HIST_SEGMENTS:
        hist |= set(range(lo, hi + 1))
    res["A57-12"] = {"pass": not (seg & hist), "n_overlap": len(seg & hist),
                     "segments_checked": HIST_SEGMENTS}
    ok = all(v["pass"] for v in res.values())
    res["_summary"] = {"all_pass": ok, "n_pass": sum(1 for v in res.values() if v["pass"]),
                       "n_total": 12}
    with open(os.path.join(OUT, "DIR57r10_stageA.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for k in sorted(res):
        if k.startswith("A57-"):
            print("%s  %s" % (k, "PASS" if res[k]["pass"] else "★FAIL"))
    print("── Stage A：%d/%d %s" % (res["_summary"]["n_pass"], 12,
                                   "全部通过" if ok else "★ 有失败，禁止正式轮"))


if __name__ == "__main__":
    main()