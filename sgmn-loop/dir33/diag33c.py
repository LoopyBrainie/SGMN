#!/opt/pytorch/bin/python
r"""DIR-33 诊断（三）：A33-6 该用**总体统计量**还是**实例最大值**？

`diag33b.py` 否证了「值饱和」假说（$P_v$ 尺度 1.0→0.25 只把目标共线从 0.889 动到 0.877）。
真正的原因：$g\in\mathbb R^{d_v}$ 且 $d_v{=}4$，而 DIR-32 的 A32-6 取的是
**24 实例 × 12 对比较 = 288 个抽样**的**最大值** ⇒ 这是 $R^4$ 里的小样本极值，**没有构造性上界**：
DIR-32 抽到 0.8434（过），DIR-33 抽到 0.9666（不过）—— 同一断言、同一装置族，只差一次抽样。

本文件报**每个构造**的 mean / median / max 三个量，供开卡时把一个**与判据同口径**（对 24 个配对种子取平均）
的总体统计量冻结为断言，并把 max 降级为「诊断量」。
"""
import json
import statistics
import sys
import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_task33 as R
import R1_grad as M

OUT = "/home/ec2-user/sgmn-loop/dir33/out"
_SQ = M.D_H ** 0.5


def period2_ortho(sd):
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(2, R.D_H, generator=g)
    x[1] = x[1] - (x[1] @ x[0]) / (x[0] @ x[0]) * x[0]
    x = x / x.norm(dim=1, keepdim=True) * _SQ
    return torch.stack([x[t % 2] for t in range(R.T)])


def stats(kind):
    per_inst, pooled = [], []
    for sd in R.SEEDS_F:
        P = R.build(sd, "V0")
        if kind == "ortho":
            P["a"] = period2_ortho(sd)
        else:
            R.period2(P, sd)
        g = R.targets(P)
        cs = []
        for i in range(len(g)):
            for j in range(len(g)):
                if (i - j) % 2 == 1:
                    cs.append(abs(float(g[i] @ g[j] / (g[i].norm() * g[j].norm()))))
        per_inst.append(max(cs))          # 每实例内取 max（实例内 12 对）
        pooled += cs
    return {"inst_mean": statistics.fmean(per_inst), "inst_median": statistics.median(per_inst),
            "inst_max": max(per_inst), "pooled_mean": statistics.fmean(pooled),
            "pooled_max": max(pooled), "n_pooled": len(pooled)}


def main():
    res = {k: stats(k) for k in ("draw", "ortho")}
    for k, v in res.items():
        print(f"{k:6s} 每实例max: mean {v['inst_mean']:.4f} · median {v['inst_median']:.4f} · "
              f"max {v['inst_max']:.4f}   |  合并 {v['n_pooled']} 抽: mean {v['pooled_mean']:.4f} "
              f"· max {v['pooled_max']:.4f}")
    json.dump(res, open(f"{OUT}/DIR33_diag_tgtstats.json", "w"), indent=1)
    print("\n读法：判据对 24 个配对种子**取平均** ⇒ 与之同口径的量是 inst_mean；"
          "inst_max 是 288 抽样的小样本极值，无构造上界。")


if __name__ == "__main__":
    main()
