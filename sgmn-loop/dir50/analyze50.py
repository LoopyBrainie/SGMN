#!/opt/pytorch/bin/python
r"""DIR-50 · **机制分析**（报告用，不入判据）：从正式轮 jsonl 计算
① 跨模态**衰减链**（层 1 读出 → 层 1 注入后隐 → 层 2 注入前隐 → 层 2 的 (k2,v2)）的相对位移中位数；
② 各臂/剂量/层的 `rel` 曲线均值；
③ 级联在每个 `d` 上的配对差。
"""
import glob, json, math, os, sys

ROOT = "/home/ec2-user/sgmn-loop"
MED = ("y1", "h1p", "h2pre", "k2", "v2")
STEP = 1          # med[] 索引 1 ⇒ step T_HIT+1 = 5（第一个受级联影响的步）


def load(tag):
    rows = {}
    for f in sorted(glob.glob(os.path.join(ROOT, f"dir50/out/DIR50r10_arm_{tag}_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows[int(r["seed"])] = r
    return rows


def med(v):
    s = sorted(v); n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    out = {"chain": {}, "rel_curve": {}, "cascade_by_d": {}}
    base = load("none")
    # ① 衰减链：`hit1` 臂 vs `none`，step 5
    for dose in ("0.0", "1.0"):
        h = load(f"hit1_{dose}")
        for kk in MED:
            vals = []
            for sd, r in h.items():
                b = base.get(sd)
                if b is None or STEP >= len(r["med"][kk]) or STEP >= len(b["med"][kk]):
                    continue
                yh, yn = r["med"][kk][STEP], b["med"][kk][STEP]
                if not all(math.isfinite(z) for z in list(yh) + list(yn)):
                    continue
                if max(abs(z) for z in list(yh) + list(yn)) > 1e100:
                    continue
                den = math.sqrt(sum(z * z for z in yn))
                if not (den > 0 and math.isfinite(den)):
                    continue
                num = math.sqrt(sum((x - y) ** 2 for x, y in zip(yh, yn)))
                v = num / den
                if math.isfinite(v):
                    vals.append(v)
            out["chain"][f"hit1|{dose}|{kk}"] = {"median": med(vals), "n": len(vals)}
    # ② rel 曲线
    def rel(r, li, d):
        c0 = r["c0"][li]
        if not (math.isfinite(c0) and c0 > 0):
            return None
        c = r["curve"][li][d - 1]
        return c / c0 if math.isfinite(c) else None
    for tag in ("none", "hit1_0.0", "hit1_1.0", "hit2_0.0", "hit2_1.0"):
        rows = load(tag)
        for li in (0, 1):
            for d in range(1, 8):
                vals = [rel(r, li, d) for r in rows.values()]
                vals = [v for v in vals if v is not None]
                out["rel_curve"][f"{tag}|L{li+1}|d{d}"] = {"mean": sum(vals) / len(vals), "n": len(vals)}
    # ③ 级联逐 d
    h = load("hit1_1.0")
    for d in range(1, 8):
        diffs = []
        for sd, r in h.items():
            b = base.get(sd)
            if b is None:
                continue
            x, y = rel(r, 1, d), rel(b, 1, d)
            if x is not None and y is not None:
                diffs.append(x - y)
        n = len(diffs); m = sum(diffs) / n
        sd_ = (sum((z - m) ** 2 for z in diffs) / (n - 1)) ** 0.5 if n > 1 else 0.0
        out["cascade_by_d"][f"d{d}"] = {"mean": m, "3se": 3 * sd_ / math.sqrt(n) if n else None, "n": n}
    json.dump(out, open(os.path.join(ROOT, "dir50/out/DIR50_analysis.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
