    # ══════════════════ ★★【DIR-21】判据 H21-1/1b/2/3/4 + 报告项 Rp-1…Rp-4（卡片 rev 1.1 §3 冻结）═══
    # —— 口径（写死，防漂移）——
    #   曲线横轴 = **在线步数**（step）⇒ 跨粒度可比（文档数随 m 变 ⇒ 不用文档轴）
    #   Δ(Δnll)(A,B) = mean(nll_B − nll_A)（>0 ⇒ A 更好）；配对单位 = (种子, 步点)
    #   R(step) ≡ fid_mean(step) / fid_mean(0)   （归一化内容保持；量纲无关）
    #   半衰期 ≡ 首个 R(step) ≤ 0.5 的 step
    #   容量峰值 K* ≡ 跨种子平均的「可检索条目数 n_ret」随**文档数**的 argmax
    K_LIST = [int(s) for s in STEP_GRID]

    def curve(name, seed, field):
        c = all_rows.get(name, {}).get(seed, {})
        out = {}
        for k, v in c.items():
            try:
                out[int(k)] = float(v.get(field, float("nan")))
            except (TypeError, ValueError):
                out[int(k)] = float("nan")
        return out

    def mean_curve(name, field):
        out = {}
        for K in K_LIST:
            vals = []
            for sd in seeds:
                v = curve(name, sd, field).get(K, float("nan"))
                if not math.isnan(v):
                    vals.append(v)
            if vals:
                out[K] = float(np.mean(vals))
        return out

    def paired_curves(a, b, field):
        """配对单位 = (种子, 步点) ⇒ n = 种子数 × 步点数。"""
        A, B = [], []
        for sd in seeds:
            ca, cb = curve(a, sd, field), curve(b, sd, field)
            for K in K_LIST:
                if K in ca and K in cb and not math.isnan(ca[K]) and not math.isnan(cb[K]):
                    A.append(ca[K])
                    B.append(cb[K])
        if len(A) < 2:
            return {"mean": float("nan"), "t": float("nan"), "p": 1.0, "pos": 0,
                    "n": len(A), "consistent": False}
        return paired(A, B)

    def seed_delta_curves(a, b, field):
        out = []
        for sd in seeds:
            ca, cb = curve(a, sd, field), curve(b, sd, field)
            ds = [cb[K] - ca[K] for K in K_LIST if K in ca and K in cb
                  and not math.isnan(ca[K]) and not math.isnan(cb[K])]
            if ds:
                out.append(float(np.mean(ds)))
        return out

    def norm_curve(name, field="fid_mean"):
        """R(step) = field(step) / field(0)（**逐种子**归一后再跨种子平均）。"""
        per_seed = []
        for sd in seeds:
            c = curve(name, sd, field)
            if 0 not in c or math.isnan(c[0]) or abs(c[0]) < 1e-12:
                continue
            per_seed.append({K: c[K] / c[0] for K in K_LIST if K in c and not math.isnan(c[K])})
        out = {}
        for K in K_LIST:
            vals = [p[K] for p in per_seed if K in p]
            if vals:
                out[K] = float(np.mean(vals))
        return out

    def curve_absdiff(a, b, field="fid_mean"):
        ca, cb = norm_curve(a, field), norm_curve(b, field)
        ks = [K for K in K_LIST if K in ca and K in cb]
        if not ks:
            return float("nan"), float("nan"), 0
        d = [abs(ca[K] - cb[K]) for K in ks]
        return float(np.mean(d)), float(np.max(d)), len(ks)

    def half_life(name, field="fid_mean"):
        """首个 R(step) ≤ 0.5 的 step（跨种子中位数）；未掉到 ⇒ 记为**删失**（超出扫描范围）。"""
        ks, censored = [], 0
        for sd in seeds:
            c = curve(name, sd, field)
            if 0 not in c or math.isnan(c[0]) or abs(c[0]) < 1e-12:
                continue
            R = {K: c[K] / c[0] for K in K_LIST if K in c and not math.isnan(c[K])}
            kk = None
            for K in K_LIST:
                if K in R and R[K] <= 0.5:
                    kk = K
                    break
            if kk is None:
                kk = K_LIST[-1]
                censored += 1
            ks.append(float(kk))
        if not ks:
            return float("nan"), censored
        return float(np.median(ks)), censored

    def capacity_knee(name):
        """★ 容量峰值 K*：跨种子平均的 `n_ret` 随**文档数**的 argmax（返回 (K*, 峰值)）。"""
        per_seed = []
        for sd in seeds:
            c = all_rows.get(name, {}).get(sd, {})
            pts = []
            for k, v in c.items():
                try:
                    pts.append((int(v.get("n_docs", 0)), float(v.get("n_ret", 0))))
                except (TypeError, ValueError):
                    continue
            if pts:
                pts.sort()
                per_seed.append(dict(pts))
        if not per_seed:
            return float("nan"), float("nan")
        docs = sorted({d for p in per_seed for d in p})
        mean_by_doc = {}
        for D in docs:
            vals = [p[D] for p in per_seed if D in p]
            mean_by_doc[D] = float(np.mean(vals))
        Dbest = max(mean_by_doc, key=lambda k: mean_by_doc[k])
        return float(Dbest), float(mean_by_doc[Dbest])

    # —— H21-1〔粒度不可互换：任务层〕——
    dd = paired_curves("G_tok", "G_seg4", "nll_tgt")
    sd21 = seed_delta_curves("G_tok", "G_seg4", "nll_tgt")
    _ok1 = (not math.isnan(dd["mean"])) and abs(dd["mean"]) > 0.5 and dd["p"] < 0.05 \
        and (all(x > 0 for x in sd21) or all(x < 0 for x in sd21))
    rec("H21-1 〔(a) 粒度不可互换 · 任务层〕G_tok（token 级）相对 G_seg4（段级）有显著差异",
        _ok1,
        f"|Δ(Δnll)|={abs(dd['mean']):.6f}（阈值 > 0.5，方向不限） t={dd['t']:.2f} "
        f"p={dd['p']:.3g} pos={dd['pos']}/{dd['n']} 逐种子={['%+.4f' % x for x in sd21]}",
        tol=0.5, value=dd["mean"])

    # —— H21-1b〔容量：粒度改变容量峰值位置〕——
    _k_tok, _v_tok = capacity_knee("G_tok")
    _k_doc, _v_doc = capacity_knee("G_doc")
    _ok1b = (not math.isnan(_k_tok)) and (not math.isnan(_k_doc)) and _k_tok > 0 \
        and _k_doc >= 8.0 * _k_tok
    rec("H21-1b 〔(a) 容量 · 峰值位置随粒度移动〕G_doc 的容量峰值文档数 ≥ 8× G_tok 的",
        _ok1b,
        f"K*(G_tok)={_k_tok:.0f} 篇（峰值 {_v_tok:.1f} 条）· K*(G_doc)={_k_doc:.0f} 篇"
        f"（峰值 {_v_doc:.1f} 条）⇒ 比值={(_k_doc/_k_tok if _k_tok else float('nan')):.1f}×"
        f"（阈值 ≥ 8×；解析预测 ≈ N_ENTRIES/m）",
        tol=8.0, value=(_k_doc / _k_tok if _k_tok else float("nan")))

    # —— H21-2〔衰减只认步数：同 m、不同调度〕——
    _mn, _mx, _nk = curve_absdiff("G_seg4", "G_seg4_alt")
    _ok2 = (not math.isnan(_mn)) and _mn <= 0.05 and _mx <= 0.10
    rec("H21-2 〔(b) 衰减只认步数〕G_seg4 与 G_seg4_alt（同 m=同一步数、不同触发调度）的归一化内容保持曲线一致",
        _ok2,
        f"mean|R(seg4)−R(seg4_alt)|={_mn:.4f}（阈值 ≤ 0.05） max={_mx:.4f}（阈值 ≤ 0.10）"
        f" 步点数={_nk}",
        tol=0.05, value=_mn)

    # —— H21-3〔半衰期由步数决定〕——
    _hl = {}
    _cen = {}
    for _a in ("G_tok", "G_seg4", "G_doc"):
        _hl[_a], _cen[_a] = half_life(_a)
    _vals = [v for v in _hl.values() if not math.isnan(v) and v > 0]
    _ratio = (max(_vals) / min(_vals)) if len(_vals) == 3 and min(_vals) > 0 else float("nan")
    _ok3 = (not math.isnan(_ratio)) and _ratio <= 3.0
    rec("H21-3 〔(b) 半衰期由步数决定〕三个粒度的内容半衰期（换算成在线步）彼此相差 ≤ 3×",
        _ok3,
        f"半衰期(step)：" + " · ".join(f"{a}={_hl[a]:.0f}" + (f"(删失{_cen[a]})" if _cen[a] else "")
                                     for a in ("G_tok", "G_seg4", "G_doc"))
        + f" ⇒ max/min={_ratio:.2f}×（阈值 ≤ 3；解析视界 1/(ηλ)=200 步）",
        tol=3.0, value=_ratio)

    # —— H21-4〔η_S ≡ m·η_W：两个速率是同一个量〕——
    _mn4, _mx4, _nk4 = curve_absdiff("G_tok_rate", "G_seg4")
    _ok4 = (not math.isnan(_mn4)) and _mn4 <= 0.05 and _mx4 <= 0.10
    rec("H21-4 〔(d) η_S ≡ m·η_W〕G_tok_rate（m 大、η=η/17）与 G_seg4（m 小、η）的归一化保持曲线一致",
        _ok4,
        f"mean|R(tok_rate)−R(seg4)|={_mn4:.4f}（阈值 ≤ 0.05） max={_mx4:.4f}（阈值 ≤ 0.10）"
        f" 步点数={_nk4}（两臂每篇累积 (ηλ) 相同 ⇒ 若曲线一致 ⇒ 「累积步数×速率」是唯一控制量）",
        tol=0.05, value=_mn4)

    # —— 报告项 ——
    _mtok = mean_curve("G_tok", "hit_mean")
    rec("Rp-1 〔报告项〕各臂的保持曲线（跨种子平均）", True,
        "；".join(f"{a}: " + " ".join(f"{s}:hit={mean_curve(a, 'hit_mean').get(s, float('nan')):.3f}"
                                    f"/fid={mean_curve(a, 'fid_mean').get(s, float('nan')):.3f}"
                                    f"/nll={mean_curve(a, 'nll_tgt').get(s, float('nan')):.2f}"
                                    for s in (0, 128, 512, 2048, 4096))
              for a, _, _, _ in ARMS21))
    rec("Rp-2 〔报告项〕各臂的容量峰值 K* 与半衰期", True,
        "；".join(f"{a}: K*={capacity_knee(a)[0]:.0f}篇(峰{capacity_knee(a)[1]:.1f}条)"
                  f" 半衰期={half_life(a)[0]:.0f}步" for a, _, _, _ in ARMS21))
    def _m_of(name):
        for sd in seeds:
            c = all_rows.get(name, {}).get(sd, {})
            if c:
                return c[str(K_LIST[0])].get("n_steps_doc", float("nan"))
        return float("nan")
    rec("Rp-3 〔报告项〕各臂每篇的写规则触发次数 m 与累积 (ηλ)", True,
        "；".join(f"{a}: m={_m_of(a)}" for a, _, _, _ in ARMS21))
    rec("Rp-4 〔报告项〕训练态在各臂间共享（读出映射相同）", True,
        "所有臂在同一种子下**共用同一训练态快照**（`restore_all`）⇒ 读出映射 (W_o, γ) 逐位相同"
        "⇒ 臂间唯一差异 = 写步调度与写侧 η")

    print("\n【DIR-21 汇总】", flush=True)
    for (name, eta, stride, pick) in ARMS21:
        _h = mean_curve(name, "hit_mean")
        _f = mean_curve(name, "fid_mean")
        _n = mean_curve(name, "nll_tgt")
        _kk, _vv = capacity_knee(name)
        _hlv, _cenv = half_life(name)
        print(f"  {name:14s} m={_m_of(name):>4} nll(step0)={_n.get(0, float('nan')):.3f} "
              f"hit(0/512/4096)={_h.get(0, float('nan')):.3f}/{_h.get(512, float('nan')):.3f}"
              f"/{_h.get(4096, float('nan')):.3f} "
              f"fid(0/512/4096)={_f.get(0, float('nan')):.3f}/{_f.get(512, float('nan')):.3f}"
              f"/{_f.get(4096, float('nan')):.3f} K*={_kk:.0f} 半衰期={_hlv:.0f}步", flush=True)
