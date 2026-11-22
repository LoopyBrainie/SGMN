# ══════════════════ ★★【DIR-20】判据 H1″…H7″ + 报告项 R-1…R-5（卡片 rev 1.1 §3 冻结）══════════════════
    # —— 口径（写死，防漂移）——
    #   benefit(arm) = NLL(X11_newinst_noexport) − NLL(arm)   （NLL 口径：越低越好 ⇒ 正 = 有收益）
    #   Δ(Δnll)(A,B) = paired(flat(A), flat(B))["mean"] = mean(nll_B − nll_A) ⇒ **>0 ⇒ A 更好**
    #   （与 DIR-19 的 `paired()` 同义：`paired(A,B)` 返回 B − A）
    MAIN = "X1_full_state"
    BASE = "X11_newinst_noexport"      # ★ 参照臂 = 全新实例 + 无状态（卡片 rev 1.1 §3）
    SAME = "X10_same_session"          # ★ 同会话基线
    CW = "X4_content_wrong"
    WS = "X5_wrong_session"
    VB = "X6_verbatim"
    LIN = "X8_linear"
    RN = "X7_renorm"

    def ben(name):
        if name not in agg or BASE not in agg:
            return float("nan")
        return agg[BASE]["nll"] - agg[name]["nll"]

    def flat(name, field):
        return [r[field] for rs in all_rows.get(name, {}).values() for r in rs if field in r]

    def seed_delta_gen(a, b, field="nll"):
        """逐种子的均值差（= nll_B − nll_A ⇒ >0 ⇒ A 更好）。"""
        out = []
        for sd in seeds:
            ra = all_rows.get(a, {}).get(sd, [])
            rb = all_rows.get(b, {}).get(sd, [])
            n = min(len(ra), len(rb))
            if n:
                out.append(float(np.mean([rb[i]["nll"] - ra[i]["nll"] for i in range(n)])))
        return out

    def crit(hname, label, a, b, thr=0.5):
        """★ 通用判据：**a 相对 b 更优**，Δ(Δnll) > thr、p < 0.05、三种子同向。"""
        if a not in agg or b not in agg:
            rec(hname, False, f"缺少臂 {a} 或 {b}", tol=thr, value=float("nan"))
            return None
        dd = paired(flat(a, "nll"), flat(b, "nll"))
        sd = seed_delta_gen(a, b)
        ok = (dd["mean"] > thr) and (dd["p"] < 0.05) and all(x > 0 for x in sd)
        rec(hname, ok,
            f"Δ(Δnll)={dd['mean']:+.6f}（阈值 > {thr}） t={dd['t']:.2f} p={dd['p']:.3g} "
            f"pos={dd['pos']}/{dd['n']} 逐种子={['%+.4f' % x for x in sd]} "
            f"（nll: {a}={agg[a]['nll']:.4f} vs {b}={agg[b]['nll']:.4f}）",
            tol=thr, value=dd["mean"])
        return dd

    # —— H1″〔跨会话可用〕——
    crit("H1″ 〔跨会话可用〕X1_full_state 相对 X11_newinst_noexport（同一「新实例」，唯一差别=有没有状态）有显著收益",
         "", MAIN, BASE, 0.5)
    # —— H2″〔内容特异 · 记忆内对照〕——
    crit("H2″ 〔★ 内容特异 · 记忆内对照〕X1_full_state 显著优于 X4_content_wrong（只差内容，地址逐位相同）",
         "", MAIN, CW, 0.5)
    # —— H3″〔跨会话折损有界〕——
    if MAIN in agg and SAME in agg:
        _loss = float(agg[MAIN]["nll"] - agg[SAME]["nll"])     # ★ 折损 = nll_X1 − nll_X10
        _lsd = []
        for sd in seeds:
            ra = all_rows.get(MAIN, {}).get(sd, [])
            rb = all_rows.get(SAME, {}).get(sd, [])
            if ra and rb:
                _lsd.append(float(np.mean([r["nll"] for r in ra]) - np.mean([r["nll"] for r in rb])))
        ok3 = (_loss <= 1.0) and all(x <= 1.0 for x in _lsd)
        rec("H3″ 〔★ 跨会话折损有界〕X1_full_state 相对 X10_same_session 的收益折损 ≤ 事先声明的界 1.0 nats",
            ok3, f"折损 = nll(X1) − nll(X10) = {agg[MAIN]['nll']:.4f} − {agg[SAME]['nll']:.4f} "
                 f"= {_loss:+.4f} nats（阈值 ≤ 1.0）逐种子={['%+.4f' % x for x in _lsd]}；"
                 f"benefit(X1)={ben(MAIN):+.4f}，benefit(X10)={ben(SAME):+.4f}",
            tol=1.0, value=_loss)
    else:
        rec("H3″ 〔跨会话折损有界〕", False, f"缺少臂 {MAIN} 或 {SAME}", tol=1.0, value=float("nan"))
    # —— H4″〔键尺度归一 ⇒ 跨会话寻址稳定〕——
    crit("H4″ 〔★ 键尺度归一 ⇒ 跨会话寻址稳定〕X1_full_state(ortho) 显著优于 X8_linear（原尺度）",
         "", MAIN, LIN, 0.5)
    # —— H5″〔会话归属〕——
    crit("H5″ 〔★ 会话归属〕X1_full_state 显著优于 X5_wrong_session（导入另一个种子的状态）",
         "", MAIN, WS, 0.5)
    # —— H6″〔零效度门槛〕——
    crit("H6″ 〔★ 零效度门槛〕X6_verbatim（按构造内容正确）的跨会话收益显著 > 0（参照 X11）",
         "", VB, BASE, 0.5)
    # —— H7″〔状态成分的必要性〕——
    _h7 = []
    for _nm2 in ("X2_no_readmap", "X3_no_addr"):
        if _nm2 in agg and MAIN in agg:
            _d7 = paired(flat(MAIN, "nll"), flat(_nm2, "nll"))
            _s7 = seed_delta_gen(MAIN, _nm2)
            _o7 = (_d7["mean"] > 0.5) and (_d7["p"] < 0.05) and all(x > 0 for x in _s7)
            _h7.append((_nm2, _d7, _s7, _o7))
    ok7 = any(x[3] for x in _h7)
    rec("H7″ 〔状态成分的必要性〕至少一个成分缺失（X2_no_readmap / X3_no_addr）显著降低跨会话收益",
        ok7, ("；".join(f"{n2}: Δ(Δnll)={d2['mean']:+.4f}(t={d2['t']:.2f},p≈{d2['p']:.3g},n={d2['n']}) "
                       f"逐种子={['%+.4f' % x for x in s2]} ⇒ {'PASS' if o2 else 'no'}"
                       for n2, d2, s2, o2 in _h7) if _h7 else "（无可用读数）"),
        tol=0.5, value=float(min([d2["mean"] for _n2, d2, _s2, _o2 in _h7]) if _h7 else float("nan")))

    # ══════════════════ ★★ 报告项 R-1…R-5（不进入 PASS/FAIL）══════════════════
    def _g(name, field):
        v = agg.get(name, {}).get(field)
        return float("nan") if v is None else float(v)

    rec("R-1 〔报告项〕X1_full_state 跨会话后的寻址/内容读数", True,
        f"hit={_g(MAIN, 'hit'):.4f} addr_slot={_g(MAIN, 'addr_slot'):.4f} "
        f"fid_true={_g(MAIN, 'fid_true'):.4f} cos_y_vtrue={_g(MAIN, 'cos_y_vtrue'):.4f} "
        f"（★ 寻址是否幸存 + 是否取回内容）", tol=0.0, value=_g(MAIN, 'hit'))
    rec("R-2 〔报告项〕X8_linear（原尺度）的寻址/内容读数", True,
        f"hit={_g(LIN, 'hit'):.4f} addr_slot={_g(LIN, 'addr_slot'):.4f} "
        f"fid_true={_g(LIN, 'fid_true'):.4f} cos_y_vtrue={_g(LIN, 'cos_y_vtrue'):.4f}",
        tol=0.0, value=_g(LIN, 'hit'))
    _r3 = _g(BASE, "nll") - _g("X0_no_state", "nll")
    rec("R-3 〔报告项〕X11_newinst_noexport 相对 X0_no_state 的差（「新实例」与「状态搬运」的分解）", True,
        f"nll(X11)={_g(BASE, 'nll'):.4f}  nll(X0)={_g('X0_no_state', 'nll'):.4f}  差={_r3:+.4f} nats "
        f"（★ 基座冻结 ⇒ 参数逐位相同 ⇒ 预期 ≈ 0 ⇒ 检验「重建模型实例本身」是否改变读数）",
        tol=0.0, value=_r3)
    if RN in agg and LIN in agg:
        _dR = paired(flat(RN, "nll"), flat(LIN, "nll"))
        rec("R-4 〔报告项〕X7_renorm vs X8_linear（第二个归一配置的读数）", True,
            f"Δ(Δnll)={_dR['mean']:+.4f} t={_dR['t']:.2f} p≈{_dR['p']:.3g} n={_dR['n']} "
            f"逐种子={['%+.4f' % x for x in seed_delta_gen(RN, LIN)]}",
            tol=0.0, value=_dR["mean"])
    if MAIN in agg and SAME in agg:
        rec("R-5 〔报告项〕X1（跨会话）vs X10（同会话）的寻址读数对照", True,
            f"hit: X1={_g(MAIN, 'hit'):.4f} vs X10={_g(SAME, 'hit'):.4f}；"
            f"addr_slot: X1={_g(MAIN, 'addr_slot'):.4f} vs X10={_g(SAME, 'addr_slot'):.4f}；"
            f"fid_true: X1={_g(MAIN, 'fid_true'):.4f} vs X10={_g(SAME, 'fid_true'):.4f}",
            tol=0.0, value=_g(MAIN, 'addr_slot'))

