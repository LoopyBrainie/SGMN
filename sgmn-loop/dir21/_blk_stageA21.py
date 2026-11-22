    # ══════════════════ ★★【DIR-21】Stage A 新增断言 A21-1…A21-9（A21-10 = 继承 DIR-20 的 44 项）══════
    # （卡片 rev 1.1 §4 冻结；★ **判据阈值与文字一字未改**）
    print("\n[Stage A · DIR-21]", flush=True)

    # —— 造一篇与 DIR-15…20 同构的评测文档（只需结构，不必是语料真样本）——
    _ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    _L = 69

    def _mk(stride, pick, eta=None):
        _s = make_stack(model, d, "cue", seed=20261028, mem_kind="hebbian_full",
                        eta=(eta if eta is not None else ETA), key_mode="ortho",
                        variant="normal")
        for _m in _s.mems.values():
            _m.write_stride = int(stride)
            _m.stride_pick = str(pick)
        return _s

    def _doc_ids():
        # 一篇「长度 69 的文档」：前 68 个是填充，最后一个是码 token
        _g = torch.Generator(device="cpu"); _g.manual_seed(999)
        _body = torch.randint(100, 5000, (_L - 1,), generator=_g).tolist()
        return _body + [7777]

    _did = _doc_ids()
    _pairs = {}
    _wcs = {}
    _counts = {}
    for (_nm, _S) in (("S1", 1), ("S17", 17), ("S68", 68)):
        _st = _mk(_S, "last")
        _trig = list(_st.mems[list(_st.mems)[0]].trigger_positions(_L))
        _c0 = int(_st.mems[list(_st.mems)[0]].core.count)
        _wc0 = int(_st.mems[list(_st.mems)[0]].write_calls)
        with torch.no_grad():
            _st.model(input_ids=torch.tensor([_did], device=DEV))
        _m = _st.mems[list(_st.mems)[0]]
        _pairs[_nm] = {int(t): (m_k.detach().clone(), m_v.detach().clone())
                       for t, m_k, m_v in zip(_trig, _m.keys_written, _m.vals_written)}
        _wcs[_nm] = int(_m.write_calls) - _wc0
        _counts[_nm] = int(_m.core.count) - _c0
        _st.remove()
        del _st
    gc.collect()
    torch.cuda.empty_cache()

    # —— A21-1：同一 token 位置在任何粒度下 (k, v) 逐位相同 ——
    _common = sorted(set(_pairs["S1"]) & set(_pairs["S17"]) & set(_pairs["S68"]))
    _ok1 = len(_common) > 0
    _bad = []
    for _t in _common:
        _k1, _v1 = _pairs["S1"][_t]
        for _o in ("S17", "S68"):
            _k2, _v2 = _pairs[_o][_t]
            if not (torch.equal(_k1, _k2) and torch.equal(_v1, _v2)):
                _bad.append((_t, _o))
    _ok1 = _ok1 and not _bad
    rec("A21-1〔同一 token 位置在任何粒度下 (k,v) 逐位相同 ⇒ 唯一自变量=写步调度〕", _ok1,
        f"共同触发位置 {len(_common)} 个（如 { _common[:6] }）；不一致 {len(_bad)} 处"
        + (f"：{_bad[:5]}" if _bad else ""),
        tol=0.0, value=float(len(_bad)))

    # —— A21-2：写步数精确（**一次前视** = 68 / 4 / 1 次触发；rev 1.2 §0.2 的「每篇 m」
    #   = 2 × 这个数：一次 `write_both` = **两次前视** = 目标 + 干扰）——
    _exp = {"S1": 68, "S17": 4, "S68": 1}
    _ok2 = all(_wcs[_k] == _exp[_k] for _k in _exp)
    rec("A21-2〔写步数精确：每篇前视的触发数〕", _ok2,
        f"实测 {dict(_wcs)}；期望 {dict(_exp)}（stride 1/17/68 ⇒ **每篇前视** 68/4/1 次；"
        f"每篇累积 m = 2× ⇒ 136/8/2，rev 1.2 §0.2）",
        tol=0.0, value=float(sum(abs(_wcs[_k] - _exp[_k]) for _k in _exp)))

    # —— A21-3：段边界确定性（同一 token 序列 ⇒ 同一分段）——
    _st = _mk(17, "last")
    _tr1 = list(_st.mems[list(_st.mems)[0]].trigger_positions(_L))
    _tr2 = list(_st.mems[list(_st.mems)[0]].trigger_positions(_L))
    _ok3 = (_tr1 == _tr2) and (_tr1 == [17, 34, 51, 68])
    rec("A21-3〔段边界确定性：同一 token 序列 ⇒ 同一分段 ⇒ 触发位置可复现〕", _ok3,
        f"stride=17 ⇒ 触发位置 {_tr1}（两次调用一致：{_tr1 == _tr2}）",
        tol=0.0, value=float(len(_tr1)))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # —— A21-4：解析视界可算 ——
    _horizon = 1.0 / (ETA * LAM)
    _ok4 = _horizon > 0
    rec("A21-4〔解析视界 1/(ηλ) 可从配置算出并记录〕", _ok4,
        f"η={ETA} · λ={LAM} ⇒ 1/(ηλ) = {_horizon:.1f} 步；"
        f"每步衰减因子 (1−ηλ) = {1 - ETA * LAM:.6f}；半衰期 ≈ "
        f"{math.log(0.5) / math.log(1 - ETA * LAM):.1f} 步",
        tol=0.0, value=_horizon)

    # —— A21-5：每次触发恰好一次写更新（core.count 的增量 == 触发数）——
    _ok5 = all(_counts[_k] == _exp[_k] for _k in _exp)
    rec("A21-5〔每次触发恰好一次写更新（core.count 增量 == 触发数）〕", _ok5,
        f"实测 {dict(_counts)}；期望 {dict(_exp)}",
        tol=0.0, value=float(sum(abs(_counts[_k] - _exp[_k]) for _k in _exp)))

    # —— A21-6：归一化后内容可写（写保真 cos(Wk,v) ≥ 0.95）——
    # ★ rev 1.2 §0.4 修法：Stage A 用 **ETA_BASE=1e−1**（`ortho` 可用的强学习率）造核，
    #   否则用全局 ETA=1e−5 ⇒ W≈W₀（随机）⇒ cos(Wk,v)≈0 ⇒ **会把「没写完」误报成「写不可写」**。
    _st = _mk(1, "last", eta=ETA_BASE)
    with torch.no_grad():
        _st.model(input_ids=torch.tensor([_did], device=DEV))
    _m = _st.mems[list(_st.mems)[0]]
    _fids = []
    for _i in range(min(int(_m.core.count), int(_m.core.N))):
        try:
            _fids.append(float(_m.core.fidelity(_i)))
        except Exception:
            pass
    _fid_min = float(np.nanmin(_fids)) if _fids else float("nan")
    _ok6 = (not math.isnan(_fid_min)) and _fid_min >= 0.95
    rec("A21-6〔归一化后内容可写：写保真 cos(Wk,v) ≥ 0.95〕", _ok6,
        f"逐槽写保真：min={_fid_min:.4f} mean={np.nanmean(_fids) if _fids else float('nan'):.4f} "
        f"（共 {len(_fids)} 槽）",
        tol=0.95, value=_fid_min)
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # —— A21-7：等比对照等价性（η/17 逐位生效 + 仍在稳定上界内）——
    _st = _mk(1, "last", eta=ETA / 17.0)
    _etas = [float(_m.eta) for _m in _st.mems.values()] + \
            [float(_m.core.eta) for _m in _st.mems.values()]
    _ok7 = all(abs(_e - ETA / 17.0) < 1e-15 for _e in _etas)
    rec("A21-7〔等比对照等价性：η′=η/17 逐位生效（层与核都一致）〕", _ok7,
        f"层 η={_etas[:len(_etas)//2]} · 核 η={_etas[len(_etas)//2:]}；期望 {ETA/17.0:.8g}",
        tol=0.0, value=float(abs(max(_etas) - ETA / 17.0)))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # —— A21-8：无残留（基座仍被冻结 + 显存已回收）——
    _allfrozen = all(not q.requires_grad for q in model.parameters())
    _grads = sum(1 for q in model.parameters() if q.grad is not None and
                  float(q.grad.abs().sum()) > 0)
    _mem = int(torch.cuda.memory_allocated())
    _ok8 = _allfrozen and _grads == 0
    rec("A21-8〔无残留：基座仍被冻结 + 无残留梯度〕", _ok8,
        f"全部 requires_grad=False：{_allfrozen}；非零梯度参数数：{_grads}；"
        f"显存 { _mem/2**20:.0f} MiB",
        tol=0.0, value=float(_grads))

    # —— A21-9：只读评测不写 ——
    _st = _mk(1, "last")
    _m = _st.mems[list(_st.mems)[0]]
    with torch.no_grad():
        _st.model(input_ids=torch.tensor([_did], device=DEV))
    _c0 = int(_m.core.count)
    _w0 = int(_m.write_calls)
    _sp = {"prefix": _did[:-1], "code_ids": [_did[-1]], "true_idx": _L - 2,
           "true_slot": _c0 - 1, "dist_code": "0"}
    _r = eval_arm_readonly(_st, _sp, None)
    _c1 = int(_m.core.count)
    _w1 = int(_m.write_calls)
    _ok9 = (_c0 == _c1) and (_w0 == _w1)
    rec("A21-9〔只读评测不写：count / write_calls 不变〕", _ok9,
        f"count {_c0} → {_c1}；write_calls {_w0} → {_w1}；读出 nll={_r.get('nll')}",
        tol=0.0, value=float(abs(_c1 - _c0) + abs(_w1 - _w0)))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    print("[Stage A · DIR-21] 9 项断言完成", flush=True)
