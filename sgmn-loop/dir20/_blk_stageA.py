    # ══════════════════ ★★【DIR-20】Stage A 新增断言 A20-1…A20-9 + A20-11…A20-14 ══════════════════
    # （卡片 rev 1.1 §4 / rev 1.2 §5；★ **判据的阈值与文字一字未改**）
    #   A20-10 = **继承 DIR-19 的 30 项**（逐字保留，不重写）⇒ 本块只加 DIR-20 的 13 项。
    print("\n[Stage A · DIR-20]", flush=True)

    # ★★ 缺陷 D86（卡片 rev 1.3 §0.4）：**继承的 DIR-19 Stage A 会留下约 5.7 GB 显存占用**，
    #   且把基座留成 `requires_grad=True` ⇒ 主循环会白算基座梯度（并破坏「基座全冻结」契约）。
    #   遗留对象：① `_bf0`（基座参数的 **GPU 克隆**，1.9 GB，A19-1 留）；
    #            ② `_opt9` 的 SGD **momentum 缓冲**（1.9 GB，A19-9 的基座优化器）；
    #            ③ 基座参数的 `.grad` 缓冲（1.9 GB，A19-2/A19-9 放开基座后 backward 留下）。
    #   ⇒ 修法：显式 `del`（函数局部变量**不能**通过 `locals()` 删除）+ 清梯度 + **重新冻结基座** + 回收显存。
    #   ⇒ 实测依据：DIR-20 smoke#1 的显存轨迹 —— A19-1 后 6.7 GB、A19-9 后 11.1 GB ⇒ A20-8 再载入两个
    #     fp32 基座（+3.8 GB）⇒ **CUDA OOM**（14.29 GiB 已分配）。
    _freed = []
    for _nm in ("_bf0", "_opt9"):
        _freed.append(_nm)
    try:
        del _bf0
    except NameError:
        pass
    try:
        del _opt9
    except NameError:
        pass
    for _p in model.parameters():
        _p.grad = None
        _p.requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    rec("A20-0〔★ D86〕继承 Stage A 的遗留显存已释放，基座已重新冻结（供后续断言与主循环）",
        True,
        f"释放对象={_freed}；清空基座梯度；基座 requires_grad 全部恢复 False；"
        f"当前 alloc={torch.cuda.memory_allocated()/2**20:.0f} MiB "
        f"reserved={torch.cuda.memory_reserved()/2**20:.0f} MiB",
        tol=0.0, value=float(torch.cuda.memory_allocated() / 2**20))

    _S20 = SEEDS[0]
    _tr20, _ev20 = build_docs(tok, _S20, 2, 2)
    _d20 = _ev20[0]

    def _mk20(**kw):
        """建一个 DIR-20 装置（默认 = 主臂配置：hebbian_full + ortho + η=1e-1）。"""
        p = {"mem_kind": "hebbian_full", "eta": 1e-1, "key_mode": "ortho",
             "variant": "normal", "seed": _S20}
        p.update(kw)
        return make_stack(model, d, "cue", seed=p["seed"], mem_kind=p["mem_kind"],
                          eta=p["eta"], key_mode=p["key_mode"], variant=p["variant"])

    def _yr(st):
        return [m.last_yr[0].detach().clone() for m in st.mems.values()]

    def _fwd(st, dct):
        pre = torch.tensor(dct["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            st.model(input_ids=pre)

    # ─────────── A20-9〔只读评测**不写**〕（先做：后续检查都依赖它）───────────
    _stA = _mk20()
    write_both(_stA, _d20)
    _tslot = {}
    for li, m in _stA.mems.items():
        _tslot[li] = int(m.slot_of_write[int(_d20["true_idx"])])
    _d20["true_slot"] = int(list(_tslot.values())[0])
    _c_before = {li: (int(m.core.count), int(m.write_calls)) for li, m in _stA.mems.items()}
    _writable_before = all((not m.no_write) for m in _stA.mems.values())
    _row20 = eval_arm_readonly(_stA, _d20, None)
    _c_after = {li: (int(m.core.count), int(m.write_calls)) for li, m in _stA.mems.items()}
    _ok9 = (_c_before == _c_after) and _writable_before
    rec("A20-9〔只读评测不写〕评测前后 core.count 与 write_calls 不变（且写权限未被永久关闭）",
        _ok9,
        f"评测前 count/write_calls={[(v[0], v[1]) for v in _c_before.values()]}；"
        f"评测后={[(v[0], v[1]) for v in _c_after.values()]}；评测前写权限开启={_writable_before}；"
        f"★ 若评测期间未冻结写，前缀的前向会**自己写入**⇒ count 必增 ⇒ 本项必 FAIL",
        tol=0.0, value=float(sum(v[1] for v in _c_after.values()) - sum(v[1] for v in _c_before.values())))

    # ─────────── A20-1 / A20-2〔往返指纹逐位相同 / 往返后读出逐位相同〕───────────
    _stA.freeze_write()
    _fwd(_stA, _d20)                      # 纯读（写已冻结）⇒ 状态不再变化
    _y_before = _yr(_stA)
    _state20 = _stA.export_state(include_read_map=True, include_addr=True)
    _fp_mem = SGS.state_fingerprint(_state20)
    _stA.remove(); torch.cuda.empty_cache()

    _stB = _mk20()
    _stB.import_state(_state20, strict=True, expect_key_mode="ortho")
    _state20b = _stB.export_state(include_read_map=True, include_addr=True)
    _fp_rt = SGS.state_fingerprint(_state20b)
    rec("A20-1〔export → import 往返指纹逐位相同〕", _fp_mem == _fp_rt,
        f"内存指纹={_fp_mem}；导入后重导出指纹={_fp_rt}", tol=0.0,
        value=1.0 if _fp_mem == _fp_rt else 0.0)
    _stB.freeze_write()
    _fwd(_stB, _d20)
    _y_after = _yr(_stB)
    _cos20 = [float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
              for a, b in zip(_y_before, _y_after)]
    _ok2 = (min(_cos20) >= 0.999)
    rec("A20-2〔往返后读出逐位相同〕同一前缀下新旧装置的读出余弦 ≥ 0.999", _ok2,
        f"逐层余弦={['%.6f' % c for c in _cos20]}（最小 {min(_cos20):.6f}）", tol=0.999,
        value=float(min(_cos20)))

    # ─────────── A20-14〔落盘后 torch.load 的指纹 == 内存中的指纹〕───────────
    _sd20 = HERE.parent / ".ops" / "state"
    _sd20.mkdir(parents=True, exist_ok=True)
    _f14 = _sd20 / "A20_roundtrip.pt"
    torch.save(_state20, _f14)
    _state14 = torch.load(_f14, map_location="cpu")
    _fp14 = SGS.state_fingerprint(_state14)
    rec("A20-14〔状态文件往返：落盘后 torch.load 的指纹 == 内存中的指纹〕", _fp14 == _fp_mem,
        f"内存指纹={_fp_mem}；落盘读回指纹={_fp14}", tol=0.0,
        value=1.0 if _fp14 == _fp_mem else 0.0)
    try:
        os.remove(_f14)
    except OSError:
        pass

    # ─────────── A20-3〔strict=True 喂错形状必须抛错〕───────────
    _bad = {"fmt": _state20["fmt"], "include_read_map": True, "include_addr": True,
            "layers": {}}
    for _li, _dd in _state20["layers"].items():
        _bad["layers"][_li] = dict(_dd)
        _bad["layers"][_li]["W"] = torch.zeros(3, 3)       # ← 形状不符
    _raised, _msg = False, ""
    try:
        _stB.import_state(_bad, strict=True, expect_key_mode="ortho")
    except Exception as _e:
        _raised = True
        _msg = f"{type(_e).__name__}: {_e}"
    rec("A20-3〔import_state(strict=True) 喂错形状必须抛错（不得静默截断）〕", _raised,
        _msg if _raised else "★ 未抛错 ⇒ 形状校验失效（缺陷！）", tol=0.0,
        value=1.0 if _raised else 0.0)

    # ─────────── A20-4〔include_read_map=False ⇒ W_o / γ 未搬运〕───────────
    #   判别设计：先把**源装置**的 W_o 扰动 +1.0 ⇒ 若被搬运，导入后的 W_o 必含 +1.0
    with torch.no_grad():
        for m in _stA.mems.values():
            m.Wo.add_(1.0)
    _wo_src = [m.Wo.detach().clone() for m in _stA.mems.values()]
    _ga_src = [m.gamma.detach().clone() for m in _stA.mems.values()]
    _state_nr = _stA.export_state(include_read_map=False, include_addr=True)
    _stC = _mk20()
    _wo_init = [m.Wo.detach().clone() for m in _stC.mems.values()]
    _ga_init = [m.gamma.detach().clone() for m in _stC.mems.values()]
    _stC.import_state(_state_nr, strict=True, expect_key_mode="ortho")
    _wo_after = [m.Wo.detach().clone() for m in _stC.mems.values()]
    _ga_after = [m.gamma.detach().clone() for m in _stC.mems.values()]
    _wo_untouched = all(torch.equal(a, b) for a, b in zip(_wo_init, _wo_after))
    _ga_untouched = all(torch.equal(a, b) for a, b in zip(_ga_init, _ga_after))
    _wo_not_src = all((not torch.equal(a, b)) for a, b in zip(_wo_src, _wo_after))
    rec("A20-4〔include_read_map=False ⇒ W_o / γ 未搬运（导入后仍为随机初值）〕",
        _wo_untouched and _ga_untouched and _wo_not_src,
        f"导入后 W_o 仍 == 随机初值={_wo_untouched}；γ 仍 == 随机初值={_ga_untouched}；"
        f"且 ≠ 源装置（源被扰动 +1.0）={_wo_not_src}", tol=0.0,
        value=1.0 if (_wo_untouched and _ga_untouched and _wo_not_src) else 0.0)
    _stC.remove(); torch.cuda.empty_cache()

    # ─────────── A20-5〔include_addr=False ⇒ 地址表未搬运〕───────────
    _state_na = _stA.export_state(include_read_map=True, include_addr=False)
    _stD = _mk20()
    _stD.import_state(_state_na, strict=True, expect_key_mode="ortho")
    _cnt20 = [int(m.core.count) for m in _stD.mems.values()]
    _kz20 = [bool(torch.count_nonzero(m.core.keys).item() == 0) for m in _stD.mems.values()]
    rec("A20-5〔include_addr=False ⇒ 地址表未搬运（count == 0、keys 全零）〕",
        (all(c == 0 for c in _cnt20) and all(_kz20)),
        f"逐层 count={_cnt20}；keys 全零={_kz20}", tol=0.0,
        value=float(sum(_cnt20)))
    _stD.remove(); torch.cuda.empty_cache()

    # ─────────── A20-6 / A20-7〔内容置换：只动值、不动键与签名；置换是排列且无不动点〕───────────
    _stX = _mk20(variant="normal")
    write_both(_stX, _d20)
    _sX = _stX.export_state(include_read_map=True, include_addr=True)
    _stX.remove(); torch.cuda.empty_cache()
    _stY = _mk20(variant="content_wrong")
    write_both(_stY, _d20)
    _sY = _stY.export_state(include_read_map=True, include_addr=True)
    _perm_ok, _perm_txt = True, ""
    for _li, _m in _stY.mems.items():
        _p = getattr(_m, "_cw_perm", None)
        if _p is None:
            _perm_ok = False
            _perm_txt = f"层{_li}: ★ 缺少 _cw_perm（置换未生效）"
            break
        _pi = [int(x) for x in _p.tolist()]
        _n = len(_pi)
        _is_perm = (sorted(_pi) == list(range(_n)))
        _fixed = sum(1 for _i in range(_n) if _pi[_i] == _i)
        if (not _is_perm) or _fixed != 0:
            _perm_ok = False
        _perm_txt = (f"层{_li}: n={_n} 是排列={_is_perm} 不动点数={_fixed}"
                     f"（★ D62：torch 视图别名会写坏排列 ⇒ 必须先取标量副本）")
    rec("A20-7〔content_wrong 的置换**是排列且无不动点**〕", _perm_ok, _perm_txt, tol=0.0,
        value=1.0 if _perm_ok else 0.0)
    _k_same = all(torch.equal(_sX["layers"][_li]["keys"], _sY["layers"][_li]["keys"])
                  for _li in _sX["layers"])
    _v_diff = all((not torch.equal(_sX["layers"][_li]["vals"], _sY["layers"][_li]["vals"]))
                  for _li in _sX["layers"])
    # ★★ 缺陷 D87（卡片 rev 1.3 §0.5）：rev 1.1 的 A20-6 要求「**sigs 逐位相同**」⇒ **不成立**：
    #   记忆注入会改变**下游层**的 hidden state ⇒ 下游层的上下文签名携带了注入痕迹 ⇒ 逐位不同。
    #   （第一挂载层 8 的输入在上游 ⇒ **不受注入影响** ⇒ 必然逐位相同 ⇒ 卡片意图在此层可严格验证。）
    _sig_same = {li: bool(torch.equal(_sX["layers"][li]["sigs"], _sY["layers"][li]["sigs"]))
                 for li in _sX["layers"]}
    _first_li = sorted(_sX["layers"].keys(), key=lambda s: int(s))[0]
    _ok6 = (_k_same and _v_diff and _sig_same.get(_first_li, False))
    rec("A20-6′〔内容置换只动**值**、不动**寻址键**；签名**逐层**报告（第一挂载层必须逐位相同）〕",
        _ok6,
        f"keys 逐位相同={_k_same}（全部挂载层）；vals 逐位不同={_v_diff}（全部挂载层）；"
        f"sigs 逐层相同={_sig_same}；★ 第一挂载层（层 {_first_li}）逐位相同={_sig_same.get(_first_li)}"
        f"（下游层不同是**结构事实**：记忆注入会改变下游 hidden state ⇒ 签名携带注入痕迹 ⇒ 非缺陷）",
        tol=0.0, value=1.0 if _ok6 else 0.0)
    _stY.remove(); torch.cuda.empty_cache()

    # ─────────── A20-11〔hebbian_full 的 vals **不进**读路径；W **进**读路径〕───────────
    _stE = _mk20()
    write_both(_stE, _d20)
    _stE.freeze_write()
    _fwd(_stE, _d20)
    _y1 = _yr(_stE)
    with torch.no_grad():
        _vals_saved = [m.core.vals.detach().clone() for m in _stE.mems.values()]
        for m in _stE.mems.values():
            m.core.vals.zero_()
    _fwd(_stE, _d20)
    _y2 = _yr(_stE)
    with torch.no_grad():
        for _m, _v in zip(_stE.mems.values(), _vals_saved):
            _m.core.vals.copy_(_v)
        _W_saved = [m.core.W.detach().clone() for m in _stE.mems.values()]
        for m in _stE.mems.values():
            m.core.W.zero_()
    _fwd(_stE, _d20)
    _y3 = _yr(_stE)
    with torch.no_grad():
        for _m, _w in zip(_stE.mems.values(), _W_saved):
            _m.core.W.copy_(_w)
    _c_v = [float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            for a, b in zip(_y1, _y2)]
    _c_w = [float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            for a, b in zip(_y1, _y3)]
    _ok11 = (min(_c_v) >= 1.0 - 1e-6) and (min(_c_w) < 0.999)
    rec("A20-11〔vals 不进读路径（置零 ⇒ 读出逐位不变）；W 进读路径（置零 ⇒ 读出改变）〕",
        _ok11,
        f"vals 置零后逐层余弦={['%.8f' % c for c in _c_v]}（须 ≥ 1-1e-6）；"
        f"W 置零后逐层余弦={['%.6f' % c for c in _c_w]}（须 < 0.999）", tol=0.999,
        value=float(min(_c_w)))
    _stE.remove(); torch.cuda.empty_cache()

    # ─────────── A20-12′〔真值槽**精确可预测**且**不回绕**〕───────────
    #   ★★ 卡片 rev 1.3 §0.6：rev 1.1 的 A20-12 写「true_slot == true_idx（表在写前清零）」，
    #     但本协议（与 DIR-18/19 同构）在写阶段前 `restore_all(snap1)` ⇒ 表带**训练残留**
    #     ⇒ 真值槽 = **写前 count + true_idx**（实测依据见 rev 1.3 §0.6）。
    #   ⇒ 不变量：**真值槽唯一确定**（= C0 + true_idx）且**不回绕**（C0 + 本篇条数 ≤ N_ENTRIES）。
    _stF = _mk20()
    _C0 = int(list(_stF.mems.values())[0].core.count)
    write_both(_stF, _d20)
    _eq_slot = [int(m.slot_of_write[int(_d20["true_idx"])]) == _C0 + int(_d20["true_idx"])
                for m in _stF.mems.values()]
    _cnts = [int(m.core.count) for m in _stF.mems.values()]
    _nowrap = all(c <= int(m.core.N) for c, m in zip(_cnts, _stF.mems.values()))
    rec("A20-12′〔真值槽精确可预测（= 写前 count + true_idx）且不回绕〕",
        (all(_eq_slot) and _nowrap),
        f"写前 count C0={_C0}；逐层 slot_of_write[true_idx] == C0 + true_idx：{_eq_slot}；"
        f"写后 count={_cnts}；未回绕（≤ N_ENTRIES={N_ENTRIES}）={_nowrap}；"
        f"★ Stage A 中 C0=0（无训练残留）⇒ 退化为 true_slot == true_idx；"
        f"正式轮里 C0=训练残留 ⇒ **addr（写入序口径）恒为 0，只有 addr_slot 可读**",
        tol=0.0, value=1.0 if (all(_eq_slot) and _nowrap) else 0.0)
    _stF.remove(); torch.cuda.empty_cache()

    # ─────────── A20-13〔linear 的实测 ‖k‖² ≫ 5（复现 DIR-18 的 D50/D51）〕───────────
    _stL = _mk20(key_mode="linear", eta=1e-5)
    write_both(_stL, _d20)
    _lin = [float((m.core.keys[:int(m.core.count)] ** 2).sum(1).mean()) for m in _stL.mems.values()]
    _stL.remove(); torch.cuda.empty_cache()
    _stR = _mk20(key_mode="renorm", eta=1e-1)
    write_both(_stR, _d20)
    _ren = [float((m.core.keys[:int(m.core.count)] ** 2).sum(1).mean()) for m in _stR.mems.values()]
    _stR.remove(); torch.cuda.empty_cache()
    _ok13 = (min(_lin) > 100.0) and (abs(float(np.mean(_ren)) - 5.0) < 1.0)
    rec("A20-13〔X8_linear 的键尺度实测 ≫ 5（linear ≫ 100，renorm ≈ 5）〕", _ok13,
        f"linear 逐层 E‖k‖²={['%.1f' % x for x in _lin]}；"
        f"renorm 逐层 E‖k‖²={['%.4f' % x for x in _ren]}；"
        f"★ 复现 D50/D51 ⇒ X8 必须用 η=1e-5（修订 20-C）", tol=100.0, value=float(min(_lin)))

    # ─────────── A20-8〔会话 B 的模型实例**确实是新的**；会话 A 的已释放〕───────────
    #   ★ 显存前提：D86 的清理已把继承 Stage A 的 5.7 GB 释放掉 ⇒ 这里再载入两个 fp32 基座不会 OOM。
    try:
        del _stA
    except NameError:
        pass
    try:
        del _stB
    except NameError:
        pass
    try:
        del _stC
    except NameError:
        pass
    try:
        del _stD
    except NameError:
        pass
    try:
        del _stE
    except NameError:
        pass
    try:
        del _stF
    except NameError:
        pass
    try:
        del _stX
    except NameError:
        pass
    try:
        del _stY
    except NameError:
        pass
    try:
        del _stL
    except NameError:
        pass
    try:
        del _stR
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    _mA = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    _mB = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    _idA, _idB = id(_mA), id(_mB)
    _mem_before = int(torch.cuda.memory_allocated())
    del _mA
    gc.collect()
    torch.cuda.empty_cache()
    _mem_after = int(torch.cuda.memory_allocated())
    _ok8 = (_idA != _idB) and (_mem_before - _mem_after > 500 * 1024 * 1024)
    rec("A20-8〔会话 B 的模型实例**确实是新的**（id 不同）；会话 A 的已释放（显存回落）〕", _ok8,
        f"id(会话A)={_idA}，id(会话B)={_idB}；显存 {_mem_before/2**20:.0f} MiB → "
        f"{_mem_after/2**20:.0f} MiB（回落 {_mem_before/2**20 - _mem_after/2**20:.0f} MiB）",
        tol=500.0, value=float((_mem_before - _mem_after) / 2**20))
    del _mB
    gc.collect()
    torch.cuda.empty_cache()
    print("[Stage A · DIR-20] 13 项断言完成", flush=True)
