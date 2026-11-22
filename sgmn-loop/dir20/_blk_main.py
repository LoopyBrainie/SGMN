    # ---------------- 主循环（★★ DIR-20：两段式会话协议 · **按臂**推进）
    #   ★★【缺陷 D96】交接包的版本是「**先跑完所有臂的会话 A，再跑所有臂的会话 B**」
    #   ⇒ ⇒ 磁盘上**同时**存在 11 臂 × 3 种子 × 60 篇 = 1980 个状态文件（≈ 9.2 MB/个 ⇒ **≈ 18 GB**）
    #   ⇒ ⇒ 实验卷（25 GB，已用 3.3 GB）**写满** ⇒ 正式轮两次崩在 `torch.save`
    #   （`RuntimeError: unexpected pos 18353792 vs 18353684`）⇒ 失败轮日志留档。
    #   ⇒ ⇒ 修法：改成**按臂推进**（该臂的会话 A 全部种子 → 该臂的会话 B → 立刻删该臂的状态）
    #   ⇒ ⇒ **协议语义完全不变**（`X5_wrong_session` 只需要**同一臂**内另一个种子的状态 ⇒ 仍在同臂内可得）
    #   ⇒ ⇒ 磁盘峰值 ≈ 3 种子 × 60 篇 × 9.2 MB ≈ **1.7 GB**。
    print("\n[主循环]", flush=True)
    print(f"\n[规格] smoke={args.smoke}  n_train={n_train}  n_eval={n_eval}  seeds={seeds}  臂数={len(ARMS)}",
          flush=True)
    _tag = "smoke" if args.smoke else "full"
    all_rows = _load_incremental(_tag)  # ★ 续跑：跳过已完成的 (臂, 种子) 组合
    if all_rows:
        _done = sum(len(v) for v in all_rows.values())
        print(f"\n[主循环] ★ 复用增量结果：{len(all_rows)} 臂 / {_done} 个 (臂,种子) 组合已完成 ⇒ 跳过",
              flush=True)
    STATE_DIR = HERE.parent / ".ops" / "state"      # ★ 纪律 #37：按方向命名空间隔离
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # ★★ 缺陷 D95（卡片 rev 1.6 §0.15）：**每轮开始时清空遗留状态文件** ⇒
    #   崩溃的轮次走不到「臂末尾删除」⇒ 文件累积（实测 963 个 / 22 GB ⇒ 磁盘满 ⇒ 正式轮崩）。
    #   ⇒ 全部可再生（续跑时未完成的臂会在会话 A 重新导出）⇒ 直接清空。
    _n_stale = 0
    for _f in STATE_DIR.glob("DIR20_*.pt"):
        try:
            _f.unlink()
            _n_stale += 1
        except OSError:
            pass
    if _n_stale:
        print(f"[主循环] ★ 清空遗留状态文件 {_n_stale} 个（缺陷 D95）", flush=True)

    def _load_model():
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
        for q in m.parameters():
            q.requires_grad_(False)
        return m

    def _drop_model():
        gc.collect()
        torch.cuda.empty_cache()

    def _spath(arm, seed, i):
        return STATE_DIR / f"DIR20_{arm}_{seed}_{i}.pt"

    def _free_mb():
        return shutil.disk_usage(HERE.parent).free / (1024 * 1024)

    _smeta = {}          # (arm, seed) -> {doc_idx: {...}}
    _srows = {}          # (arm, seed) -> rows（`same` 臂在会话 A 内直接只读读出）
    _keepmodel = {}      # (arm, seed) -> modelA（`same_empty` 臂保留会话 A 的实例）

    # ══════════ 按臂推进：会话 A（该臂全部种子）→ 会话 B（该臂）→ 删状态 ══════════
    for (name, mkind, aeta, kmode, variant, export_mode, session_mode) in ARMS:
        # ══════════ 阶段 1：会话 A（写 + 训练 + 导出状态）══════════
        #   ★ 每个臂的**所有种子**的会话 A 先跑完（状态全部落盘）⇒ ⇒ `X5_wrong_session`
        #     才读得到「另一个种子的会话」的状态（卡片 rev 1.1 §2）。
        for seed in seeds:
            if name in all_rows and seed in all_rows[name]:
                _rr = all_rows[name][seed]
                print(f"  seed={seed} {name:22s} (增量复用) nll="
                      f"{np.mean([_num(r['nll']) for r in _rr]):.4f}", flush=True)
                continue
            tr, ev = build_docs(tok, seed, n_train, n_eval)
            ctrl_ids = build_ctrl_ids(tok, seed, tr, ev)
            modelA = _load_model()
            st = make_stack(modelA, d, "cue", seed=seed, mem_kind=mkind, eta=aeta,
                            key_mode=kmode, variant=variant)
            # —— 训练阶段（与 DIR-19 的 N1_frozen_base **逐字同口径**）——
            traj, rep = train_arm(st, tr, "cue", variant=variant)
            snap1 = st.snapshot_all()
            meta, rows = {}, []
            for i, dd in enumerate(ev):
                st.unfreeze_write()
                st.restore_all(snap1)          # ★ 每篇从「训练后的干净底」出发（同 DIR-19 评测协议）
                write_both(st, dd)
                ts = None
                for li, m in st.mems.items():
                    s_ = int(m.slot_of_write[int(dd["true_idx"])])
                    ts = s_ if ts is None else ts
                dd["true_slot"] = int(ts)
                meta[i] = {"true_slot": int(ts)}      # ★ D85-a：**所有**臂都记录真值槽
                if export_mode != "none":
                    state = st.export_state(include_read_map=(export_mode != "no_readmap"),
                                            include_addr=(export_mode != "no_addr"))
                    fp = SGS.state_fingerprint(state)
                    sp = _spath(name, seed, i)
                    torch.save(state, sp)
                    meta[i].update({"fp": fp,
                                     "md5": hashlib.md5(open(sp, "rb").read()).hexdigest(),
                                     "path": str(sp)})
                    del state
                if session_mode == "same":
                    # ★ 同会话臂：写完**立刻**只读读出（不导出、不重建）⇒ 同会话基线
                    rows.append(eval_arm_readonly(st, dd, ctrl_ids))
            st.remove()
            del st
            _smeta[(name, seed)] = meta
            _srows[(name, seed)] = rows
            # ★★ 缺陷 D90（卡片 rev 1.5 §0.10）：`_keepmodel` 按 **(臂, 种子)** 索引；
            #   只有 `same_empty`（X0 = 同实例 + 空栈）**需要**保留会话 A 的模型实例 ——
            #   `same`（X10）的读出已在会话 A 内就地完成 ⇒ **不保留** ⇒ ★ 省显存。
            if session_mode == "same_empty":
                _keepmodel[(name, seed)] = modelA
            else:
                del modelA
                _drop_model()
            _m0 = next((v.get("md5") for v in meta.values() if v.get("md5")), None)
            print(f"  [会话A] seed={seed} {name:22s} 状态 {len(meta)} 份"
                  + (f"  md5[0]={_m0[:10]}" if _m0 else "  （本臂不导出状态）")
                  + f"  磁盘余 {_free_mb():.0f} MiB", flush=True)

        # ══════════ 阶段 2：会话 B（跨会话 / 只读）══════════
        for seed in seeds:
            if name in all_rows and seed in all_rows[name]:
                continue
            tr, ev = build_docs(tok, seed, n_train, n_eval)
            ctrl_ids = build_ctrl_ids(tok, seed, tr, ev)
            # ★★ 缺陷 D85（卡片 rev 1.2 §0.2）：会话 B **重建**的 ev 文档没有 `true_slot`
            #   ⇒ `eval_arm_readonly` 会 KeyError ⇒ ⇒ 修法：从会话 A 的 meta 取回**精确槽号**
            #   （会话归属的对照臂 `X5_wrong_session` 也用它 ⇒ 槽号必须逐篇精确）。
            _mm = _smeta.get((name, seed), {})
            for _i, _dd in enumerate(ev):
                if _i in _mm:
                    _dd["true_slot"] = int(_mm[_i]["true_slot"])
            if session_mode == "same":
                rows = _srows[(name, seed)]
            elif session_mode == "same_empty":
                modelA = _keepmodel.pop((name, seed), None)
                if modelA is None:
                    raise RuntimeError(f"★ 会话 A 的模型实例丢失（臂={name}, 种子={seed}）")
                st = make_stack(modelA, d, "cue", seed=seed, mem_kind=mkind, eta=aeta,
                                key_mode=kmode, variant=variant)
                rows = [eval_arm_readonly(st, dd, ctrl_ids) for dd in ev]
                st.remove()
                del st
                del modelA
                _drop_model()
            else:
                # ★★ 会话 B：**重建模型实例**（from_pretrained ⇒ 隐状态历史清零）
                modelB = _load_model()
                stB = None
                rows = []
                _srcseed = seed
                if session_mode == "cross_wrongseed":
                    _srcseed = seeds[(seeds.index(seed) + 1) % len(seeds)]   # ★ 导入**另一个种子**的状态
                for i, dd in enumerate(ev):
                    if stB is None:
                        stB = make_stack(modelB, d, "cue", seed=seed, mem_kind=mkind, eta=aeta,
                                         key_mode=kmode, variant=variant)
                    if session_mode in ("cross", "cross_wrongseed"):
                        sp = _spath(name, _srcseed, i)
                        state = torch.load(sp, map_location="cpu")
                        stB.import_state(state, strict=True, expect_key_mode=kmode)
                        del state
                    rows.append(eval_arm_readonly(stB, dd, ctrl_ids))
                if stB is not None:
                    stB.remove()
                    del stB
                del modelB
                _drop_model()
            all_rows.setdefault(name, {})[seed] = rows
            # ★ 增量落盘（每臂）：中断不丢已完成臂
            try:
                _incf = HERE.parent / ".ops" / f"formal_incremental_{_DIR_TAG}_{_tag}.json"   # ★ D78
                _incf.parent.mkdir(parents=True, exist_ok=True)
                with open(_incf, "w") as _f:
                    json.dump({"rows": _jsonable(all_rows)}, _f, indent=1, ensure_ascii=False)
            except Exception as _e:
                print(f"[增量落盘失败] {_e}", flush=True)
            print(f"  seed={seed} {name:22s} nll={np.mean([r['nll'] for r in rows]):.4f} "
                  f"lp={np.mean([r['lp'] for r in rows]):+.4f} "
                  f"hit={np.nanmean([r.get('hit', np.nan) for r in rows]):.4f} "
                  f"addr={np.nanmean([r.get('addr', np.nan) for r in rows]):.4f} "
                  f"addr_slot={np.nanmean([r.get('addr_slot', np.nan) for r in rows]):.4f} "
                  f"fid_true={np.nanmean([r.get('fid_true', np.nan) for r in rows]):.4f} "
                  f"cos_y_vtrue={np.nanmean([r.get('cos_y_vtrue', np.nan) for r in rows]):.4f}"
                  f"  磁盘余 {_free_mb():.0f} MiB", flush=True)

        # ★ 状态文件用完即删（磁盘可控 ⇒ 只保留 md5 / 指纹记录）
        if export_mode != "none":
            for seed in seeds:
                for i in range(n_eval):
                    try:
                        os.remove(_spath(name, seed, i))
                    except OSError:
                        pass
            print(f"  [清理] 臂 {name} 的状态文件已删除；磁盘余 {_free_mb():.0f} MiB", flush=True)

