    # ---------------- DIR-21 主循环：训练一次 / 种子 ⇒ 5 个粒度臂**共用训练态** + **步轴**扫描
    print("\n[主循环]", flush=True)
    print(f"\n[规格] smoke={args.smoke}  n_train={n_train}  n_eval={n_eval}  seeds={seeds} "
          f"STEP_GRID={STEP_GRID}", flush=True)
    _tag = "smoke" if args.smoke else "full"
    # ★ DIR-21：主循环结构已改（步轴曲线）⇒ 增量续跑不适用，每轮从头跑。
    all_rows = {}
    STATE_DIR = HERE.parent / ".ops" / "state"      # ★ 纪律 #37/#42：命名空间隔离
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # ★★ 缺陷 D95 的修法：每轮开始清空遗留状态文件（全部可再生）
    _n_stale = 0
    for _f in STATE_DIR.glob("DIR21_*.pt"):
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

    def _set_arm(st, eta, stride, pick):
        """★ 设置该臂的**写侧速率**与**写步粒度**（其余一切不变）。"""
        for m in st.mems.values():
            m.eta = float(eta)
            m.core.eta = float(eta)
            m.write_stride = int(stride)
            m.stride_pick = str(pick)

    def _entry_read_specs(d, st, count_before):
        """★【DIR-21】文档 `d` 的**逐条目读出规格**。

        每个**已写入**条目 = (前缀 = `d['ids'][:t]`, 码 = `d['ids'][t]`, 真值槽)。
        ★ 目标位置 `t = L−1`（码 token）**必在**触发集合里 ⇒ ⇒ 每个粒度下**任务都可作答** ✔
        ★ 真值槽 = `count_before + (该条目在写入序列中的序号)`（**精确槽号映射**，沿用 D54 的做法）✔
        """
        li0 = list(st.mems)[0]
        m = st.mems[li0]
        L = len(d["ids"])
        trig = list(m.trigger_positions(L))       # ★ 目标文档的触发位置（升序）
        specs = []
        for i, t in enumerate(trig):
            specs.append({"prefix": list(d["ids"][:t]),
                          "code_ids": [int(d["ids"][t])],
                          "true_idx": int(t) - 1,
                          "true_slot": int((int(count_before) + i) % int(m.core.N)),
                          "dist_code": d.get("dist_code", "0")})
        return specs

    for seed in seeds:
        if all(arm in all_rows and seed in all_rows[arm] for arm, _, _, _ in ARMS21):
            print(f"  seed={seed} （全部臂已由增量完成 ⇒ 跳过）", flush=True)
            continue
        tr, ev = build_docs(tok, seed, n_train, n_eval)
        ctrl_ids = build_ctrl_ids(tok, seed, tr, ev)
        model = _load_model()
        # ★ 训练：**所有臂共用**（token 级 / η=1e−1 / ortho）⇒ 读出映射跨臂相同 ⇒ 臂间可比
        st = make_stack(model, d, "cue", seed=seed, mem_kind="hebbian_full", eta=ETA,
                        key_mode="ortho", variant="normal")
        traj, rep = train_arm(st, tr, "cue", variant="normal")
        snap = st.snapshot_all()
        print(f"  [训练] seed={seed} 完成（{n_train} 篇）", flush=True)
        for (name, eta, stride, pick) in ARMS21:
            if name in all_rows and seed in all_rows[name]:
                print(f"  seed={seed} {name:14s} （增量复用）", flush=True)
                continue
            # ★★ 复位到训练态（**状态**复位；**读出映射** W_o/γ 保留 ⇒ 训练好的那份）
            st.restore_all(snap)
            _set_arm(st, eta, stride, pick)
            _li0 = list(st.mems)[0]
            _m0 = st.mems[_li0]
            _trig = list(_m0.trigger_positions(len(ev[0]["ids"])))
            _m_doc = 2 * len(_trig)          # ★ 每篇的写规则触发总次数（目标文档 + 干扰文档）
            _count0 = int(_m0.core.count)
            # —— 写文档 1（为了拿它的逐条目读出规格与真值槽）——
            st.unfreeze_write()
            write_both(st, ev[0])
            specs = _entry_read_specs(ev[0], st, _count0)
            _written = 1
            curves = {}
            for _s in STEP_GRID:
                _D = int(_s // _m_doc) + 1            # ★ 步轴 ⇒ 文档数（**跨臂可比**）
                while _written < _D:
                    _written += 1
                    st.unfreeze_write()
                    write_both(st, ev[_written - 1])
                # —— 只读：读**文档 1** 的已写入条目（最老的那篇 ⇒ 年龄 = _s 步）——
                tgt = None
                per_entry = []
                for sp in specs:
                    r = eval_arm_readonly(st, sp, ctrl_ids)
                    per_entry.append(r)
                    if int(sp["true_idx"]) == len(ev[0]["ids"]) - 2:
                        tgt = r          # ★ 目标条目（码 token）⇒ 配对的 NLL 口径
                hits = [float(r.get("hit", float("nan"))) for r in per_entry]
                fids = [float(r.get("fid_true", float("nan"))) for r in per_entry]
                rowsum = {
                    "step": int(_s),
                    "n_docs": int(_written),
                    "nll_tgt": float(tgt["nll"]) if tgt else float("nan"),
                    "nll_mean": float(np.nanmean([float(r["nll"]) for r in per_entry])),
                    "hit_mean": float(np.nanmean(hits)),
                    "fid_mean": float(np.nanmean(fids)),
                    "n_ret": int(round(float(np.nansum(hits)))),
                    "n_entries": int(len(per_entry)),
                    "n_steps_doc": int(_m_doc),
                }
                curves[str(_s)] = rowsum
            all_rows.setdefault(name, {})[seed] = curves
            # ★ 增量落盘（每臂）：中断不丢已完成臂
            try:
                _incf = HERE.parent / ".ops" / f"formal_incremental_{_DIR_TAG}_{_tag}.json"
                _incf.parent.mkdir(parents=True, exist_ok=True)
                with open(_incf, "w") as _f:
                    json.dump({"rows": _jsonable(all_rows)}, _f, indent=1, ensure_ascii=False)
            except Exception as _e:
                print(f"[增量落盘失败] {_e}", flush=True)
            _c1 = curves[str(STEP_GRID[0])]
            _cl = curves[str(STEP_GRID[-1])]
            print(f"  seed={seed} {name:14s} m={_m_doc:4d} | step0: nll={_c1['nll_tgt']:.3f} "
                  f"hit={_c1['hit_mean']:.3f} fid={_c1['fid_mean']:.3f} n_ret={_c1['n_ret']:3d}"
                  f" | step{STEP_GRID[-1]}: nll={_cl['nll_tgt']:.3f} hit={_cl['hit_mean']:.3f} "
                  f"fid={_cl['fid_mean']:.3f} n_ret={_cl['n_ret']:3d}", flush=True)
        st.remove()
        del st
        del model
        _drop_model()
