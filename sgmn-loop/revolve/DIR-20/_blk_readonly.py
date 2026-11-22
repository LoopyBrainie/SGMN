def eval_arm_readonly(st, d, ctrl_ids=None):
    """★【DIR-20】**只读**评测：只前向一次 ⇒ ⇒ **不写、不训练**。

    ★ 与 DIR-19 的 `eval_arm` 的**唯一**差别：**评测前不写** ⇒ ⇒ 记忆是会话结束时
      **冻结**的状态 ⇒ ⇒ 所有指标都必须从「**导入的状态**」里算出来：
        · `y`（读出）      —— 读阶段前向时由 `m.last_yr` 给出；
        · 真值值 `v_j`    —— 地址表里存的 `core.vals`（`hebbian_full` 的 `vals` 不进读路径，
                             只用于写保真度测量 ⇒ 卡片 rev 1.1 §0.3 / A20-11）；
        · 真值槽          —— 会话 A 写入时记录的 `d["true_slot"]`（精确槽号映射）。
    ★ 只读 ⇒ `core.count` 与 `write_calls` **不变** ⇒ Stage A **A20-9** 当场断言。
    """
    pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
    with torch.no_grad():
        lg = st.model(input_ids=pre).logits.float()
    lp_all = F.log_softmax(lg[0, -1], dim=-1)
    lp = float(lp_all[d["code_ids"][0]])
    true_slot = int(d["true_slot"])
    hits, fids, fids_true, cys, addr_ok, addr_slot_ok = [], [], [], [], [], []
    n_used = 0
    for li, m in st.mems.items():
        yr = m.last_yr[0] if m.last_yr is not None else None
        j_sel = m.last_cue_idx
        # —— D46 口径 ①：写入顺序空间（DIR-20 协议下与槽空间**恒等**，A20-12 断言）
        addr_ok.append(1.0 if (j_sel is not None and int(j_sel) == int(d["true_idx"])) else 0.0)
        # —— D46 口径 ②：槽空间（精确槽号映射）
        addr_slot_ok.append(1.0 if (j_sel is not None and int(j_sel) == true_slot) else 0.0)
        n = min(int(m.core.count), int(m.core.N))
        Vs = m.core.vals.detach()[:n]
        n_used = int(Vs.shape[0])
        if yr is None or Vs.shape[0] == 0:
            hits.append(0.0)
            fids.append(float("nan"))
            fids_true.append(float("nan"))
            cys.append(float("nan"))
            continue
        cs = F.cosine_similarity(Vs, yr.unsqueeze(0), dim=1)
        hits.append(1.0 if int(torch.argmax(cs)) == true_slot else 0.0)
        if hasattr(m.core, "fidelity"):
            fids.append(float(m.core.fidelity(int(j_sel))) if j_sel is not None else float("nan"))
            fids_true.append(float(m.core.fidelity(true_slot)))
        else:
            fids.append(float("nan"))
            fids_true.append(float("nan"))
        cys.append(float(F.cosine_similarity(
            Vs[true_slot].unsqueeze(0), yr.unsqueeze(0), dim=1).item()))
    row = {"lp": lp, "nll": -lp, "nll_code": -lp,
           "hit": float(np.mean(hits)),
           "addr": float(np.mean(addr_ok)) if addr_ok else float("nan"),
           "addr_slot": float(np.mean(addr_slot_ok)) if addr_slot_ok else float("nan"),
           "fid": (float(np.nanmean(fids)) if (len(fids) and not all(math.isnan(f) for f in fids)) else float("nan")),
           "fid_true": (float(np.nanmean(fids_true)) if (len(fids_true) and not all(math.isnan(f) for f in fids_true)) else float("nan")),
           "cos_y_vtrue": (float(np.nanmean(cys)) if (len(cys) and not all(math.isnan(f) for f in cys)) else float("nan")),
           "n_entries": n_used,
           "count": int(st.mems[list(st.mems)[0]].core.count)}
    if ctrl_ids is not None:
        row["nll_ctrl"] = float(-lp_all[ctrl_ids].mean())
    row["nll_dcode"] = float(-lp_all[int(d["dist_code"])])
    return row


