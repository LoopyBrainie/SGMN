# DIR-53 · **证据行 `D-76` 填表骨架**（`EVIDENCE-ROW-SKELETON.md`；**不是证据行**）

> 证据表 `evidence/table.md` **只增行**；写坏只能加勘误行（审计器**不校验列数**）。
> 定论当天只做「抄数 + 套固定措辞」，**每格必须带 `〔来源〕`，当天不得现算**。
> ★★ `SUCCEEDED` **±60 字符**内**只许**出现挂计数本身（`DEF-52-3`：`200/200`、`52/52` 会把审计推导值顶成荒谬值）；
> 填行时**不要**在"结论""判定表"两行写任何 `k/k` 形式。

## 块格式（照抄 `### **D-75**` 的字段顺序）

```
### **D-76** · DIR-53：$P_k$ 的「离线学习」是否有独立收益 —— {结局短语}
```

| 字段 | 内容 | 〔来源〕（当天只抄不算） |
|---|---|---|
| **方向出处** | 框架 §7.2「$P_k,P_v$ 的**固定、离线学习**或其他来源」的其余分支（D-65 授权行；DIR-41 只测了"固定随机"与"上限"） | 卡 rev1.2 §1 |
| **卡片** | `revolve/DIR-53/DIRECTION_rev53_v10.md`（rev 1.0，含 `DEF-53-1`）· `v11.md`（`DEF-53-2`）· `v12.md`（**判据冻结版**，md5 `c5fc757c7e462eb0b85f6844c1caa4c1`） | `DIR53r12_stageA.json: guards.A53-2_card_md5` |
| **主判据量** | `Δ_j = L_FIX,j − L_OFF,j`（同格同初值，仅 `Pk1` 来源不同）· `R_j = L_FIX,j − L_ORACLE,j`（可寻址区间）· `C_j = L_OFF_rc,j − L_OFF,j`（内容置换） | 卡 rev1.2 §2.1/§3.1 |
| **冻结值** | `3se` 门槛（P1）· 地板 `0.20·R_j`（P2）· `N_TRAIN=96`、`N_EVAL=400`、`n_rep=16`、`250 步`、`lr 0.03`、种子段 `train 21000..21095` / `eval 21500..21899`、`≥4/5 格` | `DIR53r12_formal.json: sig_k/frac_headroom/pass_rate_min,n_train,n_eval,n_rep,steps` |
| **装置** | 逐字派生 DIR-41（`dir40.train_arm40/eval_arm40`、`dir37._mk`、`dir33` 任务层、`dir31` 装置）；**唯一自变量 = `Pk1` 来源**；四文件 md5 未变 | `DIR53r12_stageA.json: guards.A53-1_md5` |
| **★ 正式轮读数** | 5 格 × 3 臂 = **15 次头训练**；逐格 `L_FIX/L_OFF/L_OFF_rc/L_ORACLE`、`R_j`、地板、`Δ_j`、`3se(Δ_j)`、`C_j`、`3se(C_j)` | `DIR53r12_adjudication.json: cells.<cell>.{L.*,R,floor,delta,pd.3se,pc.mean,pc.3se}` |
| **★ 判定表（按序）** | ①**有效性闸**（缺臂/`n` 不符/`OFF` 惰臂/同源错/非有限 ⇒ `INCOMPLETE` 或 `DIVERGED`）· ②`R_j ≤ 0` ⇒ `INDETERMINATE` · ③过拟合支（训练段改善但评估段无收益）· ④`P1∧P2∧P3` 逐格 · ⑤`pass_count ≥ 4` ⇒ `PASS` | `DIR53r12_adjudication.json: {missing,bad_n,inert_arms,sha_mismatch,nonfinite,pass_count,outcome,counted}` |
| ★ **机制（非判据）** | ①`P_k` 学到什么：逐格 `rel_dPk`、`row_norm_dev_max`（须 `<1e-12`）、`L_off_first20→last20`；②收益走哪条路：`cos_q_k`、`Wmax` 随 `P_k` 的变化、`C_j` 是否使收益消失；③`R_j` 的量级（本格上限） | `DIR53r12_<cell>_OFF.json: off_meta.*`、`DIR53r12_<cell>_{FIX,ORACLE}.json: arms.*.{cos_q_k,Wmax}` |
| ★ **结论（只能这样说）** | 按骨架 `VERDICT-SCAFFOLD.md §1` 的**六支固定措辞**择一，逐字抄；`PASS` 才计入 `SUCCEEDED` | `DIR53r12_adjudication.json: outcome,counted` |
| **边界（不得外推）** | 卡 §8 五条**原文**（本玩具参数组 / 「离线学习」= 冻结的那个程序 / `OFF` 多 128 个可训参数 ⇒ 只答设计问题 / `P_v` 未测 / `FIXED-SUFFICES` 的合法表述） | 卡 rev1.2 §8 |
| **缺陷与纪律** | `DEF-53-1`（Freeze C ⇒ 任务损失对 `P_k` 梯度恒零）· `DEF-53-2`（rev1.1 维度非法）· `DEF-53-3`（投影广播）· `DEF-53-4`（裁决器层→臂映射错，演练抓到）· `DEF-53-5`（`stdev` 遇 NaN 抛异常）· 纪律候选 123（前提机器化 + 补丁必带 assert） | 本节 + `decisions/DECISIONS.md` 决定九十一/九十二 |

## 交叉核对（填行前必须成立）

1. `DIR53r12_adjudication.json: merge_agreement.agree == true`（含 `outcome_equal`、`counted_equal`、`numeric_diffs == []`）
2. `DIR53r12_adjudication.json: n_cells_read == 5`
3. `DIR53r12_stageA.json: ok == true`（`A53-1..12`）
4. `dir53/out/` 下 `DIR53r12_P*_{FIX,OFF,ORACLE}.json` **恰 15 个**，且每个 `n_train/n_eval/n_rep/steps` 与冻结值一致