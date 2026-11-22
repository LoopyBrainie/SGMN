# DIR-62 · $W^0$ 的具体构造 = **预载通道**：存活谱由闭式代数决定，而**秩预算压在 $W$ 还是压在 $\Delta W$** 是两种不同架构

> **卡片版本**：rev 1.1（**冻结**，2026-11-17）。本版只修 **2 处 Stage A 守卫**（DEF-62-1/2），**§4 的四条判据与阈值（$10^{-9}$/$10^{-6}$）一字未改**；rev 1.0 原文留档于 `DIRECTION_rev62_v10.md`（md5 `f4433ac63ce9815a09c8681276211532`）。
>
> **修订记录（rev 1.0 → 1.1；均因 rev 1.0 的守卫自身不可执行，非因结果不合意）**
> - **DEF-62-1（A62-5 的"$L_A$ 逐位为 0"是假断言）**：$L_A(r{=}1)=\tfrac12\sum_{i>1}\sigma_i^2(\bar MR)$ 中 $\sigma_5,\sigma_6\sim10^{-17}$ 是 SVD 的**数值零**，平方后约 $10^{-34}$，故 $L_A$ **不会**逐位等于 0。实测 `L_A = 6.03e-34 / 2.03e-32`（`DIR62r10_stageA.json`）⇒ rev 1.0 的 A62-5 FAIL。**处置**：改为"$\operatorname{rank}\bar M\le1$ **且** $L_A(r{=}1)\le10^{-30}$（数值零）"。
> - **DEF-62-2（A62-6 的扰动尺度标定错，守卫无量测力）**：rev 1.0 取 $\varepsilon{=}10^{-2}$ 并断言偏差 $>10^{-3}$，实测偏差只有 $6.83\times10^{-5}$（非正交的效应是**二阶**的：$k^{\top}k_P=\varepsilon\|k\|^2$ 先进入数据项再被 $\lambda/(s+\lambda)$ 稀释）⇒ FAIL。**处置**：取 $\varepsilon{=}0.3$，断言偏差 $>10^{-2}$ **且** $\mathrm{dev}(0.3)/\mathrm{dev}(0.01)\ge100$（**标定值**：$\mathrm{dev}(0.01){=}6.83\times10^{-5}$、$\mathrm{dev}(0.3){=}5.64\times10^{-2}$，比值 $826$；标定脚本 `dir62/calib62_sens.py` 留档）。此改动只动**守卫**，不动任何判据。

> **分工**：`D-07`（$W^0{=}0,r{=}1$ 的在线更新）、`D-31`（增量初始秩 $r_0$，明写**不得外推** $W^0\neq0$）、`D-25/D-30`（$W^0{=}0$ 作基线）都不覆盖「$W^0$ 本身是什么、能不能预载、预载要付什么」。**证据表零覆盖**（已核：`W^0` 仅作"零点/基线"出现 15 处，无一处把它当自变量）。

## 1 · 一句话

框架里 $W^0$ 同时出现在两个位置——§5.1 的**正则中心**（平衡点 $W^*=(B+\lambda W^0)(C+\lambda I)^{-1}$）与 §3.3 的**实现参数化**（$W=UV^{\top}$，秩 $\le r$）——而这两处的**秩预算作用对象不同**：§5.1 把秩预算压在 $\Delta W=W-W^0$ 上（预载**免费**），实现把秩预算压在 $W$ 上（预载**必须挤进** $r$）。本轮把这个差别做成可测的：$W^0$ 的存活**逐方向**由闭式谱律决定，且"预载是否免费"是一个**参数化选择**，不是数值细节。

## 2 · 代数预言（写卡前已推完，并已用 `dir62/ref62.py` 单算一遍）

**装置（$d{=}6$，单键恒定输入）**：$k=e_1+e_2$（$s=\|k\|^2=2$）、$v=e_3+e_4$；预载
$$W^0=\sum_{j=1}^{p}\gamma_j\,\frac{v_jk_j^{\top}}{\|k_j\|^2},\qquad k_j\in\{e_3,e_4,e_5\}\perp k,\quad v_j\in\{e_1,e_2,e_5\}\perp v .$$
目标 $J(W)=\tfrac12\|Wk-v\|^2+\tfrac\lambda2\|W-W^0\|_F^2$，$\lambda=0.05$。秩 $r\in\{1,2,3,4\}$。

- **P1（正交键存活律，精确、与 $\lambda,s$ 无关）**：若 $k_j\perp k$，则 $W^*k_j=W^0k_j$ ⇒ 读数增益 $\mathrm{gain}(k_j)\equiv\gamma_j$（**逐位等于 $\gamma_j$**）。推导一行：$(C+\lambda I)^{-1}k_j=k_j/\lambda$、$Bk_j=0$ ⇒ $W^*k_j=(\lambda W^0k_j)/\lambda=W^0k_j$。
- **P2（训练键稀释律，精确）**：若把预载放在**训练键**上（$W^0=\gamma\,v_Pk^{\top}/s$），则
  $$W^*k=\frac{s}{s+\lambda}v+\frac{\lambda}{s+\lambda}v_P\ \Rightarrow\ \text{系数}=(0.975609756098,\ 0.024390243902).$$
  推论：**训练键上的饱和增益上限是 $s/(s+\lambda)=0.9756$，永远到不了 1**（正则把读出按 $\lambda/(s+\lambda)$ 泄向预载；$W^0k=0$ 时泄向零）。
- **P3（争秩律）**：若秩预算压在 $W$ 上（实现形 $W=UV^{\top}$），最优解是 $W_r=\operatorname{SVD}_r(W^*R)R^{-1}$（$R=(C+\lambda I)^{1/2}$），**预载与数据同场按 $\sigma(W^*R)$ 竞争**，且
  $$J(W_r)-J(W^*)=\tfrac12\sum_{i>r}\sigma_i^2(W^*R)\quad(\text{已核到 }4\times10^{-18}).$$
  由于 $\sigma(W^*R)=\{s/\sqrt{s+\lambda}\}\cup\{\gamma_j\sqrt\lambda\}$（$=1.396861\cup0.2236\gamma_j$），**分配只看谱值，不看"是数据还是预载"**：强预载可以**挤掉任务数据**。
- **P4（$\Delta W$ 参数化 ⇒ 预载免费）**：若秩预算压在 $\Delta W$ 上（$W=W^0+UV^{\top}$），则 §5.1 的 $M_r^*=\operatorname{SVD}_r(\bar MR)R^{-1}$ 给
  $$J(W^0+M_r^*)-J(W^*)=\tfrac12\sum_{i>r}\sigma_i^2(\bar MR),\qquad \bar M=(B-W^0C)(C+\lambda I)^{-1},$$
  而本装置 $W^0k=0\Rightarrow\bar M=vk^{\top}/(s+\lambda)$ ⇒ **$\operatorname{rank}\bar M=1$** ⇒ **任意 $r\ge1$ 时 $L_A\equiv0$**：预载一个方向也不丢。

**算例（`ref62.py` 与 `pilot62b.py` 实算，全部留档）**

| 装置 | $\sigma(W^*R)$ | $J^*$ | $r{=}1$ | $r{=}2$ | $r{=}3$ | $r{=}4$ |
|---|---|---|---|---|---|---|
| `WEAK` $\gamma{=}(1.4,0.5,0.15)$ | $1.396861,\ 0.313050,\ 0.111803,\ 0.033541$ | 0.0243902439 | 0.0802027439 | 0.0312027439 | 0.0249527439 | 0.0243902439 |
| `STRONG` $\gamma{=}(8,6,1)$ | $1.788854,\ 1.396861,\ 1.341641,\ 0.223607$ | 0.0243902439 | **1.9250000000** | 0.9493902439 | 0.0493902439 | 0.0243902439 |
| $L_A$（$\Delta W$ 预算） | — | 0 | 0 | 0 | 0 | 0 |

- `WEAK` 的逐级截断：$r{=}1$ 只剩数据（$\mathrm{gain}_{\rm data}{=}0.975610$，三个预载全为 $0$）；$r{=}2$ 数据+$\gamma_1$；$r{=}3$ 再+$\gamma_2$；$r{=}4$ 全存活。
- `STRONG` 的**驱逐**：$r{=}1$ 只留 $\gamma_1{=}8$ 的预载，**任务数据被挤掉**（$\mathrm{gain}_{\rm data}{=}0$）；$r{=}2$ 数据回来但 $\gamma_2$ 仍被挤掉。
- 两种装置的 $J^*$ **相同**（$0.0243902439=\tfrac12\cdot\tfrac{\lambda}{s+\lambda}\|v\|^2$）：预载正交且装得下时，地面水位只由"数据被 $\lambda$ 稀释"决定。
- 因子化动力学可达性（预卡试跑 `pilot62b.py`，$\eta{=}0.05$、$N{=}2\times10^5$、3 个种子）：**8 个臂全部到 $J_r$ 到 $10^{-16}$ 相对误差**，逐方向增益逐位命中。

## 3 · 自变量、装置与唯一改动

- **唯一自变量 ①**：预载强度 $\gamma_j$（`WEAK`/`STRONG` 两档）× 秩预算 $r\in\{1,2,3,4\}$。
- **唯一自变量 ②**：**秩预算的对象**——`DeviceW`（$W=UV^{\top}$，实现形，§3.3）vs `DeviceD`（$W=W^0+UV^{\top}$，$\Delta W$ 形，§5.1）。
- **装置**：逐字派生 `dir61/R1_init61.py` 的同步因子化步（$G=(Wk{-}v)k^{\top}+\lambda(W{-}W_0)$、$U\leftarrow U-\eta GV$、$V\leftarrow V-\eta G^{\top}U$），**唯一改动**：$W_0$ 由"$U_0V_0^{\top}$"换成"**外生的 $W^0$**"，且 `DeviceD` 的读出矩阵是 $W^0+UV^{\top}$。$\eta=0.05$、$\lambda=0.05$、$N=2\times10^5$、种子段 `20261118 + 0..9`。
- **初始化**：$U_0,V_0\sim\mathcal N(0,0.3^2)$（**不得用零初始化**：DIR-61 已证 $U_0{=}V_0{=}0\Rightarrow GV_0{=}G^{\top}U_0{=}0$ 仍是驻点；A62-8 把这个陷阱做成本卡的守卫）。

## 4 · 判据与阈值（冻结，不改）

1. **① 存活律**（`DeviceW`，`WEAK`，$r{=}4\ge\operatorname{rank}W^*$，5 个种子）：$\max_j|\mathrm{gain}(k_j)-\gamma_j|\le10^{-9}$ **且** $|\mathrm{gain}_{\rm data}-s/(s+\lambda)|\le10^{-9}$。
2. **② 稀释律**（预载落在训练键上，$r{=}2$，5 个种子）：$|c_{\rm data}-s/(s+\lambda)|\le10^{-9}$ **且** $|c_{P}-\lambda/(s+\lambda)|\le10^{-9}$（$c$ 在正交基 $\{\hat v,\hat v_P\}$ 上分解）。
3. **③ 争秩律**（`DeviceW`，`WEAK` 与 `STRONG` 各自的 $r\in\{1,2,3,4\}$，5 个种子）：每臂
   - $|J_N-J_r|/J_r\le10^{-6}$，**且**
   - $\max_j|\mathrm{gain}_j-\mathrm{gain}_j^{\rm ref}|\le10^{-6}$、$|\mathrm{gain}_{\rm data}-\mathrm{gain}_{\rm data}^{\rm ref}|\le10^{-6}$；
   - **并且** `STRONG`/$r{=}1$ 必须复现**驱逐**：$|\mathrm{gain}_{\rm data}|\le10^{-6}$。
4. **④ 参数化对照**（`DeviceD`，`WEAK` 与 `STRONG`，$r{=}1$，5 个种子）：$|J_N-J^*|/J^*\le10^{-6}$ **且** $\max_j|\mathrm{gain}_j-\gamma_j|\le10^{-6}$、$|\mathrm{gain}_{\rm data}-s/(s+\lambda)|\le10^{-6}$（**预载免费**：$r{=}1$ 也一个不丢）。
5. **判据外必报量**：$\Delta J_{\rm param}=J_{\rm DeviceW}(r{=}1)-J^*$（`STRONG` 应为 $1.9006$、`WEAK` 为 $0.0558$）；$\operatorname{rank}\bar M$；$L_A$ 与 $L_B$ 的比值。
6. 任一条命中失败 ⇒ 该子句 FAIL；**四条全 PASS 才 `SUCCEEDED`**（主假设 H62 = ①∧②∧③∧④）。

## 5 · Stage A 守卫（硬门，任一 FAIL 不得进正式轮）

- **A62-1 平衡点锚**：$\|W^*(C+\lambda I)-(B+\lambda W^0)\|_F\le10^{-12}$ 且 $\|\nabla J(W^*)\|_F\le10^{-10}$（`WEAK` 与 `STRONG`）。
- **A62-2 有限差分梯度锚**：随机方向 $\Delta$，$(J(W{+}h\Delta)-J(W{-}h\Delta))/2h$ 与 $\langle\nabla J,\Delta\rangle$ 相对差 $\le10^{-6}$。
- **A62-3 §5.1 恒等式（$\Delta W$ 预算）**：$\big|J(W^0+M_r^*)-J(W^*)-\tfrac12\sum_{i>r}\sigma_i^2(\bar MR)\big|\le10^{-12}$，$r=1,2,3$。
- **A62-4 $W$ 预算恒等式**：$\big|J(W_r)-J(W^*)-\tfrac12\sum_{i>r}\sigma_i^2(W^*R)\big|\le10^{-12}$，$r=1,2,3$。
- **A62-5 $\operatorname{rank}\bar M=1$ 与 $L_A\equiv0$**（rev 1.1 修 DEF-62-1）：$\operatorname{rank}\bar M\le1$ 且 $L_A(r{=}1)\le10^{-30}$（**数值零**；不要求逐位为 0——SVD 的数值零 $\sigma_{5,6}$ 平方后约 $10^{-34}$）。
- **A62-6 敏感性负对照**（rev 1.1 修 DEF-62-2）：把 $\gamma_1$ 的键改成 $k_P=e_3+\varepsilon e_1$，取 $\varepsilon{=}0.3$（**不再正交**，$k^{\top}k_P{=}0.6$）⇒ ① 的预测必须**失效**：$|\mathrm{gain}_1-\gamma_1|>10^{-2}$ **且** $\mathrm{dev}(0.3)/\mathrm{dev}(0.01)\ge100$（标定：$5.64\times10^{-2}$ vs $6.83\times10^{-5}$）⇒ 证明"⊥"条件是律的必需前提、且量测在效应尺度上有分辨力。
- **A62-7 确定性**：同一配置连跑两次，读出逐位相同。
- **A62-8 DIR-61 陷阱继承**：$U_0{=}V_0{=}0$ 在本装置仍冻结（$\max_t\|W_t-W_0\|_F\le10^{-12}$）；正式轮全部臂必须用随机初始化（否则读数无意义）。

## 6 · 结果文件与判据量定义

- `dir62/ref62.py`（闭式参照 + 算例，已跑通，输出留档）
- `dir62/pilot62b.py`（预卡可达性试跑，已跑通：8/8 臂到 $10^{-16}$）
- `dir62/R1_preload62.py`（正式轮装置：`--arm {WEAK,STRONG} --r {1,2,3,4} --device {W,D} --seed`）
- `dir62/stageA62.py`（A62-1…8）
- `dir62/adjudicate62.py`（机械套用 §4）
- 输出：`dir62/out/DIR62r10_{stageA,formal_*,adjudication}.json`
- 判据量定义（代码里逐字实现）：$\mathrm{gain}(k_j)=\dfrac{(W k_j)^{\top}v_j}{\|v_j\|^2}$、$\mathrm{gain}_{\rm data}=\dfrac{(Wk)^{\top}v}{\|v\|^2}$、$c_{\rm data},c_P$ 为 $Wk$ 在 $\{\hat v,\hat v_P\}$ 上的正交分解系数。

## 7 · 证据边界（预注册，写进 VERDICT 与证据行）

1. 只测**单键恒定输入**（$C,B$ 秩 1）、$d{=}6$、$\lambda=\eta=0.05$、float64；**多键/满秩 $C$** 时 $\operatorname{null}(C)$ 变空 ⇒ P1 的"100% 存活"**不外推**。
2. 预载被构造成**严格正交**于训练键/值；一般位置（非正交）的存活律由 $B-W^0C$ 与 $(C+\lambda I)^{-1}$ 共同决定，**本卡不测、不猜**（A62-6 只证明正交性是必需的）。
3. $W^*$ 与 $W_r$ 都是**局部二次目标的**量，**不得**换算成语言任务性能（同 §5.1 原话）。
4. `DeviceD` 的"预载免费"只在 $\operatorname{rank}\bar M\le r$ 时成立；本卡 $\operatorname{rank}\bar M=1$ ⇒ **不得**外推为"任何 $W^0$ 都免费"。
5. 本轮**不改** v2.4 Freeze A–E、不新增组件；$Wk$/$Wk_j$ 都是**关联状态诊断探针**，未经 $\mathcal Q_l,\mathcal O_l$ 任务读取链路。
6. "强预载驱逐数据"只在本玩具的**恒定单键**下成立；**不得**读成"预载一定有害"——`WEAK` 下数据始终在（$r\ge1$）。

## 8 · 缺陷与修订流程

- 开卡前自查（`DEF-61-2/3` 的教训）：①每个 Stage A 断言**先在纸上验真假**（A62-3/4/5 已核到 $10^{-18}$，A62-8 已由 DIR-61 直接证）；②判据只用**单一响应尺度**（本卡全部是"读数 vs 闭式值"的绝对/相对差，**无**跨族模型比较）。
- 若 Stage A 出现 FAIL：先判"卡片缺陷"还是"实现错"，分别记 `DEF-62-x`；卡片缺陷照 DIR-61 流程开 rev 1.1（**阈值不动**）。
