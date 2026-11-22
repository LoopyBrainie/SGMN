# DIR-26 · 方向卡片 **rev 1.3** —— 撤回"$A\equiv C$ 幅度不变性"这条**错误断言**（Stage A 实跑推翻）

> 订正页，不是重写。rev 1.0 / 1.1 / 1.2 其余条款继续有效。
> **时间戳**：仅 Stage A 数据（`dir26/D26_R1_stageA.log`），**无 smoke / formal、无判据读数**。

## 触发：A26-6 在**真实门控率**下 FAIL（$|{\rm ret}_1(A)-{\rm ret}_1(C)|=0.1143$，阈值 0.02）

rev 1.2 之前的 Stage A 用探针阈值 $\tau{=}0.25$ 跑管道检查，门控只跳 5/560 次 ⇒ $\bar\alpha\approx0.99$ ⇒ C 臂几乎等于 A 臂，
断言"看起来"通过（0.0000）。换成 §C4 标定出的真实 $c_\tau$（跳过率 0.40，$\bar\alpha{=}0.664$）后立刻暴露 **0.1143**。

**根因（装置的逐字代码）**：`dir22/sgmn_module_hebb_v7.py` 的更新是
$$G=\underbrace{\alpha\cdot\operatorname{outer}(Wk-v,\ k)}_{\text{★ 只这一项被 }\alpha\text{ 缩放}}+\ \lambda\,(W-W^0)$$
⇒ $\alpha$ **不缩放**衰减项。于是
$$\alpha\,H+\lambda\,(W-W^0)=\alpha\Big[H+\frac{\lambda}{\alpha}(W-W^0)\Big]$$
**恒定门控严格等价于 $\lambda\to\lambda/\alpha$**（正是框架 §3.5 第 301–306 行的说法），而 $\lambda$ 变大会让 $P$ 的**方向**改变，
不只是幅度缩小。⇒ $C$ 臂**不是** $A$ 臂的等比缩放，$\mathrm{ret}_1(C)\neq\mathrm{ret}_1(A)$ 是**正确行为**。

**我的错误**：把"$\mathrm{ret}_1$ 对 $P\to cP$ 不变"（$\arg\max$ 的数学性质，成立）错写成"$\mathrm{ret}_1(A)=\mathrm{ret}_1(C)$"（装置层面的断言，**不**成立）。
在 $d{=}16,r_0{=}4,N\le12$ 的算例上它恰好差 0（$\lambda/\alpha$ 的变化还没大到翻转任何 $\arg\max$），于是把假性质当成了已验证的事实。

## §C7 · 撤回并替换 rev 1.0 §4 的 **(d)** 子句 与 §5 的 **A26-6**

**撤回**：`(d) 测量自校验 |ret1(A)−ret1(C)| ≤ 0.02`、`A26-6 幅度不变性（A≡C）`。二者**作废**，留档不删改。

**替换后（A26-6′ 与 (d′)）**：对**固定的** $P$，检验指标本身的**纯标量缩放不变性**：
$$\forall c\in\{0.37,\ 1,\ 2.5\}:\quad \mathrm{ret}_1(cP)=\mathrm{ret}_1(P)\quad\text{（逐条目 $\arg\max$ 相同）}$$
逐格点在 A 臂末态上执行（每个格点 3 次，$O(N^2d)$，开销可忽略）。这是 $\arg\max$ 的数学性质 ⇒ **它通过不代表装置正确，只代表指标写对了** —— 定位为**护栏**，不作为判别力来源。

## §C8 · 新增**报告项**（原 (d) 的位置，不参与 H26 判定）

$$\Delta_{\text{rate}}:=\mathrm{ret}_1(C)-\mathrm{ret}_1(A)\quad(\text{同流、等质量、只差"恒定门控"} )$$
按框架 §3.5，$C$ 等价于 $\lambda\to\lambda/\bar\alpha$ ⇒ $\Delta_{\text{rate}}$ 是"**恒定门控/速率效应本身**在检索读上的量级"，
它与 $\Delta_{\text{content}}$ 的**分列报告**就是用户规则要求的"内容性收益 vs 通用注入收益"不混报。

**判定合成（替换 rev 1.1 §C1 末段）**：H26 CONFIRMED ⇔ (a′) PASS **且** (b) 四条按 §C1 阈值全 PASS **且** (c) PASS **且** (d′：A26-6′ 全部通过) PASS。

## §C9 · 记录一条**由本卡实测得到**的框架层事实（进 DEF 表）
框架 §3.5 的"恒定门控 ⇔ $\lambda$ 变更"在**本装置上可被实测区分**：$\mathrm{ret}_1$ 差 **+0.1143**（$r{=}8,N{=}35,R{=}4$，$\bar\alpha{=}0.664$）。
⇒ 后续任何把"少写一点"当作"均匀弱化"的方向都会读错（**DEF-26-1**）。
