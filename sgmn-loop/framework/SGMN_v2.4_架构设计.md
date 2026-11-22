# SGMN v2.4 架构设计：低秩在线可塑记忆与隔离读取

> **版本性质**：只定义架构对象、数据流、梯度归属和状态动力学；不指定基座模型、控制器结构、初始化方案、数值超参数或工程实现。
>
> **事实依据**：《SGMN v2.3 架构文档（简版）》《SGMN v2.3 框架——本轮实验完整总结报告》《SGMN 实验计划 v1》。外部 reviewer 意见仅用于问题核对，不作为证据。脑电动力学文档属于不同研究对象，不向本版引入其模型或结论。
>
> **编号说明**：本版 Freeze A–E 为重新组织后的 v2.4 架构定义编号，不自动继承 v2.3 中同名 Freeze 的语义；历史实验结论只按其原始公式、假设和适用范围引用。

## 1. 设计定位与证据边界

v2.4 的核心定位是：

$$
\boxed{\text{冻结基座上的低秩在线可塑关联状态，以及与其梯度隔离的任务驱动读取控制器。}}
$$

若沿用“持久关联”这一表述，“持久”仅表示状态可沿在线步保留，不预设跨会话持久化，也不等价于已证明的长期记忆能力。

本轮证据只支持以下结论。

| 证据 | 实际支持的范围 | 不能外推为 |
|---|---|---|
| EXP-01 / D-01 | 标量、iid 高斯输入、$\alpha\equiv1$ 的全矩阵在线更新；三种稳定性口径、小步长适应标度、均方失稳的罕见轨迹机制 | 任意矩阵谱、低秩因子化、时序相关输入或完整 LLM 的统一稳定界 |
| EXP-02 / D-02 | 指定确定性二次目标上的全参数与因子化 GD；二阶展开、初始化相关的有界性、特定配置下收敛到秩 $r$ 最优解 | 随机因子化更新的通用均方界、通用修正因子 $c(s)$、任意初始化下必收敛到全局低秩解 |
| EXP-06 / D-06 | 指定标量 iid 模型的自洽噪声底、小 $\eta$ 的 SNR 标度及有限 $\eta$ 修正 | AR(1) 噪声、真实信息寿命、长期干扰和跨会话保持 |
| EXP-03 及后续实验 | 尚无直接证据 | 梯度隔离已验收、记忆已改善 LLM、联合训练可行 |

因此，v2.4 可以冻结**数学定义和接口契约**，但必须把“已推导”“已在特定实验中验证”“集成后仍待验收”分开。31 项回归通过不等于端到端架构通过。

本版相对 v2.3 的关键修订是：

1. 统一时间与层级索引，保持 $h_{t-1}\!\to h_t$ 的**时间因果对**，不改写成层间预测。
2. 以局部目标为唯一核心，区分全矩阵参考动力学与低秩因子自身动力学。
3. 区分随机均方稳定界、确定性全参数 GD 界和因子化有界性临界步长。
4. 将 $k,v$、读取上下文与读取状态的 stop-gradient 和参数所有权写成自洽的一组约束，并明确读取查询 $q$ 与写入键 $k$ 的共享键空间。
5. $S$、replay、人格、跨会话持久化及 LLM 能力不进入已验证核心。

## 2. Freeze A：状态对象与逐层数据流

### 2.1 符号和状态

- $l$：Transformer 层索引；$t$：在线事件索引，不与层索引混用。
- $h_{l,t}$：第 $l$ 层注入前的隐状态；$h_{l,t}^{+}$：注入后的隐状态。
- $\theta_l$：冻结基座参数；冻结表示不更新参数，不表示删除其对输入的 Jacobian。
- $W_l^0\in\mathbb{R}^{d_v\times d_k}$：固定的基准关联矩阵，其与具体 LLM 权重的构造关系不在本版敲定。
- 可塑关联状态为

$$
W_{l,t}=W_l^0+\Delta W_{l,t},\qquad
\Delta W_{l,t}=U_{l,t}V_{l,t}^{\top},
$$

其中 $U_{l,t}\in\mathbb{R}^{d_v\times r}$、$V_{l,t}\in\mathbb{R}^{d_k\times r}$，故 $\operatorname{rank}(\Delta W_{l,t})\le r$。

- $\operatorname{sg}(\cdot)$：前向保持原值，反向梯度为零；它不是前向断开，也不消除状态之间的函数依赖。

### 2.2 基座、读取和注入

取 $h_{0,t}=h_{0,t}^{+}$ 为模型输入。对挂载 SGMN 的层，定义

$$
h_{l,t}=F_l(h_{l-1,t}^{+};\theta_l),
$$

先定义读取上下文 $c_{l,t}=\operatorname{sg}(h_{l-1,t})$。为闭合读取上下文与关联矩阵的维度关系，v2.4 将读取器固定为以下最小接口：

$$
q_{l,t}=\mathcal Q_l(c_{l,t};\phi_l^q)\in\mathbb{R}^{d_k},
$$

$$
y_{l,t}=\operatorname{sg}(W_{l,t})q_{l,t}\in\mathbb{R}^{d_v},
\qquad
m_{l,t}=\mathcal O_l(c_{l,t},y_{l,t};\phi_l^o),
$$

$$
\boxed{h_{l,t}^{+}=h_{l,t}+\gamma_l m_{l,t}.}
$$

其中 $\phi_l=(\phi_l^q,\phi_l^o)$ 仍只属于读取控制器，$\gamma_l$ 是注入强度，$m_{l,t}$ 位于第 $l$ 层残差空间。未挂载 SGMN 的层等价于 $m_{l,t}=0$。

$q_{l,t}$ 是当前事件的读取查询，$k_{l,t}$ 是写入历史关联的键；二者共享键空间 $\mathbb{R}^{d_k}$，但不要求 $q_{l,t}=k_{l,t}$，也不要求 $\mathcal Q_l=P_{k,l}$。这里的共享键空间是坐标兼容契约，而非仅要求维度相等；如何通过投影或训练满足该契约仍开放，该契约本身也不保证检索有效。$y_{l,t}$ 是维度为 $d_v$ 的关联读取结果，$\mathcal O_l$ 将其转化为残差空间输出。核心读取据此固定为线性关联接口 $y=\operatorname{sg}(W)q$；$\mathcal Q_l,\mathcal O_l$ 的内部结构不指定，更复杂的记忆读取算子不进入本版核心。

读取使用 $W_{l,t}$，写入产生 $W_{l,t+1}$；同一步不采用“先更新记忆、再由更新后的记忆定义本步目标”的隐式顺序。

$S$ 不进入 v2.4 的已定义核心。未来若加入情节状态，只能以 $\operatorname{sg}(S_{l,t})$ 作为额外界定清楚的读取输入；其更新方程、容量、遗忘和持久化必须另行定义。

### 2.3 写入键与目标

为保持 v2.3 的时间因果语义，并避免当前层注入直接构成自身写入目标，定义

$$
\boxed{
k_{l,t}=\operatorname{sg}\!\left(P_{k,l}(h_{l,t-1})\right),\qquad
v_{l,t}=\operatorname{sg}\!\left(P_{v,l}(h_{l,t})\right).
}
$$

其中 $k_{l,t}\in\mathbb{R}^{d_k}$、$v_{l,t}\in\mathbb{R}^{d_v}$，故 $W_{l,t}k_{l,t}$ 与 $v_{l,t}$ 同维，也与读取端 $q_{l,t}\in\mathbb{R}^{d_k}$、$y_{l,t}\in\mathbb{R}^{d_v}$ 的契约一致。这里两者都取注入前隐状态。于是 $k_{l,t}$ 与 $v_{l,t}$ 构成同层相邻在线事件的关联，而不是 $h_{l-1,t}\!\to h_{l,t}$ 的层间关联。

该定义同时澄清两件事：

- 局部写入损失不能经 $k,v$ 反向训练上游隐状态或投影参数；
- $v_{l,t}$ 不通过当前层的 $\gamma_l m_{l,t}$ 直接依赖当前 $W_{l,t}$。但前层记忆和过去记忆仍可通过基座前向影响它，故跨层、跨时间耦合并未被 stop-gradient 删除。

## 3. Freeze B：局部写入目标与低秩状态动力学

### 3.1 唯一局部目标

定义当步局部目标

$$
\ell_{l,t}(W)=
\frac{\alpha_{l,t}}{2}\|Wk_{l,t}-v_{l,t}\|^2
+\frac{\lambda}{2}\|W-W_l^0\|_F^2,
\qquad \lambda>0,
$$

其中 $\alpha_{l,t}\in[0,1]$ 只门控数据项，并在局部更新中视为 detach 的外生量。其梯度为

$$
G_{l,t}:=\nabla_W\ell_{l,t}
=\nabla_{\Delta W}\ell_{l,t}
=\alpha_{l,t}(W_{l,t}k_{l,t}-v_{l,t})k_{l,t}^{\top}
+\lambda(W_{l,t}-W_l^0).
$$

该式把“学什么”固定为局部因果关联预测加基准回归，不把最终任务损失隐式并入写入目标。

### 3.2 全矩阵参考动力学

在不施加低秩参数化时，局部梯度步为

$$
\boxed{
W_{l,t+1}=W_{l,t}-\eta G_{l,t}.
}
$$

展开即 v2.3 的更新式。它是后续平衡点、标量随机稳定性和均值时间尺度分析的参考对象。

### 3.3 低秩因子动力学

对 $\Delta W=UV^{\top}$，由链式法则

$$
\nabla_U\ell=GV,\qquad \nabla_V\ell=G^{\top}U.
$$

v2.4 将低秩核心的同步局部梯度步明确写为

$$
\boxed{
U_{l,t+1}=U_{l,t}-\eta G_{l,t}V_{l,t},\qquad
V_{l,t+1}=V_{l,t}-\eta G_{l,t}^{\top}U_{l,t}.
}
$$

省略层索引，更新后的记忆增量严格满足

$$
\Delta W^+-\Delta W
=-\eta\left(GVV^{\top}+UU^{\top}G\right)
+\eta^2GVU^{\top}G.
$$

因此，全矩阵递推不是低秩因子递推的等价改写。前者不能把自己的稳定界直接授予后者，二阶项也不能在精确展开中遗漏。

两个仅由数学即可确定的边界是：

- $U=V=0$ 时，$\nabla_U\ell=\nabla_V\ell=0$，双线性因子化可能停留在零增量；
- 因子化目标一般非凸，特定实验中收敛到全局秩 $r$ 最优解，不构成任意初始化和步长下的收敛保证。

本版据此只冻结“初始化必须纳入动力学分析与稳定性认证”，不冻结具体初始化分布或尺度。

## 4. Freeze C：读取梯度、参数所有权与隔离契约

令任务损失为 $\mathcal L_{\mathrm{task}}$，从注入后的隐状态传回的梯度为

$$
\delta_{l,t}^{+}=\nabla_{h_{l,t}^{+}}\mathcal L_{\mathrm{task}}.
$$

则由链式法则

$$
\boxed{
\frac{\partial\mathcal L_{\mathrm{task}}}{\partial\gamma_l}
=\langle\delta_{l,t}^{+},m_{l,t}\rangle,
\qquad
\nabla_{\phi_l}\mathcal L_{\mathrm{task}}
=\gamma_l\left(\frac{\partial m_{l,t}}{\partial\phi_l}\right)^{\top}
\delta_{l,t}^{+}.
}
$$

这里 $\phi_l=(\phi_l^q,\phi_l^o)$，$\partial m_{l,t}/\partial\phi_l$ 是包含 $q_{l,t}$ 路径在内的全导数。$c_{l,t}$ 与 $W_{l,t}$ 的 stop-gradient 不作用于 $\mathcal Q_l,\mathcal O_l$ 自身参数，因此任务梯度可以训练读取查询和残差输出，但不能沿这两个被截断的分支进入上游隐状态或写入状态。

读取分支中的 $W_{l,t}$ 被 stop-gradient，因此

$$
\left.\frac{\partial\mathcal L_{\mathrm{task}}}{\partial W_{l,t}}\right|_{\mathrm{read}}=0.
$$

与此同时，$W_{l,t}$ 的**数值**仍经 $y_{l,t}=\operatorname{sg}(W_{l,t})q_{l,t}$ 参与前向计算。隔离的目标正是“前向可用、任务梯度不进入写入”，而不是把记忆从计算图中实际移除。

当前层 $c_{l,t}$ 的截断只作用于读取支路；冻结基座仍须允许任务梯度经主路径的输入 Jacobian 向低层注入点传播，否则多层读取控制器不能从最终任务损失获得信号。参数冻结与输入梯度是否保留是两件事。

| 对象 | 任务损失更新 | 局部在线更新 | v2.4 归属 |
|---|---:|---:|---|
| $\theta_l$ | 否 | 否 | 冻结基座；必要时保留输入梯度 |
| $\phi_l^q,\phi_l^o,\gamma_l$ | 是 | 否 | 读取侧元参数 |
| $U_{l,t},V_{l,t}$，亦即 $W_{l,t}$ | 否 | 是 | 只由显式局部规则驱动 |
| $P_{k,l},P_{v,l}$ | 核心内否 | 否 | 视为给定写入映射；固定或离线学习的选择仍开放 |
| $\alpha_{l,t}$ 及其产生机制 | 核心内否 | 只作为 detach 门控 | 视为给定外生机制；产生规则开放 |
| $S$ | 不定义 | 不定义 | 不属于当前已定义核心 |

若以后让任务损失训练写入投影或写入门控，就会引入“任务目标影响未来写入数据”的外层通道。那将是新的耦合系统，不能在未重新分析的情况下继续沿用本版的独立局部动力学结论。

即使写入映射保持固定，读取侧 $\phi,\gamma$ 的任务训练也会通过改变后续隐状态，间接改变未来的键值分布。它不构成自动微分意义上的梯度泄漏，但意味着外层训练期间的 $C,B$ 未必平稳；本版的平衡点和随机稳态公式只能按给定数据过程条件化使用。

还有两个退化点必须保留在架构边界中：

- $\gamma_l=0$ 时，当前注入对前向输出无影响，且 $\nabla_{\phi_l}\mathcal L_{\mathrm{task}}=0$；$\gamma_l$ 自身是否能脱离零点取决于 $\langle\delta_{l,t}^{+},m_{l,t}\rangle$。
- 非零梯度、有限梯度和记忆扰动确实改变输出，都不是 stop-gradient 定义本身能够保证的，仍须由 EXP-03 类接口实验验收。

## 5. Freeze D：可保留的数学结论及其适用条件

以下公式是架构的解析参照，不是对所有数据、所有层和所有参数化的无条件保证。

### 5.1 全矩阵平衡点、秩约束最优解与容量损失

先取 $\alpha\equiv1$，定义期望局部目标

$$
J(W)=\frac12\mathbb{E}\|Wk-v\|^2+\frac{\lambda}{2}\|W-W^0\|_F^2,
$$

以及

$$
C=\mathbb{E}[kk^{\top}],\qquad B=\mathbb{E}[vk^{\top}].
$$

由 $\nabla_WJ=WC-B+\lambda(W-W^0)=0$，全矩阵平衡点为

$$
\boxed{W^*=(B+\lambda W^0)(C+\lambda I)^{-1}.}
$$

令

$$
\bar M=W^*-W^0=(B-W^0C)(C+\lambda I)^{-1},\qquad
R=(C+\lambda I)^{1/2}.
$$

由于 $\lambda>0$，$R$ 可逆。忽略与 $\Delta W$ 无关的常数，

$$
J(W^0+\Delta W)=\frac12\|(\Delta W-\bar M)R\|_F^2+\text{const}.
$$

作 $Z=\Delta W\,R$，秩预算在变换后保持不变，故最佳秩 $r$ 解为

$$
\boxed{
M_r^*=\operatorname{SVD}_r(\bar M R)R^{-1},
}
$$

其中 $\operatorname{SVD}_r$ 表示 Frobenius 范数下的最佳秩 $r$ 截断。相应容量损失严格为

$$
\boxed{
J(W^0+M_r^*)-J(W^0+\bar M)
=\frac12\sum_{i>r}\sigma_i^2(\bar M R).
}
$$

$\tfrac12$ 系数由上述恒等式直接成立，不依赖某次数值复现。该损失只是局部二次目标的空间容量代价，不能直接换算成语言任务性能损失；$M_r^*$ 是秩约束最优解，也不等于因子化在线动力学必然到达的平衡点。

### 5.2 写入门控的代数作用

对给定的外生门控过程，定义

$$
C_\alpha=\mathbb{E}[\alpha kk^{\top}],\qquad
B_\alpha=\mathbb{E}[\alpha vk^{\top}].
$$

全矩阵平均向量场的任何不动点必须满足

$$
\boxed{W_\alpha^*(C_\alpha+\lambda I)=B_\alpha+\lambda W^0.}
$$

若 $C_\alpha,B_\alpha$ 不依赖当前 $W$，这就是显式平衡点公式；若门控依赖当前状态，则两者可能随 $W$ 改变，上式只是隐式不动点关系，不能当作闭式解。

对恒定 $\bar\alpha>0$，令 $\beta=\lambda/\bar\alpha$，则

$$
W_{\bar\alpha}^*=(B+\beta W^0)(C+\beta I)^{-1},
\qquad
W_{\bar\alpha}^*-W^0=(B-W^0C)(C+\beta I)^{-1}.
$$

所以恒定门控会改变有效正则强度。令 $D_0=B-W^0C$，则

$$
\frac{d}{d\beta}\left\|D_0(C+\beta I)^{-1}\right\|_F^2
=-2\operatorname{tr}\!\left[D_0(C+\beta I)^{-3}D_0^{\top}\right]\le0.
$$

因此方向是明确的：$\bar\alpha$ 增大时，$\|W_{\bar\alpha}^*-W^0\|_F$ 非减；$\bar\alpha$ 减小时，该范数非增。实验计划中若采用“随 $\bar\alpha$ 增大而下降”的判据，应更正。

内容相关门控是否带来独立于速率效应的收益，仍属 EXP-07 的开放问题。

### 5.3 稳定性：三种口径与两种参数化不得混用

在给定外生数据过程、且 $C_\alpha,B_\alpha$ 相对当前状态固定时，对全矩阵更新，围绕满足上述不动点关系的 $W^*$ 定义 $\delta W_t=W_t-W^*$。令

$$
A_t:=I_{d_k}-\eta(\alpha_tk_tk_t^{\top}+\lambda I_{d_k}),
$$

则

$$
\delta W_{t+1}=\delta W_tA_t+\eta\Xi_t,
$$

其中自洽创新项为

$$
\boxed{
\Xi_t=W^*(C_\alpha-\alpha_tk_tk_t^{\top})
+(\alpha_tv_tk_t^{\top}-B_\alpha),
\qquad \mathbb{E}[\Xi_t]=0.
}
$$

在标量、$\alpha\equiv1$、iid 高斯 $k_t$ 下，写 $\mu=\mathbb{E}[k^2]$，不是标准差；则

$$
\delta_{t+1}=A_t\delta_t+\eta\xi_t,\qquad
A_t=1-\eta(k_t^2+\lambda),
$$

$$
\xi_t=(\mu-k_t^2)W^*+v_tk_t-B.
$$

令

$$
D=\mathbb{E}[(k^2+\lambda)^2]=3\mu^2+2\mu\lambda+\lambda^2,
$$

则齐次误差二阶矩的传播因子为

$$
\mathbb{E}[A^2]=1-2\eta(\mu+\lambda)+\eta^2D.
$$

因此均方稳定要求 $\mathbb{E}[A^2]<1$，即

$$
\boxed{
0<\eta<\eta_{\mathrm{ms}}:=\frac{2(\mu+\lambda)}{3\mu^2+2\mu\lambda+\lambda^2}.
}
$$

齐次误差传播的均值稳定和几乎必然稳定分别由

$$
|1-\eta(\mu+\lambda)|<1,
\qquad
\mathbb{E}[\log|A|]<0
$$

刻画，三者不可互换。本轮配置 $\mu=5,\lambda=0.05$ 的数值锚点为 $\eta_{\mathrm{ms}}=0.13377$、均值界 $0.39604$、几乎必然界约 $0.491$；它们不是架构默认学习率。

矩阵情形必须先固定状态空间与向量化约定。采用列优先 $x_t=\operatorname{vec}(\delta W_t)$，由 $\operatorname{vec}(\delta W_tA_t)=(A_t^{\top}\otimes I_{d_v})x_t$，得

$$
x_{t+1}=\widetilde A_tx_t+\eta\widetilde\Xi_t,
\qquad
\widetilde A_t=A_t^{\top}\otimes I_{d_v},
\quad
\widetilde\Xi_t=\operatorname{vec}(\Xi_t).
$$

齐次二阶矩传播由线性算子

$$
\mathcal T(P)=\mathbb{E}[\widetilde A_tP\widetilde A_t^{\top}]
$$

决定，判据是 $\rho(\mathcal T)<1$，而不是把标量 $\mathbb{E}[A^2]$ 直接换成一个矩阵。若当前 $(A_t,\Xi_t)$ 与过去状态独立、稳态误差均值为零且交叉项消失，则稳态协方差写成

$$
P_\infty^x=\mathcal T(P_\infty^x)+\eta^2Q_\Xi,
\qquad
Q_\Xi=\mathbb{E}[\widetilde\Xi_t\widetilde\Xi_t^{\top}].
$$

时序相关或状态相关门控下，上述独立性与交叉项前提不自动成立。

另需区分 $\alpha\equiv1$ 时确定性期望目标上的全参数 GD：其线性误差模式给出的界是

$$
0<\eta<\frac{2}{\lambda_{\max}(C+\lambda I)}.
$$

EXP-02 的 $0.19914$ 及因子化临界步长属于这一类确定性有界性实验。因子化临界值依赖初始化与瞬态放大；本轮只证明“不能沿用全参数界”，没有给出可跨配置冻结的通用公式或数值表。

### 5.4 iid 噪声底与有限步长 SNR

在上述标量 iid 稳态条件下，记 $\sigma_\xi^2=\mathbb{E}[\xi^2]$，则

$$
P_\infty
=\frac{\eta^2\sigma_\xi^2}{1-\mathbb{E}[A^2]}
=\frac{\eta\sigma_\xi^2}{2(\mu+\lambda)(1-\eta\kappa)},
\qquad
\kappa:=\frac{D}{2(\mu+\lambda)}.
$$

这里 $\sigma_\xi^2$ 必须来自完整自洽创新项，不能只保留 $\alpha vk^{\top}-B_\alpha$ 而丢掉 $W^*(C_\alpha-\alpha kk^{\top})$。

按报告中的 Memory SNR 定义，在信号项和数据统计固定时，

$$
\boxed{
\mathrm{SNR}(\eta)\propto
\sqrt{\frac{1-\eta\kappa}{\eta}}
=\eta^{-1/2}\sqrt{1-\eta\kappa}.
}
$$

所以 $\eta^{-1/2}$ 是小步长极限，不是整个稳定区间内的精确律。类似地，$T_{\mathrm{adapt}}\propto[\eta(\mu+\lambda)]^{-1}$ 只保留为本轮已验证的小步长平均时间尺度，不外推为整个稳定区间内的精确定律。

### 5.5 正则衰减不等于已验证的信息寿命

仅在全矩阵、无写入 $\alpha=0$、$0<\eta\lambda<1$ 的参考系统中，

$$
\Delta W_{t+n}=(1-\eta\lambda)^n\Delta W_t,
$$

故可定义参考衰减时间

$$
\tau_{\mathrm{ref}}=-\frac1{\log(1-\eta\lambda)}
\approx\frac1{\eta\lambda}\quad(\eta\lambda\ll1).
$$

它不是低秩因子在线系统、时序相关数据或 LLM 任务中已经测得的 $T_{\mathrm{retain}}$。AR(1) 修正、谱时间尺度、真实保持曲线和干扰仍保持开放。

## 6. 跨层动力传播与整体评估

### 6.1 读取和写入不是两个前向上互不相干的系统

梯度隔离只限制反向传播。前向上，第 $j$ 层记忆仍可沿以下路径影响后续状态：

$$
W_{j,t}\longrightarrow y_{j,t}\longrightarrow m_{j,t}\longrightarrow h_{j,t}^{+}
\longrightarrow h_{l,t}\ (l>j)
\longrightarrow (k_{l,\cdot},v_{l,\cdot})
\longrightarrow W_{l,\cdot+1}.
$$

因此，当前层写入虽然只优化局部目标，其键、目标和有效数据分布仍可能受到前层记忆影响；过去记忆也可能通过时间上下文影响未来写入。$\operatorname{sg}$ 不会消除这些依赖。

令 $z_t$ 汇总各层低秩记忆状态，$a_t$ 表示外部输入，$\vartheta$ 汇总读取元参数。把基座与读取前向记为 $\mathcal H$，局部写入映射记为 $\mathcal F$，则堆叠系统抽象为

$$
h_t=\mathcal H(z_t,a_t;\vartheta),\qquad
z_{t+1}=\mathcal F(z_t,h_t,a_t).
$$

在某工作点附近线性化，

$$
\delta z_{t+1}
\approx
\underbrace{\left(
\mathcal F_z+\mathcal F_h\mathcal H_z
\right)}_{\mathcal J_{\mathrm{stack}}}
\delta z_t+\text{创新项}.
$$

即使 $\mathcal F_z$ 所代表的各层独立更新分别稳定，$\mathcal F_h\mathcal H_z$ 仍引入跨层反馈。故“单层在 exogenous iid 输入下稳定”不是“多层在内生、时序相关隐状态下稳定”的充分条件。随机情形还须研究相应的二阶矩传播，而不能只看 $\mathcal J_{\mathrm{stack}}$ 的典型轨迹。

### 6.2 架构评价

- **局部关联目标：可冻结。** 更新规则具有明确二次目标，平衡点、噪声和参考时间尺度可以解析追踪。
- **低秩表示：可冻结；低秩优化仅有条件可靠。** 秩约束最优解和容量损失清楚，但因子化带来非凸性、预条件效应和二阶耦合。
- **读写解耦：设计自洽，集成未验收。** 参数所有权和前向、反向路径现在可以无歧义地定义；EXP-03 尚未证明真实模型同时满足隔离与前向有效性。
- **整机动力学：尚未闭合。** 单模块证书没有覆盖跨层反馈、内生输入分布和时间相关性。
- **记忆语义与 LLM 能力：不作能力宣称。** 当前证据支持的是可塑记忆数学内核，不是完整长期记忆系统。

## 7. v2.4 冻结清单与开放接口

### 7.1 本版冻结

| 冻结项 | v2.4 结论 |
|---|---|
| A. 状态与数据流 | $W=W^0+UV^{\top}$；区分注入前、后隐状态；保持同层时间因果键值对；读取使用 $W_t$，写入产生 $W_{t+1}$；$q,k\in\mathbb{R}^{d_k}$ 分别承担读取查询与写入键，$y=\operatorname{sg}(W)q\in\mathbb{R}^{d_v}$ |
| B. 写入动力学 | 唯一局部二次目标；$\alpha$ 只门控数据项并 detach；全矩阵更新仅作参考，低秩因子按各自梯度更新并保留二阶展开 |
| C. 读取与梯度 | 任务损失只更新读取侧 $\phi^q,\phi^o,\gamma$；$k,v$、读取上下文 $c$ 与读取记忆状态均 stop-gradient；这不阻断 $\mathcal Q,\mathcal O$ 自身参数训练；冻结基座不更新参数但保留所需输入梯度 |
| D. 数学纪律 | 区分秩约束最优解与动力学平衡点；区分均方、均值、几乎必然和确定性 GD 稳定性；使用完整自洽创新项及有限 $\eta$ 修正 |
| E. 组件边界 | 不加入早期 Push/Pop、动态栈深度或 replay 组件；$S$ 只保留未来扩展位置，不赋予已验证动力学 |

其中 C 是**必须满足的接口定义**，不是“已通过 EXP-03”的事实陈述。

### 7.2 保持开放

| 开放项 | 本版不敲定的内容 |
|---|---|
| 写入映射与门控 | $P_k,P_v$ 的固定、离线学习或其他来源；$\alpha$ 的产生规则及内容选择收益 |
| 数值与结构选择 | $r,\eta,\lambda$ 的取值，因子初始化，挂载层集合，$W^0$ 的具体构造，$\mathcal Q_l,\mathcal O_l$ 的内部结构、参数共享及查询坐标对齐方式 |
| 稳定性推广 | 因子化随机在线更新的操作域、跨层耦合稳定性、非 iid 与 AR(1) 修正 |
| 时间与情节状态 | token/segment 到在线步的映射，$\eta_S,\eta_W$ 关系，$S$ 的更新、检索、容量和遗忘 |
| 记忆语义 | 信息寿命分解、长期干扰、replay 作用、模态持续度及“人格”命题 |
| 集成接口验收 | 前向记忆敏感性、读取参数的非零有限梯度、任务梯度不进入写入通道，以及冻结基座不发生意外更新 |
| 持久化与 LLM 协同 | 跨会话状态语义、前向任务收益、长期基座能力保持和受保护联合训练 |

这些开放项不是隐含默认值。尤其是：本轮 $c(s)$ 表不能变成通用学习率表，$r=4$ 或旧文档的 $r=8$ 不能变成架构默认秩，$(r,T_{\mathrm{retain}},\mathrm{SNR})$ 不能命名为已证明的容量定理。

## 8. 最终架构陈述

$$
\boxed{
\begin{aligned}
\text{SGMN v2.4}={}&\text{冻结基座}
+\text{由显式局部目标驱动的低秩在线关联状态}\\
&+\text{前向连通、反向隔离的任务驱动读取控制器}.
\end{aligned}}
$$

它已经是一个定义更自洽的混合状态系统：写入负责形成关联，读取负责学习如何使用关联，二者在前向耦合、在任务梯度上分离。其数学内核可以依赖，但因子化在线稳定性、跨层耦合、接口验收和实际 LLM 收益必须分别证明，不能由任一单项结果代替。
