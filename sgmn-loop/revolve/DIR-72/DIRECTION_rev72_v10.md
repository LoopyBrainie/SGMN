# DIR-72 · rev 1.0 · $\eta^\ast_{\rm top}$ 是否由**不动点处的线性谱**决定？（2026-11-17）

- **方向出处**：D-92 §3 派生（"下一层要问的已不在指数是多少，而是边界是否由不动点处的线性谱决定"）+ §7.2 跨层耦合稳定性子项的余项。**非自造**。
- **装置**：`dir70/ref70.py` **原样未动**（只读 import）；新增 `dir72/step72.py`（把 `ref70.run_stack` 的**单步函数体逐字**复制成 `step()`，并加一个 `field()`：把同一梯度场作用在 $(W_1,W_2)$ 上）。**唯一新增变量 = 求谱的方式（有限差分）**，不引入任何新动力学。

## 1 · 主假设与否证子句

**H72**：$\eta^\ast_{\rm top}$ 是**不动点 $\;(W_1^\ast,W_2^\ast)=(u_1^\ast P,u_2^\ast P)\;$ 的线性稳定边界**，即
$$\eta^\ast_{\rm pred}=\min_{k:\,\mathrm{Re}\lambda_k>0}\;\frac{2\,\mathrm{Re}\,\lambda_k}{|\lambda_k|^2},\qquad \lambda_k\in\mathrm{spec}\big(J\big|_{\rm factor}\big),$$
其中 $J$ 是**因子空间** $(U_1,V_1,U_2,V_2)\in\mathbb R^{48}$ 上单步映射在不动点处的 Jacobian。
（$\mathrm{Re}\lambda>0$ 上的这一形式是离散 Euler 的**精确**稳定界：实特征值给 $2/\lambda$，复特征对给 $2a/(a^2+b^2)$ —— 后者在 $|b|\gtrsim a$ 时比 $2/a$ **小一半**，这是本卡与"$2/\lambda_{\max}$"写法的唯一差别，见 §4 的玩具算例。）

**否证子句**：若任一格 $\big|\eta^\ast_{\rm meas}/\eta^\ast_{\rm pred}-1\big|>1.0$（即比值出 $[0.5,2]$）⇒ 该格的"发散"**不是**不动点的线性失稳（而是非线性/瞬态机制）⇒ H72 在该格被否证。

## 2 · MVP 标准（最小产物）与数值阈值

**最小产物**：5 个数 —— `target` $\gamma\in\{3,10,30\}$ 与 `keys` $\gamma\in\{3,10\}$ 五格各一个 $\eta^\ast_{\rm pred}$，与 D-92 已**二分**的 $\eta^\ast_{\rm meas}$（**直接读已落盘 JSON，不重跑**）逐格比。

- **① 主判据**：五格**全部** $\big|\eta^\ast_{\rm meas}/\eta^\ast_{\rm pred}-1\big|\le0.30$ ⇒ **PASS**。
- **② 空间归因**：同一五格用**矩阵空间** $72$ 维 Jacobian（$J$ 作用在 $(W_1,W_2)$ 的梯度场上）算 $\eta^\ast_{\rm pred}^{\rm mat}$，"比值落在 $[0.5,2]$ 的格数 $\le1$" ⇒ **PASS**（即：矩阵空间**不能**解释边界，因子空间才行）。
- **③ $C$ 的非单调**：由因子空间预言反算 $C_{\rm pred}(\gamma)=\eta^\ast_{\rm pred}\gamma^2$，须有 $C_{\rm pred}(10)>C_{\rm pred}(3)$ **且** $C_{\rm pred}(10)>C_{\rm pred}(30)$（峰值落在 $\gamma{=}10$，与 D-92 实测排序同）⇒ **PASS**。

**裁决规则（冻结）**：① PASS ⇒ `SUCCEEDED`；① FAIL 但 ② 或 ③ PASS ⇒ `PARTIAL`；①②③ 全 FAIL ⇒ `FALSIFIED`。判据"从未被计算"（如某格 Jacobian 不可算）**不得**记 PASS/FAIL，须单列。

## 3 · 量测对象与算例

- **不动点取法**：以 $\eta_{\rm warm}=0.3\,\eta^\ast_{\rm meas}$ 迭代 `step()`（不动点处 $G\equiv0$ ⇒ 任何因子化都是不动点）直到 $\|W_1-u_1^\ast P\|/\|u_1^\ast P\|<10^{-11}$（上限 $5\times10^5$ 步），取那时的 $(U_1,V_1,U_2,V_2)$ 为 $\mathbf x^\ast$。**不得**用解析构造代替（不同因子化的 Jacobian 不同，见 A3）。
- **Jacobian**：中心差分 $J_{:,j}=\big(\mathrm{step}(\mathbf x^\ast+h\mathbf e_j)-\mathrm{step}(\mathbf x^\ast-h\mathbf e_j)\big)/(2h)$，$h=10^{-6}$（A4 检验 $h$ 减半稳定）。$\mathrm{step}$ 内 $\eta$ 取 $10^{-3}$（**只作求谱的标度**：$J$ 的谱含 $\eta$ 因子，$\eta^\ast_{\rm pred}$ 与所选 $\eta$ 无关，A4/A7 检验这一点）。
- **算例（写于读数前，闭式）**：一维玩具 $w=uv$、$G=(c+\lambda)w-b$、$u'=u-\eta Gv$、$v'=v-\eta Gu'$，在不动点 $uv=b/(c+\lambda)$ 处
$$J=\begin{pmatrix}1-\eta b&-\eta b\\-\eta b&1-\eta b\end{pmatrix}\ \Rightarrow\ \lambda\in\{1-2\eta b,\,1\}\ \Rightarrow\ \eta^\ast_{\rm toy}=\frac1b .$$
取 $b=c_b=1$：$\eta^\ast_{\rm toy}=1$，而矩阵空间给 $2/(c_b+\lambda)=1.9048$ ⇒ **两空间相差恰好 2 倍**。A1 用这个闭式校验 FD 机器。

## 4 · 预言（**写于任何 FD 读数之前**）

**代数预言（已算，`dir72/out/algebra72.json`）**：矩阵空间在 $P$-方向上的 $2\times2$ 块 $M=\begin{pmatrix}c_b+\lambda&\gamma c_b\\ 2c_b(u_2^\ast-1)&c_b u_1^{\ast2}+\lambda\end{pmatrix}$（因 $C=c_bI$、$B=c_bP$、$C_2=c_bu_1^{\ast2}P$、$P^2=P$；对称部分重数 3，反对称/补空间给 $c_b+\lambda$）⇒

| mode | $\gamma$ | $u_1^\ast$ | $\lambda_{\max}(M)$ | $\eta^\ast_{\rm pred}{}^{\rm mat}$ | $\eta^\ast_{\rm meas}$（D-92） | 实测/矩阵 |
|---|---|---|---|---|---|---|
| target | 3 | 3.79966 | 14.4859 | 0.138065 | 0.0654901 | **0.474** |
| target | 10 | 10.47185 | 109.7096 | 0.018230 | 0.00733154 | **0.402** |
| target | 30 | 29.52217 | 871.6085 | 0.002295 | 0.000574097 | **0.250** |
| keys | 3 | 0.25845 | 0.5834 | 3.428187 | 0.200414 | **0.058** |
| keys | 10 | 0.09498 | 0.5545 | 3.606782 | 0.0272972 | **0.0076** |

- **P1**：因子空间（用 §1 的复特征对形式）五格比值**全部**落 $[0.7,1.3]$ ⇒ ① PASS。理由：玩具算例的 2 倍与复特征对的"再小一半"都指向"边界 = 线性稳定界"，D-92 的 $10^{-15}$ 收敛又证明不动点确实在吸引。
- **P2**：矩阵空间只有 `target γ=3`（0.474，勉强擦边）可能落 $[0.5,2]$ ⇒ ② PASS（$\le1$ 格）。
- **P3**：因子空间零模数 $\in[8,12]$（每层固定点流形 4 维 ⇒ 8；退化因子化再多 0–4），非零模 $\ge36$。
- **P4**：$C_{\rm pred}$ 峰值在 $\gamma{=}10$ ⇒ ③ PASS。
- **风险（写死）**：若 `keys` 两格比值 $\ll0.3$ 而 `target` 三格 $\approx1$ ⇒ H72 只在"目标注入"这一种耦合上成立 ⇒ **不得**写成"边界由线性谱决定"的一般结论，须按模式分述。

## 5 · Stage A 守卫（任一 FAIL 阻断正式轮）

- **A1** 玩具闭式自检：FD Jacobian 的谱 $=\{1-2\eta b,1\}$ 到 $10^{-8}$，且 $\eta^\ast_{\rm toy}=1/b$ 到 $10^{-10}$。
- **A2** 矩阵空间 FD（72 维）$\lambda_{\max}$ 与 §4 的解析 $2\times2$ 块**逐格**吻合到相对 $10^{-6}$（`target` $\gamma=3$ 取 14.4859）。
- **A3** 因子空间零模计数 $\in[8,12]$、非零模 $\ge36$（$|\lambda|<10^{-8}$ 记零）。
- **A4** 步长与标度不变性：$h{=}10^{-6}$ vs $5\times10^{-7}$ 的 $\eta^\ast_{\rm pred}$ 相对变化 $<10^{-4}$；**求谱用的 $\eta$ 由 $10^{-3}$ 改为 $10^{-4}$ 时 $\eta^\ast_{\rm pred}$ 相对变化 $<10^{-6}$**（谱随 $\eta$ 线性缩放，判据必须标度不变）。
- **A5** $\gamma{=}0$ 锚：因子空间预言与记录值 $\eta^\ast_{\rm rec}=0.88751$（DIR-67）相差 $\le25\%$（**锚检查**，不是判据；超界说明取点/映射有 gross error）。
- **A6** $\gamma{=}0$ 逐位：`step()` 与 `ref70.run_stack` 在 $\gamma{=}0$ 下逐步同轨迹（$20\,000$ 步，$\max|\Delta W_1|=0$）。
- **A7** 无可算格：五格的 Jacobian 全部有限、$|\lambda|$ 无 NaN/Inf。

## 6 · 边界（预先声明）

- 只 $c{=}D{=}6$、$\lambda{=}0.05$、$a{=}0.1$、单种子、$R{=}2$、$L{=}2$。
- **"线性稳定界"≠"$\eta$ 的全局安全界"**：即使 H72 成立，也只说明"$\eta>\eta^\ast$ 后不动点失稳"，**不**给出从 $W(0)$ 出发的瞬态安全裕度。
- 若 H72 被否证，**不得**反推"边界由某个非线性量决定"的具体形式（本卡不测该形式）。
- 不改 v2.4 Freeze A–E；不含 $S$、不含 LLM。
