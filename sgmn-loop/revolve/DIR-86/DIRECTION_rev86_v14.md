# DIRECTION · DIR-86 · rev 1.4 —— 守卫层更正（**判据一字未改**）

**版本**：rev **1.4**（Cycle 143 冻结）。supersedes rev 1.3（保留原样）。rev 1.0（`DEF-86-1`，作废）/1.1/1.2/1.3 全部留档。
**本 rev 只改 Stage A 守卫的三处文字**（A86-15 / A86-7′ / 新增 A86-18）。
★ **判据层（P86、C86-a…d、$\Delta_{\rm base}$ 定义、判定规则、状态词、阈值、边界）一字未改**。

---

## §0 `DEF-86-4`：A86-15 **在正确实现上必然 FAIL**（rev 1.2/1.3 的守卫写错）

### 0.1 事实（从装置源码逐字读出）

`sgmn_module.py` `_install()` 的 `fwd_hook` 内，**每层每前向的执行顺序**是：

```python
m = mem.read(hs_in, self.drop_sg)      # 读：用 W_t
hs_out = hs_out + mem.gamma * m        # 注入
mem.write(hs_in, self.drop_sg)         # 写：产生 W_{t+1}
mem.cache(hs_in)                       # prev_hs = hs_in（为下一步准备）
```

注意 **`write()` 在 `cache()` 之前**，且 `write()` 首行是

```python
if self.prev_hs is None or self.no_write:
    return        # 首个时间步无 k_{l,0}（声明条件）；或写入被显式关闭
```

⇒ **第一个 chunk 的 `prev_hs` 为 `None`**（`cache` 尚未被调用过），`write()` 在**自增 `write_calls` 之前**就返回。

### 0.2 结论

$$\text{喂入 } N \text{ 块} \;\Longrightarrow\; \text{实际生效的写事件} = \max(0,\,N-1)$$

- `write_calls == steps == max(0, N-1)`（两者同在 `write()` 内自增）；
- 消费的写位置数 $=64\cdot(N-1)$。

⇒ rev 1.2 §5 / rev 1.3 §3 的 A86-15 写的是「`写入位置数 / 64 == N`」，**在 $N{=}150$ 时给出 $149\ne150$ ⇒ 必然 FAIL**。
**这不是实现错误，是守卫写错。** 若我照此实现并照此断言，一个**正确**的实现会被判阻断。

### 0.3 为什么这仍是**真守卫**（不是把守卫放松到无意义）

错的方向是「少 1」，而 $N$ 轴上的真实错法（rev 1.2 `DEF-86-3` 记的量纲错）是**差 64 倍**。两者相差两个数量级。
⇒ 更正的守卫写成**精确等式**（不是不等式、不是容差），仍能一眼区分 $N-1$ 与 $64N$、与 $N$、与常数。

---

## §1 更正后的守卫（替换 rev 1.3 §3 的对应三行）

| 编号 | 更正后的断言 | 通过条件 |
|---|---|---|
| **A86-15′**（替换 A86-15） | **写事件记账**：喂入 $N$ 块后，每个挂载层的 `write_calls` 与 `steps` 均等于 $\max(0,N-1)$ | **精确相等**（三层全部）。写关闭的臂（`INJ`）恒为 $0$ |
| **A86-15″**（同一条，单位面） | 消费写位置数 $=64\cdot\max(0,N-1)$ | **精确**；且断言 $64(N-1)\ne N$（记录 $N$ 是**块**、不是 token —— 量纲仍被钉死） |
| **A86-7″**（替换 A86-7′，**弱化到实际可证的内容**） | `INJ`（`no_write=True`）四个 $N$ 上的 NLL **逐位相同** | max abs diff $=0$。★ **它证明的是**：① `no_write` 确实生效（否则 $W$ 会变、读数会变）② 评测确定性。★ **它不证明**「`INJ` 没有误读 $W$」—— 因为 `no_write` 下 $W$ 本就不变，误读也看不出来。**该职责归 A86-17。** |
| **A86-17**（不变） | `no_write=True` 跑满 600 块后 $U,V$ 与初值逐位不变 | max abs diff $=0$（精确） |
| **A86-18**（★ 新增） | **`BASE` 的 NLL 与 $N$ 无关**（代数必然：$\gamma{=}0\Rightarrow h^+=h$，前向不依赖 $W$） | 四个 $N$ 的 NLL 两两 max abs diff $=0$（**精确**）。★ **若不为 0 ⇒ 某层 $\gamma$ 未被真正置 0**（或注入含 NaN/Inf）⇒ 阻断 |

★ **A86-18 的额外价值**：它把基线从「4 次测量」压成「1 次测量 + 1 条精确性断言」，且**顺带**证明 $\gamma$ 的置零确实作用到全部 3 层（与 A86-11′ 互为交叉校验）。

---

## §2 由此明确的两条结构性事实（写进实现与 VERDICT，不是新判据）

1. **$\mathrm{NLL}_{\rm BASE}(N)\equiv\mathrm{NLL}_{\rm BASE}$**（常数，与 $N$ 无关）
   ⇒ $\Delta_{\rm base}(N)=\mathrm{NLL}_{\rm MEM}(N)-\mathrm{NLL}_{\rm BASE}$ 的 $N$ 依赖**全部**来自 `MEM` 臂；
   ⇒ C86-b（$\Delta_{\rm base}(B)-\Delta_{\rm base}(B/4)>3\mathrm{se}$）在代数上**等价于** $\mathrm{NLL}_{\rm MEM}(B)-\mathrm{NLL}_{\rm MEM}(B/4)>3\mathrm{se}$。
   ★ **不因此简化掉基线**：仍须实测 `BASE` 并用 A86-18 证明其为常数——**否则**若 $\gamma$ 未真正置零，上式等价性不成立而无人察觉。

2. **$B$ 的口径**：$B=600$ = **喂入块数**（检查点 $\{0,150,300,600\}$ 均按块计）；实际生效写事件 $=599$。
   ⇒ $B\ge128$ 的门限（rev 1.1 §1.1）**两个口径下都成立**，且检查点按块计与 `R1_joint_read.py:50` 的 `N_CHUNKS=600` **同一单位**。本 rev 明确：**卡片中一切 $N$、$B$ 均指「喂入块数」**。

---

## §3 与 rev 1.3 的一致性

| 项 | 状态 |
|---|---|
| 装置（`R1_joint_read.py` / 根 `sgmn_module.py` + 两个 md5） | **不变** |
| 常量（$\Gamma_{\rm ref}{=}0.05$、$B{=}600$、`CHUNK=64`、$R{=}8$、`MOUNT=[8,12,16]`、`ETA=0.01`、3 seeds、60×512 评测） | **不变** |
| 三臂（`BASE`/`INJ`=`no_write`/`MEM`） | **不变** |
| $\gamma\in\{0.05,0.15\}$、3 层同值且 `requires_grad_(False)` | **不变** |
| **判据 P86 / C86-a…d、$\Delta_{\rm base}$、判定规则、边界** | **一字未改** |
| A86-15 → **A86-15′/15″**；A86-7′ → **A86-7″**（弱化到可证）；**新增 A86-18** | **本 rev 的改动** |

---

## §4 实现要点（`dir86/R1_task86.py`，本 rev 冻结口径）

- **import 复用**（不复制）：`R1_joint_read` 的 `model_and_tok / load_corpus / build_docs / build_chunks / eval_docs / rec`；根 `sgmn_module` 的 `SGMNStack`。
  ★ `R1_joint_read.py` 末尾有 `if __name__ == "__main__": main()` ⇒ **import 无副作用**（已核）。
- **基座只加载一次**：`model_and_tok()` 内部有 `_M` 缓存，且 `m.eval()` + 全部参数 `requires_grad_(False)` ⇒ **评测确定性**（已核）。每臂用 `SGMNStack` 直接构造（等价于 `make_stack` 的 6 行，仅为复用同一 `model` 实例），每臂结束 `st.remove()` 摘钩子。
- **臂**：`BASE`(`γ=0`，写开启) / `INJ`(`γ`=目标，`no_write=True`) / `MEM`(`γ`=目标，写开启)。
- **γ 施加**：三层全部 `m.gamma.data.fill_(target)` 且 `m.gamma.requires_grad_(False)`；回填三个实测值（A86-11′）。
- **检查点**：喂块**累积**推进（0→150→300→600），每次 `eval_docs` 取**逐 doc NLL**；`eval_docs` 自带 `no_write` 与 `prev_hs` 的保存/恢复 ⇒ 分段评测**不得改数学**（A86-14 实测核）。
- **落盘**：增量写 JSON（每个 `(arm,γ,seed,N)` 完成即追加），`python -u` + `setsid nohup`。
- **统计**：按 `R1_joint_read.compare` 的配对口径取 $3\mathrm{se}$；se 用**逐 doc 配对差**的标准误（$n{=}60$）。

---

## §5 下一 cycle 执行顺序

1. 核两个 md5（`sgmn_module.py` = `23759d5e…`、`R1_joint_read.py` = `57b850d4…`，trainer=本地）；
2. 写 `dir86/R1_task86.py`（按 §4）；
3. 跑 **A86-1…A86-18**（重点 15′/15″/7″/17/18/14/11′）；任一 FAIL 先修，**不进正式轮**；
4. 工作例（rev 1.0 §3.3）；
5. smoke（单 seed、$N{=}150$）——**不产生裁决**；
6. 正式轮：`{BASE,INJ,MEM} × γ{0.05,0.15} × N{0,150,300,600} × 3 seeds`，增量落盘；
7. 裁决 → `VERDICT.md` → 证据行 **`D-106`** → 台账 → `rebuild_full`/`repack`/`audit_consistency`。