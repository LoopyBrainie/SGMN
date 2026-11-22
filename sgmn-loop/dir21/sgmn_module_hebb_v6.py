"""
sgmn_module_structures.py —— DIR-14 装置：把「记忆结构」做成可插拔的核心

派生自 DIR-16 的 `sgmn_module_generalize.py`（**逐字派生**，base md5 = b5384895c75ab6e6a4d0d6535be63c93），**写规则语义未改**：
  * 本文件相对 DIR-16 装置**只增不改**：
    ① `read_params_for(mode)`：**按查询方式返回真正参与前向的读参数**
       （cue/misaddress/oracle 模式下 $W_q$ **不参与** ⇒ 优化器必须排除它，否则「训练」是空的）；
    ② `freeze_write()` / `unfreeze_write()`：读阶段冻结写（供 DIR-17 的两阶段协议用）；
    ③ `snapshot_all()` / `restore_all()`：跨训练步的记忆快照与恢复。
  * 写规则、显式 KV 核、地址机制（`addr_mode`）、读出映射 $W_o$、梯度隔离、诊断量**全部逐字沿用**。

派生自 DIR-15 的 `sgmn_module_cue.py`（**逐字派生**，base md5 = a80e637c4a6d92b6055607a98ac34797），**写规则语义未改**：
  * 本文件相对 DIR-15 装置**只改两处**：
    ① 【D29】读侧地址签名的取位：`[L-2]` → **`[L-1]`**（与写侧「事实首 token 的前一位」的约定对齐；
       DIR-15 用的是 `[L-2]`，差一位 ⇒ 但 DIR-15 仍拿到命中 1.0000 ⇒ 匹配对 ±1 位**稳健**，这一点须在报告里说清）；
    ② 新增 `addr_mode ∈ {sig, key}`：`sig` = 用**原始上下文签名**做地址（DIR-15 的做法）；
       `key` = 用**键** $P_k h$ 做地址（**阴性对照** DIR-16 的 C5 臂）。
  * 写规则、显式 KV 核、读出映射 $W_o$、梯度隔离、`last_yr`/`last_q`/`last_y` 诊断量**全部逐字沿用**。

派生自 DIR-14 的 `sgmn_module_structures.py`（**逐字派生**，base md5 = 51e870607c85c6f026cc9de239a55da7），**写规则语义未改**：
  * 本文件相对 DIR-14 装置**只改「查询的构造」与「显式 KV 核多存一列上下文签名」**；
    写规则、因子化核、多槽核、稀疏字典核、读出映射 $W_o$、梯度隔离（叶张量 + `requires_grad`）全部**逐字沿用**。
  * 新增：`query_mode ∈ {context, oracle, cue, misaddress}`；`ExplicitKVCore.sigs`（每条关联的**上下文签名** $h_{t-1}$）。
  * 动机：DIR-14 证明「写规则 / 读接口 / 键几何 / 读的选择性」四层全部排除 ⇒ 只剩 query↔key 的对应关系；
    本装置直接检验「把查询换成由**要取回的内容/线索**构造」是否可行。

派生自 DIR-13 的 `sgmn_module_keys.py`（逐字派生 + 新增 `mem_kind`），**写规则语义未改**：
所有结构都保持 v2.4 §3.1 的"局部、只看 (k,v)、只更新与该 (k,v) 相关的部分状态"。

可插拔的核心（每层一个）：
  * `dense`       —— v2.4 §3.2 稠密矩阵：W ← W − η(α(Wk−v)k^T + λ(W−W^0))；读 y = W q
  * `multi_slot`  —— 多槽（K 块 W^(j) + 槽原型 s_j）：写按 cos(k,s_j) 选槽、槽内照 §3.1 更新；
                     读按 softmax(β cos(q,s_j)) 对各槽输出加权
  * `sparse_dict` —— 稀疏字典（M 个原型 π_j + 关联值 v_j）：写更新 top-m 原型的值；读按 softmax(β cos(q,π_j)) 加权
  * `explicit_kv` —— 显式键-值分离（N 对 (k_i,v_i)）：写轮转追加；读按 softmax(β cos(q,k_i)) 加权
  * ★ DIR-18 新增：`hebbian_full` / `hebbian_fact` —— **v2.4 实际的写规则**（§3.1/§3.2/§3.3 的
    $G=\alpha(Wk-v)k^\top+\lambda(W-W^0)$ 梯度 + 全矩阵参考步 / 同步因子步），
    外加内容寻址所需的**地址索引**（keys/sigs），读为 **$y=\mathrm{sg}(W)q$**（§2.2 的线性读）

其余（读上下文 c=sg(h^+)、注入 h^+ = h + γm、位置级写、Freeze C 的 sg 隔离、阴性对照 drop_sg、
读时克隆修 D1、每篇文档前重置记忆）均与 DIR-13 一致。
"""
import math
import torch
import torch.nn.functional as F


def sg(x, drop=False):
    """v2.4 §2.1：前向保持原值、反向梯度为零。drop=True 用于阴性对照。"""
    return x if drop else x.detach()


# ====================================================================== 记忆核心
class DenseCore:
    """v2.4 §3.2：单个稠密矩阵 W（d_v × d_k）。"""

    kind = "dense"

    def __init__(self, d_k, d_v, alpha, lam, eta, gen, device):
        self.d_k, self.d_v = d_k, d_v
        self.alpha, self.lam, self.eta = alpha, lam, eta
        # 【D18】状态必须是【叶张量 + requires_grad】⇒ 阴性对照（去掉 sg）时读路径的梯度
        #   才能泄漏到写入侧 ⇒ Freeze C 的 ③ 检查才有区分力（否则恒为 0，检查形同虚设）。
        self.W = torch.zeros(d_v, d_k, device=device).requires_grad_(True)  # 写侧从 W^0=0 出发
        self.W0 = self.W.detach().clone()
        self.last_delta = 0.0

    def write(self, k, v):
        with torch.no_grad():
            G = self.alpha * torch.outer(self.W @ k - v, k) + self.lam * (self.W - self.W0)
            new = self.W - self.eta * G
            self.last_delta = float((new - self.W).norm())
            self.W.copy_(new)

    def read(self, q, beta=16.0, drop_sg=False):
        # 【D1】读时克隆（写会原地改状态，克隆切断存储共享）；【D18】drop_sg 时【不 detach】
        #   ⇒ 阴性对照的梯度可以泄漏到写入侧 ⇒ Freeze C ③ 的检查才有区分力。
        W = self.W.clone() if drop_sg else self.W.detach().clone()
        return q @ W.T                                          # (N,d_v)

    def stats(self):
        return {"slots_used": 1, "entries": 1, "state_norm": float(self.W.norm())}


class MultiSlotCore:
    """多槽：K 个槽，每槽一个 rank-r 因子化状态 (U_j, V_j)（W^(j)=U_j V_j^T）
    + 槽原型 s_j（地址）。写按键相似度选槽、槽内照 §3.1/§3.3 更新；读按 softmax(β cos(q,s_j)) 加权各槽输出。

    【设计声明】每槽 rank-r（r=8）⇒ 槽内容本身有秩上限 ⇒ 若本臂失败，
    须区分「槽结构不行」与「槽内 rank-8 容量不足」⇒ 后者另立一轮（全秩槽）。
    【§3.3 同步步】U_j ← U_j − η G V_j，V_j ← V_j − η G^T U_j（都用旧值）。"""

    kind = "multi_slot"

    def __init__(self, d_k, d_v, alpha, lam, eta, gen, device, n_slots=64, r=8,
                 proto_step=0.05):
        self.d_k, self.d_v = d_k, d_v
        self.alpha, self.lam, self.eta = alpha, lam, eta
        self.K, self.r, self.proto_step = n_slots, r, proto_step
        # 【D15】因子化槽状态必须【非零初始化】：U=V=0 时双线性更新恒为 0
        #   （U←U−η(G@V)、V←V−η(G^T@U) 两边都乘着对方的旧值 ⇒ 零是不动点）
        #   ⇒ 这是 DIR-11 N7「因子化动力学无法从零记忆出发」在多槽结构上的复现。
        #   ⇒ 采用 DIR-04 资产的初始化尺度（randn*0.1）。
        self.U = (torch.randn(n_slots, d_v, r, generator=gen) * 0.1).to(device).requires_grad_(True)
        self.V = (torch.randn(n_slots, d_k, r, generator=gen) * 0.1).to(device).requires_grad_(True)
        self.protos = torch.randn(n_slots, d_k, generator=gen).to(device)
        self.protos = self.protos / self.protos.norm(dim=1, keepdim=True).clamp_min(1e-30)
        self.used = torch.zeros(n_slots, device=device)
        self.last_delta = 0.0

    def write(self, k, v):
        with torch.no_grad():
            s = self.protos @ k
            s = s / (self.protos.norm(dim=1) * k.norm().clamp_min(1e-30)).clamp_min(1e-30)
            j = int(torch.argmax(s))
            self.used[j] += 1
            U, V = self.U[j], self.V[j]
            W = U @ V.T
            G = self.alpha * torch.outer(W @ k - v, k) + self.lam * W
            newU = U - self.eta * (G @ V)
            newV = V - self.eta * (G.T @ U)
            self.last_delta = float((newU - U).norm() + (newV - V).norm())
            self.U[j].copy_(newU)
            self.V[j].copy_(newV)
            pr = self.protos[j] + self.proto_step * k
            self.protos[j] = pr / pr.norm().clamp_min(1e-30)

    def read(self, q, beta=16.0, drop_sg=False):
        U = self.U.clone() if drop_sg else self.U.detach().clone()
        V = self.V.clone() if drop_sg else self.V.detach().clone()
        s = (q @ self.protos.T)
        s = s / (q.norm(dim=1, keepdim=True) * self.protos.norm(dim=1).unsqueeze(0)).clamp_min(1e-30)
        b = torch.softmax(beta * s, dim=-1)                       # (N,K)
        coef = torch.einsum("nd,kdr->nkr", q, V)                  # (N,K,r)  槽内系数 V_j^T q
        y = torch.einsum("nk,nkr,kvr->nv", b, coef, U)           # (N,d_v)  Σ_j b_j U_j (V_j^T q)
        return y

    def stats(self):
        return {"slots_used": int((self.used > 0).sum()), "entries": int(self.used.sum()),
                "state_norm": float(self.U.norm() + self.V.norm())}


class SparseDictCore:
    """稀疏字典：M 个原型 π_j + 关联值 v_j；写更新 top-m 原型的值，读按原型相似度加权。"""

    kind = "sparse_dict"

    def __init__(self, d_k, d_v, alpha, lam, eta, gen, device, n_proto=64, m_update=4,
                 proto_step=0.05):
        self.d_k, self.d_v = d_k, d_v
        self.alpha, self.lam, self.eta = alpha, lam, eta
        self.M, self.m_update, self.proto_step = n_proto, m_update, proto_step
        self.protos = torch.randn(n_proto, d_k, generator=gen).to(device)
        self.protos = self.protos / self.protos.norm(dim=1, keepdim=True).clamp_min(1e-30)
        self.vals = torch.zeros(n_proto, d_v, device=device).requires_grad_(True)
        self.used = torch.zeros(n_proto, device=device)
        self.last_delta = 0.0

    def write(self, k, v):
        with torch.no_grad():
            s = (self.protos @ k)
            s = s / (self.protos.norm(dim=1) * k.norm().clamp_min(1e-30)).clamp_min(1e-30)
            idx = torch.topk(s, self.m_update).indices
            tot = 0.0
            for j in idx:
                j = int(j)
                self.used[j] += 1
                w = float(s[j].clamp_min(0.0))
                upd = self.eta * w * (self.alpha * (self.vals[j] - v) + self.lam * self.vals[j])
                new = self.vals[j] - upd
                tot += float((new - self.vals[j]).norm())
                self.vals[j].copy_(new)
                p = (1 - self.proto_step) * self.protos[j] + self.proto_step * w * k
                self.protos[j] = p / p.norm().clamp_min(1e-30)
            self.last_delta = tot

    def read(self, q, beta=16.0, drop_sg=False):
        s = (q @ self.protos.T)
        s = s / (q.norm(dim=1, keepdim=True) * self.protos.norm(dim=1).unsqueeze(0)).clamp_min(1e-30)
        a = torch.softmax(beta * s, dim=-1)
        return a @ (self.vals.clone() if drop_sg else self.vals.detach().clone())

    def stats(self):
        return {"slots_used": int((self.used > 0).sum()), "entries": int(self.used.sum()),
                "state_norm": float(self.vals.norm())}


class ExplicitKVCore:
    """显式键-值分离：N 对 (k_i, v_i)；写轮转追加；读按 softmax(β cos(q,k_i)) 加权。"""

    kind = "explicit_kv"

    def __init__(self, d_k, d_v, alpha, lam, eta, gen, device, n_entries=64, d_model_sig=0):
        self.d_k, self.d_v = d_k, d_v
        self.alpha, self.lam, self.eta = alpha, lam, eta
        self.N = n_entries
        self.keys = torch.zeros(n_entries, d_k, device=device).requires_grad_(True)
        self.vals = torch.zeros(n_entries, d_v, device=device).requires_grad_(True)
        # 【DIR-15 新增】上下文签名旁表：第 i 条关联写入时的上下文签名 h_{t-1}（与键 k 的**同一**输入）。
        # 用途：以「地址（写侧上下文）」而不是「键的线性投影」来匹配查询 ⇒ 内容寻址式读取。
        self.sigs = torch.zeros(n_entries, d_model_sig, device=device)
        self.count = 0
        self.last_delta = 0.0

    def write(self, k, v, sig=None):
        with torch.no_grad():
            i = self.count % self.N
            self.count += 1
            self.last_delta = float((k.norm() ** 2 + v.norm() ** 2) ** 0.5)
            self.keys[i].copy_(k)
            self.vals[i].copy_(v)
            if sig is not None:
                self.sigs[i].copy_(sig)

    def read(self, q, beta=16.0, drop_sg=False):
        # 【D21】键是**写侧量**（§2.3 的 k = sg(P_k h)）⇒ 必须从计算图里摘出，
        #   否则「写侧原地更新 keys」会破坏跨越该前向的反向传播（version counter），
        #   且正常的 (sg) 模式下写侧参数会拿到梯度 ⇒ 违反 Freeze C。
        #   值同理：正常模式 detach，阴性对照 (drop_sg=True) 保留克隆 ⇒ 泄漏可被检出。
        K = self.keys.detach()
        s = (q @ K.T)
        s = s / (q.norm(dim=1, keepdim=True) * K.norm(dim=1).unsqueeze(0)).clamp_min(1e-30)
        a = torch.softmax(beta * s, dim=-1)
        return a @ (self.vals.clone() if drop_sg else self.vals.detach().clone())

    def stats(self):
        return {"slots_used": min(self.count, self.N), "entries": int(self.count),
                "state_norm": float(self.vals.norm())}


# ====================================================================== 层与栈
class HebbianCore:
    """★ DIR-18 新增：**v2.4 实际的写规则**（§3.1/§3.2/§3.3）+ 内容寻址所需的**地址索引**。

    核状态（可塑）：
      * `hebbian_full` —— 全矩阵参考步（§3.2）：$W \leftarrow W - \eta G$；
      * `hebbian_fact` —— 同步因子步（§3.3）：$U \leftarrow U-\eta GV$，$V \leftarrow V-\eta G^\top U$（**用旧 U、旧 V**）。
      其中 $G = \alpha(Wk-v)k^\top + \lambda(W-W^0)$（§3.1，逐字沿用 DIR-08 的实现）。

    地址索引（**不是记忆内容**，只用于「地址选键」）：`keys`（写入时的键）、`sigs`（上下文签名 $h_{t-1}$）。
    ★ `vals` 额外保留写入的 $v_i$，**仅用于测量写保真度** $\cos(Wk_i,v_i)$，不参与读路径。

    读：$y = \mathrm{sg}(W)\,q$（§2.2 的线性读，作用在地址选出的键 $q$ 上）。
    """

    kind = "hebbian_full"

    def __init__(self, d_k, d_v, alpha, lam, eta, gen, device, factorized=False,
                 n_entries=64, d_model_sig=0, d_k_model=0):
        self.d_k, self.d_v = d_k, d_v
        self.alpha, self.lam, self.eta = alpha, lam, eta
        self.N = n_entries
        self.factorized = factorized
        if factorized:
            self.kind = "hebbian_fact"
            self.U = (torch.randn(d_v, d_k, generator=gen) / math.sqrt(d_k)).to(device)
            self.V = (torch.randn(d_k, d_v, generator=gen) / math.sqrt(d_v)).to(device)
            self.W = torch.zeros(d_v, d_k, device=device)
        else:
            self.kind = "hebbian_full"
            self.W = (torch.randn(d_v, d_k, generator=gen) / math.sqrt(d_k)).to(device)
            self.U = None
            self.V = None
        self.W0 = self.W.detach().clone()          # §3.1 的参考点（$W-W^0$ 项）
        self.keys = torch.zeros(n_entries, d_k, device=device)
        self.vals = torch.zeros(n_entries, d_v, device=device)   # ★ 仅用于写保真度测量
        self.sigs = torch.zeros(n_entries, d_model_sig, device=device)
        self.count = 0
        self.last_delta = 0.0
        self.data_norm = 0.0        # 最后一步的数据项范数
        self.reg_norm = 0.0         # 最后一步的正则项范数

    def _W(self):
        return (self.W0 + self.U @ self.V.T) if self.factorized else self.W

    def write(self, k, v, sig=None):
        with torch.no_grad():
            i = self.count % self.N
            self.count += 1
            self.last_delta = float((k.norm() ** 2 + v.norm() ** 2) ** 0.5)
            self.keys[i].copy_(k)
            self.vals[i].copy_(v)
            if sig is not None:
                self.sigs[i].copy_(sig)
            W = self._W()
            G = self.alpha * torch.outer(W @ k - v, k) + self.lam * (W - self.W0)   # §3.1
            self.data_norm = float((self.alpha * torch.outer(W @ k - v, k)).norm())
            self.reg_norm = float((self.lam * (W - self.W0)).norm())
            if self.factorized:
                # §3.3 同步因子步：**用旧 U、旧 V**
                self.U.copy_(self.U - self.eta * (G @ self.V))
                self.V.copy_(self.V - self.eta * (G.T @ self.U))
            else:
                # §3.2 全矩阵参考步
                self.W.copy_(self.W - self.eta * G)

    def read(self, q, beta=16.0, drop_sg=False):
        """§2.2 的线性读：$y = \mathrm{sg}(W) q$。（$q$ 由上层「地址选键」给出。）"""
        W = self._W()
        return (sg(W, drop_sg) @ q.T).T            # (N, d_v)

    def fidelity(self, j):
        """写保真度：$\cos(W k_j,\; v_j)$（$j$ 为槽下标）。"""
        with torch.no_grad():
            W = self._W()
            a = W @ self.keys[j]
            b = self.vals[j]
            if a.norm() < 1e-30 or b.norm() < 1e-30:
                return 0.0
            return float((a @ b) / (a.norm() * b.norm()))

    def stats(self):
        return {"slots_used": min(self.count, self.N), "entries": int(self.count),
                "state_norm": float(self._W().norm()),
                "w0_norm": float(self.W0.norm()),
                "data_norm": self.data_norm, "reg_norm": self.reg_norm}


class LayerMem:
    def __init__(self, d_model, d_k, d_v, r, alpha, lam, eta, gamma, device, gen,
                 use_factorized=True, no_write=False, perpos=False,
                 mem_kind="dense", n_slots=64, n_proto=64, n_entries=64,
                 key_mode="linear"):
        self.d_model, self.d_k, self.d_v, self.r = d_model, d_k, d_v, r
        self.alpha, self.lam, self.eta = alpha, lam, eta
        self.use_factorized = use_factorized
        self.no_write = no_write
        self.perpos = perpos
        self.mem_kind = mem_kind
        self.read_beta = 16.0        # 地址 softmax 的锐度
        # 【DIR-15 新增】查询的构造方式：
        #   context     —— v2.4 现状：q = W_q · c（用当前位置上下文的线性投影）
        #   oracle      —— 诊断上界：q = 写阶段记录的目标键
        #   cue         —— 内容寻址：先用「上下文签名」匹配出条目，取该条目的键作查询
        #   misaddress  —— 严格对照：取线索匹配结果的邻居条目的键
        self.query_mode = "context"
        # 【DIR-16 新增】地址的构造方式：
        #   sig —— 用**原始上下文签名** h_{t-1} 做地址（DIR-15 的做法）
        #   key —— 用**键** P_k h_{t-1} 做地址（阴性对照：DIR-16 的 C5 臂）
        self.addr_mode = "sig"
        self.oracle_q = None         # (d_k,) 诊断上界用
        self.last_cue_idx = None     # 最近一次线索匹配选中的条目
        self.last_cue_scores = None  # (N,) 签名余弦
        self.key_mode = key_mode     # linear | sparse | ortho | renorm（★ DIR-18 rev 1.2 修订 54）
        # ★★【DIR-18 rev 1.5 · 修订 68】**内容置换**开关（记忆内「内容错」对照）
        #   True ⇒ 写入记忆的**值**取自文档内另一个位置 ⇒ **内容错**；**地址完全不动**。
        self.content_wrong = False
        self._cw_gen = torch.Generator(device="cpu"); self._cw_gen.manual_seed(20261106)
        self.key_topk = 32
        # ★【DIR-18】ortho 键模式：键取自**随机正交基**的列（按写入序号取）⇒ 写入键两两正交
        #   ⇒ 隔离「键共线」（DIR-13 实测成对余弦 0.987–0.995、有效槽位 PR≈1.01）这一已知障碍。
        self.ortho_basis = None      # (d_model, d_model) 正交矩阵（惰性生成）
        self.ortho_idx = 0           # 正交基的取列游标
        self.last_q = None
        self.last_y = None      # 诊断用：实际注入的量（Wo y）—— 与 DIR-11…14 的口径一致
        self.last_yr = None     # 【D22】诊断用：**记忆读出 y**（未过 Wo）⇒ 命中指标的正确对象
        self.keys_written = []
        self.vals_written = []
        self.slot_of_write = []      # ★ 每次写占用的槽号（精确映射，见 D54）
        # ★【DIR-18 rev 1.2 修订 53 · D47】写侧自监督的**记录槽**：
        #   `W4b` 臂用它构造 `L_write = −cos(W k_j, v_j)`，**只更新 P_v**。
        #   记录的是**写入时的键/值/签名**（用于反查真值槽），不含任何梯度图。
        self.write_pairs = []          # [(k, v, sig), ...]
        self.hs_for_write = []      # [h_t, ...]（值映射 P_v 的输入；写侧自监督要重算 P_v·h_t）
        self.prev_hs = None
        self.write_calls = 0
        self.read_calls = 0
        self.steps = 0
        self.pk_scale = 1.0
        mk = lambda *s: (torch.randn(*s, generator=gen) / math.sqrt(s[0])).to(device).requires_grad_(True)
        self.Pk = mk(d_model, d_k)
        self.Pv = mk(d_model, d_v)
        self.Wq = mk(d_k, d_model)
        self.Wo = mk(d_model, d_v)
        self.gamma = torch.tensor(float(gamma), device=device, requires_grad=True)
        if mem_kind == "dense":
            self.core = DenseCore(d_k, d_v, alpha, lam, eta, gen, device)
        elif mem_kind == "multi_slot":
            self.core = MultiSlotCore(d_k, d_v, alpha, lam, eta, gen, device, n_slots=n_slots)
        elif mem_kind == "sparse_dict":
            self.core = SparseDictCore(d_k, d_v, alpha, lam, eta, gen, device, n_proto=n_proto)
        elif mem_kind in ("hebbian_full", "hebbian_fact"):
            self.core = HebbianCore(d_k, d_v, alpha, lam, eta, gen, device,
                                    factorized=(mem_kind == "hebbian_fact"),
                                    n_entries=n_entries, d_model_sig=d_model)
        elif mem_kind == "explicit_kv":
            self.core = ExplicitKVCore(d_k, d_v, alpha, lam, eta, gen, device, n_entries=n_entries,
                                       d_model_sig=d_model)
        else:
            raise ValueError(mem_kind)

    # ------------------------------------------------------------------ 读
    def _cue_match(self, hs_in):
        """线索匹配：读侧的**地址**与写入时记录的地址比较（余弦），返回 (best_idx, scores)。

        【D29】读侧地址取 `hs_in[0][L-1]`（**最后一位**）——与写侧约定对齐：
        写侧在事实首 token 位置 t 写入时，记的签名是 $h_{t-1}$（**事实首 token 的前一位**）；
        读阶段前缀的最后一位正是「事实首 token 的前一位」⇒ 两侧取同一个量。
        （DIR-15 误用 `[L-2]`，差一位；DIR-15 仍拿到命中 1.0 ⇒ 匹配对 ±1 位稳健。）

        `addr_mode='sig'`：地址 = 原始上下文签名 $h$（d_model 维）；
        `addr_mode='key'`：地址 = 键 $P_k h$（d_k 维）⇒ **阴性对照**。
        """
        core = self.core
        n = min(core.count, core.N)
        if n == 0:
            return None, None
        L = hs_in.shape[1]
        h_last = hs_in[0, L - 1].detach()                       # 【D29】读侧地址的取位
        if self.addr_mode == "key":
            addr_q = (self.Pk.detach() @ h_last)
            stored = core.keys[:n].detach()
        else:
            addr_q = h_last
            stored = core.sigs[:n]
        sc = stored @ addr_q
        sc = sc / (stored.norm(dim=1) * addr_q.norm()).clamp_min(1e-30)
        return int(torch.argmax(sc).item()), sc

    def read(self, hs_in, drop_sg=False):
        c = sg(hs_in, drop_sg)
        cf = c.reshape(-1, c.shape[-1])
        if self.query_mode == "context":
            q = (self.Wq @ cf.T).T                                # (N,d_k)
        elif self.query_mode == "oracle":
            assert self.oracle_q is not None, "oracle 模式必须提供 oracle_q"
            q = self.oracle_q.reshape(1, -1).expand(cf.shape[0], -1).contiguous()
        elif self.query_mode in ("cue", "misaddress"):
            best, sc = self._cue_match(hs_in)
            self.last_cue_idx, self.last_cue_scores = best, sc
            if best is None:
                q = (self.Wq @ cf.T).T
            else:
                n = min(self.core.count, self.core.N)
                j = best if self.query_mode == "cue" else (best + 1) % n
                q = self.core.keys[j].detach().reshape(1, -1).expand(cf.shape[0], -1).contiguous()
        else:
            raise ValueError(self.query_mode)
        if self.key_mode == "sparse":
            thr = q.abs().topk(self.key_topk, dim=-1).values[:, -1:]
            q = q * (q.abs() >= thr).to(q.dtype)
        y = self.core.read(q, self.read_beta, drop_sg)             # (N,d_v) 记忆读出
        # 【D17】v2.4 §2.2：m = W_o y（读出映射）—— 原实现漏掉了这一层 ⇒ W_o 拿不到梯度，
        #   且注入的量纲不对。修复：m = Wo y。
        m = (self.Wo @ y.T).T                                      # (N,d_v)
        self.last_q = q.detach()
        self.last_yr = y.detach()     # 【D22】记忆读出 y（d_v 空间，与写入值同空间）
        self.last_y = m.detach()      # 诊断用：实际注入的量（Wo y）
        return m

    # ------------------------------------------------------------------ 写
    def write(self, hs_in, drop_sg=False):
        """位置级写（v2.4 §2.3/§3.1/§3.2 的逐步语义）：t=1..L-1 逐位置写入。"""
        if self.no_write:
            return
        if hs_in.shape[0] != 1 or not self.perpos:
            raise RuntimeError("DIR-14 装置只实现位置级写（B=1）")
        L = hs_in.shape[1]
        if L < 2:
            return
        with torch.no_grad():
            for t in range(1, L):
                if self.key_mode == "ortho":
                    # ★【DIR-18】正交键：k = 正交基的第 j 列，标定到与线性键同样的 ‖k‖²≈5
                    if self.ortho_basis is None:
                        g = torch.Generator(device="cpu"); g.manual_seed(1234 + int(self.ortho_idx))
                        A = torch.randn(self.d_model, self.d_model, generator=g)
                        Q, _ = torch.linalg.qr(A)
                        self.ortho_basis = Q.to(self.Pk.device)
                    j = self.ortho_idx % self.ortho_basis.shape[1]
                    self.ortho_idx += 1
                    k = self.ortho_basis[:, j].to(self.Pk.dtype)
                    k = k * (math.sqrt(5.0) / float(k.norm().clamp_min(1e-30)))
                else:
                    k = sg(hs_in[0, t - 1].reshape(1, -1) @ self.Pk, drop_sg).mean(dim=0)
                    if self.key_mode == "sparse":
                        thr = k.abs().topk(self.key_topk).values[-1]
                        k = k * (k.abs() >= thr).to(k.dtype)
                    if self.key_mode == "renorm":
                        # ★【DIR-18 rev 1.2 修订 54 · D50/D51】**逐写归一化**到 ‖k‖²=5。
                        #   根因（D50）：装置的 `pk_scale` 是用**随机 N(0,1) 向量**标定的
                        #   （E‖k‖²=5），而真实 hidden state 的范数远大于随机向量
                        #   ⇒ 实测 ‖k‖ mean=131 / max=3451（离散 26×）⇒
                        #   稳定上界 η<2/(λ+α‖k‖²) 与学习所需 η 窗口**不相交**。
                        #   逐写归一化 ⇒ 键尺度**恒为 5** ⇒ 两个窗口重新相交。
                        k = k * (math.sqrt(5.0) / float(k.norm().clamp_min(1e-30)))
                    else:
                        k = k * self.pk_scale
                # ★★【DIR-18 rev 1.5 · 修订 68】**内容置换**：只换「值」的来源位置，
                #   **键 k 与签名 sig 完全不动** ⇒ 寻址不变、内容变错 ⇒ 干净的「记忆内」对照。
                #   置换 = 文档内**无不动点**的随机置换（derangement）⇒ 每个写入的值都来自别的 token。
                if self.content_wrong:
                    _L1 = L - 1                                   # 值位置集合 = {1..L-1}
                    if not hasattr(self, "_cw_perm") or self._cw_perm is None                             or self._cw_perm.numel() != _L1:
                        _p = torch.randperm(_L1, generator=self._cw_gen)   # 0..L-2
                        for _i in range(_L1):                      # 消除不动点（derangement）
                            if int(_p[_i]) == _i:
                                _j = (_i + 1) % _L1
                                # ★ 必须先取**标量副本**再交换：torch 的 `p[i]` 是**视图**，
                                #   `p[i], p[j] = p[j], p[i]` 会因别名而写坏排列（实测：不再是排列）。
                                _a, _b = int(_p[_i]), int(_p[_j])
                                _p[_i] = _b; _p[_j] = _a
                        self._cw_perm = _p
                    _src = 1 + int(self._cw_perm[t - 1])          # 映射到 1..L-1
                else:
                    _src = t
                v = sg(hs_in[0, _src].reshape(1, -1) @ self.Pv, drop_sg).mean(dim=0)
                self.keys_written.append(k.detach().clone())
                self.vals_written.append(v.detach().clone())
                # ★【DIR-18 rev 1.2 修订 53】记录**值映射的输入 h_t**（写侧自监督要重算 P_v·h_t）
                self.hs_for_write.append(hs_in[0, _src].detach().clone())   # ★ 与置换后的 v 一致
                sig = hs_in[0, t - 1].detach()                    # 【DIR-15】上下文签名（键的同一输入）
                # ★【DIR-18 rev 1.2 修订 53】记录写对（供 `W4b` 的写侧自监督反查真值槽）
                self.write_pairs.append((k.detach().clone(), v.detach().clone(),
                                         sig.clone()))
                # ★【DIR-18 rev 1.2 修订 65 · D54】记录本次写**实际占用的槽号**
                #   （`core.write` 内部是 `i = count % N; count += 1` ⇒ 槽号可精确预测）
                #   ⇒ ⇒ 「真值槽」不再靠「值内容反查」（在有陈旧残留时反查会歧义），
                #   而是**精确的写入序号 → 槽号映射** ⇒ ⇒ 两口径（写入序 / 槽空间）可分离。
                _slot = int(self.core.count % self.core.N)
                self.slot_of_write.append(_slot)
                if hasattr(self.core, "sigs"):
                    self.core.write(k, v, sig=sig)
                else:
                    self.core.write(k, v)
                self.write_calls += 1
                self.steps += 1

    def write_side_loss(self, j_true):
        """【DIR-18 rev 1.2 修订 53 · D47】**写侧自监督**损失（只对 `P_v` 有梯度）。

        L_write = −cos(W k_j, v_j)，其中 `v_j = P_v · h_t`（**重新计算 ⇒ 梯度流入 P_v**），
        `W` 与 `k_j` 均为**常量**（核状态 no_grad ⇒ 梯度**不流入 W**）。

        ⇒ 这是 v2.4 §7.2「受保护的联合训练」在 **Freeze C 下唯一可实现**的形态：
        写侧有自己的目标、读侧有自己的目标，二者**互不污染**。

        `j_true`：真值条目在**写入顺序**中的下标（= `true_idx`）。
        """
        if not self.write_pairs or j_true is None or j_true >= len(self.write_pairs):
            return None
        k_j, _v_j, _s_j = self.write_pairs[j_true]
        W = self.core._W().detach()
        # 真值槽的值 = P_v · h_t：**重算**以获得对 P_v 的梯度（h_t 由 `vals_written` 的来源给出）
        h_t = self.hs_for_write[j_true] if hasattr(self, "hs_for_write") else None
        if h_t is None:
            return None
        v_j = self.Pv @ h_t
        cos = F.cosine_similarity((W @ k_j).unsqueeze(0), v_j.unsqueeze(0), dim=1)
        return -cos.mean()

    def cache(self, hs_in):
        self.prev_hs = hs_in.detach()

    # ------------------------------------------------------------------ 参数分组
    def read_params(self):
        return [self.Wq, self.Wo, self.gamma]

    def write_params(self):
        return []

    def named_read_params(self):
        return [("Wq", self.Wq), ("Wo", self.Wo), ("gamma", self.gamma)]

    def named_write_params(self):
        out = [("Pk", self.Pk), ("Pv", self.Pv)]
        if self.mem_kind == "dense":
            out += [("W", self.core.W)]
        elif self.mem_kind == "multi_slot":
            out += [("U", self.core.U), ("V", self.core.V), ("protos", self.core.protos)]
        elif self.mem_kind == "sparse_dict":
            out += [("protos", self.core.protos), ("vals", self.core.vals)]
        elif self.mem_kind in ("hebbian_full", "hebbian_fact"):
            out += [("W", self.core.W)]
            if self.core.factorized:
                out += [("U", self.core.U), ("V", self.core.V)]
        else:
            out += [("keys", self.core.keys), ("vals", self.core.vals)]
        return out

    def state_snapshot(self):
        if self.mem_kind == "dense":
            return self.core.W.detach().clone()
        if self.mem_kind in ("hebbian_full", "hebbian_fact"):
            return (self.core.W.detach().clone(),
                    (self.core.U.detach().clone() if self.core.factorized else None),
                    (self.core.V.detach().clone() if self.core.factorized else None),
                    self.core.keys.detach().clone(), self.core.vals.detach().clone(),
                    self.core.sigs.detach().clone(), int(self.core.count))
        if self.mem_kind == "explicit_kv":
            return (self.core.keys.detach().clone(), self.core.vals.detach().clone(),
                    self.core.sigs.detach().clone(), int(self.core.count))
        if self.mem_kind == "multi_slot":
            return (self.core.U.detach().clone(), self.core.V.detach().clone(),
                    self.core.protos.detach().clone(), self.core.used.detach().clone())
        if self.mem_kind == "sparse_dict":
            return (self.core.vals.detach().clone(), self.core.protos.detach().clone(),
                    self.core.used.detach().clone())
        return (self.core.keys.detach().clone(), self.core.vals.detach().clone(), self.core.count)

    def state_restore(self, snap):
        with torch.no_grad():
            if self.mem_kind in ("hebbian_full", "hebbian_fact"):
                W, U, V, K, Vv, S, cnt = snap
                self.core.W.copy_(W)
                if self.core.factorized:
                    self.core.U.copy_(U); self.core.V.copy_(V)
                self.core.keys.copy_(K); self.core.vals.copy_(Vv); self.core.sigs.copy_(S)
                self.core.count = int(cnt)
            elif self.mem_kind == "dense":
                self.core.W.copy_(snap)
            elif self.mem_kind == "multi_slot":
                self.core.U.copy_(snap[0]); self.core.V.copy_(snap[1])
                self.core.protos.copy_(snap[2]); self.core.used.copy_(snap[3])
            elif self.mem_kind == "sparse_dict":
                self.core.vals.copy_(snap[0]); self.core.protos.copy_(snap[1])
                self.core.used.copy_(snap[2])
            elif self.mem_kind == "explicit_kv":
                self.core.keys.copy_(snap[0]); self.core.vals.copy_(snap[1])
                self.core.sigs.copy_(snap[2]); self.core.count = int(snap[3])
            else:
                self.core.keys.copy_(snap[0]); self.core.vals.copy_(snap[1])
                self.core.count = snap[2]
        self.write_calls = 0
        self.steps = 0
        self.keys_written.clear()
        self.vals_written.clear()
        self.slot_of_write.clear()      # ★ DIR-18 rev 1.2 修订 65 · D54
        self.write_pairs.clear()        # ★ DIR-18 rev 1.2 修订 53
        self.hs_for_write.clear()      # ★ DIR-18 rev 1.2 修订 53

    def state_norm(self):
        return float(self.state_snapshot().__class__ and 0) if False else \
            float(self.core.stats()["state_norm"])


def grad_report(named):
    out = []
    for n, t in named:
        g = t.grad
        if g is None:
            out.append((n, 0.0, True, False))
        else:
            gv = float(g.abs().max())
            bad = bool(torch.isnan(g).any() or torch.isinf(g).any())
            out.append((n, gv, False, bad))
    return out


class SGMNStack:
    def __init__(self, model, mount_layers, d_k, d_v, r, alpha, lam, eta, gamma,
                 device, seed=0, use_factorized=True, drop_sg=False, no_write=False,
                 perpos=True, mem_kind="dense", n_slots=64, n_proto=64, n_entries=64,
                 key_mode="linear"):
        self.model = model
        self.device = device
        self.drop_sg = drop_sg
        gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
        self.mems = {}
        self._hs_in = {}
        for li in mount_layers:
            m = LayerMem(model.config.hidden_size, d_k, d_v, r, alpha, lam, eta, gamma,
                         device, gen, use_factorized, no_write=no_write, perpos=perpos,
                         mem_kind=mem_kind, n_slots=n_slots, n_proto=n_proto,
                         n_entries=n_entries, key_mode=key_mode)
            with torch.no_grad():
                x = torch.randn(20000, model.config.hidden_size, generator=gen).to(device)
                kk = (x @ m.Pk.detach()) * m.pk_scale
                m.pk_scale = m.pk_scale / float((kk ** 2).mean().sqrt()) * math.sqrt(5.0)
            self.mems[li] = m
        self._handles = []
        self._install()

    def _layers(self):
        for attr in ("model", "transformer"):
            base = getattr(self.model, attr, None)
            if base is not None and hasattr(base, "layers"):
                return base.layers
        return self.model.layers

    def _install(self):
        layers = self._layers()
        for li, mem in self.mems.items():
            layer = layers[li]

            def pre_hook(module, args, kwargs, li=li):
                hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                self._hs_in[li] = hs
                return None

            def fwd_hook(module, args, kwargs, output, li=li):
                mem = self.mems[li]
                hs_in = self._hs_in[li]
                hs_out = output[0] if isinstance(output, tuple) else output
                m = mem.read(hs_in, self.drop_sg)
                hs_out = hs_out + mem.gamma * m
                mem.write(hs_in, self.drop_sg)
                mem.cache(hs_in)
                if isinstance(output, tuple):
                    return (hs_out,) + tuple(output[1:])
                return hs_out

            self._handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            self._handles.append(layer.register_forward_hook(fwd_hook, with_kwargs=True))

    def named_read_params(self):
        out = []
        for li, m in self.mems.items():
            out += [(f"L{li}.{n}", t) for n, t in m.named_read_params()]
        return out

    # ------------------------------------------------------------------ DIR-17 新增
    def read_params_for(self, mode=None):
        """返回**真正参与该查询方式前向**的读参数（用于优化器构造）。

        cue / misaddress / oracle 模式：$q$ 由匹配结果直接给出 ⇒ **$W_q$ 不参与前向**
        ⇒ 它的梯度恒为 0 ⇒ **必须**从优化器里排除（否则「训练」什么都没训）。
        context 模式：$q = W_q c$ ⇒ $W_q$ 参与。
        """
        out = []
        for li, m in self.mems.items():
            mm = mode if mode is not None else m.query_mode
            out.append((f"L{li}.Wo", m.Wo))
            out.append((f"L{li}.gamma", m.gamma))
            if mm == "context":
                out.append((f"L{li}.Wq", m.Wq))
        return out

    def freeze_write(self):
        for m in self.mems.values():
            m.no_write = True

    def unfreeze_write(self):
        for m in self.mems.values():
            m.no_write = False

    def snapshot_all(self):
        return {li: m.state_snapshot() for li, m in self.mems.items()}

    def restore_all(self, snap):
        for li, m in self.mems.items():
            m.state_restore(snap[li])
            m.prev_hs = None

    def named_write_params(self):
        out = []
        for li, m in self.mems.items():
            out += [(f"L{li}.{n}", t) for n, t in m.named_write_params()]
        return out

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ================================================================ DIR-19 新增
    # —— ① 可训基座参数集合（保护旋钮 N1/N3/N4）——
    def trainable_base_params(self, mode="none", layers=None):
        """按**保护旋钮**给出「可训基座参数集合」。

        ★ **先全部冻结**，再按 mode 放开 ⇒ ⇒ 「没被放开的参数」梯度**恒为 0**（纪律 ⑭ 的生效性检查）。
        mode:
          "none"    ⇒ 空（基座全冻结 = DIR-17/18 的做法）
          "all"     ⇒ 全部基座参数
          "layers"  ⇒ 只放开 `layers` 指定层的参数（旋钮 N4）
          "lowrank" ⇒ 只放开 2-D 权重的**低秩子空间**内的分量（旋钮 N3，由 `lowrank_projections` 约束）
        """
        for p in self.model.parameters():
            p.requires_grad_(False)
        out = []
        for name, p in self.model.named_parameters():
            keep = False
            if mode == "all":
                keep = True
            elif mode == "layers" and layers:
                keep = any((f"layers.{li}." in name) or (f".layers.{li}." in name)
                           or (f"h.{li}." in name) for li in layers)
            elif mode == "lowrank":
                keep = (p.dim() == 2)
            if keep:
                p.requires_grad_(True)
                out.append((name, p))
        return out

    # —— ② 低秩子空间投影（旋钮 N3）——
    def lowrank_projections(self, r=8, layers=None):
        """对每个 2-D 基座权重 W 计算 (W0, U_r)（U_r = SVD 前 r 个左奇异向量）。

        ⇒ 允许的更新子空间 = span(U_r)；投影矩阵 P = U_r U_rᵀ。
        ★ **SVD 代价**：每个矩阵约 1–2 s ⇒ ⇒ **必须缓存**（基座初值全臂相同 ⇒ 只算一次）。
        """
        out = {}
        for name, p in self.model.named_parameters():
            if p.dim() != 2:
                continue
            if layers is not None:
                if not any((f"layers.{li}." in name) or (f".layers.{li}." in name)
                           or (f"h.{li}." in name) for li in layers):
                    continue
            W0 = p.detach().clone()
            U, S, Vh = torch.linalg.svd(W0.float(), full_matrices=False)
            # ★★ 缺陷 D67：**存到 CPU**（GPU 上 169 个 896×4864 的 W0 ≈ 2.9 GB ⇒ OOM）
            #   ⇒ 投影时**逐参数**搬到设备上、用完即释放（瞬时 ~35 MB）。
            out[name] = (W0.cpu(), U[:, :r].contiguous().cpu())
        return out

    def project_base_to_subspace(self, projs):
        """把基座权重投影回 `W0 + span(U_r)` ⇒ ⇒ 漂移**严格**落在低秩子空间内。"""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name not in projs:
                    continue
                W0, U = projs[name]
                _dev = p.device
                W0 = W0.to(_dev, non_blocking=True)
                U = U.to(_dev, non_blocking=True)
                W = p.detach()
                dW = W - W0
                dW = U @ (U.T @ dW)          # 投影到 span(U)
                p.copy_(W0 + dW)
                del W0, U, dW

    def subspace_residual(self, projs):
        """返回 `max ‖(I−P)(W−W0)‖`（应 ≈ 0 ⇒ 生效性检查用）。"""
        worst = 0.0
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name not in projs:
                    continue
                W0, U = projs[name]
                _dev = p.device
                W0 = W0.to(_dev, non_blocking=True)
                U = U.to(_dev, non_blocking=True)
                dW = (p.detach() - W0).float()
                res = dW - U @ (U.T @ dW)
                worst = max(worst, float(res.norm()))
                del W0, U, dW, res
        return worst

    # —— ③ 记忆钩子的临时摘除 / 复原（供 KL 的「不含记忆」参考分布用）——
    def suspend_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return len(self._handles)

    def resume_hooks(self):
        if not self._handles:
            self._install()
        return len(self._handles)

    # —— ④ 基座漂移的 KL 正则项 ——
    def kl_to_reference(self, model_ref, ids):
        """KL( π_θ0^base(·|x) ‖ π_θ^base(·|x) )，两侧**都不含记忆**。

        ★ 口径（DIR-19 修订 19-B）：**摘除记忆钩子**后前向 ⇒ ⇒ 该项**纯粹度量基座漂移**。
        ★ `model_ref` = 训练开始前的**冻结快照**（fp32）。
        """
        import torch.nn.functional as _F
        self.suspend_hooks()
        try:
            with torch.no_grad():
                # ★★ 缺陷 D73：`logits` 全序列 = [1, ~870, 151936] fp32 ≈ **520 MB**
                #   ⇒ 与基座梯度（2 GB）叠加 ⇒ **T4 15 GB OOM**。
                #   修法：`logits_to_keep=1` ⇒ 只算**最后一位**的 logits（≈0.6 MB）。
                try:
                    lg_ref = model_ref(input_ids=ids, logits_to_keep=1).logits.float()
                except TypeError:
                    lg_ref = model_ref(input_ids=ids).logits.float()
            try:
                lg_cur = self.model(input_ids=ids, logits_to_keep=1).logits.float()
            except TypeError:
                lg_cur = self.model(input_ids=ids).logits.float()
        finally:
            self.resume_hooks()
        # 在**目标位置**（最后一个非 pad 位）上算 KL
        _pos = lg_cur.shape[1] - 1
        lpr = _F.log_softmax(lg_ref[0, _pos], dim=-1)
        lpc = _F.log_softmax(lg_cur[0, _pos], dim=-1)
        pr = lpr.exp()
        # ★★ 缺陷 D70：原实现 `kl = float((pr * (lpr - lpc)).sum())` ⇒ 返回 **float**
        #   ⇒ **无梯度** ⇒ KL 正则对基座**完全不起作用** ⇒ 旋钮 N2 是**空操作**！
        #   修法：返回**带图的张量**（不 detach）⇒ 梯度从 KL 流入基座参数。
        kl = (pr * (lpr - lpc)).sum()
        return kl, lpc

    # ★★★ 装置 v5（DIR-20 · 2026-11-07）：v4 的**逐字派生**，**只改状态搬运层**
#     （缺陷 D89：`explicit_kv` 核没有 W ⇒ `export_state` 崩 ⇒ 改为 W 可缺省）。
#     写规则、读规则、梯度隔离、诊断量**一字未改**。
# ★★★ 装置 v6（DIR-20 · 2026-11-07）：v5 的**逐字派生**，**只改 import 的 W 形状校验归属**
#     （缺陷 D91：`W=None` 时 `d["W"].shape` 崩）。写规则、读规则**一字未改**。
# ================================================================ DIR-20 新增
    # —— 跨会话状态语义：把「记忆的状态」**序列化**成可搬运的字节流 ——
    #
    # ★★ 为什么需要 export/import（而不是进程内的 snapshot_all/restore_all）：
    #   `snapshot_all` 是**同一进程内**的（DIR-16…DIR-19 全是单会话）⇒ ⇒ 隐状态历史**未清零**
    #   ⇒ ⇒ 「会话 A 写、会话 B 读」在同进程里**测不到跨会话**（模型实例没变）。
    #   ★ 本方法导出的是**状态**（核状态 W + 地址表 + 读出映射），**不含**模型参数
    #   ⇒ ⇒ 会话 B 用**全新加载的模型实例**导入该状态 ⇒ ⇒ 模型侧的隐状态历史被清零
    #   ⇒ ⇒ 这就是「跨会话」与「同会话」的**唯一**差别。
    def export_state(self, include_read_map=True, include_addr=True):
        """把记忆状态导出为**可序列化**的字典（torch.save 友好；全部在 CPU 上）。

        成分可分别关闭（用于「跨会话需要搬运什么」的对照）：
          `include_read_map=False` ⇒ **不导出读出映射** $W_o,\gamma$
              ⇒ ⇒ 会话 B 的读出映射停在**随机初值** ⇒ ⇒ 读出无意义；
          `include_addr=False`     ⇒ **不导出地址表**（keys / sigs / count）
              ⇒ ⇒ 会话 B **无法寻址**（读不到正确的槽）。
        """
        out = {"fmt": "SGMN_XSESSION_v1", "include_read_map": bool(include_read_map),
               "include_addr": bool(include_addr), "layers": {}}
        for li, m in self.mems.items():
            core = m.core
            d = {
                "mem_kind": m.mem_kind, "key_mode": m.key_mode,
                "query_mode": m.query_mode, "addr_mode": m.addr_mode,
                "read_beta": float(m.read_beta),
                # ★★【DIR-20 · 装置 v5 · 缺陷 D89】核心**可能没有 W**：
                #   `explicit_kv` 核（X6_verbatim 臂）的内容在地址表的 `vals` 里 ⇒ 没有 W。
                #   ⇒ ⇒ 没有 W 时写 `None`（导入时跳过）⇒ 状态搬运对**所有核**都可用。
                "W": (core.W.detach().cpu().clone() if hasattr(core, "W") else None),
            }
            if include_addr:
                for _k in ("keys", "vals", "sigs"):
                    if hasattr(core, _k):
                        d[_k] = getattr(core, _k).detach().cpu().clone()
                d["count"] = int(core.count)
            if include_read_map:
                d["Wo"] = m.Wo.detach().cpu().clone()
                d["gamma"] = m.gamma.detach().cpu().clone()
            out["layers"][str(li)] = d
        return out

    def import_state(self, state, strict=True, expect_key_mode=None):
        """从 `export_state` 的产物恢复状态。

        ★ `strict=True` ⇒ 配置/形状不符即**抛错**（不做静默截断）⇒ ⇒ 跨会话搬运的
          「状态确实到位」由**形状逐位相等**保证 ⇒ ⇒ 与 Stage A 的往返检查配对。
        """
        if state.get("fmt") != "SGMN_XSESSION_v1":
            raise ValueError(f"state fmt 不符：{state.get('fmt')!r}")
        for li, m in self.mems.items():
            d = state["layers"].get(str(li))
            if d is None:
                raise KeyError(f"缺少层 {li} 的状态")
            if expect_key_mode is not None and d.get("key_mode") != expect_key_mode:
                raise ValueError(f"key_mode 不符：{d.get('key_mode')!r} != {expect_key_mode!r}")
            with torch.no_grad():
                # ★★【DIR-20 · 装置 v6 · 缺陷 D91】`W` 可缺省（`explicit_kv` 核没有 W）：
                #   ⇒ 形状校验**必须**与复制逻辑同在「`W` 存在」的分支内
                #   ⇒ （v5 把校验漏在了分支外 ⇒ `d["W"].shape` 对 None 崩）。
                if d.get("W") is not None:
                    if not hasattr(m.core, "W"):
                        raise ValueError(f"层 {li}：当前核没有 W，无法导入")
                    if d["W"].shape != m.core.W.shape:
                        raise ValueError(f"层 {li}：W 形状不符 {tuple(d['W'].shape)} vs {tuple(m.core.W.shape)}")
                    m.core.W.copy_(d["W"].to(m.core.W.device))
                for _k in ("keys", "vals", "sigs"):
                    if _k in d:
                        if not hasattr(m.core, _k):
                            raise ValueError(f"层 {li}：当前核没有 {_k}，无法导入")
                        _t = getattr(m.core, _k)
                        if d[_k].shape != _t.shape:
                            raise ValueError(f"层 {li}：{_k} 形状不符")
                        _t.copy_(d[_k].to(_t.device))
                if "count" in d:
                    m.core.count = int(d["count"])
                if "Wo" in d:
                    m.Wo.copy_(d["Wo"].to(m.Wo.device))
                    m.gamma.copy_(d["gamma"].to(m.gamma.device))
        return self

    @staticmethod
    def state_fingerprint(state):
        """状态的**内容指纹**（对所有张量按 md5 汇总）⇒ ⇒ 往返检查的判据。"""
        import hashlib
        h = hashlib.md5()
        h.update(str(state.get("fmt")).encode())
        for li in sorted(state["layers"].keys()):
            d = state["layers"][li]
            h.update(li.encode())
            for k in sorted(d.keys()):
                v = d[k]
                if torch.is_tensor(v):
                    h.update(k.encode())
                    h.update(v.detach().cpu().numpy().tobytes())
                else:
                    h.update(f"{k}={v}".encode())
        return h.hexdigest()
