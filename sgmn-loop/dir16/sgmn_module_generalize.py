"""
sgmn_module_structures.py —— DIR-14 装置：把「记忆结构」做成可插拔的核心

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

其余（读上下文 c=sg(h^+)、注入 h^+ = h + γm、位置级写、Freeze C 的 sg 隔离、阴性对照 drop_sg、
读时克隆修 D1、每篇文档前重置记忆）均与 DIR-13 一致。
"""
import math
import torch


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
class LayerMem:
    def __init__(self, d_model, d_k, d_v, r, alpha, lam, eta, gamma, device, gen,
                 use_factorized=True, no_write=False, perpos=False,
                 mem_kind="dense", n_slots=64, n_proto=64, n_entries=64):
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
        self.key_mode = "linear"     # linear | sparse
        self.key_topk = 32
        self.last_q = None
        self.last_y = None      # 诊断用：实际注入的量（Wo y）—— 与 DIR-11…14 的口径一致
        self.last_yr = None     # 【D22】诊断用：**记忆读出 y**（未过 Wo）⇒ 命中指标的正确对象
        self.keys_written = []
        self.vals_written = []
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
                k = sg(hs_in[0, t - 1].reshape(1, -1) @ self.Pk, drop_sg).mean(dim=0)
                if self.key_mode == "sparse":
                    thr = k.abs().topk(self.key_topk).values[-1]
                    k = k * (k.abs() >= thr).to(k.dtype)
                k = k * self.pk_scale
                v = sg(hs_in[0, t].reshape(1, -1) @ self.Pv, drop_sg).mean(dim=0)
                self.keys_written.append(k.detach().clone())
                self.vals_written.append(v.detach().clone())
                sig = hs_in[0, t - 1].detach()                    # 【DIR-15】上下文签名（键的同一输入）
                if hasattr(self.core, "sigs"):
                    self.core.write(k, v, sig=sig)
                else:
                    self.core.write(k, v)
                self.write_calls += 1
                self.steps += 1

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
        else:
            out += [("keys", self.core.keys), ("vals", self.core.vals)]
        return out

    def state_snapshot(self):
        if self.mem_kind == "dense":
            return self.core.W.detach().clone()
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
            if self.mem_kind == "dense":
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
                 perpos=True, mem_kind="dense", n_slots=64, n_proto=64, n_entries=64):
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
                         n_entries=n_entries)
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

    def named_write_params(self):
        out = []
        for li, m in self.mems.items():
            out += [(f"L{li}.{n}", t) for n, t in m.named_write_params()]
        return out

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
