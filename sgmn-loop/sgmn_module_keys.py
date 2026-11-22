"""
sgmn_module.py —— SGMN v2.4 的可微实现（DIR-04 专用装置）

v2.4 依据：
  §2.1  sg(.)：前向保持原值、反向梯度为零；【不是前向断开】，也不消除状态之间的函数依赖
  §2.2  q = Q_l(sg(c)) ;  y = sg(W) q ;  h^+ = h + gamma*m
        「读取使用 W_{l,t}，写入产生 W_{l,t+1}；同一步不采用'先更新记忆、再由更新后的记忆
         定义本步目标'的隐式顺序」
  §2.3  k_{l,t} = sg(P_k(h_{l,t-1})) ,  v_{l,t} = sg(P_v(h_{l,t}))
  §3.1  G = alpha(W k - v) k^T + lambda (W - W^0)
  §3.3  U^+ = U - eta G V ,  V^+ = V - eta G^T U   （【同步】步：都用旧 U、旧 V）
  §4    任务损失只更新读取侧 phi=(phi^q, phi^o) 与 gamma；写入通道只由显式局部规则驱动

语义约定（本轮显式声明，v2.4 未规定）：
  * 记忆状态 W 是【模型级】的，跨 batch 共享；读是【逐 token】的；
    写用 batch 平均后的 k、v（因此 G 是单个 (d_v,d_k) 矩阵）。
  * 形状约定：hs_in 为 (B, L, d_model)；所有投影在最后一维上做。

装置设计要点（为让「③ 任务梯度不进入写入通道」成为【有区分力】的检查）：
  * U, V, W^0, P_k, P_v 均以 requires_grad=True 的【叶子张量】实现，但不被任何优化器更新；
    正确实现下它们在 backward 后 .grad 应为 None/精确 0；
    若任何一处 stop-gradient 被漏掉，梯度就会真的流到它们身上 -> 检查失败。
  * 读取上下文 c = sg(h_{l-1}^+)：截断【只作用于读支路】，基座主路径的输入梯度仍回传
    （v2.4 §4 明文要求），故 W_q/W_o/gamma 仍能拿到非零有限梯度。
  * 写入的 G 用【带梯度】计算（依赖 k,v 自身的 sg 来隔离），更新动作在 no_grad 下做。
    这样 k/v 的 sg 是【承重】的，漏掉就会被 ③ 抓到。
"""
import math
import torch


def sg(x, drop=False):
    """v2.4 §2.1：前向保持原值、反向梯度为零。drop=True 用于阴性对照。"""
    return x if drop else x.detach()


class LayerMem:
    """单个挂载层的记忆状态与读写参数。"""

    def __init__(self, d_model, d_k, d_v, r, alpha, lam, eta, gamma, device, gen,
                 use_factorized=True, no_write=False, perpos=False):
        self.d_model, self.d_k, self.d_v, self.r = d_model, d_k, d_v, r
        self.alpha, self.lam, self.eta = alpha, lam, eta
        self.use_factorized = use_factorized
        self.no_write = no_write      # True => 不执行写入（用于分离写介导的效应）
        self.perpos = perpos          # True => 逐位置写入（v2.4 §2.3/§3.2 的逐步语义）
        self.clone_read = (not use_factorized)   # D1 修复：全矩阵分支读取时克隆（见 read()）
        # DIR-12：读取侧的竞争式检索算子（beta=0 ⇒ 与原线性读逐位相同）
        self.last_q = None
        self.last_y = None
        self.read_beta = 16.0     # 竞争的锐度（余弦分数的倍数）
        self.read_lambda = 0.0    # 竞争强度：0 ⇒ 原线性读；1 ⇒ 纯余弦注意力读
        self.oracle_key = None      # 诊断用：{layer: d_k 向量}，直接作为查询键
        # ---- DIR-13：键的构造方式（§7.2「写入映射与门控」开放项；写/读规则一字未改）
        self.key_mode = "linear"    # linear | sparse | ortho
        self.key_topk = 32          # sparse 模式保留的分量数
        self.ortho_bank = None      # ortho 模式：严格正交的键库 (d_k, d_k)
        self.ortho_idx = 0
        # 【D13】记录【实际写入】的键与值（诊断必须用真实写入的键，而不是脚本里另算的线性投影）
        self.keys_written = []
        self.vals_written = []
        mk = lambda *s: (torch.randn(*s, generator=gen) / math.sqrt(s[0])).to(device).requires_grad_(True)
        self.Pk = mk(d_model, d_k)          # 写入映射（固定）
        self.Pv = mk(d_model, d_v)
        self.W0 = mk(d_v, d_k)               # 基准关联矩阵（固定）
        self.U = (torch.randn(d_v, r, generator=gen) * 0.1).to(device).requires_grad_(True)
        self.V = (torch.randn(d_k, r, generator=gen) * 0.1).to(device).requires_grad_(True)
        # 全矩阵参考支路的可塑状态（v2.4 §3.2：W_{t+1}=W_t-eta G，W 本身是可塑状态）
        self.Wfull = self.W0.detach().clone().requires_grad_(True) if not use_factorized else None
        self.Wq = mk(d_k, d_model)          # 读取侧 phi^q
        self.Wo = mk(d_model, d_v)           # 读取侧 phi^o
        self.gamma = torch.tensor(float(gamma), device=device, requires_grad=True)
        self.pk_scale = 1.0
        self.prev_hs = None
        self.steps = 0
        self.write_calls = 0
        self.read_calls = 0
        self.injection = None
        self.last_attn = None

    # ------------------------------------------------------------------ 读
    def read(self, hs_in, drop_sg=False):
        """c = sg(h_{l-1}^+)；q = Wq c；y = sg(W) q；m = Wo y。逐 token。"""
        self.read_calls += 1
        c = sg(hs_in, drop_sg)                                   # (B,L,d_model)
        cf = c.reshape(-1, c.shape[-1])                          # (N,d_model)
        q = (self.Wq @ cf.T).T                                  # (N,d_k)
        if self.key_mode == "sparse":
            # 与写入侧同一门控 ⇒ 查询与键的"地址"（支撑集）可比
            thr = q.abs().topk(self.key_topk, dim=-1).values[:, -1:]
            q = q * (q.abs() >= thr).to(q.dtype)
        W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
        # D1 修复（扩展）：读取用到的 W【一律取克隆】——不仅线性读，检索算子的键基也用它。
        # 原因：位置级写对 Wfull 的原地修改会 bump 版本号，而计算图保存了 W ⇒ 反向报错。
        # 语义不变：读的 W 本就 stop-gradient（sg），克隆只是切断存储共享。
        Wsb = sg(W, drop_sg)
        if self.clone_read:
            Wsb = Wsb.clone()
        if self.read_lambda > 0.0:
            # DIR-12 检索算子（尺度无关版）：查询先在【记忆的键基】（W 的列）上做 softmax 竞争，
            #   竞争分数用【余弦】（q 与列的夹角）⇒ 有界于 [-1,1] ⇒ 不随记忆幅度溢出；
            #   q~ = sum_j a_j W[:,j]；
            #   最终查询 = (1-lambda)*q + lambda*q~  ⇒ lambda=0 精确退化为原线性读
            #   y = sg(W) q（Freeze B 的状态与写规则一字未动）。
            qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            Wn = Wsb / Wsb.norm(dim=0, keepdim=True).clamp_min(1e-30)   # 列范数归一
            logits = self.read_beta * (qn @ Wn)                  # (N,d_k) ∈ [-beta,beta]
            logits = logits - logits.max(dim=-1, keepdim=True).values
            a = torch.softmax(logits, dim=-1)                   # 竞争权重
            self.last_attn = a
            q_att = a @ Wsb.T                                   # (N,d_k)
            q = (1.0 - self.read_lambda) * q + self.read_lambda * q_att
        if self.oracle_key is not None:
            q = self.oracle_key.detach().reshape(1, -1).expand_as(q)
        y = (Wsb @ q.T).T                                       # (N,d_v)
        self.last_q = q.detach()          # 实际使用的查询（含竞争混合）
        self.last_y = y.detach()          # 实际读出的内容（诊断必须用它）
        m = (self.Wo @ y.T).T                                    # (N,d_model)
        self.injection = m.reshape(hs_in.shape)
        return self.injection

    # ------------------------------------------------------------------ 写
    def write(self, hs_in, drop_sg=False):
        """【DIR-11 装置变体：逐位置写入】忠实实现 v2.4 §2.3/§3.1/§3.2 的逐步语义：
            k_{l,t} = sg(P_k h_{l,t-1}),  v_{l,t} = sg(P_v h_{l,t})   （t = 1..L-1）
            G_t    = alpha(W_t k_t - v_t) k_t^T + lam(W_t - W^0)
            W_{t+1} = W_t - eta G_t                      （同一前视内逐位置在线更新）
        §2.2：读取使用 W_{l,t}（前视开始时的记忆），写入产生 W_{l,t+1}
               ⇒ hook 的"先读后写"顺序即满足该语义。
        与 DIR-04 资产的唯一差别：DIR-04 装置把 k,v 在 batch/序列上取平均（一次前视=一个平均关联）；
        本变体逐位置写入（一次前视=L-1 个关联），故 write_calls 计的是【位置级】写次数。
        B>1 时退化为 DIR-04 的平均写（本轮所有测量均为 B=1）。"""
        if self.no_write:
            return
        if not self.perpos:
            return self._write_averaged(hs_in, drop_sg)
        if hs_in.shape[0] != 1:
            # B>1：退化为平均写（与 DIR-04 装置一致）
            return self._write_averaged(hs_in, drop_sg)
        L = hs_in.shape[1]
        if L < 2:
            return
        with torch.no_grad():
            for t in range(1, L):
                # v2.4 §2.3：k_{l,t} = sg(P_k(h_{l,t-1}))，h_{l,t-1} 是【同一前视内】的前一位置
                if self.key_mode == "ortho":
                    # DIR-13 K4：人为【严格正交】的键（分离"键的几何"与"写规则"）
                    j = self.ortho_idx % self.ortho_bank.shape[0]
                    self.ortho_idx += 1
                    k = self.ortho_bank[j].clone() * self.pk_scale
                else:
                    k = sg(hs_in[0, t - 1].reshape(1, -1) @ self.Pk, drop_sg).mean(dim=0)
                    if self.key_mode == "sparse":
                        # DIR-13 K2：稀疏 top-k 门控（非线性的键构造）
                        thr = k.abs().topk(self.key_topk).values[-1]
                        k = k * (k.abs() >= thr).to(k.dtype)
                    k = k * self.pk_scale
                v = sg(hs_in[0, t].reshape(1, -1) @ self.Pv, drop_sg).mean(dim=0)
                self.keys_written.append(k.detach().clone())
                self.vals_written.append(v.detach().clone())
                v = sg(hs_in[0, t].reshape(1, -1) @ self.Pv, drop_sg).mean(dim=0)
                W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
                G = self.alpha * torch.outer(W @ k - v, k) + self.lam * (W - self.W0)
                if self.use_factorized:
                    self.U.copy_(self.U - self.eta * (G @ self.V))
                    self.V.copy_(self.V - self.eta * (G.T @ self.U))
                else:
                    self.Wfull.copy_(self.Wfull - self.eta * G)
                self.write_calls += 1
                self.steps += 1

    def _write_averaged(self, hs_in, drop_sg=False):
        """DIR-04 装置的原写规则（batch/序列平均 k,v）；B>1 时使用。"""
        if self.prev_hs is None or self.no_write:
            return
        self.write_calls += 1
        pf = self.prev_hs.reshape(-1, self.prev_hs.shape[-1])
        hf = hs_in.reshape(-1, hs_in.shape[-1])
        if self.key_mode == "ortho":
            j = self.ortho_idx % self.ortho_bank.shape[0]
            self.ortho_idx += 1
            k = self.ortho_bank[j].clone() * self.pk_scale
        else:
            k = sg(pf @ self.Pk, drop_sg).mean(dim=0)
            if self.key_mode == "sparse":
                thr = k.abs().topk(self.key_topk).values[-1]
                k = k * (k.abs() >= thr).to(k.dtype)
            k = k * self.pk_scale
        v = sg(hf @ self.Pv, drop_sg).mean(dim=0)
        W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
        G = self.alpha * torch.outer(W @ k - v, k) + self.lam * (W - self.W0)
        with torch.no_grad():
            if self.use_factorized:
                self.U.copy_(self.U - self.eta * (G @ self.V))
                self.V.copy_(self.V - self.eta * (G.T @ self.U))
            else:
                self.Wfull.copy_(self.Wfull - self.eta * G)
        self.steps += 1

    def _write_avg_legacy(self, hs_in, drop_sg=False):
        """k = sg(P_k h_{l,t-1})；v = sg(P_v h_{l,t})；G = alpha(Wk-v)k^T + lam(W-W^0)；
        同步因子步（或全矩阵参考步）。记忆跨 batch 共享 -> k,v 取 batch 平均。"""
        if self.prev_hs is None or self.no_write:
            return        # 首个时间步无 k_{l,0}（声明条件）；或写入被显式关闭
        self.write_calls += 1
        pf = self.prev_hs.reshape(-1, self.prev_hs.shape[-1])     # (N,d_model)
        hf = hs_in.reshape(-1, hs_in.shape[-1])
        if self.key_mode == "ortho":
            j = self.ortho_idx % self.ortho_bank.shape[0]
            self.ortho_idx += 1
            k = self.ortho_bank[j].clone() * self.pk_scale
        else:
            k = sg(pf @ self.Pk, drop_sg).mean(dim=0)
            if self.key_mode == "sparse":
                thr = k.abs().topk(self.key_topk).values[-1]
                k = k * (k.abs() >= thr).to(k.dtype)
            k = k * self.pk_scale     # (d_k,)
            self.keys_written.append(k.detach().clone())
            self.vals_written.append(v.detach().clone())
        v = sg(hf @ self.Pv, drop_sg).mean(dim=0)                       # (d_v,)
        W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
        G = self.alpha * torch.outer(W @ k - v, k) + self.lam * (W - self.W0)   # (d_v,d_k)
        # 读取侧注入强度与写门控解耦：alpha 只门控数据项（v2.4 §3.1）
        with torch.no_grad():
            if self.use_factorized:
                self.U.copy_(self.U - self.eta * (G @ self.V))     # 同步：用旧 U、旧 V
                self.V.copy_(self.V - self.eta * (G.T @ self.U))
            else:
                self.Wfull.copy_(self.Wfull - self.eta * G)         # 全矩阵参考：W 本身是可塑状态
        self.steps += 1

    def cache(self, hs_in):
        self.prev_hs = hs_in.detach()


class SGMNStack:
    """把 SGMN 读/写挂到 HF decoder 的若干层上。"""

    def __init__(self, model, mount_layers, d_k, d_v, r, alpha, lam, eta, gamma,
                 device, seed=0, use_factorized=True, drop_sg=False, no_write=False,
                 perpos=False):
        self.model = model
        self.device = device
        self.drop_sg = drop_sg              # 阴性对照：去掉全部 stop-gradient
        gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
        self.mems = {}
        self._hs_in = {}
        for li in mount_layers:
            m = LayerMem(model.config.hidden_size, d_k, d_v, r, alpha, lam, eta, gamma,
                         device, gen, use_factorized, no_write=no_write, perpos=perpos)
            with torch.no_grad():
                # 尺度校准：使【单位方差输入下】E[k^2]=5（与 DIR-01..03 的 mu=5 约定量级对齐）。
                # 真实隐状态尺度不同 -> 这是量级对齐，不是 v2.4 的规定。
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
                m = mem.read(hs_in, self.drop_sg)          # 读：用 W_t
                hs_out = hs_out + mem.gamma * m             # 注入：h_l^+ = h_l + gamma*m
                mem.write(hs_in, self.drop_sg)              # 写：产生 W_{t+1}
                mem.cache(hs_in)                            # 为下一步准备 h_{l,t}
                if isinstance(output, tuple):
                    return (hs_out,) + tuple(output[1:])
                return hs_out

            self._handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            self._handles.append(layer.register_forward_hook(fwd_hook, with_kwargs=True))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # --------------------------------------------------------------- 参数分组
    def read_params(self):
        ps = []
        for m in self.mems.values():
            ps += [m.Wq, m.Wo, m.gamma]
        return ps

    def write_params(self):
        ps = []
        for m in self.mems.values():
            ps += [m.U, m.V, m.W0, m.Pk, m.Pv]
            if getattr(m, "Wfull", None) is not None:
                ps.append(m.Wfull)
        return ps

    def named_write_params(self):
        out = []
        for li, m in self.mems.items():
            for nm, t in (("U", m.U), ("V", m.V), ("W0", m.W0), ("Pk", m.Pk), ("Pv", m.Pv)):
                if t is not None:
                    out.append((f"L{li}.{nm}", t))
            if getattr(m, "Wfull", None) is not None:
                out.append((f"L{li}.Wfull", m.Wfull))
        return out

    def named_read_params(self):
        out = []
        for li, m in self.mems.items():
            for nm, t in (("Wq", m.Wq), ("Wo", m.Wo), ("gamma", m.gamma)):
                out.append((f"L{li}.{nm}", t))
        return out


def grad_report(named_tensors):
    """[(name, grad_max_abs, grad_is_none, has_nan_or_inf)]"""
    out = []
    for nm, t in named_tensors:
        g = t.grad
        if g is None:
            out.append((nm, 0.0, True, False))
        else:
            out.append((nm, float(g.abs().max()), False,
                        bool(torch.isinf(g).any() or torch.isnan(g).any())))
    return out
