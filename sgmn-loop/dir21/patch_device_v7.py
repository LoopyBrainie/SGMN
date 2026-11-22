#!/usr/bin/env python3
"""DIR-21 · 装置 v6 ⇒ **v7**（卡片 rev 1.1）的**逐字派生补丁**（在训练机上跑）。

★ **只加两件事**，写规则（v2.4 §3.2 的 `W ← W − ηG`）、读规则、梯度隔离、诊断量**一字未改**：

① **写步粒度**（`write_stride` / `stride_pick`）—— ★ **写规则何时触发**：
     触发位置集合 = ``{S, 2S, ...} ∩ [1, L−1]`` **∪ {L−1}**
     （★ 目标位置 `L−1`（码 token）**必写** ⇒ ⇒ 每个粒度下**任务都可作答**
      ⇒ ⇒ 臂间的 NLL **可比** —— 否则粗粒度臂会因为「答案根本没写进去」而**平凡地**变差）。
② **键索引按 token 位置**：正交键取第 `(base + t − 1)` 列（`base` = 本次前视开始时的计数）
     ⇒ ⇒ **同一个 token 在任何粒度下拿到逐位相同的 (k, v)** ⇒ ⇒ **唯一自变量 = 写步调度** ✔
     （★ `write_stride = 1` 时与 DIR-15…20 的协议**逐位一致** ⇒ 跨方向可比 ✔）

③ （必要的配套）`state_snapshot` / `state_restore` 里带上 `ortho_idx`
     ⇒ ⇒ 各臂恢复训练态后**从同一个键索引锚点继续** ⇒ ⇒ A21-1（同一 token 键逐位相同）成立 ✔
"""
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "sgmn_module_hebb_v6.py"
DST = HERE / "sgmn_module_hebb_v7.py"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def must_replace(text, old, new, tag, count=1):
    n = text.count(old)
    if n != count:
        raise SystemExit(f"[补丁失败] {tag}: 命中 {n} 次（期望 {count}）")
    return text.replace(old, new, count)


def main():
    t = SRC.read_text(encoding="utf-8")
    log = []

    # ── ① 构造器：加两个参数 + 两个属性 ──────────────────────────────────────
    t = must_replace(
        t,
        "    def __init__(self, d_model, d_k, d_v, r, alpha, lam, eta, gamma, device, gen,\n"
        "                 use_factorized=True, no_write=False, perpos=False,\n"
        "                 mem_kind=\"dense\", n_slots=64, n_proto=64, n_entries=64,\n"
        "                 key_mode=\"linear\"):\n",
        "    def __init__(self, d_model, d_k, d_v, r, alpha, lam, eta, gamma, device, gen,\n"
        "                 use_factorized=True, no_write=False, perpos=False,\n"
        "                 mem_kind=\"dense\", n_slots=64, n_proto=64, n_entries=64,\n"
        "                 key_mode=\"linear\", write_stride=1, stride_pick=\"last\"):\n",
        "ctor-signature", 1)
    t = must_replace(
        t,
        "        self.key_mode = key_mode     # linear | sparse | ortho | renorm（★ DIR-18 rev 1.2 修订 54）\n",
        "        self.key_mode = key_mode     # linear | sparse | ortho | renorm（★ DIR-18 rev 1.2 修订 54）\n"
        "        # ★★【DIR-21 · 卡片 rev 1.1】**写步粒度**：每隔 `write_stride` 个 token 触发一次写规则。\n"
        "        #   触发集合 = {S, 2S, ...} ∩ [1, L−1] ∪ {L−1}（★ 目标位置必写 ⇒ 任务可作答）。\n"
        "        #   `stride_pick` = \"last\"（段尾）| \"first\"（段首）。\n"
        "        self.write_stride = int(write_stride)\n"
        "        self.stride_pick = str(stride_pick)\n",
        "ctor-attrs", 1)
    log.append("① 构造器 += write_stride / stride_pick")

    # ── ② 写步调度 + 键索引按 token 位置 ─────────────────────────────────────
    t = must_replace(
        t,
        "    def write(self, hs_in, drop_sg=False):\n"
        "        \"\"\"位置级写（v2.4 §2.3/§3.1/§3.2 的逐步语义）：t=1..L-1 逐位置写入。\"\"\"\n"
        "        if self.no_write:\n"
        "            return\n"
        "        if hs_in.shape[0] != 1 or not self.perpos:\n"
        "            raise RuntimeError(\"DIR-14 装置只实现位置级写（B=1）\")\n"
        "        L = hs_in.shape[1]\n"
        "        if L < 2:\n"
        "            return\n"
        "        with torch.no_grad():\n"
        "            for t in range(1, L):\n",

        "    def _trigger(self, t, L):\n"
        "        \"\"\"★【DIR-21】**写步调度**：本 token 是否触发写规则。\n\n"
        "        触发位置 = ``{S, 2S, ...} ∩ [1, L−1]`` **∪ {L−1}**。\n"
        "        ★ 目标位置 `L−1`（码 token）**必写** ⇒ 每个粒度下**任务都可作答**\n"
        "          ⇒ ⇒ 臂间 NLL 可比（否则粗粒度臂会因「答案没写进去」而平凡地变差）。\n"
        "        `stride_pick`：`\"last\"` = 每段的**尾** token；`\"first\"` = 每段的**首** token。\n"
        "        \"\"\"\n"
        "        S = int(self.write_stride)\n"
        "        if S <= 1:\n"
        "            return True\n"
        "        if int(t) == int(L) - 1:\n"
        "            return True\n"
        "        if self.stride_pick == \"first\":\n"
        "            return ((int(t) - 1) % S) == 0\n"
        "        return (int(t) % S) == 0\n\n"
        "    def write(self, hs_in, drop_sg=False):\n"
        "        \"\"\"位置级写（v2.4 §2.3/§3.1/§3.2 的逐步语义）：t=1..L-1 逐位置写入。\n\n"
        "        ★★【DIR-21 · 卡片 rev 1.1】**写步粒度**（`write_stride` / `stride_pick`）\n"
        "          —— ★ **只改「写规则何时触发」**，写规则本身（§3.2 的 `W ← W − ηG`）**一字未改**。\n"
        "        ★★ **键索引按 token 位置**：正交键取第 `(base + t − 1)` 列（`base` = 本次前视开始时的\n"
        "          计数）⇒ ⇒ **同一个 token 在任何粒度下拿到逐位相同的 (k, v)**\n"
        "          ⇒ ⇒ **唯一自变量 = 写步调度**（Stage A 的 A21-1）。\n"
        "          （`write_stride = 1` ⇒ 与 DIR-15…20 的协议**逐位一致** ✔）\n"
        "        \"\"\"\n"
        "        if self.no_write:\n"
        "            return\n"
        "        if hs_in.shape[0] != 1 or not self.perpos:\n"
        "            raise RuntimeError(\"DIR-14 装置只实现位置级写（B=1）\")\n"
        "        L = hs_in.shape[1]\n"
        "        if L < 2:\n"
        "            return\n"
        "        _base = int(self.ortho_idx)      # ★ DIR-21：键索引的锚点（本次前视起点）\n"
        "        with torch.no_grad():\n"
        "            for t in range(1, L):\n"
        "                _do = self._trigger(t, L)   # ★ DIR-21：写步调度\n",

        "write-head", 1)
    log.append("② write() += 写步调度（_trigger）")

    # ── ③ 正交键的列索引：按 token 位置（`_base + t − 1`）────────────────────
    t = must_replace(
        t,
        "                    j = self.ortho_idx % self.ortho_basis.shape[1]\n"
        "                    self.ortho_idx += 1\n",
        "                    # ★★【DIR-21】列索引**按 token 位置** ⇒ 与粒度无关 ⇒ A21-1\n"
        "                    j = (_base + t - 1) % self.ortho_basis.shape[1]\n"
        "                    self.ortho_idx += 1          # ★ 每个 token 都推进（不论是否触发写）\n",
        "ortho-index", 1)
    log.append("③ 正交键列索引 ⇒ 按 token 位置（_base + t − 1）")

    # ── ④ 粒度：未触发写的 token 要 `continue`（放在键算完之后）──────────────
    t = must_replace(
        t,
        "                # ★★【DIR-18 rev 1.5 · 修订 68】**内容置换**：只换「值」的来源位置，\n",
        "                if not _do:\n"
        "                    continue        # ★ DIR-21：本 token 不触发写步（粒度调度）\n"
        "                # ★★【DIR-18 rev 1.5 · 修订 68】**内容置换**：只换「值」的来源位置，\n",
        "stride-continue", 1)
    log.append("④ 未触发的 token ⇒ continue（不写）")

    # ── ⑤ 快照带上 `ortho_idx`（各臂复位后从同一锚点继续）────────────────────
    t = must_replace(
        t,
        "            return (self.core.W.detach().clone(),\n"
        "                    (self.core.U.detach().clone() if self.core.factorized else None),\n"
        "                    (self.core.V.detach().clone() if self.core.factorized else None),\n"
        "                    self.core.keys.detach().clone(), self.core.vals.detach().clone(),\n"
        "                    self.core.sigs.detach().clone(), int(self.core.count))\n",
        "            return (self.core.W.detach().clone(),\n"
        "                    (self.core.U.detach().clone() if self.core.factorized else None),\n"
        "                    (self.core.V.detach().clone() if self.core.factorized else None),\n"
        "                    self.core.keys.detach().clone(), self.core.vals.detach().clone(),\n"
        "                    self.core.sigs.detach().clone(), int(self.core.count),\n"
        "                    int(self.ortho_idx))   # ★ DIR-21：键索引锚点（各臂复位后从同一起点继续）\n",
        "snapshot-ortho", 1)
    t = must_replace(
        t,
        "            if self.mem_kind in (\"hebbian_full\", \"hebbian_fact\"):\n"
        "                W, U, V, K, Vv, S, cnt = snap\n",
        "            if self.mem_kind in (\"hebbian_full\", \"hebbian_fact\"):\n"
        "                W, U, V, K, Vv, S, cnt = snap[:7]\n"
        "                if len(snap) >= 8:\n"
        "                    self.ortho_idx = int(snap[7])   # ★ DIR-21：恢复键索引锚点\n",
        "restore-ortho", 1)
    log.append("⑤ 快照/恢复带上 ortho_idx（键索引锚点）")

    # ── ⑥ Stage A 用的触发位置集合（供脚本自检）──────────────────────────────
    t = must_replace(
        t,
        "    def write(self, hs_in, drop_sg=False):\n",
        "    def trigger_positions(self, L):\n"
        "        \"\"\"★【DIR-21】本篇的**写步触发位置**（与 `write()` 的调度**同一函数** ⇒ 不会漂移）。\"\"\"\n"
        "        return [t for t in range(1, int(L)) if self._trigger(t, int(L))]\n\n"
        "    def write(self, hs_in, drop_sg=False):\n",
        "trigger-positions", 1)
    log.append("⑥ trigger_positions()（Stage A 用）")

    DST.write_text(t, encoding="utf-8")
    print("v6 md5 =", md5(SRC))
    print("v7 md5 =", md5(DST))
    print("行数 v6/v7 =", len(SRC.read_text(encoding='utf-8').splitlines()),
          "/", len(t.splitlines()))
    for line in log:
        print("  ·", line)


if __name__ == "__main__":
    main()
