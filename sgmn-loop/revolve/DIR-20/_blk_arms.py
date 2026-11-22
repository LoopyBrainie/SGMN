# ★ DIR-20 臂表（11 臂；卡片 rev 1.1 §2 冻结）
#   字段：(name, mem_kind, eta, key_mode, variant, export_mode, session_mode)
#   export_mode : full | no_readmap | no_addr | none
#   session_mode: cross（重建实例 + 导入） | same（同实例，写完即测） |
#                 same_empty（同实例 + 空栈） | cross_noimport（重建实例 + 不导入） |
#                 cross_wrongseed（重建实例 + 导入**另一个种子**的状态）
#   ★ `X8_linear` 的 η = 1e-5（**尺度适配档**；卡片 rev 1.1 §0.4 / 修订 20-C）：
#     `linear` 键的实测 ‖k‖² ≈ 1.7e4（DIR-18 的 D50/D51）⇒ 稳定上界 η < 2/(λ+α‖k‖²) ≈ 1.2e-4
#     ⇒ ⇒ η=1e-1 会让写规则**发散** ⇒ ⇒ H4″ 会测成「发散 vs 正常」⇒ ⇒ 不可接受。
#   ★ `X6_verbatim` 用 `explicit_kv`：其内容在地址表的 `vals` 里 ⇒ ⇒ **按构造内容正确**
#     ⇒ ⇒ 作为「零效度门槛」对照；且 `explicit_kv` 的读**对键尺度不敏感**（余弦）
#     ⇒ ⇒ **X6 不参与 H4″**（卡片 rev 1.1 §3 写死）。
ARMS = [
    # ① 下界 / 主臂
    ("X0_no_state",          "hebbian_full", 1e-1, "ortho",  "normal",        "none",       "same_empty"),
    ("X1_full_state",        "hebbian_full", 1e-1, "ortho",  "normal",        "full",       "cross"),
    # ② 成分必要性
    ("X2_no_readmap",        "hebbian_full", 1e-1, "ortho",  "normal",        "no_readmap", "cross"),
    ("X3_no_addr",           "hebbian_full", 1e-1, "ortho",  "normal",        "no_addr",    "cross"),
    # ③ 内容 / 会话归属
    ("X4_content_wrong",     "hebbian_full", 1e-1, "ortho",  "content_wrong", "full",       "cross"),
    ("X5_wrong_session",     "hebbian_full", 1e-1, "ortho",  "normal",        "full",       "cross_wrongseed"),
    # ④ 键尺度
    ("X7_renorm",            "hebbian_full", 1e-1, "renorm", "normal",        "full",       "cross"),
    ("X8_linear",            "hebbian_full", 1e-5, "linear", "normal",        "full",       "cross"),
    # ⑤ 零效度对照
    ("X6_verbatim",          "explicit_kv",  None, "linear", "normal",       "full",       "cross"),
    # ⑥ 基线
    ("X10_same_session",     "hebbian_full", 1e-1, "ortho",  "normal",        "none",       "same"),
    ("X11_newinst_noexport", "hebbian_full", 1e-1, "ortho",  "normal",        "none",       "cross_noimport"),
]
