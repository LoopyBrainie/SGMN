#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-20 · Stage A 显存轨迹探针（只读诊断；不改被测脚本的字节内容）。

做法：import 装配后的脚本 ⇒ monkey-patch `rec` ⇒ 每条断言后打印当前显存。
"""
import importlib.util
import pathlib
import sys

import torch

SCRIPT = "/home/ec2-user/sgmn-loop/dir20/R1_xsession.py"

spec = importlib.util.spec_from_file_location("R1_xsession_probe", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["R1_xsession_probe"] = mod
spec.loader.exec_module(mod)

_orig = mod.rec


def rec(name, passed, detail, tol=None, value=None):
    _orig(name, passed, detail, tol, value)
    try:
        print("   [mem] alloc=%8.1f MiB reserved=%8.1f MiB | %s"
              % (torch.cuda.memory_allocated() / 2**20,
                 torch.cuda.memory_reserved() / 2**20, name[:60]), flush=True)
    except Exception:
        pass


mod.rec = rec
sys.argv = ["R1_xsession.py", "--smoke"]
try:
    mod.main()
except SystemExit as e:
    print("[probe] SystemExit", e.code, flush=True)
except Exception as e:
    print("[probe] exception:", type(e).__name__, e, flush=True)
