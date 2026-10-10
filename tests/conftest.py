# -*- coding: utf-8 -*-
"""pytest 兼容配置：把项目根与 tests 目录加进 sys.path。

本项目的测试既支持内置迷你框架（`python tests/run_tests.py`，零依赖），
也支持 pytest（`pytest tests`，有它当然更好）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
