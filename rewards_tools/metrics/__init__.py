# rewards_tools/metrics/__init__.py
"""Metrics Module - 相似度计算模块"""

from .dataflow_match import DataFlowAnalyzer
from .syntax_match import SyntaxAnalyzer, SyntaxMatchScorer

def __getattr__(name):
    if name == 'EnvRecomReexe':
        from .env_recom_reexe import EnvRecomReexe
        return EnvRecomReexe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'DataFlowAnalyzer',
    'SyntaxMatchScorer',
    'SyntaxAnalyzer',
    'EnvRecomReexe'
]