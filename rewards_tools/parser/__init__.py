# rewards_tools/parser/__init__.py
"""Parser Module - 代码解析模块"""

# 从utils导入所有有用的函数
from .utils import (
    remove_c_cpp_comments,
)

# 从DFG导入数据流图相关函数
from .DFG import (
    SemanticMatchScore
)

# 定义公开的接口
__all__ = [
    'remove_c_cpp_comments',
    "SemanticMatchScore"
]
