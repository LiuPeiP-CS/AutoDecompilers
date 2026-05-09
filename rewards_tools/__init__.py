__version__ = "0.1.0"

# 可以方便地导入常用模块
from . import code_prepro
from . import metrics
from . import parser

# 或者直接导入最常用的函数
try:
    from .parser.utils import remove_c_cpp_comments
    from .parser.DFG import extract_data_flow_graph
except ImportError:
    # 如果导入失败，提供提示信息
    import warnings
    warnings.warn("部分模块导入失败，请确保所有依赖已安装")