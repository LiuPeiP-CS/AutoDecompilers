__version__ = "0.1.0"

from . import code_prepro
from . import metrics
from . import parser

try:
    from .parser.utils import remove_c_cpp_comments
    from .parser.DFG import extract_data_flow_graph
except ImportError:
    import warnings
    warnings.warn("importing some modules fails, please make sure that all the dependencies are installed")