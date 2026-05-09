from __future__ import annotations
import os
import sys
import itertools
import math
import typing as T
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from tree_sitter import Language, Parser, Node
from typing import Optional

# this code is for semantic matching score based on dfg

"""
# -------------------- 语言库路径 --------------------
def get_language_library():
    # 获取或构建 tree-sitter 语言库
    # 尝试多种可能的路径
    possible_paths = [
        os.path.abspath("build/my-languages.so"),
        os.path.join(os.path.dirname(__file__), "build/my-languages.so"),
        "/usr/local/lib/my-languages.so",
    ]

    for lib_path in possible_paths:
        if os.path.exists(lib_path):
            return lib_path

    # 如果找不到，尝试从源码构建（需要 git 和编译器）
    try:
        return build_language_library()
    except Exception as e:
        print(f"无法构建语言库: {e}")
        print("请手动构建 tree-sitter C++ 语言库")
        print("或从 https://github.com/tree-sitter/tree-sitter-cpp 获取预编译版本")
        sys.exit(1)

def build_language_library():
    # 从源码构建 tree-sitter 语言库
    import tempfile
    import subprocess
    import shutil

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    try:
        # 克隆 tree-sitter-cpp 仓库
        cpp_repo = os.path.join(temp_dir, "tree-sitter-cpp")
        subprocess.run([
            "git", "clone", 
            "https://github.com/tree-sitter/tree-sitter-cpp.git",
            cpp_repo
        ], check=True, capture_output=True)

        # 构建语言库
        lib_path = os.path.abspath("build/my-languages.so")
        os.makedirs(os.path.dirname(lib_path), exist_ok=True)

        Language.build_library(
            lib_path,
            [cpp_repo]
        )

        return lib_path
    finally:
        shutil.rmtree(temp_dir)

LANG_LIB = get_language_library()
try:
    CPP_LANGUAGE = Language(LANG_LIB, "cpp")
except Exception as e:
    print(f"加载 C++ 语言失败: {e}")
    sys.exit(1)
"""

# -------------------- 节点分类表 --------------------
# NODE_CATEGORIES = {
#     'DEFINITION': {'init_declarator', 'parameter_declaration', 'declaration', 'field_declaration', 'template_parameter', 'alias_declaration'},
#     'ASSIGNMENT': {'assignment_expression'},
#     'INCREMENT': {'update_expression', 'prefix_unary_expression', 'postfix_unary_expression'},
#     'CONTROL_FLOW': {'if_statement', 'switch_statement', 'else', 'case_statement', 'default_statement', 'conditional_expression'},
#     'LOOP': {'while_statement', 'for_statement', 'do_statement', 'range_based_for_statement'},
#     'FUNCTION': {'function_definition', 'return_statement', 'lambda_expression', 'template_declaration', 'function_declarator'},
#     'COMPOUND': {'struct_specifier', 'class_specifier', 'union_specifier', 'enum_specifier', 'namespace_definition', 'field_declaration_list'},
#     'MEMORY': {'pointer_expression', 'reference_expression', 'new_expression', 'delete_expression', 'sizeof_expression', 'address_expression'},
#     'EXPRESSION': {'binary_expression', 'unary_expression', 'parenthesized_expression', 'subscript_expression', 'field_expression', 'call_expression'}
# }

# ============================ 节点分类 ============================

NODE_CATEGORIES = {
    'DEFINITION': {
        'init_declarator', 'parameter_declaration', 'declaration',
        'field_declaration', 'template_parameter', 'alias_declaration',
        'type_identifier', 'primitive_type'
    },
    'ASSIGNMENT': {
        'assignment_expression', 'augmented_assignment_expression'
    },
    'INCREMENT': {
        'update_expression', 'prefix_unary_expression',
        'postfix_unary_expression', 'unary_expression'
    },
    'CONTROL_FLOW': {
        'if_statement', 'switch_statement', 'else',
        'case_statement', 'default_statement',
        'conditional_expression', 'break_statement',
        'continue_statement', 'goto_statement'
    },
    'LOOP': {
        'while_statement', 'for_statement', 'do_statement',
        'range_based_for_statement', 'for_range_loop'
    },
    'FUNCTION': {
        'function_definition', 'return_statement',
        'lambda_expression', 'template_declaration',
        'function_declarator', 'call_expression',
        'parameter_list'
    },
    'COMPOUND': {
        'struct_specifier', 'class_specifier', 'union_specifier',
        'enum_specifier', 'namespace_definition',
        'field_declaration_list', 'compound_statement'
    },
    'MEMORY': {
        'pointer_expression', 'reference_expression',
        'new_expression', 'delete_expression',
        'sizeof_expression', 'address_expression',
        'dereference_expression'
    },
    'EXPRESSION': {
        'binary_expression', 'unary_expression',
        'parenthesized_expression', 'subscript_expression',
        'field_expression', 'call_expression',
        'conditional_expression', 'comma_expression'
    },
    'LITERAL': {
        'number_literal', 'string_literal',
        'char_literal', 'true', 'false', 'nullptr'
    }
}


# ============================ DFG 结构 ============================

@dataclass(frozen=True)
class DFGEdge:
    src_id: int
    dst_id: int
    edge_type: str
    src_pos: T.Tuple[T.Tuple[int, int], T.Tuple[int, int]]
    dst_pos: T.Tuple[T.Tuple[int, int], T.Tuple[int, int]]


class VariableInfo:
    __slots__ = ('var_id', 'var_name', 'var_type',
                 'scope_level', 'positions', 'usage_count')

    def __init__(self, var_id: int, var_name: str,
                 var_type: str = "", scope_level: int = 0):
        self.var_id = var_id
        self.var_name = var_name
        self.var_type = var_type
        self.scope_level = scope_level
        self.positions = []
        self.usage_count = 0

    def add_usage(self, pos):
        self.positions.append(pos)
        self.usage_count += 1


# ============================ Scope ============================

class EnhancedScope:
    def __init__(self, builder,
                 parent: Optional["EnhancedScope"] = None,
                 scope_type="block", scope_name=""):
        self.builder = builder
        self.parent = parent
        self.scope_type = scope_type
        self.scope_name = scope_name
        self.variables = {}
        self.children = []
        self.scope_level = (parent.scope_level + 1) if parent else 0

        if parent:
            parent.children.append(self)

    def declare(self, name, node=None, var_type=""):
        if name in self.variables:
            info = self.variables[name]
            if node:
                info.add_usage((node.start_point, node.end_point))
            return info

        var_id = next(self.builder._var_id_counter)
        info = VariableInfo(var_id, name, var_type, self.scope_level)

        if node:
            info.add_usage((node.start_point, node.end_point))

        self.variables[name] = info
        return info

    def lookup(self, name):
        cur = self
        while cur:
            if name in cur.variables:
                return cur.variables[name]
            cur = cur.parent
        return None

    def all_vars(self):
        out = {}
        cur = self
        while cur:
            out.update({v.var_id: v for v in cur.variables.values()})
            cur = cur.parent
        return out


# ============================ DFG Builder ============================

class CppDFGBuilder:

    def __init__(self):
        self._reset()

    def _reset(self):
        self.edges = []
        self.node_pos = {}
        self._var_id_counter = itertools.count(1)
        self.cur_scope = EnhancedScope(self)

    def build(self, root):
        self._reset()
        self._analyze(root, self.cur_scope)
        return self.edges, self.cur_scope.all_vars()

    def _add_edge(self, src, dst, typ, src_pos, dst_pos):
        self.edges.append(
            DFGEdge(src, dst, typ, src_pos, dst_pos)
        )

    def _analyze(self, node, scope):
        if not node:
            return set()

        kind = node.type

        if kind == "identifier":
            return self._handle_identifier(node, scope)

        if kind in NODE_CATEGORIES['LITERAL']:
            return set()

        for cat, members in NODE_CATEGORIES.items():
            if kind in members:
                handler = getattr(self, f"_handle_{cat.lower()}", None)
                if handler:
                    return handler(node, scope)

        return self._handle_generic(node, scope)

    def _handle_identifier(self, node, scope):
        name = node.text.decode()
        var = scope.lookup(name)
        if var:
            var.add_usage((node.start_point, node.end_point))
            return {var.var_id}
        return set()

    def _handle_assignment(self, node, scope):
        lhs = node.child_by_field_name("left")
        rhs = node.child_by_field_name("right")
        if not lhs or not rhs:
            return set()

        srcs = self._analyze(rhs, scope)
        dsts = self._analyze(lhs, scope)

        for s in srcs:
            for d in dsts:
                self._add_edge(s, d, "computedFrom", None, None)

        return dsts

    def _handle_increment(self, node, scope):
        arg = node.child_by_field_name("argument")

        if arg is None and node.children:
            arg = node.children[0]

        if arg is None:
            return set()

        vars_ = self._analyze(arg, scope)
        for v in vars_:
            self._add_edge(v, v, "selfUpdate", None, None)

        return vars_

    def _handle_definition(self, node, scope):
        out = set()

        type_node = node.child_by_field_name("type")
        var_type = type_node.text.decode() if type_node else ""

        for child in node.children:
            if child.type in {"init_declarator", "declarator"}:
                ident = self._find_ident(child)
                if not ident:
                    continue

                name = ident.text.decode()
                var = scope.declare(name, ident, var_type)
                out.add(var.var_id)

                val = child.child_by_field_name("value")
                if val:
                    srcs = self._analyze(val, scope)
                    for src in srcs:
                        self._add_edge(src, var.var_id,
                                       "computedFrom", None, None)

        return out

    def _handle_generic(self, node, scope):
        res = set()
        for ch in node.children:
            res.update(self._analyze(ch, scope))
        return res

    def _find_ident(self, node):
        if node.type == "identifier":
            return node
        for ch in node.children:
            r = self._find_ident(ch)
            if r:
                return r
        return None


# ============================ α 归一化 ============================

def _build_alpha_table(vars_dict):
    scope2vars = defaultdict(list)

    for v in vars_dict.values():
        scope2vars[v.scope_level].append(v)

    name_map = {}

    for level in sorted(scope2vars.keys()):
        level_vars = sorted(
            scope2vars[level],
            key=lambda vv: vv.positions[0] if vv.positions else (0, 0)
        )
        for idx, var in enumerate(level_vars):
            new_name = f"s{level}_v{idx}"
            name_map[(var.scope_level, var.var_name)] = new_name

    return name_map


def apply_alpha_conversion(vars_dict):
    alpha = _build_alpha_table(vars_dict)

    for v in vars_dict.values():
        key = (v.scope_level, v.var_name)
        if key in alpha:
            v.var_name = alpha[key]

    return vars_dict


# ============================ 匹配 ============================

class DFGMatcher:

    def semantic_match(self,
                       edges_pred,
                       edges_ref,
                       vars_pred,
                       vars_ref):

        if not edges_ref:
            return 1.0 if not edges_pred else 0.0

        vars_pred_dict = dict(vars_pred)
        vars_ref_dict = dict(vars_ref)

        g_pred = self._build_graph(edges_pred, vars_pred_dict)
        g_ref = self._build_graph(edges_ref, vars_ref_dict)

        matched = self._advanced_matching(
            edges_pred, edges_ref, g_pred, g_ref
        )

        return len(matched) / len(edges_ref)

    def _build_graph(self, edges, vars_):
        g = {
            "nodes": vars_,
            "adj": defaultdict(list),
            "deg": defaultdict(int)
        }

        for e in edges:
            g["adj"][e.src_id].append((e.dst_id, e.edge_type))
            g["deg"][e.src_id] += 1
            g["deg"][e.dst_id] += 1

        return g

    def _advanced_matching(self, ep, er, gp, gr):
        matched = set()

        for re in er:
            best = None
            best_score = 0.7

            for pe in ep:
                if pe in matched:
                    continue
                s = self._edge_sim(pe, re, gp, gr)
                if s > best_score:
                    best = pe
                    best_score = s

            if best:
                matched.add(best)

        return matched

    def _edge_sim(self, e1, e2, g1, g2):
        if e1.edge_type != e2.edge_type:
            return 0.0

        struct = self._struct_sim(e1, e2, g1, g2)
        return struct

    def _struct_sim(self, e1, e2, g1, g2):

        def neighbors(g, e):
            res = set()
            for dst, typ in g["adj"].get(e.src_id, []):
                v = g["nodes"].get(dst)
                name = v.var_name if v else None
                res.add((name, typ))
            return res

        n1 = neighbors(g1, e1)
        n2 = neighbors(g2, e2)

        if not n1 and not n2:
            return 1.0

        inter = len(n1 & n2)
        union = len(n1 | n2)

        return inter / union if union else 0.0


class SemanticMatchScore:
    def __init__(self, parser):
        self.parser = parser
        self.builder = CppDFGBuilder()
        self.matcher = DFGMatcher()

    # -------------------- CLI 入口 --------------------
    def parse_code(self, code: str) -> Node:
        """解析代码并返回 AST 根节点"""
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree.root_node
        except Exception as e:
            print(f"解析代码失败: {e}")
            return None

    def print_dfg_info(self, edges, variables, name="DFG"):
        """打印 DFG 信息用于调试"""
        print(f"\n=== {name} 信息 ===")
        print(f"变量数量: {len(variables)}")
        print(f"边数量: {len(edges)}")

        print("\n变量列表:")
        for var_id, var_info in variables.items():
            print(f"  {var_id}: {var_info.var_name} (类型: {var_info.var_type}, 作用域: {var_info.scope_level})")

        print("\n数据流边:")
        for edge in edges[:10]:  # 只显示前10条边
            src_var = variables.get(edge.src_id, None)
            dst_var = variables.get(edge.dst_id, None)
            src_name = src_var.var_name if src_var else f"未知({edge.src_id})"
            dst_name = dst_var.var_name if dst_var else f"未知({edge.dst_id})"
            print(f"  {src_name} -> {dst_name} [{edge.edge_type}]")

    def semantic_matching_score(self, code1, code2):
        # 解析代码
        root1, root2 = self.parse_code(code1), self.parse_code(code2)
        if not root1 or not root2:
            print("解析代码失败")
            sys.exit(1)

        # 构建 DFG
        edges1, vars1 = self.builder.build(root1)
        edges2, vars2 = self.builder.build(root2)

        # ---------- 变量名 α-归一化 ----------
        vars1 = apply_alpha_conversion(vars1)
        vars2 = apply_alpha_conversion(vars2)

        # 调试信息
        if len(edges1) == 0 or len(edges2) == 0:
            print("警告: 一个或两个文件的 DFG 为空")
            # self.print_dfg_info(edges1, vars1, "文件1")
            # self.print_dfg_info(edges2, vars2, "文件2")

        # 计算匹配度
        score = self.matcher.semantic_match(
            tuple(edges1), tuple(edges2),
            tuple(vars1.items()), tuple(vars2.items())
        )

        print(f"语义匹配度: {score:.4f}")
        return score

#
# def main():
#     if len(sys.argv) != 3:
#         print("用法: python dfg_match.py <file1.cpp> <file2.cpp>")
#         print("示例: python dfg_match.py example1.cpp example2.cpp")
#         sys.exit(1)
#
#     file1, file2 = sys.argv[1], sys.argv[2]
#
#     try:
#         with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
#             code1, code2 = f1.read(), f2.read()
#     except FileNotFoundError as e:
#         print(f"文件未找到: {e}")
#         sys.exit(1)
#     except UnicodeDecodeError:
#         # 尝试其他编码
#         try:
#             with open(file1, 'r', encoding='latin-1') as f1, open(file2, 'r', encoding='latin-1') as f2:
#                 code1, code2 = f1.read(), f2.read()
#         except Exception as e:
#             print(f"读取文件失败: {e}")
#             sys.exit(1)
#
#     # 解析代码
#     root1, root2 = parse_code(code1), parse_code(code2)
#     if not root1 or not root2:
#         print("解析代码失败")
#         sys.exit(1)
#
#     # 构建 DFG
#     builder = CppDFGBuilder()
#     edges1, vars1 = builder.build(root1)
#     edges2, vars2 = builder.build(root2)
#
#     # ---------- 变量名 α-归一化 ----------
#     vars1 = apply_alpha_conversion(vars1)
#     vars2 = apply_alpha_conversion(vars2)
#
#     # 调试信息
#     if len(edges1) == 0 or len(edges2) == 0:
#         print("警告: 一个或两个文件的 DFG 为空")
#         print_dfg_info(edges1, vars1, "文件1")
#         print_dfg_info(edges2, vars2, "文件2")
#
#     # 计算匹配度
#     matcher = DFGMatcher()
#     score = matcher.semantic_match(
#         tuple(edges1), tuple(edges2),
#         tuple(vars1.items()), tuple(vars2.items())
#     )
#
#     print(f"语义匹配度: {score:.4f}")
#
#     # 显示详细匹配信息
#     if score < 0.8:  # 如果匹配度较低，显示更多信息
#         print(f"\n详细分析:")
#         print(f"文件1: {len(edges1)} 条边, {len(vars1)} 个变量")
#         print(f"文件2: {len(edges2)} 条边, {len(vars2)} 个变量")
#         print(f"匹配边数: {int(score * len(edges2))} / {len(edges2)}")
#
#
# if __name__ == "__main__":
#     """
#     echo "int x=1; int y=x+2;" > a.cpp
#     echo "int a=1; int b=a+2;" > b.cpp
#     python dfg_match.py a.cpp b.cpp
#     # → 语义匹配度: 1.00
#     """
#     main()
#
# if __name__ == '__main__':
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     # 对1的所有代码
#     pair1 = [
#         """// 代码A - 先计算再输出
# void process_data(int x, int y) {
#     int sum = x + y;        // 定义: sum
#     int prod = x * y;        // 定义: prod
#     int result = sum + prod; // 使用: sum, prod
#     printf("%d", result);    // 使用: result
# }""",
#         """// 代码B - 相同的数据依赖，不同顺序
# void process_data(int x, int y) {
#     int sum = x + y;         // 定义: sum
#     int result = sum + x*y;  // 使用: sum，直接使用x*y
#     printf("%d", result);    // 使用: result
# }"""
#     ]
#
#     # 对2的所有代码
#     pair2 = [
#         """// 代码A - 链式数据依赖
# int chain_deps(int a, int b, int c) {
#     int x = a + b;      // x 依赖 a,b
#     int y = x * c;      // y 依赖 x,c
#     int z = y - a;      // z 依赖 y,a
#     return z;
# }""",
#         """// 代码B - 并行数据依赖
# int parallel_deps(int a, int b, int c) {
#     int x = a + b;      // x 依赖 a,b
#     int y = a * c;      // y 依赖 a,c
#     int z = x + y;      // z 依赖 x,y
#     return z;
# }"""
#     ]
#
#     # 对3的所有代码
#     pair3 = [
#         """// 代码A - 浅层数据流
# int shallow_flow(int n) {
#     int result = 0;
#     for (int i = 0; i < n; i++) {
#         result += i;     // 每次循环只依赖result和i
#     }
#     return result;
# }""",
#         """// 代码B - 深层数据流
# int deep_flow(int n) {
#     int result = 0;
#     int temp = 1;
#     for (int i = 0; i < n; i++) {
#         temp = temp * i;      // temp依赖自身
#         result += temp;       // result依赖temp
#     }
#     return result;
# }"""
#     ]
#
#     # 对4的所有代码
#     pair4 = [
#         """// 代码A - 整型数组处理
# int process_int_array(int arr[], int size) {
#     int sum = 0;
#     for (int i = 0; i < size; i++) {
#         sum += arr[i];        // 读arr，写sum
#         arr[i] = sum;         // 读sum，写arr
#     }
#     return sum;
# }""",
#         """// 代码B - 浮点数组处理
# float process_float_array(float arr[], int size) {
#     float sum = 0.0;
#     for (int i = 0; i < size; i++) {
#         sum += arr[i];        // 读arr，写sum
#         arr[i] = sum;         // 读sum，写arr
#     }
#     return sum;
# }"""
#     ]
#
#     # 对5的所有代码
#     pair5 = [
#         """// 代码A - 直接数据传递
# int direct_pass(int x) {
#     int a = x + 1;      // a依赖x
#     int b = a * 2;      // b依赖a
#     int c = b - 3;      // c依赖b
#     return c;           // 返回c
# }""",
#         """// 代码B - 间接数据传递
# int indirect_pass(int x) {
#     int a = x + 1;      // a依赖x
#     int b = x * 2;      // b依赖x
#     int c = a + b;      // c依赖a,b
#     return c;           // 返回c
# }"""
#     ]
#
#     # 对6的所有代码
#     pair6 = [
#         """// 代码A - 简单索引依赖
# void array_index_simple(int a[], int n) {
#     for (int i = 1; i < n; i++) {
#         a[i] = a[i-1] + 1;    // 每个元素依赖前一个
#     }
# }""",
#         """// 代码B - 复杂索引依赖
# void array_index_complex(int a[], int n) {
#     for (int i = 2; i < n; i++) {
#         a[i] = a[i-1] + a[i-2];  // 每个元素依赖前两个
#     }
# }"""
#     ]
#
#     # 对7的所有代码
#     pair7 = [
#         """// 代码A - 条件数据流
# int conditional_flow(int x, int y) {
#     int result = 0;
#     if (x > y) {
#         result = x + y;     // result依赖x,y
#     } else {
#         result = x - y;     // result依赖x,y
#     }
#     return result;          // 使用result
# }""",
#         """// 代码B - 无条件数据流
# int unconditional_flow(int x, int y) {
#     int result = x + y;     // result依赖x,y
#     if (x > y) {
#         result += x;        // result依赖自身
#     }
#     return result;          // 使用result
# }"""
#     ]
#
#     # 对8的所有代码
#     pair8 = [
#         """// 代码A - 计数器终止
# int counter_termination(int n) {
#     int result = 0;
#     for (int i = 0; i < n; i++) {
#         result += i;        // result依赖自身，循环由i控制
#     }
#     return result;
# }""",
#         """// 代码B - 数据依赖终止
# int data_termination(int n) {
#     int result = 0;
#     int i = 0;
#     while (result < n) {    // 循环条件依赖result
#         result += i;        // result依赖自身
#         i++;                // i依赖自身
#     }
#     return result;
# }"""
#     ]
#
#     # 对9的所有代码
#     pair9 = [
#         """// 代码A - 加法累积
# int accumulate_sum(int n) {
#     int sum = 0;
#     for (int i = 0; i <= n; i++) {
#         sum = sum + i;      // 累积模式
#     }
#     return sum;
# }""",
#         """// 代码B - 乘法累积
# int accumulate_product(int n) {
#     int prod = 1;
#     for (int i = 1; i <= n; i++) {
#         prod = prod * i;    // 相同的累积模式，不同操作
#     }
#     return prod;
# }"""
#     ]
#
#     # 对10的所有代码
#     pair10 = [
#         """// 代码A - 树状数据流
# int tree_flow(int a, int b, int c, int d) {
#     int x = a + b;          // x依赖a,b
#     int y = c + d;          // y依赖c,d
#     int z = x * y;          // z依赖x,y
#     return z;
# }""",
#         """// 代码B - 链状数据流
# int chain_flow(int a, int b, int c, int d) {
#     int x = a + b;          // x依赖a,b
#     int y = x * c;          // y依赖x,c
#     int z = y - d;          // z依赖y,d
#     return z;
# }"""
#     ]
#
#     # 将所有对组合成一个大的列表，方便批量处理
#     all_pairs = [pair1, pair2, pair3, pair4, pair5, pair6, pair7, pair8, pair9, pair10]
#
#
#     def remove_special_tokens(code_string):
#         lines = code_string.split("NEW_LINE")
#         lines = [item.strip() for item in lines]
#
#         curr_indent = 0
#         new_lines = []
#         for line in lines:
#             indent_count = line.count('INDENT')
#             dedent_count = line.count('DEDENT')
#             curr_indent += indent_count - dedent_count
#             wo_indent = re.sub('INDENT\s?', '', line)
#             wo_dedent = re.sub('DEDENT\s?', '', wo_indent)
#             new_lines.append('\t' * curr_indent + wo_dedent)
#         return ("\n").join(new_lines)
#
#
#     from tree_sitter import Language, Parser
#     from rewards_tools.code_prepro.c_processor import *
#
#     try:
#         LANGUAGE = Language(so_path, 'c')
#         parser = Parser()
#         parser.set_language(LANGUAGE)
#     except Exception as e:
#         logger.error(f"Failed to initialize parser: {e}")
#         raise
#
#     processor = CProcessor(parser)
#     # c_tokenizer = processor.tokenize_code
#     c_detokenizer = processor.detokenize_code
#
#     # syntax_analyzer_llm = SyntaxMatchScorer(parser)
#     dfg_analyzer_llm = SemanticMatchScore(parser)
#     #
#     # syntax_analyzer_ppo = SyntaxAnalyzer(parser)
#     # dfg_analyzer_ppo = DataFlowAnalyzer(parser)
#
#     for each_pair in all_pairs:
#         pre_code = c_detokenizer(each_pair[0])
#         pre_code = remove_special_tokens(pre_code)  # 去掉代码中的多余字符
#         gt_code = remove_special_tokens(each_pair[1])
#         print(f"this is the pre_code: \n{pre_code}")
#         print(f"this is the gt_code: \n{gt_code}")
#
#         # calculate syntax match and dataflow match
#         try:
#             # syntax_match_score = syntax_analyzer_llm.calculate_syntax_score(pre_code, gt_code)
#             dataflow_match_score = dfg_analyzer_llm.semantic_matching_score(pre_code, gt_code)
#             # syntax_match_score = syntax_analyzer_ppo.calculate_singe_ref_can(gt_code, pre_code)
#             # dataflow_match_score = dfg_analyzer_ppo.calculate_singe_ref_can(gt_code, pre_code)
#             print(f"the dataflow_match_score is {dataflow_match_score}")
#         except Exception as e:
#             logger.info(f"There is wrong when we compute syntax match and dfg match: {e}.")