# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from rewards_tools.parser.utils import remove_c_cpp_comments
from apted import APTED, Config
from apted.helpers import Tree
import re
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import List, Tuple, Dict, Set, Any, Optional, Union


class SyntaxAnalyzer:
    """
    语法分析器，用于提取和比较代码的语法结构
    """

    def __init__(self, parser):
        self.parser = parser

    def calculate_singe_ref_can(self, reference: str, candidate: str) -> float:
        return self.calculate_single_syntax_match([reference], candidate)

    def calculate_single_syntax_match(self, references: List[str], candidate: str) -> float:
        return self.calculate_syntax_similarity([references], [candidate])

    def calculate_syntax_similarity(self, references: List[List[str]], candidates: List[str]) -> float:
        """
        Args:
            references: 参考代码列表，每个元素是一组参考代码
            candidates: 候选代码列表

        Returns:
            float: 相似度得分 (0-1)
        """
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")

        match_count = 0
        total_count = 0

        try:
            for i, candidate in enumerate(candidates):
                reference_group = references[i]
                candidate_subtrees = [x[0] for x in self._get_all_subtrees(candidate)]

                for reference in reference_group:
                    reference_subtrees = self._get_all_subtrees(reference)

                    if not reference_subtrees:
                        logger.warning("No syntax subtrees extracted from reference code")
                        continue

                    # 计算匹配的子树数量
                    for subtree, depth in reference_subtrees:
                        if subtree in candidate_subtrees:
                            match_count += 1
                    total_count += len(reference_subtrees)

            if total_count == 0:
                logger.warning("No reference syntax subtrees extracted from the whole corpus")
                return 0.0

            return match_count / total_count
        except Exception as e:
            print(e)
            return 0


    def _get_all_subtrees(self, code: str):
        """
        get the sub-tree expression
        """
        try:
            cleaned_code = self._remove_comments(code)
            tree = self.parser.parse(bytes(cleaned_code, 'utf8'))
            root_node = tree.root_node
            return self._extract_all_subtrees(root_node)
        except Exception as e:
            logger.error(f"Error extracting syntax trees: {e}")
            return []

    def _remove_comments(self, code: str) -> str:
        """
        clean the notes
        """
        try:
            return remove_c_cpp_comments(code)
        except Exception as e:
            logger.warning(f"Failed to remove comments: {e}")
            return code

    def _extract_all_subtrees(self, root_node):
        """
        提取AST中的所有子树（保持与原代码相同的逻辑）

        Args:
            root_node: AST根节点

        Returns:
            子树表达式和深度列表
        """
        node_stack = []
        sub_tree_sexp_list = []
        depth = 1
        node_stack.append([root_node, depth])

        while len(node_stack) != 0:
            cur_node, cur_depth = node_stack.pop()
            # sub_tree_sexp_list.append([cur_node.sexp(), cur_depth])
            sub_tree_sexp_list.append([str(cur_node), cur_depth])

            for child_node in cur_node.children:
                if len(child_node.children) != 0:
                    depth = cur_depth + 1
                    node_stack.append([child_node, depth])

        return sub_tree_sexp_list

# ==========================================================================================================
# the methods 2 by APTED, but there is still a bug
class SyntaxMatchScorer:
    def __init__(self, parser):
        self.parser = parser
        self._LITERAL_RE = re.compile(
            r'^(integer_literal|float_literal|char_literal|string_literal|true|false|nullptr)$')
        self._PRUNE_SET = {'comment', 'preproc_include', 'preproc_def'}

    def normalize_ast_to_apted_tree(self, node, source_bytes=None):
        """
        将 Tree-sitter AST 转成 APTED Tree，保证返回的永远是 Tree 对象或 None。
        source_bytes: 源码字节串，用于从 node.start_byte/end_byte 截取节点文本（py-tree-sitter 的 Node 无 .text 属性）。
        """
        if node is None:
            return None

        node_type = getattr(node, "type", str(node))
        if node_type in self._PRUNE_SET:
            return None

        # 标签处理：从 source_bytes 截取节点文本（py-tree-sitter 需手动切片）
        if self._LITERAL_RE.match(node_type):
            try:
                if source_bytes is not None and hasattr(node, 'start_byte') and hasattr(node, 'end_byte'):
                    value = source_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                else:
                    value = getattr(node, 'text', b'').decode('utf8', errors='ignore') if hasattr(node, 'text') else ""
            except Exception:
                value = ""
            label = f"{node_type}[{value}]"
        elif node_type in {'identifier', 'type_identifier', 'function_identifier'}:
            label = f"{node_type}[*]"
        else:
            label = node_type

        # 递归处理所有子节点
        children = []
        for child in getattr(node, "children", []):
            child_tree = self.normalize_ast_to_apted_tree(child, source_bytes)
            if child_tree is not None:
                if isinstance(child_tree, Tree):
                    children.append(child_tree)
                else:
                    children.append(Tree(str(child_tree)))

        return Tree(label, *children)

    def debug_tree(self, tree, depth=0):
        """安全地调试树结构"""
        if tree is None:
            return

        # APTED Tree 是一个 namedtuple，有 name 和 children 属性
        print("  " * depth + f"Node: {tree.name} (type: {type(tree)})")

        # 安全地访问 children
        children = getattr(tree, 'children', [])
        for child in children:
            if isinstance(child, Tree):
                self.debug_tree(child, depth + 1)
            else:
                print("  " * (depth + 1) + f"WARNING: Non-Tree child: {type(child)} - {child}")

    def count_nodes(self, tree):
        """计算树中的节点数"""
        if tree is None:
            return 0
        return 1 + sum(self.count_nodes(child) for child in tree.children)

    def calculate_syntax_score(self, generated_code, reference_code):
        """计算语法匹配分数"""
        try:
            gen_bytes = generated_code.encode() if isinstance(generated_code, str) else generated_code
            ref_bytes = reference_code.encode() if isinstance(reference_code, str) else reference_code
            tree_gen = self.parser.parse(gen_bytes)
            tree_ref = self.parser.parse(ref_bytes)

            # 转换为APTED树（传入 source_bytes 以便从节点 byte 范围取文本）
            apted_tree_gen = self.normalize_ast_to_apted_tree(tree_gen.root_node, gen_bytes)
            apted_tree_ref = self.normalize_ast_to_apted_tree(tree_ref.root_node, ref_bytes)

            # print("\nDebugging generated tree:")
            # self.debug_tree(apted_tree_gen)

            # print("\nDebugging reference tree:")
            # self.debug_tree(apted_tree_ref)

            # 检查有效性
            if apted_tree_gen is None or apted_tree_ref is None:
                return 0.0

            # 验证树结构的完整性
            def validate_tree(tree, path=""):
                if not isinstance(tree, Tree):
                    print(f"ERROR: Invalid tree at {path}: {type(tree)}")
                    return False

                for i, child in enumerate(tree.children):
                    if not isinstance(child, Tree):
                        print(f"ERROR: Non-Tree child at {path}.children[{i}]: {type(child)}")
                        return False
                    if not validate_tree(child, f"{path}.children[{i}]"):
                        return False
                return True

            if not validate_tree(apted_tree_gen):
                print("Generated tree validation failed")
                return 0.0

            if not validate_tree(apted_tree_ref):
                print("Reference tree validation failed")
                return 0.0

            # 计算编辑距离
            config = ValueAwareConfig()
            apted = APTED(apted_tree_gen, apted_tree_ref, config)
            edit_distance = apted.compute_edit_distance()

            # 转换为浮点数
            if isinstance(edit_distance, str):
                try:
                    edit_distance = float(edit_distance)
                except ValueError:
                    print(f"Cannot convert {edit_distance} to float")
                    return 0.0

            # 计算树大小
            size_gen = self.count_nodes(apted_tree_gen)
            size_ref = self.count_nodes(apted_tree_ref)

            # 计算归一化分数
            max_size = max(size_gen, size_ref)
            if max_size == 0:
                return 1.0

            normalized_distance = edit_distance / max_size
            score = max(0.0, 1.0 - normalized_distance)

            return score

        except Exception as e:
            print(f"Error in calculate_syntax_score: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


class ValueAwareConfig(Config):
    """自定义成本配置"""
    valuecls = float

    def rename(self, node1, node2):
        """重命名操作的成本"""
        if node1 is None or node2 is None:
            return 1.0

        # 确保我们处理的是 Tree 对象的 name 属性
        name1 = node1.name if hasattr(node1, 'name') else str(node1)
        name2 = node2.name if hasattr(node2, 'name') else str(node2)

        def parse_label(label):
            if '[' in label and label.endswith(']'):
                type_part, value_part = label.split('[', 1)
                value_part = value_part.rstrip(']')
                return type_part, value_part
            return str(label), ""

        type1, val1 = parse_label(name1)
        type2, val2 = parse_label(name2)

        # 类型不同 -> 高成本
        if type1 != type2:
            return 1.0

        # 标识符：类型相同即可完全匹配
        if type1 in ['identifier', 'type_identifier', 'function_identifier']:
            return 0.0

        # 字面量：需要值匹配
        if type1 in {'integer_literal', 'float_literal'}:
            try:
                f1, f2 = float(val1), float(val2)
                return 0.0 if abs(f1 - f2) < 1e-10 else 0.8
            except ValueError:
                return 0.0 if val1 == val2 else 0.8

        if type1 in {'char_literal', 'string_literal', 'true', 'false', 'nullptr'}:
            return 0.0 if val1 == val2 else 0.8

        # 其他节点：类型相同即可
        return 0.0

    def delete(self, node):
        return 1.0

    def insert(self, node):
        return 1.0

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # 对1的所有代码
    pair1 = [
        """// 代码A - 先计算再输出
void process_data(int x, int y) {
    int sum = x + y;        // 定义: sum
    int prod = x * y;        // 定义: prod
    int result = sum + prod; // 使用: sum, prod
    printf("%d", result);    // 使用: result
}""",
        """// 代码B - 相同的数据依赖，不同顺序
void process_data(int x, int y) {
    int sum = x + y;         // 定义: sum
    int result = sum + x*y;  // 使用: sum，直接使用x*y
    printf("%d", result);    // 使用: result
}"""
    ]

    # 对2的所有代码
    pair2 = [
        """// 代码A - 链式数据依赖
int chain_deps(int a, int b, int c) {
    int x = a + b;      // x 依赖 a,b
    int y = x * c;      // y 依赖 x,c
    int z = y - a;      // z 依赖 y,a
    return z;
}""",
        """// 代码B - 并行数据依赖
int parallel_deps(int a, int b, int c) {
    int x = a + b;      // x 依赖 a,b
    int y = a * c;      // y 依赖 a,c
    int z = x + y;      // z 依赖 x,y
    return z;
}"""
    ]

    # 对3的所有代码
    pair3 = [
        """// 代码A - 浅层数据流
int shallow_flow(int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        result += i;     // 每次循环只依赖result和i
    }
    return result;
}""",
        """// 代码B - 深层数据流
int deep_flow(int n) {
    int result = 0;
    int temp = 1;
    for (int i = 0; i < n; i++) {
        temp = temp * i;      // temp依赖自身
        result += temp;       // result依赖temp
    }
    return result;
}"""
    ]

    # 对4的所有代码
    pair4 = [
        """// 代码A - 整型数组处理
int process_int_array(int arr[], int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];        // 读arr，写sum
        arr[i] = sum;         // 读sum，写arr
    }
    return sum;
}""",
        """// 代码B - 浮点数组处理
float process_float_array(float arr[], int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];        // 读arr，写sum
        arr[i] = sum;         // 读sum，写arr
    }
    return sum;
}"""
    ]

    # 对5的所有代码
    pair5 = [
        """// 代码A - 直接数据传递
int direct_pass(int x) {
    int a = x + 1;      // a依赖x
    int b = a * 2;      // b依赖a
    int c = b - 3;      // c依赖b
    return c;           // 返回c
}""",
        """// 代码B - 间接数据传递
int indirect_pass(int x) {
    int a = x + 1;      // a依赖x
    int b = x * 2;      // b依赖x
    int c = a + b;      // c依赖a,b
    return c;           // 返回c
}"""
    ]

    # 对6的所有代码
    pair6 = [
        """// 代码A - 简单索引依赖
void array_index_simple(int a[], int n) {
    for (int i = 1; i < n; i++) {
        a[i] = a[i-1] + 1;    // 每个元素依赖前一个
    }
}""",
        """// 代码B - 复杂索引依赖
void array_index_complex(int a[], int n) {
    for (int i = 2; i < n; i++) {
        a[i] = a[i-1] + a[i-2];  // 每个元素依赖前两个
    }
}"""
    ]

    # 对7的所有代码
    pair7 = [
        """// 代码A - 条件数据流
int conditional_flow(int x, int y) {
    int result = 0;
    if (x > y) {
        result = x + y;     // result依赖x,y
    } else {
        result = x - y;     // result依赖x,y
    }
    return result;          // 使用result
}""",
        """// 代码B - 无条件数据流
int unconditional_flow(int x, int y) {
    int result = x + y;     // result依赖x,y
    if (x > y) {
        result += x;        // result依赖自身
    }
    return result;          // 使用result
}"""
    ]

    # 对8的所有代码
    pair8 = [
        """// 代码A - 计数器终止
int counter_termination(int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        result += i;        // result依赖自身，循环由i控制
    }
    return result;
}""",
        """// 代码B - 数据依赖终止
int data_termination(int n) {
    int result = 0;
    int i = 0;
    while (result < n) {    // 循环条件依赖result
        result += i;        // result依赖自身
        i++;                // i依赖自身
    }
    return result;
}"""
    ]

    # 对9的所有代码
    pair9 = [
        """// 代码A - 加法累积
int accumulate_sum(int n) {
    int sum = 0;
    for (int i = 0; i <= n; i++) {
        sum = sum + i;      // 累积模式
    }
    return sum;
}""",
        """// 代码B - 乘法累积
int accumulate_product(int n) {
    int prod = 1;
    for (int i = 1; i <= n; i++) {
        prod = prod * i;    // 相同的累积模式，不同操作
    }
    return prod;
}"""
    ]

    # 对10的所有代码
    pair10 = [
        """// 代码A - 树状数据流
int tree_flow(int a, int b, int c, int d) {
    int x = a + b;          // x依赖a,b
    int y = c + d;          // y依赖c,d
    int z = x * y;          // z依赖x,y
    return z;
}""",
        """// 代码B - 链状数据流
int chain_flow(int a, int b, int c, int d) {
    int x = a + b;          // x依赖a,b
    int y = x * c;          // y依赖x,c
    int z = y - d;          // z依赖y,d
    return z;
}"""
    ]

    # 将所有对组合成一个大的列表，方便批量处理
    all_pairs = [pair1, pair2, pair3, pair4, pair5, pair6, pair7, pair8, pair9, pair10]


    def remove_special_tokens(code_string):
        lines = code_string.split("NEW_LINE")
        lines = [item.strip() for item in lines]

        curr_indent = 0
        new_lines = []
        for line in lines:
            indent_count = line.count('INDENT')
            dedent_count = line.count('DEDENT')
            curr_indent += indent_count - dedent_count
            wo_indent = re.sub(r'INDENT\s?', '', line)
            wo_dedent = re.sub(r'DEDENT\s?', '', wo_indent)
            new_lines.append('\t' * curr_indent + wo_dedent)
        return ("\n").join(new_lines)


    from tree_sitter import Language, Parser
    from rewards_tools.code_prepro.c_processor import *

    # 跨平台：优先在 rewards_tools/parser 下查找 my-languages/old_my-languages 的 .so（Linux）或 .pyd（Windows）
    _metrics_dir = os.path.dirname(os.path.abspath(__file__))
    _parser_dir = os.path.join(os.path.dirname(_metrics_dir), 'parser')
    so_path = None
    for base in ['old_my-languages', 'my-languages']:
        for ext in ('.pyd', '.so'):  # Windows 多为 .pyd
            p = os.path.join(_parser_dir, base + ext)
            if os.path.isfile(p):
                so_path = p
                break
        if so_path:
            break
    if not so_path:
        # 尝试自动构建：使用 tree-sitter-c / tree-sitter-cpp 目录
        ext = '.pyd' if sys.platform == 'win32' else '.so'
        out_lib = os.path.join(_parser_dir, 'my-languages' + ext)
        c_repo = os.path.join(_parser_dir, 'tree-sitter-c')
        cpp_repo = os.path.join(_parser_dir, 'tree-sitter-cpp')
        if os.path.isdir(c_repo) and os.path.isdir(cpp_repo):
            try:
                logger.info("正在构建 parser 库 my-languages%s ...", ext)
                Language.build_library(out_lib, [c_repo, cpp_repo])
                so_path = out_lib
            except Exception as build_err:
                logger.warning("自动构建 parser 库失败: %s", build_err)
        if not so_path or not os.path.isfile(so_path):
            so_path = os.path.join(_parser_dir, 'my-languages.so')
            if not os.path.isfile(so_path):
                logger.error(
                    "Parser 库未找到。请在 rewards_tools/parser 下构建 C 解析库，例如：\n"
                    "  cd %s\n"
                    "  参考 build_lang.py 使用 Language.build_library 生成 my-languages.so（或 Windows 下 my-languages.pyd）",
                    _parser_dir,
                )
                raise FileNotFoundError("Parser library not found under %s" % _parser_dir)
    try:
        # LANGUAGE = Language(so_path, 'c')
        # parser = Parser()
        # parser.set_language(LANGUAGE)
        import tree_sitter_cpp as tscpp
        LANGUAGE = Language(tscpp.language())
        parser = Parser(LANGUAGE)
    except Exception as e:
        logger.error(f"Failed to initialize parser: {e}")
        raise

    processor = CProcessor(parser)
    # c_tokenizer = processor.tokenize_code
    c_detokenizer = processor.detokenize_code

    syntax_analyzer_llm = SyntaxMatchScorer(parser)
    # dfg_analyzer_llm = SemanticMatchScore(parser)
    #
    # syntax_analyzer_ppo = SyntaxAnalyzer(parser)
    # dfg_analyzer_ppo = DataFlowAnalyzer(parser)

    for each_pair in all_pairs:
        pre_code = c_detokenizer(each_pair[0])
        pre_code = remove_special_tokens(pre_code)  # 去掉代码中的多余字符
        gt_code = remove_special_tokens(each_pair[1])
        print(f"this is the pre_code: \n{remove_c_cpp_comments(pre_code)}")
        print(f"this is the gt_code: \n{remove_c_cpp_comments(gt_code)}")

        # calculate syntax match and dataflow match
        try:
            syntax_match_score = syntax_analyzer_llm.calculate_syntax_score(pre_code, gt_code)
            # dataflow_match_score = dfg_analyzer_llm.semantic_matching_score(pre_code, gt_code)
            # syntax_match_score = syntax_analyzer_ppo.calculate_singe_ref_can(gt_code, pre_code)
            # dataflow_match_score = dfg_analyzer_ppo.calculate_singe_ref_can(gt_code, pre_code)
            print(f"the dataflow_match_score is {syntax_match_score}")
        except Exception as e:
            logger.info(f"There is wrong when we compute syntax match and dfg match: {e}.")