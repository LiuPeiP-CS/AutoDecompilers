import logging
from functools import lru_cache
from typing import List, Tuple, Dict, Set, Any, Optional, Union
from tree_sitter import Language, Parser, Node
from rewards_tools.parser.utils import remove_c_cpp_comments

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFlowAnalyzer:
    """
    数据流分析器，用于提取和比较代码的数据流图
    """

    # 定义各种语句类型
    ASSIGNMENT_STATEMENTS = ['assignment_expression']
    DECLARATION_STATEMENTS = ['variable_declarator']
    INCREMENT_STATEMENTS = ['postfix_unary_expression']
    CONTROL_FLOW_STATEMENTS = {
        'if': ['if_statement', 'else'],
        'for': ['for_statement'],
        'foreach': ['for_each_statement'],
        'while': ['while_statement']
    }

    def __init__(self, parser):
        self.parser = parser

    def calculate_singe_ref_can(self, reference: str, candidate: str) -> float:
        return self.calculate_single_dfg_match([reference], candidate)

    def calculate_single_dfg_match(self, references: List[str], candidate: str) -> float:
        return self.calculate_dataflow_similarity([references], [candidate])

    # @lru_cache(maxsize=500)
    def calculate_dataflow_similarity(self, references: List[List[str]], candidates: List[str]) -> float:
        """
        计算参考代码和候选代码之间的数据流相似度

        Args:
            references: 参考代码列表，每个元素是一组参考代码
            candidates: 候选代码列表

        Returns:
            float: 相似度得分 (0-1)
        """
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")

        total_matches = 0
        total_flows = 0

        try:
            for i, candidate in enumerate(candidates):
                reference_group = references[i]
                candidate_flows = self._get_normalized_dataflow(candidate)

                for reference in reference_group:
                    reference_flows = self._get_normalized_dataflow(reference)

                    if not reference_flows:
                        logger.warning("No data flows extracted from reference code")
                        continue

                    if len(reference_flows) > 0:
                        total_flows += len(reference_flows)
                        matches = self._count_matching_flows(reference_flows, candidate_flows)
                        total_matches += matches

            if total_flows == 0:
                logger.warning("No reference data flows extracted from the whole corpus")
                return 0.0

            return total_matches / total_flows
        except Exception as e:
            print(e)
            return 0.0

    def calculate_single_dataflow_match(self, references: List[str], candidate: str) -> float:
        """
        计算单个候选代码与参考代码的数据流匹配度

        Args:
            references: 参考代码列表
            candidate: 候选代码

        Returns:
            float: 匹配度得分
        """
        return self.calculate_dataflow_similarity([references], [candidate])

    def _get_normalized_dataflow(self, code: str) -> List[Tuple[str, str, List[str]]]:
        """
        获取标准化后的数据流图（带缓存）

        Args:
            code: 源代码字符串

        Returns:
            标准化后的数据流列表
        """
        try:
            cleaned_code = self._remove_comments(code)
            raw_dfg = self._extract_dataflow(cleaned_code)
            return self._normalize_dataflow(raw_dfg)
        except Exception as e:
            logger.error(f"Error processing code: {e}")
            return []

    def _remove_comments(self, code: str) -> str:
        """
        移除代码中的注释

        Args:
            code: 包含注释的代码

        Returns:
            清理后的代码
        """
        try:
            # 这里调用你的 remove_c_cpp_comments 函数
            # 由于没有提供实现，暂时返回原代码
            code = remove_c_cpp_comments(code)
            return code
        except Exception as e:
            logger.warning(f"Failed to remove comments: {e}")
            return code

    def _extract_dataflow(self, code: str) -> List[Tuple]:
        """
        从代码中提取原始数据流图

        Args:
            code: 源代码

        Returns:
            原始数据流图
        """
        try:
            tree = self.parser.parse(bytes(code, 'utf8'))
            root_node = tree.root_node

            # 提取令牌索引和代码令牌
            tokens_index = self._tree_to_token_index(root_node)
            code_lines = code.split('\n')
            code_tokens = [self._index_to_code_token(x, code_lines) for x in tokens_index]

            # 创建索引到代码的映射
            index_to_code = {}
            for idx, (index, token) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, token)

            # 提取数据流图
            try:
                dfg, _ = self._extract_dataflow_from_node(root_node, index_to_code, {})
                dfg.sort(key=lambda x: x[1])
            except:
                logger.info("There is wrong when extracting the DFG!")

            # 过滤有效的数据流节点
            valid_indexes = self._get_valid_dataflow_indexes(dfg)
            filtered_dfg = [d for d in dfg if d[1] in valid_indexes]

            # 合并重复节点
            merged_dfg = self._merge_dataflow_nodes(filtered_dfg)

            return merged_dfg

        except Exception as e:
            logger.error(f"Error extracting dataflow: {e}")
            return []

    def _extract_dataflow_from_node(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """
        从 AST 节点递归提取数据流

        Args:
            node: AST 节点
            index_to_code: 索引到代码的映射
            states: 变量状态

        Returns:
            (数据流图, 更新后的状态)
        """
        states = states.copy()

        # 处理叶子节点
        if self._is_leaf_node(node):
            return self._handle_leaf_node(node, index_to_code, states)

        # 根据节点类型分派处理
        node_type = node.type
        if node_type in self.DECLARATION_STATEMENTS:
            return self._handle_declaration(node, index_to_code, states)
        elif node_type in self.ASSIGNMENT_STATEMENTS:
            return self._handle_assignment(node, index_to_code, states)
        elif node_type in self.INCREMENT_STATEMENTS:
            return self._handle_increment(node, index_to_code, states)
        elif any(node_type in types for types in self.CONTROL_FLOW_STATEMENTS.values()):
            return self._handle_control_flow(node, node_type, index_to_code, states)
        else:
            return self._handle_generic_node(node, index_to_code, states)

    def _is_leaf_node(self, node: Node) -> bool:
        """检查是否为叶子节点"""
        return (len(node.children) == 0 or node.type == 'string') and node.type != 'comment'

    def _handle_leaf_node(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理叶子节点"""
        idx, code = index_to_code.get((node.start_point, node.end_point), (None, None))

        if idx is None or code is None:
            return [], states

        if node.type == code:  # 字面量
            return [], states
        elif code in states:  # 已定义的变量
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:  # 新变量
            if node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states

    def _handle_declaration(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理变量声明"""
        if len(node.children) == 2:
            name_node, value_node = node.children[0], node.children[1]
        else:
            name_node, value_node = node.children[0], None

        dfg = []

        if value_node is None:  # 无初始值的声明
            name_indexes = self._tree_to_variable_index(name_node, index_to_code)
            for index in name_indexes:
                idx, code = index_to_code[index]
                dfg.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
        else:  # 带初始值的声明
            value_dfg, states = self._extract_dataflow_from_node(value_node, index_to_code, states)
            dfg.extend(value_dfg)

            name_indexes = self._tree_to_variable_index(name_node, index_to_code)
            value_indexes = self._tree_to_variable_index(value_node, index_to_code)

            for name_index in name_indexes:
                idx1, code1 = index_to_code[name_index]
                for value_index in value_indexes:
                    idx2, code2 = index_to_code[value_index]
                    dfg.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                states[code1] = [idx1]

        return sorted(dfg, key=lambda x: x[1]), states

    def _handle_assignment(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理赋值语句"""
        left_node = node.child_by_field_name('left')
        right_node = node.child_by_field_name('right')

        dfg = []
        right_dfg, states = self._extract_dataflow_from_node(right_node, index_to_code, states)
        dfg.extend(right_dfg)

        left_indexes = self._tree_to_variable_index(left_node, index_to_code)
        right_indexes = self._tree_to_variable_index(right_node, index_to_code)

        for left_index in left_indexes:
            idx1, code1 = index_to_code[left_index]
            for right_index in right_indexes:
                idx2, code2 = index_to_code[right_index]
                dfg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]

        return sorted(dfg, key=lambda x: x[1]), states

    def _handle_increment(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理前缀和后缀自增自减语句"""
        dfg = []

        # 判断是前缀还是后缀操作
        is_prefix = True
        operator = None
        variable_node = None

        for i, child in enumerate(node.children):
            if child.type in ['++', '--']:
                operator = child.type
                # 如果操作符在变量前面，是前缀操作
                is_prefix = i == 0
            elif child.type == 'identifier':
                variable_node = child

        if variable_node is None or operator is None:
            logger.warning(f"Invalid increment/decrement statement: {node.text}")
            return dfg, states

        # 获取变量信息
        var_indexes = self._tree_to_variable_index(variable_node, index_to_code)

        for var_index in var_indexes:
            idx, var_name = index_to_code[var_index]

            if is_prefix:
                # 前缀操作 (++i, --i): 先计算后使用
                relationship = 'computedFrom'
            else:
                # 后缀操作 (i++, i--): 先使用后计算
                relationship = 'computedFrom'  # 但使用场景不同

            # 变量从自身的前一个状态计算得到新值
            dfg.append((var_name, idx, relationship, [var_name], [idx]))

            # 更新状态
            if var_name in states:
                # 如果是前缀操作，需要记录新的状态
                states[var_name].append(idx)
            else:
                states[var_name] = [idx]

        return sorted(dfg, key=lambda x: x[1]), states

    def _handle_control_flow(self, node: Node, node_type: str, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理控制流语句"""
        if node_type in self.CONTROL_FLOW_STATEMENTS['if']:
            return self._handle_if_statement(node, index_to_code, states)
        elif node_type in self.CONTROL_FLOW_STATEMENTS['for']:
            return self._handle_for_statement(node, index_to_code, states)
        elif node_type in self.CONTROL_FLOW_STATEMENTS['foreach']:
            return self._handle_foreach_statement(node, index_to_code, states)
        elif node_type in self.CONTROL_FLOW_STATEMENTS['while']:
            return self._handle_while_statement(node, index_to_code, states)
        else:
            return self._handle_generic_node(node, index_to_code, states)

    def _handle_if_statement(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理 if 语句（与原代码逻辑完全一致）"""
        dfg = []
        current_states = states.copy()
        other_states = []
        found_control_flow = False  # 对应原代码的flag
        has_else = False  # 对应原代码的tag

        # 检查是否有else
        if 'else' in node.type:
            has_else = True

        for child in node.children:
            # 检查else
            if 'else' in child.type:
                has_else = True

            # 完全复制原代码逻辑
            if child.type not in self.CONTROL_FLOW_STATEMENTS['if'] and not found_control_flow:
                # 第一个非控制流子节点（条件部分）
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)
            else:
                found_control_flow = True
                child_dfg, new_states = self._extract_dataflow_from_node(child, index_to_code, states)
                dfg.extend(child_dfg)
                other_states.append(new_states)

        # 状态合并（完全复制原代码逻辑）
        other_states.append(current_states)
        if not has_else:
            other_states.append(states)

        merged_states = self._merge_states(other_states)
        return sorted(dfg, key=lambda x: x[1]), merged_states

    def _handle_for_statement(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理 for 循环"""
        dfg = []
        current_states = states.copy()

        # 处理所有子节点
        for child in node.children:
            child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
            dfg.extend(child_dfg)

        # 处理循环体（在local_variable_declaration之后的部分）
        found_declaration = False
        for child in node.children:
            if found_declaration:
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)
            elif child.type == "local_variable_declaration":
                found_declaration = True

        # 合并重复的数据流
        merged_dfg = self._merge_duplicate_flows(dfg)
        return merged_dfg, current_states

    def _handle_foreach_statement(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理 foreach 循环"""
        dfg = []
        current_states = states.copy()

        name_node = node.child_by_field_name('left')
        value_node = node.child_by_field_name('right')
        body_node = node.child_by_field_name('body')

        # 模拟两次迭代
        for i in range(2):
            # 处理值表达式
            if value_node:
                value_dfg, current_states = self._extract_dataflow_from_node(value_node, index_to_code, current_states)
                dfg.extend(value_dfg)

            # 建立变量关联
            if name_node and value_node:
                name_indexes = self._tree_to_variable_index(name_node, index_to_code)
                value_indexes = self._tree_to_variable_index(value_node, index_to_code)

                for name_index in name_indexes:
                    idx1, code1 = index_to_code[name_index]
                    for value_index in value_indexes:
                        idx2, code2 = index_to_code[value_index]
                        dfg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                    current_states[code1] = [idx1]

            # 处理循环体
            if body_node:
                body_dfg, current_states = self._extract_dataflow_from_node(body_node, index_to_code, current_states)
                dfg.extend(body_dfg)

        # 合并重复的数据流
        merged_dfg = self._merge_duplicate_flows(dfg)
        return merged_dfg, current_states

    def _handle_while_statement(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理 while 循环"""
        dfg = []
        current_states = states.copy()

        # 模拟两次迭代
        for i in range(2):
            for child in node.children:
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)

        # 合并重复的数据流
        merged_dfg = self._merge_duplicate_flows(dfg)
        return merged_dfg, current_states

    def _handle_generic_node(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        """处理通用节点"""
        dfg = []
        current_states = states.copy()
        do_first_statements = []  # 需要优先处理的语句类型

        # 优先处理特定类型的子节点
        for child in node.children:
            if child.type in do_first_statements:
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)

        # 处理其余子节点
        for child in node.children:
            if child.type not in do_first_statements:
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)

        return sorted(dfg, key=lambda x: x[1]), current_states

    def _merge_duplicate_flows(self, dfg: List) -> List:
        """合并重复的数据流（与原代码逻辑一致）"""
        flow_dict = {}
        for flow in dfg:
            key = (flow[0], flow[1], flow[2])
            if key not in flow_dict:
                flow_dict[key] = [flow[3], flow[4]]
            else:
                # 合并父变量列表
                flow_dict[key][0] = list(set(flow_dict[key][0] + flow[3]))
                # 合并索引列表并排序
                flow_dict[key][1] = sorted(list(set(flow_dict[key][1] + flow[4])))

        # 重新构建数据流列表
        merged_dfg = [
            (key[0], key[1], key[2], values[0], values[1])
            for key, values in sorted(flow_dict.items(), key=lambda t: t[0][1])
        ]
        return sorted(merged_dfg, key=lambda x: x[1])

    def _merge_states(self, states_list: List[Dict]) -> Dict:
        """合并多个状态字典"""
        merged = {}
        for states in states_list:
            for key, values in states.items():
                if key not in merged:
                    merged[key] = values.copy()
                else:
                    merged[key].extend(values)

        # 去重并排序
        for key in merged:
            merged[key] = sorted(list(set(merged[key])))

        return merged

    def _get_valid_dataflow_indexes(self, dfg: List) -> Set:
        """获取有效的数据流索引"""
        indexs = set()
        for d in dfg:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        return indexs

    def _merge_dataflow_nodes(self, dfg: List) -> List:
        """合并数据流节点"""
        dic = {}
        for d in dfg:
            if d[1] not in dic:
                dic[d[1]] = d
            else:
                dic[d[1]] = (d[0], d[1], d[2], list(set(dic[d[1]][3] + d[3])), list(set(dic[d[1]][4] + d[4])))

        DFG = []
        for d in dic:
            DFG.append(dic[d])
        dfg = DFG
        return dfg

    def _normalize_dataflow(self, dataflow: List) -> List[Tuple[str, str, List[str]]]:
        """标准化数据流"""
        var_dict = {}
        i = 0
        normalized_dataflow = []

        for item in dataflow:
            var_name = item[0]
            relationship = item[2]
            par_vars_name_list = item[3]

            # 完全保持原始逻辑：先处理父变量，再处理当前变量
            for name in par_vars_name_list:
                if name not in var_dict:
                    var_dict[name] = 'var_' + str(i)
                    i += 1

            if var_name not in var_dict:
                var_dict[var_name] = 'var_' + str(i)
                i += 1

            normalized_dataflow.append((
                var_dict[var_name],
                relationship,
                [var_dict[x] for x in par_vars_name_list]
            ))

        return normalized_dataflow

    def _count_matching_flows(self, reference_flows: List, candidate_flows: List) -> int:
        """
        计算匹配的数据流数量

        Args:
            reference_flows: 参考数据流
            candidate_flows: 候选数据流

        Returns:
            匹配数量
        """
        matches = 0

        for flow in reference_flows:
            if flow in candidate_flows:
                matches += 1
                candidate_flows.remove(flow)

        return matches

    # 原有的工具函数（保持原样但重新组织）
    def _tree_to_token_index(self, root_node: Node) -> List[Tuple]:
        """将 AST 转换为令牌索引"""
        if self._is_leaf_node(root_node):
            return [(root_node.start_point, root_node.end_point)]
        else:
            tokens = []
            for child in root_node.children:
                tokens.extend(self._tree_to_token_index(child))
            return tokens

    def _tree_to_variable_index(self, root_node: Node, index_to_code: Dict) -> List[Tuple]:
        """提取变量索引"""
        if self._is_leaf_node(root_node):
            index = (root_node.start_point, root_node.end_point)
            _, code = index_to_code.get(index, (None, ''))
            if root_node.type != code:
                return [index]
            else:
                return []
        else:
            indexes = []
            for child in root_node.children:
                indexes.extend(self._tree_to_variable_index(child, index_to_code))
            return indexes

    def _index_to_code_token(self, index: Tuple, code_lines: List[str]) -> str:
        """将索引转换为代码令牌"""
        start, end = index[0], index[1]

        if start[0] == end[0]:
            return code_lines[start[0]][start[1]:end[1]]
        else:
            token = code_lines[start[0]][start[1]:]
            for i in range(start[0] + 1, end[0]):
                token += code_lines[i]
            token += code_lines[end[0]][:end[1]]
            return token

#
# if __name__ == '__main__':
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
#     # dfg_analyzer_llm = SemanticMatchScore(parser)
#     #
#     # syntax_analyzer_ppo = SyntaxAnalyzer(parser)
#     dfg_analyzer_ppo = DataFlowAnalyzer(parser)
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
#             # dataflow_match_score = dfg_analyzer_llm.semantic_matching_score(pre_code, gt_code)
#             # syntax_match_score = syntax_analyzer_ppo.calculate_singe_ref_can(gt_code, pre_code)
#             dataflow_match_score = dfg_analyzer_ppo.calculate_singe_ref_can(gt_code, pre_code)
#             print(f"the dataflow_match_score is {dataflow_match_score}")
#         except Exception as e:
#             logger.info(f"There is wrong when we compute syntax match and dfg match: {e}.")