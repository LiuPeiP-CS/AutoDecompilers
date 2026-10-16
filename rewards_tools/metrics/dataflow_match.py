import logging
from functools import lru_cache
from typing import List, Tuple, Dict, Set, Any, Optional, Union
from tree_sitter import Language, Parser, Node
from rewards_tools.parser.utils import remove_c_cpp_comments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFlowAnalyzer:
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
        return self.calculate_dataflow_similarity([references], [candidate])

    def _get_normalized_dataflow(self, code: str) -> List[Tuple[str, str, List[str]]]:
        try:
            cleaned_code = self._remove_comments(code)
            raw_dfg = self._extract_dataflow(cleaned_code)
            return self._normalize_dataflow(raw_dfg)
        except Exception as e:
            logger.error(f"Error processing code: {e}")
            return []

    def _remove_comments(self, code: str) -> str:
        try:
            code = remove_c_cpp_comments(code)
            return code
        except Exception as e:
            logger.warning(f"Failed to remove comments: {e}")
            return code

    def _extract_dataflow(self, code: str) -> List[Tuple]:
        try:
            tree = self.parser.parse(bytes(code, 'utf8'))
            root_node = tree.root_node

            tokens_index = self._tree_to_token_index(root_node)
            code_lines = code.split('\n')
            code_tokens = [self._index_to_code_token(x, code_lines) for x in tokens_index]

            index_to_code = {}
            for idx, (index, token) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, token)

            try:
                dfg, _ = self._extract_dataflow_from_node(root_node, index_to_code, {})
                dfg.sort(key=lambda x: x[1])
            except:
                logger.info("There is wrong when extracting the DFG!")

            valid_indexes = self._get_valid_dataflow_indexes(dfg)
            filtered_dfg = [d for d in dfg if d[1] in valid_indexes]

            merged_dfg = self._merge_dataflow_nodes(filtered_dfg)

            return merged_dfg

        except Exception as e:
            logger.error(f"Error extracting dataflow: {e}")
            return []

    def _extract_dataflow_from_node(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        states = states.copy()

        if self._is_leaf_node(node):
            return self._handle_leaf_node(node, index_to_code, states)

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
        return (len(node.children) == 0 or node.type == 'string') and node.type != 'comment'

    def _handle_leaf_node(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        idx, code = index_to_code.get((node.start_point, node.end_point), (None, None))

        if idx is None or code is None:
            return [], states

        if node.type == code:  
            return [], states
        elif code in states:  
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:  
            if node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states

    def _handle_declaration(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        if len(node.children) == 2:
            name_node, value_node = node.children[0], node.children[1]
        else:
            name_node, value_node = node.children[0], None

        dfg = []

        if value_node is None:  
            name_indexes = self._tree_to_variable_index(name_node, index_to_code)
            for index in name_indexes:
                idx, code = index_to_code[index]
                dfg.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
        else:  
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
        
        dfg = []

        
        is_prefix = True
        operator = None
        variable_node = None

        for i, child in enumerate(node.children):
            if child.type in ['++', '--']:
                operator = child.type
                
                is_prefix = i == 0
            elif child.type == 'identifier':
                variable_node = child

        if variable_node is None or operator is None:
            logger.warning(f"Invalid increment/decrement statement: {node.text}")
            return dfg, states

        
        var_indexes = self._tree_to_variable_index(variable_node, index_to_code)

        for var_index in var_indexes:
            idx, var_name = index_to_code[var_index]

            if is_prefix:
                
                relationship = 'computedFrom'
            else:
                
                relationship = 'computedFrom'  

            
            dfg.append((var_name, idx, relationship, [var_name], [idx]))

            
            if var_name in states:
                
                states[var_name].append(idx)
            else:
                states[var_name] = [idx]

        return sorted(dfg, key=lambda x: x[1]), states

    def _handle_control_flow(self, node: Node, node_type: str, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        
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
        
        dfg = []
        current_states = states.copy()
        other_states = []
        found_control_flow = False  
        has_else = False  

        if 'else' in node.type:
            has_else = True

        for child in node.children:
            if 'else' in child.type:
                has_else = True

            if child.type not in self.CONTROL_FLOW_STATEMENTS['if'] and not found_control_flow:
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)
            else:
                found_control_flow = True
                child_dfg, new_states = self._extract_dataflow_from_node(child, index_to_code, states)
                dfg.extend(child_dfg)
                other_states.append(new_states)

        other_states.append(current_states)
        if not has_else:
            other_states.append(states)

        merged_states = self._merge_states(other_states)
        return sorted(dfg, key=lambda x: x[1]), merged_states

    def _handle_for_statement(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        dfg = []
        current_states = states.copy()

        for child in node.children:
            child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
            dfg.extend(child_dfg)

        found_declaration = False
        for child in node.children:
            if found_declaration:
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)
            elif child.type == "local_variable_declaration":
                found_declaration = True

        merged_dfg = self._merge_duplicate_flows(dfg)
        return merged_dfg, current_states

    def _handle_foreach_statement(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        dfg = []
        current_states = states.copy()

        name_node = node.child_by_field_name('left')
        value_node = node.child_by_field_name('right')
        body_node = node.child_by_field_name('body')

        for i in range(2):
            if value_node:
                value_dfg, current_states = self._extract_dataflow_from_node(value_node, index_to_code, current_states)
                dfg.extend(value_dfg)

            if name_node and value_node:
                name_indexes = self._tree_to_variable_index(name_node, index_to_code)
                value_indexes = self._tree_to_variable_index(value_node, index_to_code)

                for name_index in name_indexes:
                    idx1, code1 = index_to_code[name_index]
                    for value_index in value_indexes:
                        idx2, code2 = index_to_code[value_index]
                        dfg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                    current_states[code1] = [idx1]

            if body_node:
                body_dfg, current_states = self._extract_dataflow_from_node(body_node, index_to_code, current_states)
                dfg.extend(body_dfg)

        merged_dfg = self._merge_duplicate_flows(dfg)
        return merged_dfg, current_states

    def _handle_while_statement(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        dfg = []
        current_states = states.copy()

        for i in range(2):
            for child in node.children:
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)

        merged_dfg = self._merge_duplicate_flows(dfg)
        return merged_dfg, current_states

    def _handle_generic_node(self, node: Node, index_to_code: Dict, states: Dict) -> Tuple[List, Dict]:
        dfg = []
        current_states = states.copy()
        do_first_statements = []  

        
        for child in node.children:
            if child.type in do_first_statements:
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)

        
        for child in node.children:
            if child.type not in do_first_statements:
                child_dfg, current_states = self._extract_dataflow_from_node(child, index_to_code, current_states)
                dfg.extend(child_dfg)

        return sorted(dfg, key=lambda x: x[1]), current_states

    def _merge_duplicate_flows(self, dfg: List) -> List:
        
        flow_dict = {}
        for flow in dfg:
            key = (flow[0], flow[1], flow[2])
            if key not in flow_dict:
                flow_dict[key] = [flow[3], flow[4]]
            else:
                
                flow_dict[key][0] = list(set(flow_dict[key][0] + flow[3]))
                
                flow_dict[key][1] = sorted(list(set(flow_dict[key][1] + flow[4])))

        
        merged_dfg = [
            (key[0], key[1], key[2], values[0], values[1])
            for key, values in sorted(flow_dict.items(), key=lambda t: t[0][1])
        ]
        return sorted(merged_dfg, key=lambda x: x[1])

    def _merge_states(self, states_list: List[Dict]) -> Dict:
        
        merged = {}
        for states in states_list:
            for key, values in states.items():
                if key not in merged:
                    merged[key] = values.copy()
                else:
                    merged[key].extend(values)

        
        for key in merged:
            merged[key] = sorted(list(set(merged[key])))

        return merged

    def _get_valid_dataflow_indexes(self, dfg: List) -> Set:
        
        indexs = set()
        for d in dfg:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        return indexs

    def _merge_dataflow_nodes(self, dfg: List) -> List:
        
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
        
        var_dict = {}
        i = 0
        normalized_dataflow = []

        for item in dataflow:
            var_name = item[0]
            relationship = item[2]
            par_vars_name_list = item[3]

            
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
        
        matches = 0

        for flow in reference_flows:
            if flow in candidate_flows:
                matches += 1
                candidate_flows.remove(flow)

        return matches

    
    def _tree_to_token_index(self, root_node: Node) -> List[Tuple]:
        
        if self._is_leaf_node(root_node):
            return [(root_node.start_point, root_node.end_point)]
        else:
            tokens = []
            for child in root_node.children:
                tokens.extend(self._tree_to_token_index(child))
            return tokens

    def _tree_to_variable_index(self, root_node: Node, index_to_code: Dict) -> List[Tuple]:
        
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
        
        start, end = index[0], index[1]

        if start[0] == end[0]:
            return code_lines[start[0]][start[1]:end[1]]
        else:
            token = code_lines[start[0]][start[1]:]
            for i in range(start[0] + 1, end[0]):
                token += code_lines[i]
            token += code_lines[end[0]][:end[1]]
            return token
