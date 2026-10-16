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

    def __init__(self, parser):
        self.parser = parser

    def calculate_singe_ref_can(self, reference: str, candidate: str) -> float:
        return self.calculate_single_syntax_match([reference], candidate)

    def calculate_single_syntax_match(self, references: List[str], candidate: str) -> float:
        return self.calculate_syntax_similarity([references], [candidate])

    def calculate_syntax_similarity(self, references: List[List[str]], candidates: List[str]) -> float:
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

class SyntaxMatchScorer:
    def __init__(self, parser):
        self.parser = parser
        self._LITERAL_RE = re.compile(
            r'^(integer_literal|float_literal|char_literal|string_literal|true|false|nullptr)$')
        self._PRUNE_SET = {'comment', 'preproc_include', 'preproc_def'}

    def normalize_ast_to_apted_tree(self, node, source_bytes=None):
        if node is None:
            return None

        node_type = getattr(node, "type", str(node))
        if node_type in self._PRUNE_SET:
            return None

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
        if tree is None:
            return

        print("  " * depth + f"Node: {tree.name} (type: {type(tree)})")

        children = getattr(tree, 'children', [])
        for child in children:
            if isinstance(child, Tree):
                self.debug_tree(child, depth + 1)
            else:
                print("  " * (depth + 1) + f"WARNING: Non-Tree child: {type(child)} - {child}")

    def count_nodes(self, tree):
        if tree is None:
            return 0
        return 1 + sum(self.count_nodes(child) for child in tree.children)

    def calculate_syntax_score(self, generated_code, reference_code):
        try:
            gen_bytes = generated_code.encode() if isinstance(generated_code, str) else generated_code
            ref_bytes = reference_code.encode() if isinstance(reference_code, str) else reference_code
            tree_gen = self.parser.parse(gen_bytes)
            tree_ref = self.parser.parse(ref_bytes)

            apted_tree_gen = self.normalize_ast_to_apted_tree(tree_gen.root_node, gen_bytes)
            apted_tree_ref = self.normalize_ast_to_apted_tree(tree_ref.root_node, ref_bytes)


            if apted_tree_gen is None or apted_tree_ref is None:
                return 0.0

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

            config = ValueAwareConfig()
            apted = APTED(apted_tree_gen, apted_tree_ref, config)
            edit_distance = apted.compute_edit_distance()

            if isinstance(edit_distance, str):
                try:
                    edit_distance = float(edit_distance)
                except ValueError:
                    print(f"Cannot convert {edit_distance} to float")
                    return 0.0

            size_gen = self.count_nodes(apted_tree_gen)
            size_ref = self.count_nodes(apted_tree_ref)

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
    valuecls = float

    def rename(self, node1, node2):
        if node1 is None or node2 is None:
            return 1.0

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

        if type1 != type2:
            return 1.0

        if type1 in ['identifier', 'type_identifier', 'function_identifier']:
            return 0.0

        if type1 in {'integer_literal', 'float_literal'}:
            try:
                f1, f2 = float(val1), float(val2)
                return 0.0 if abs(f1 - f2) < 1e-10 else 0.8
            except ValueError:
                return 0.0 if val1 == val2 else 0.8

        if type1 in {'char_literal', 'string_literal', 'true', 'false', 'nullptr'}:
            return 0.0 if val1 == val2 else 0.8

        return 0.0

    def delete(self, node):
        return 1.0

    def insert(self, node):
        return 1.0
