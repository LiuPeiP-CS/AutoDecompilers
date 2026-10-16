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
    def parse_code(self, code: str) -> Node:
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree.root_node
        except Exception as e:
            print(f"parsing code fails: {e}")
            return None

    def print_dfg_info(self, edges, variables, name="DFG"):
        print(f"\n=== {name} information ===")
        print(f"the number of vars: {len(variables)}")
        print(f"the number of edges: {len(edges)}")

        print("\nvar list:")
        for var_id, var_info in variables.items():
            print(f"  {var_id}: {var_info.var_name} (type: {var_info.var_type}, scope: {var_info.scope_level})")

        print("\ndata flow edge:")
        for edge in edges[:10]:  
            src_var = variables.get(edge.src_id, None)
            dst_var = variables.get(edge.dst_id, None)
            src_name = src_var.var_name if src_var else f"unknown({edge.src_id})"
            dst_name = dst_var.var_name if dst_var else f"unknown({edge.dst_id})"
            print(f"  {src_name} -> {dst_name} [{edge.edge_type}]")

    def semantic_matching_score(self, code1, code2):
        
        root1, root2 = self.parse_code(code1), self.parse_code(code2)
        if not root1 or not root2:
            print("parsing code fails")
            sys.exit(1)

        edges1, vars1 = self.builder.build(root1)
        edges2, vars2 = self.builder.build(root2)

        vars1 = apply_alpha_conversion(vars1)
        vars2 = apply_alpha_conversion(vars2)

        
        if len(edges1) == 0 or len(edges2) == 0:
            print("warning: the dfg of one or two files is full")
            
        score = self.matcher.semantic_match(
            tuple(edges1), tuple(edges2),
            tuple(vars1.items()), tuple(vars2.items())
        )

        print(f"semantic_matching_score: {score:.4f}")
        return score
