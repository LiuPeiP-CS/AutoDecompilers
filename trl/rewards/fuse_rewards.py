import sys
import threading
from termcolor import colored
import os.path, subprocess
import os
import shutil
import re
import json
import tempfile
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
import time
# from func_timeout import func_timeout, FunctionTimedOut
from rewards_tools.code_prepro.c_processor import *
from rewards_tools.metrics.syntax_match import SyntaxAnalyzer, SyntaxMatchScorer
from rewards_tools.metrics.dataflow_match import DataFlowAnalyzer
from rewards_tools.parser.DFG import SemanticMatchScore
from rewards_tools.parser.utils import remove_c_cpp_comments
from rewards_tools.metrics.env_recom_reexe import EnvRecomReexe
from rewards_tools.metrics.humaneval_feedback import HumanEvalFeedback # 面向推理测试
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import tempfile

abs_path = "/workspace/trl-main/temp/"
os.makedirs(abs_path, exist_ok=True)

class OverallRewards:
    def __init__(self,
                 so_path=None, # "./rewards_tools/parser/my-languages.so"
                 use_apted=False,
                 judge_c=False,
                 re_exe_reward=True,
                 syntax_reward=False,
                 semantic_reward=False,
                 mode='train',
                 dataset='exebench'
                 ):

        self._so_path = so_path
        self.use_apted = use_apted
        self._thread_local = threading.local()
        self._get_thread_resources()

        self.judge_c = judge_c
        self.re_exe_reward = re_exe_reward
        self.syntax_reward = syntax_reward
        self.semantic_reward = semantic_reward

        self.mode = mode
        self.dataset = dataset

    def _get_thread_resources(self):
        """
        每个线程第一次调用时，创建该线程专属的 parser + 所有分析器。
        之后复用，不重复创建。
        """
        if hasattr(self._thread_local, 'ready'):
            return self._thread_local

        # 每个线程独立的 parser
        LANGUAGE = Language(tscpp.language())
        parser = Parser(LANGUAGE)

        # c_tokenizer / c_detokenizer
        processor = CProcessor(parser)
        self._thread_local.c_tokenizer = processor.tokenize_code
        self._thread_local.c_detokenizer = processor.detokenize_code

        # 分析器：每个线程各自持有独立 parser 的分析器实例
        if self.use_apted:
            self._thread_local.syntax_analyzer_llm = SyntaxMatchScorer(parser)
            self._thread_local.dfg_analyzer_llm = SemanticMatchScore(parser)
            self._thread_local.dfg_analyzer_ppo = DataFlowAnalyzer(parser)
        else:
            self._thread_local.syntax_analyzer_ppo = SyntaxAnalyzer(parser)
            self._thread_local.dfg_analyzer_ppo = DataFlowAnalyzer(parser)

        self._thread_local.ready = True
        return self._thread_local

    def get_code_rewards(self, dependency, response, ground_truth, train_id, cancel_event=None):
        # print(f"**************************this is the predicted pscode ***************************\n{response}")
        # print("*******************************************************************************************")

        tl = self._get_thread_resources()
        c_detokenizer_func = tl.c_detokenizer
        if self.use_apted:
            tl_syntax_analyzer_llm = tl.syntax_analyzer_llm
            tl_dfg_analyzer_ppo = tl.dfg_analyzer_ppo
        else:
            tl_syntax_analyzer_ppo = tl.syntax_analyzer_ppo
            tl_dfg_analyzer_ppo = tl.dfg_analyzer_ppo

        try:
            response = remove_c_cpp_comments(response)
            ground_truth = remove_c_cpp_comments(ground_truth)
        except Exception as e:
            logger.error(f"\nFailed to remove comments: {e}\n")

        def _do_heavy_rewards():
            reward = []
            feedback = ""

            # print(f"-------------------------------this is the response {response}-------------------------------")

            # pre_code = tl.c_detokenizer(response)
            pre_code = c_detokenizer_func(response)
            pre_code = self.remove_special_tokens(pre_code) # 去掉代码中的多余字符


            gt_code = self.remove_special_tokens(ground_truth)

            # 如果生成的代码异常长（比如是真实代码的两倍以上），直接返回最低惩罚分
            if len(pre_code) > len(gt_code) * 2.5:
                return [-1, -1, 0, 0], "Code is excessively long, indicating repetitive hallucination."

            if self.judge_c:
                judge_c_reward = self.is_c_code(pre_code) # 判断是否是c语言代码
                if judge_c_reward == -1:
                    reward.append(-1)
                    if self.re_exe_reward:
                        reward.append(-1)
                    if self.syntax_reward:
                        reward.append(0)
                    if self.semantic_reward:
                        reward.append(0)

                    feedback = "The generated code is not a C language program. Please regenerate the code."
                    return reward, feedback
                else:
                    reward.append(judge_c_reward)

            try:
                if self.re_exe_reward:
                    if self.dataset == 'humaneval' and self.mode == 'eval':
                        Re_exe = HumanEvalFeedback
                    else:
                        Re_exe = EnvRecomReexe
                    # 加tmp_path
                    # re_exe_begin = time.time()
                    with Re_exe(decom=response, dependencies=dependency, tmp_path=abs_path + str(train_id), cancel_event=cancel_event) as recom_reexe:
                        recom_reexe_feedback_dict= recom_reexe.reward()
                        recom_reexe_score, recom_reexe_feedback = recom_reexe_feedback_dict.get('score', 0), recom_reexe_feedback_dict.get('feedback', {})
                        # print(recom_reexe.log)

                    # print(f"the re-exe time cost is {time.time() - re_exe_begin}")

                    if recom_reexe_score == -1:
                        reward.append(recom_reexe_score)
                        recom_reexe_feedback_str = json.dumps(recom_reexe_feedback)
                        feedback = f"The generated code contains compilation errors, and the specific error messages are: {recom_reexe_feedback_str}. Please analyze the errors and regenerate the code."

                    elif recom_reexe_score == 1:
                        reward.append(recom_reexe_score)
                        feedback = "All right"

                    # elif recom_reexe_score == -6:
                    #     reward.append(-0.6)
                    #     # parser the dict
                    #     new_feedback_dict = {}
                    #     for key, value in recom_reexe_feedback.items():
                    #         if value.get('failure') == 'Runtime error':
                    #             new_feedback_dict[key] = value
                    #     feedback = f"The generated code contains runtime errors, and the specific error messages are: {json.dumps(new_feedback_dict)}. Please analyze the errors and regenerate the code."
                    elif recom_reexe_score == -6:
                        reward.append(-0.6)
                        new_feedback_dict = {}
                        # 兼容判断：如果底层 value 不是字典，说明传入的是单层扁平字典 (如 humaneval 的 Runtime error)
                        if not any(isinstance(v, dict) for v in recom_reexe_feedback.values()):
                            new_feedback_dict = recom_reexe_feedback
                        else:
                            # 否则认为是嵌套字典，筛选出包含 Runtime error 的子任务
                            for key, value in recom_reexe_feedback.items():
                                if isinstance(value, dict) and value.get('failure') == 'Runtime error':
                                    new_feedback_dict[key] = value

                        feedback = f"The generated code contains runtime errors, and the specific error messages are: {json.dumps(new_feedback_dict)}. Please analyze the errors and regenerate the code."

                    # elif recom_reexe_score == -3:
                    #     reward.append(-0.3)
                    #     new_feedback_dict = {}
                    #     for key, value in recom_reexe_feedback.items():
                    #         if value.get('failure') == 'Incorrect output':
                    #             new_feedback_dict[key] = value
                    #     feedback = f"There are incorrect outputs when performing unit testing on the generated code. Specifically, it is: {json.dumps(new_feedback_dict)}. Please analyze the errors and regenerate the code."
                    elif recom_reexe_score == -3:
                        reward.append(-0.3)
                        new_feedback_dict = {}

                        # 加入一层判定，防止其它执行器传回非嵌套字典
                        if not any(isinstance(v, dict) for v in recom_reexe_feedback.values()):
                            # 如果是扁平字典，直接判断外层
                            if recom_reexe_feedback.get('failure') == 'Incorrect output':
                                new_feedback_dict = recom_reexe_feedback
                        else:
                            # 嵌套字典逻辑（原逻辑加上 isinstance 保护）
                            for key, value in recom_reexe_feedback.items():
                                # 💡 关键修改：先确保 value 是 dict，再调用 .get()
                                if isinstance(value, dict) and value.get('failure') == 'Incorrect output':
                                    new_feedback_dict[key] = value

                        feedback = f"There are incorrect outputs when performing unit testing on the generated code. Specifically, it is: {json.dumps(new_feedback_dict)}. Please analyze the errors and regenerate the code."

                    else:
                        reward.append(0)
                        feedback = f"There are all incorrect outputs when performing unit testing on the generated code. Please analyze the errors and regenerate the code."
                        # logger.info("There is the wrong reward when re_exe_reward is True!")
                        # sys.exit()
            except Exception as e:
                reward.append(-0.5)
                feedback = f"In the code execution and correction process, an exception occurred. Please regenerate the code."
                logger.info(f"+++++++++++++++++++++++++++++++++++++++++++++++++There is an exception when re_exe_reward is True: {e}!++++++++++++++++++++++++++++++++++++++++++")

            if self.mode == "train":
                # calculate syntax match and dataflow match
                try:
                    if self.use_apted:
                        if self.syntax_reward:
                            try:
                                syntax_match_score = tl_syntax_analyzer_llm.calculate_syntax_score(pre_code, gt_code)
                                reward.append(syntax_match_score)
                            except:
                                reward.append(0)
                        if self.semantic_reward:
                            # dataflow_match_score = self.dfg_analyzer_llm.semantic_matching_score(pre_code, gt_code)
                            try:
                                dataflow_match_score = tl_dfg_analyzer_ppo.calculate_singe_ref_can(gt_code, pre_code)
                                reward.append(dataflow_match_score)
                            except:
                                reward.append(0)
                    else:
                        if self.syntax_reward:
                            try:
                                syntax_match_score = tl_syntax_analyzer_ppo.calculate_singe_ref_can(gt_code, pre_code)
                                reward.append(syntax_match_score)
                            except:
                                reward.append(0)
                        if self.semantic_reward:
                             try:
                                dataflow_match_score = tl_dfg_analyzer_ppo.calculate_singe_ref_can(gt_code, pre_code)
                                reward.append(dataflow_match_score)
                             except:
                                reward.append(0)
                except Exception as e:
                    feedback = 'When we compare the syntax and semantic between the decompiled pscode and the source code, it is an error. Please regenerate the code.'
                    logger.info(f"There is wrong when we compute syntax match and dfg match: {e}.")

                assert len(reward) == 4, f"The size of the reward must be 4, but now is only {reward}"
            return reward, feedback

        try:
            # 设置硬超时为 15 秒 (可以根据你的代码平均执行时间进行调整)
            return _do_heavy_rewards()
        except Exception as e:
            logger.error(f"[get_code_rewards] 🚨 Unexpected internal error: {e}")
            return [0, 0, 0, 0], "An unexpected internal error occurred during reward calculation. Please regenerate."


    def remove_newline(self, code_string):
        return re.sub('NEW_LINE\s?', '\n', code_string)

    def remove_special_tokens(self, code_string):
        lines = code_string.split("NEW_LINE")
        lines = [item.strip() for item in lines]

        curr_indent = 0
        new_lines = []
        for line in lines:
            indent_count = line.count('INDENT')
            dedent_count = line.count('DEDENT')
            curr_indent += indent_count - dedent_count
            wo_indent = re.sub('INDENT\s?', '', line)
            wo_dedent = re.sub('DEDENT\s?', '', wo_indent)
            new_lines.append('\t' * curr_indent + wo_dedent)
        return ("\n").join(new_lines)

    def dfs_parse_tree(self, node, level, count_list, verbose=False):
        if verbose:
            if node.type == 'ERROR':
                print(level, '-' * (level * 2), colored(node.type, 'red'))
            else:
                print(level, '-' * (level * 2), node.type)
        if node.type == 'ERROR':
            count_list[0] += 1
        else:
            count_list[1] += 1
        for child in node.children:
            self.dfs_parse_tree(child, level + 1, count_list, verbose)
        return

    def tree_sitter_full_compile(self, code, lang='c', verbose=False):
        root = self.parser.parse(bytes(code, 'utf-8')).root_node
        count_list = [0, 0]
        self.dfs_parse_tree(root, 0, count_list, verbose)
        return count_list  # 用tree-sitter做完整语法解析，返回 [错误节点数, 正常节点数]。

    def compile_code_string(self,code_string, print_error=True):
        code_string = self.remove_newline(code_string)

        # ************************************** 此处的函数调用缺输入输出 ************************************* #
        error, op_output = self.compile_prog(code_string)

        if print_error:
            if error == "CompError" or error == "RuntimeError":
                print(op_output)

        if error == "CompError" or error == "RuntimeError":
            return error, op_output, False
        elif error == "AllRight" or error == "PartRight":
            return error, op_output, True
        else:
            print("************** There is the wrong return from compile_prog **************")

    def _template_match(self, got: str, template: str) -> bool:
        """把模板里的 {{regex}} 替换成真正的正则，全文匹配"""
        # 先把普通字符转义，再恢复占位符
        template = re.escape(template)
        template = template.replace(r'\{\{', '{{').replace(r'\}\}', '}}')
        # 把 {{regex}} 换回去
        pattern = re.sub(r'\{\{(.+?)\}\}', r'(?P<_g\g<0>>\1)', template)
        return re.fullmatch(pattern, got) is not None

    def compile_prog(self, source: str,
                     test_inputs: list[str] = None,  # 测试时的输入数据
                     ground_outputs: list[str] = None,  # 测试时的输出数据
                     timeout: float = 2):
        """
        返回 (结论, 信息)
        结论 ∈ {"CompError", "RuntimeError", "AllRight", "PartRight"}
        """
        if len(test_inputs) != len(ground_outputs):
            raise ValueError("输入与答案数量不一致")

        # ---- 编译阶段 ----
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "main.c")
            exe_file = os.path.join(tmpdir, "main")
            with open(src_file, 'w', encoding='utf-8') as f:  # 将反编译的代码写入文件中
                f.write(source)

            comp = subprocess.run(
                ["gcc", "-std=c11", src_file, "-o", exe_file],
                capture_output=True, text=True
            )
            if comp.returncode != 0:
                return "CompError", comp.stderr.strip()
            # 或者使用下面的代码？
            # proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
            # error = [i.decode('utf-8') for i in proc.stderr.readlines()]
            # err = '\n'.join(error)

            # ---- 运行 + 比对阶段 ----
            total = len(test_inputs)
            right = 0
            run_err_info = None

            for tin, tg in zip(test_inputs, ground_outputs):
                try:
                    run = subprocess.run(
                        [exe_file],
                        input=tin,
                        text=True,
                        capture_output=True,
                        timeout=timeout
                    )
                    if run.returncode != 0:
                        # 运行时错误
                        run_err_info = run.stderr.strip() or f"non-zero exit {run.returncode}"
                        return "RuntimeError", run_err_info

                    # 输出比对（精确匹配，含换行）
                    # if run.stdout.strip() == tg.strip():
                    if self._template_match(run.stdout.strip(), tg.strip()):
                        right += 1
                except subprocess.TimeoutExpired:
                    return "RuntimeError", "Time limit exceeded"

            # ---- 结果统计 ----
            if right == total:
                return "AllRight", ""
            elif right == 0:
                return "PartRight", ""
            else:
                return "PartRight", ""


    def is_c_code(self, text: str) -> int:
        """
        判断一段文本是否为 C 语言代码，并进行打分（是 C 语言代码返回 1，否则返回 0）
        """
        text = text.strip()
        if not text:
            return 0

        score = 0.0

        # 1. 惩罚项（其他语言特征）
        # Python
        if re.search(r'\bdef\s+\w+\s*\([^)]*\)\s*:', text) or \
                re.search(r'\b(?:class|def)\s+\w+\s*:', text):
            score -= 10

        # Java/C#（排除类定义，但允许结构体）
        if re.search(r'\b(?:public|private|protected)\s+(?:class|interface)\s+\w+', text):
            score -= 10

        # JavaScript
        if re.search(r'\b(function|var|let|const|=>|console\.log)\b', text):
            score -= 10

        # Shell/Bash
        if re.search(r'^#!', text, re.MULTILINE):
            score -= 8

        # 2. 强力C特征（高权重）
        # 预处理器指令
        preproc_patterns = [
            r'#\s*include\s*[<"].*\.h[>"]',
            r'#\s*define\s+\w+',
            r'#\s*(ifdef|ifndef|endif|if|else|elif|pragma)\b'
        ]
        for pattern in preproc_patterns:
            if re.search(pattern, text):
                score += 4.0
                break

        # main函数
        main_patterns = [
            r'\bint\s+main\s*\(\s*(?:void)?\s*\)',
            r'\bint\s+main\s*\(\s*int\s+\w+\s*,\s*(?:char\s*\*\s*\w+|char\s+\*\s*\*\s*\w+)\s*\)',
            r'\bvoid\s+main\s*\(\s*(?:void)?\s*\)',
        ]
        if any(re.search(pattern, text) for pattern in main_patterns):
            score += 5.0

        # 3. 中等权重特征
        # C类型系统
        c_types = ['int', 'char', 'float', 'double', 'short', 'long',
                   'unsigned', 'signed', 'void', 'auto', 'register']
        type_matches = sum(1 for t in c_types if re.search(rf'\b{t}\b', text))
        score += type_matches * 0.4

        # 结构体/联合体
        if re.search(r'\bstruct\s+\w+\s*\{', text) or \
                re.search(r'\bunion\s+\w+\s*\{', text):
            score += 1.5

        # 枚举
        if re.search(r'\benum\s+\w+\s*\{', text):
            score += 1.0

        # 4. 控制流和关键字
        control_flow = ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default']
        control_count = sum(1 for kw in control_flow if re.search(rf'\b{kw}\b(?!\s*:)', text))
        score += control_count * 0.3

        other_c_keywords = ['break', 'continue', 'goto', 'return', 'sizeof',
                            'typedef', 'extern', 'static', 'const', 'volatile']
        keyword_count = sum(1 for kw in other_c_keywords if re.search(rf'\b{kw}\b', text))
        score += keyword_count * 0.2

        # 5. C语言特定库函数
        c_lib_funcs = ['printf', 'scanf', 'fopen', 'fclose', 'fprintf', 'fscanf',
                       'malloc', 'calloc', 'realloc', 'free', 'sizeof']
        func_count = sum(1 for func in c_lib_funcs if re.search(rf'\b{func}\s*\(', text))
        score += func_count * 0.3

        # 6. 语法结构特征
        # 指针操作
        if re.search(r'\*\s*\w+|\&\s*\w+', text):
            score += 0.5

        # 类型转换
        if re.search(r'\(\s*(int|char|float|double|void)\s*\*\s*\)', text):
            score += 0.4

        # 花括号匹配
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > 0 and close_braces > 0 and open_braces == close_braces:
            score += 0.8

        # 分号
        semicolon_count = text.count(';')
        if semicolon_count > 0:
            score += min(0.6, semicolon_count * 0.08)

        # 7. 注释
        # 排除URL中的//
        lines = text.split('\n')
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//') and not stripped.startswith('http'):
                comment_lines += 1
            elif '/*' in line and '*/' in line:
                comment_lines += 1

        if comment_lines > 0:
            score += 0.7

        # 8. 函数定义检测
        # 改进的函数定义模式
        func_pattern = rf'\b({"|".join(c_types)})\s+\w+\s*\([^)]*\)\s*(?:;|{{)'
        if re.search(func_pattern, text, re.IGNORECASE):
            score += 1.2

        # 9. 动态阈值
        lines = text.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        if len(non_empty_lines) <= 3:
            # 短代码片段
            threshold = 0.8
        elif len(non_empty_lines) <= 10:
            threshold = 1.0
        else:
            threshold = 1.2

        # 最终决策
        return 1 if score >= threshold else -1