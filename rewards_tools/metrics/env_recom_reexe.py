import os, shutil, subprocess, signal, json, traceback, threading, re, math
from typing import Optional, Tuple
from pathlib import Path
from string import Template
# from ast import literal_eval
import tree_sitter_cpp as tsc
from tree_sitter import Language, Parser
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from contextlib import contextmanager
import uuid
import time
import threading
_thread_local = threading.local()
# from openai import OpenAI, APIStatusError

abs_path = "/workspace/trl-main/rewards_tools/"

# path
# _TMP_PATH = "/workspace/temp/"
_LIB_PATH = abs_path + "metrics/exebench"
prompt_file = abs_path + "metrics/handle_prompt.txt"

# if not os.path.exists(_TMP_PATH):
#     # os.mkdir(_TMP_PATH)
#     raise ValueError("Invalid input: '_TMP_PATH' not exists")
if not os.path.exists(_LIB_PATH):
    raise ValueError("Invalid input: '_LIB_PATH' not exists")
with open(prompt_file, 'r') as f:
    _PROMPT_TEMPLATE = Template(f.read())

# settings
_DEFAULT_CMD_TIMEOUT = 30
_THREAD_NUM = 3

# infer settings
_TEMPERATURE = 0.0
_TOP_P = 1.0
ATTEMPT_TIMES = 2


_API_URL = 'http://XXXX:3456/v1/chat/completions'
_MODEL_ = '/workspace/qwen3-train/models/Qwen3-Coder-30B'

_INFER_TIMEOUT = 50
_MAX_TOKENS = 8192



# Tree-sitter settings
C_LANGUAGE = Language(tsc.language())
# C_LANGUAGE = Language(abs_path + "parser/my-languages.so")
Q_TRUE_VALUE = C_LANGUAGE.query(" (declaration (init_declarator((number_literal)))) @true_value")
Q_CALL_FUNC = C_LANGUAGE.query('(call_expression function: (identifier) @func_name)')
Q_DECL_PTR = C_LANGUAGE.query(" (declaration ( pointer_declarator (function_declarator(_)  @declar_value1 ))) ")
Q_DECL_FN = C_LANGUAGE.query(" (declaration ( function_declarator(_)  @declar_value2 ))")
C_PARSER = Parser(C_LANGUAGE)


def _fix_nested_dict(inp):
    if isinstance(inp, dict):
        for k in inp:
            inp[k] = _fix_nested_dict(inp[k])
    elif isinstance(inp, list):
        for idx, e in enumerate(inp):
            inp[idx] = _fix_nested_dict(e)
    else:
        try:
            return json.loads(inp)
        except:
            return inp
    return inp


def exebench_dict_to_dict(pairs: dict):
    keys = pairs['var']
    values = pairs['value']
    return _fix_nested_dict({k: v for k, v in zip(keys, values)})


def diff_io(observed_output, expected_output) -> bool:
    if type(observed_output) is not type(expected_output):
        return False
    if isinstance(observed_output, list):
        if len(observed_output) != len(expected_output):
            return False
        for e1, e2 in zip(observed_output, expected_output):
            ok = diff_io(e1, e2)
            if not ok:
                return False
    elif isinstance(observed_output, dict):
        for key in observed_output:
            if key not in expected_output:
                return False
            ok = diff_io(observed_output[key], expected_output[key])
            if not ok:
                return False
    elif isinstance(observed_output, float):
        ok = math.isclose(observed_output, expected_output)
        if not ok:
            return False
    else:
        ok = observed_output == expected_output
        if not ok:
            return False
    return True


class EnvRecomReexe:

    def __init__(self, decom: str, dependencies: dict, tmp_path: str, task_num: int = 3,
                 parallel: bool = False, cancel_event: threading.Event = None):
        if not decom or not dependencies:
            raise ValueError("Invalid input: 'decom' or 'dependencies' is empty")
        self.decom = decom
        self.dependencies = dependencies
        self.task_num = task_num if task_num < 30 else 30
        self.parallel = parallel
        self._cancel_event = cancel_event or threading.Event()

        self.dir = os.path.join(tmp_path, str(uuid.uuid4()))

        os.makedirs(self.dir, exist_ok=True)

        self.recom_path = Path(self.dir) / "recom"
        self.recom_path.mkdir(parents=True, exist_ok=True)
        self.reexe_path = Path(self.dir) / "reexe"
        self.reexe_path.mkdir(parents=True, exist_ok=True)

        # if os.path.exists(tmp_path):
        #     shutil.rmtree(tmp_path)
        # os.makedirs(tmp_path)
        # self.dir = tmp_path
        # self.recom_path = Path(tmp_path) / "recom"
        # self.recom_path.mkdir(parents=True, exist_ok=True)
        # self.reexe_path = Path(tmp_path) / "reexe"
        # self.reexe_path.mkdir(parents=True, exist_ok=True)

        self.han_decom = ''
        self.handled_decom = ''

        # self.log = {'fname': self.dependencies["fname"], 'recompile': {}, 'reexecute': {"success": [], "failure": []}, 'post_handle': {"handle_decom": [], "delete_conflict": [], 'chat_with_model': {'error': []}}}
        self.log = {'recompile': {}}
        if self.parallel:
            self._log_lock = threading.Lock()
            self._case_lock = [threading.Lock(), threading.Lock(), threading.Lock()]
            self._handle_decom_lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        shutil.rmtree(self.dir)
        self.log.clear()
        self.log = None

    @contextmanager
    def _log_context(self):
        if self.parallel:
            with self._log_lock:
                yield
        else:
            yield

    @contextmanager
    def _case_access(self, cato: int):
        if self.parallel:
            with self._case_lock[cato]:
                yield
        else:
            yield

    @contextmanager
    def _handle_decom(self):
        if self.parallel:
            with self._handle_decom_lock:
                yield
        else:
            yield

    def _run_command(self, command: str, timeout: Optional[int] = _DEFAULT_CMD_TIMEOUT) -> Tuple[str, str, int, str]:
        stdout, stderr = '', ''
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   errors="replace", preexec_fn=os.setsid)
        deadline = time.monotonic() + timeout
        while True:
            # 检查外部取消信号
            if self._cancel_event.is_set():
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                return '', '', -1, 'cancelled'

            # 检查自身超时
            if time.monotonic() > deadline:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                return '', '', -1, 'timeout'

            # 检查进程是否自然结束
            retcode = process.poll()
            if retcode is not None:
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    return stdout, stderr, process.returncode, ''
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    stdout, stderr = process.communicate()
                    return stdout, stderr, process.returncode, "timeout"
            time.sleep(0.2)


    def chat_with_model(self, prompt: str) -> str:
        try:
            payload = {
                "model": _MODEL_,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": _TEMPERATURE,
                "top_p": _TOP_P,
                "max_tokens": _MAX_TOKENS,
                "stream": False,
            }

            req = urllib.request.Request(
                url=_API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=_INFER_TIMEOUT) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            result = json.loads(body)

            return result["choices"][0]["message"]["content"]

        except Exception:
            # when needed to construct
            if 'chat_with_model' not in self.log.keys():
                self.log['chat_with_model'] = {'error': []}
            elif 'error' not in self.log['chat_with_model'].keys():
                self.log['chat_with_model']['error'] = []
            # when needed to construct
            self.log['chat_with_model']['error'].append(traceback.format_exc())
            return ""

    # def call_with_messages(self, prompt: str):
    #     try:
    #         chat_completion = client.chat.completions.create(
    #             model= _MODEL,
    #             messages= [
    #                 {
    #                     "role": "user",
    #                     "temperature": _TEMPERATURE,
    #                     "top_p": _TOP_P,
    #                     "content": prompt
    #                 }
    #             ]
    #         )
    #     except APIStatusError as e:
    #         if e.status_code == 408:
    #             # when needed to construct
    #             if 'chat_with_model' not in self.log.keys():
    #                 self.log['chat_with_model'] = {'error': []}
    #             elif 'error' not in self.log['chat_with_model'].keys():
    #                 self.log['chat_with_model']['error'] = []
    #             # when needed to construct
    #             self.log['chat_with_model']['error'].append(f'Openai error\n{traceback.format_exc()}')
    #             return ""
    #         else:
    #             raise e

    #     if not hasattr(chat_completion, "choices") or not chat_completion.choices:
    #         # when needed to construct
    #         if 'chat_with_model' not in self.log.keys():
    #             self.log['chat_with_model'] = {'error': []}
    #         elif 'error' not in self.log['chat_with_model'].keys():
    #             self.log['chat_with_model']['error'] = []
    #         # when needed to construct
    #         self.log['chat_with_model']['error'].append(f'Request failed\n{chat_completion}')
    #         return ""
    #     return chat_completion.choices[0].message.content

    def recom(self) -> bool:
        c_deps_com = self.dependencies.get("synth_deps", '')
        if not c_deps_com:
            self.log['recompile']["internal_error"] = "No synth_deps found in dependencies"
            return False

        decom_func = re.sub(r'^.*(?<![a-zA-Z0-9_])extern(?![a-zA-Z0-9_]).*$', '', self.decom)
        success = False
        for i in range(-1, ATTEMPT_TIMES):
            if i != -1:
                prompt = Template.substitute(_PROMPT_TEMPLATE, dependency=c_deps_com, decompiled=decom_func)
                optimized = self.chat_with_model(prompt)
                # optimized = self.call_with_messages(prompt)
                optimized_decom_func = ""
                if optimized and 'none' not in optimized:
                    optimized_content = re.search(r"# optimized code[ ]*\n((.|\n)*?)#", optimized, re.I)
                    if optimized_content:
                        optimized_c = re.search(r"```C\n((.|\n)*)```", optimized_content[1], re.I)
                        if optimized_c:
                            optimized_decom_func = optimized_c[1]
                        else:
                            # when needed to construct
                            if 'chat_with_model' not in self.log.keys():
                                self.log['chat_with_model'] = {'error': []}
                            elif 'error' not in self.log['chat_with_model'].keys():
                                self.log['chat_with_model']['error'] = []
                            # when needed to construct
                            self.log['chat_with_model']['error'].append(f"No matched in optimized_c chat_{i}")
                    else:
                        # when needed to construct
                        if 'chat_with_model' not in self.log.keys():
                            self.log['chat_with_model'] = {'error': []}
                        elif 'error' not in self.log['chat_with_model'].keys():
                            self.log['chat_with_model']['error'] = []
                        # when needed to construct
                        self.log['chat_with_model']['error'].append(f"No matched in optimized_content chat_{i}")
                else:
                    # when needed to construct
                    if 'chat_with_model' not in self.log.keys():
                        self.log['chat_with_model'] = {'error': []}
                    elif 'error' not in self.log['chat_with_model'].keys():
                        self.log['chat_with_model']['error'] = []
                    # when needed to construct
                    self.log['chat_with_model']['error'].append(f"Error response chat_{i}")

                if optimized_decom_func:
                    decom_func = optimized_decom_func
                else:
                    continue

            c_code = c_deps_com + '\n' + decom_func + '\n'
            c_code_file = self.recom_path / "recom.c"
            with c_code_file.open("w") as f:
                f.write(c_code)
            execu_file = self.recom_path / "recom"
            if execu_file.exists():
                execu_file.unlink()

            stdout, stderr, retcode, cmderr = self._run_command(f"gcc -o {execu_file} -c {c_code_file}")
            if not execu_file.exists():
                matches = re.findall(r': error: (.*)$', stderr, re.MULTILINE)
                if len(matches) > 0:
                    # when needed to construct
                    if 'chat_with_model' not in self.log.keys():
                        self.log['chat_with_model'] = {}
                    # when needed to construct
                    self.log['chat_with_model'][f'chat_{i}'] = matches
                    if any(re.search(r'undeclared \(first use in this function\)', line) for line in matches) or \
                            any(re.search(r'undeclared here \(not in a function\)', line) for line in matches) or \
                            any(re.search(r'request for member .* in something not a structure or union', line) for line
                                in matches) or \
                            any(re.search(r'unknown type name .*; did you mean', line) for line in matches):
                        continue
                    else:
                        break
                if cmderr:  # 如果超时就当失败，是否准确？？
                    break
                else:
                    # when needed to construct
                    if 'chat_with_model' not in self.log.keys():
                        self.log['chat_with_model'] = {'error': []}
                    elif 'error' not in self.log['chat_with_model'].keys():
                        self.log['chat_with_model']['error'] = []
                    # when needed to construct
                    self.log['chat_with_model']['error'].append(
                        f'Unnomal compile chat_{i}: retcode={retcode}\n{stderr}\n{stdout}')
            else:
                success = True
                break

        if not success:
            self.log['recompile']['exit_code'] = retcode
            if cmderr:
                self.log['recompile']['run_error'] = cmderr
            if stderr:
                self.log['recompile']['compile_error'] = stderr
            if stdout:
                self.log['recompile']['compile_print'] = stdout
        self.han_decom = decom_func
        self.log['recompile']['chat'] = i
        return success

    def reexe_single(self, idx: int = 0) -> bool:
        inputs = self.dependencies.get("synth_io_pairs", {}).get("input", [])
        if len(inputs) <= idx:
            with self._log_context():
                if "internal_error" not in self.log['reexecute'].keys():
                    self.log['reexecute']["internal_error"] = []
                self.log['reexecute']["internal_error"].append(f"No input {idx}")
            return False
        outputs = self.dependencies.get("synth_io_pairs", {}).get("output", [])
        if len(outputs) <= idx:
            with self._log_context():
                if "internal_error" not in self.log['reexecute'].keys():
                    self.log['reexecute']["internal_error"] = []
                self.log['reexecute']["internal_error"].append(f"No output {idx}")
            return False

        case_path = self.reexe_path / str(idx)
        case_path.mkdir(parents=True, exist_ok=True)
        input = exebench_dict_to_dict(inputs[idx])
        input_file = case_path / 'input.json'
        with input_file.open('w') as f:
            json.dump(input, f)
        ground = exebench_dict_to_dict(outputs[idx])

        def delete_conflict(dummy, ind):
            matches = re.findall(r'(\w+)__bench[ ]*\(.*?\)[ ]*{', dummy, re.MULTILINE)
            c_deps_0 = self.dependencies.get("synth_deps", '')
            func_def_0 = self.handled_decom
            if len(matches) > 0:
                for item in matches:
                    pattern = r'^.*?(?<![a-zA-Z0-9_])' + item + r'(?![a-zA-Z0-9_]).*?;.*?$'
                    c_deps_1 = re.sub(pattern, '', c_deps_0, flags=re.M)
                    if len(c_deps_1) == len(c_deps_0):
                        with self._log_context():
                            # when needed to construct
                            if 'delete_conflict' not in self.log.keys():
                                self.log['delete_conflict'] = []
                            # when needed to construct
                            self.log['delete_conflict'].append("Delete synth_deps failed {} {}".format(item, ind))
                    else:
                        c_deps_0 = c_deps_1
                    pattern = r'(?<![a-zA-Z0-9_])' + item + r'(?![a-zA-Z0-9_])[ ]*\('
                    changed = item + '__bench('
                    func_def_1 = re.sub(pattern, changed, func_def_0, flags=re.M)
                    if len(func_def_1) == len(func_def_0):
                        with self._log_context():
                            # when needed to construct
                            if 'delete_conflict' not in self.log.keys():
                                self.log['delete_conflict'] = []
                            # when needed to construct
                            self.log['delete_conflict'].append("Delete handled_decom failed {} {}".format(item, ind))
                    else:
                        func_def_0 = func_def_1
            return c_deps_0, func_def_0

        cato = idx // 10
        c_dep_file = self.reexe_path / f'case_{cato}.c'
        cpp_wrapper_file = self.reexe_path / f'case_{cato}.cpp'
        execu_file = self.reexe_path / f'case_{cato}'
        with self._case_access(cato):
            if not c_dep_file.exists() or not cpp_wrapper_file.exists():
                cpp_wrapper = self.dependencies.get("synth_exe_wrapper", '')
                if not cpp_wrapper:
                    with self._log_context():
                        if "internal_error" not in self.log['reexecute'].keys():
                            self.log['reexecute']["internal_error"] = []
                        self.log['reexecute']["internal_error"].append(f"No cpp_wrapper {cato}")
                    return False
                funcname = self.dependencies["fname"]
                if f'{funcname}__bench' in cpp_wrapper:
                    funcname = f'{funcname}__bench'
                cpp_wrapper = re.sub(r'extern\s\"C\"\s\{\s.*?\s\}',
                                     'extern "C" \n{\n#include "' + str(c_dep_file.resolve()) + '"\n}\n', cpp_wrapper)

                with self._handle_decom():
                    if not self.handled_decom:
                        self.handled_decom = self.handle_decom(self.han_decom, funcname)
                dummy_funcs = self.dependencies.get("synth_io_pairs", {}).get("dummy_funcs", [])
                if len(dummy_funcs) <= idx:
                    dummy_func = ''
                else:
                    dummy_func = dummy_funcs[idx]
                handled_synth_deps, handled_func_def = delete_conflict(dummy_func, cato)
                c_deps = handled_synth_deps + '\n' + dummy_func + '\n' + handled_func_def

                with c_dep_file.open('w') as f:
                    f.write(c_deps)
                with cpp_wrapper_file.open('w') as f:
                    f.write(cpp_wrapper)

                if execu_file.exists():
                    execu_file.unlink()
                stdout, stderr, retcode, cmderr = self._run_command(
                    f'g++ -fpermissive -o {str(execu_file)} {str(cpp_wrapper_file)} -I {str(_LIB_PATH)} -I{str(_LIB_PATH)}')
                title = f'case_{cato}'
                self.log['reexecute'][title] = {'exit_code': retcode}
                if cmderr:
                    self.log['reexecute'][title]['run_error'] = cmderr
                if stderr:
                    self.log['reexecute'][title]['compile_error'] = stderr
                if stdout:
                    self.log['reexecute'][title]['compile_print'] = stdout
                if not execu_file.exists():
                    with self._log_context():
                        if "internal_error" not in self.log['reexecute'].keys():
                            self.log['reexecute']["internal_error"] = []
                        self.log['reexecute']["internal_error"].append(f"No execu_file {cato}")
                    return False
            elif not execu_file.exists():
                return False

        output_file = case_path / 'output.json'
        stdout, stderr, retcode, cmderr = self._run_command(f'{str(execu_file)} {str(input_file)} {str(output_file)}')
        title = f'task_{idx}'
        self.log['reexecute'][title] = {'exit_code': retcode}
        if cmderr:
            self.log['reexecute'][title]['run_error'] = cmderr
        if stderr:
            self.log['reexecute'][title]['exe_error'] = stderr
        if stdout:
            self.log['reexecute'][title]['exe_print'] = stdout
        if not output_file.exists():
            self.log['reexecute'][title]['failure'] = "Runtime error"
            return False

        with output_file.open('r') as f:
            output = json.load(f)
        if not diff_io(output, ground):
            self.log['reexecute'][title]['failure'] = "Incorrect output"
            self.log['reexecute'][title]['input'] = input
            self.log['reexecute'][title]['observed_output'] = output
            self.log['reexecute'][title]['expected_output'] = ground
            return False

        return True

    def reexe_serial_terminal(self) -> int:
        for i in range(self.task_num):
            if i in self.log['reexecute']['success']:
                continue
            if i in self.log['reexecute']['failure']:
                return i
            if self.reexe_single(i):
                self.log['reexecute']['success'].append(i)
            else:
                self.log['reexecute']['failure'].append(i)
                return i
        return self.task_num

    def reexe_serial(self) -> list:
        failed = []
        for i in range(self.task_num):
            if i in self.log['reexecute']['success']:
                continue
            elif i in self.log['reexecute']['failure']:
                failed.append(i)
            elif self.reexe_single(i):
                self.log['reexecute']['success'].append(i)
            else:
                self.log['reexecute']['failure'].append(i)
                failed.append(i)
        return failed

    def reexe_parallel(self):  # to do
        failed = []
        with ThreadPoolExecutor(max_workers=_THREAD_NUM) as executor:
            future_to_idx = {
                executor.submit(self.reexe_single, i): i
                for i in range(self.task_num)
            }
            for future in as_completed(future_to_idx):
                if not future.result():
                    failed.append(future_to_idx[future])
        return failed

    def reexe(self):
        self.log['reexecute'] = {"success": [], "failure": []}
        if not self.parallel:
            failed = self.reexe_serial()
        else:
            failed = self.reexe_parallel()
        return failed

    def reward(self) -> dict:
        if "success" not in self.log['recompile'].keys():
            self.log['recompile']['success'] = self.recom()
        if not self.log['recompile']['success']:
            return {
                "score": -1,
                "feedback": {
                    k: v for k, v in self.log['recompile'].items()
                    if k not in ['exit_code', 'success', 'chat']
                }
            }

        failed = self.reexe()
        if not failed:
            return {"score": 1}

        runtime_error = False
        compile_error = False
        cato = 0
        feedback = {}
        for i in failed:
            ind = f'task_{i}'
            if ind not in self.log['reexecute'].keys():
                compile_error = True
                cato = i // 10
                break
            if self.log['reexecute'][ind].get('failure', '') == "Runtime error":
                runtime_error = True
            feedback[f'case_{i}'] = {**self.log['reexecute'][ind]}

        if compile_error:
            return {
                "score": -1,
                "feedback": {
                    k: v for k, v in self.log['reexecute'][f'case_{cato}'].items()
                    if k != 'exit_code'
                }
            }
        if runtime_error:
            return {
                "score": -6,
                "feedback": feedback
            }

        return {
            "score": -3,
            "feedback": feedback
        }

    def handle_decom(self, decom: str, funcname: str) -> str:
        try:

            tree = C_PARSER.parse(bytes(decom, "utf8"))
            root_node = tree.root_node
            code_bytes = bytearray(decom, "utf8")

            output = []
            captures = Q_TRUE_VALUE.captures(root_node)
            if captures:
                for capture in captures.values():
                    for value in capture:
                        output.append(value.text.decode("utf8"))

            call_function = []
            captures = Q_CALL_FUNC.captures(root_node)
            if captures:
                for key, capture in captures.items():
                    for value in capture:
                        call_function.append(value.text.decode("utf8"))
            captures = Q_DECL_PTR.captures(root_node)
            if captures:
                for key, capture in captures.items():
                    for value in capture:
                        call_function.append(value.text.decode("utf8"))
            captures = Q_DECL_FN.captures(root_node)
            if captures:
                for key, capture in captures.items():
                    for value in capture:
                        call_function.append(value.text.decode("utf8"))

            last_functions = []
            changes = []
            changes_filter = []
            function_filter = []

            def walk(node):
                if node.type in ['function_definition']:
                    for child in node.children:
                        if child.type == 'function_declarator':
                            for c_child in child.children:
                                if c_child.type == 'identifier':
                                    if c_child.text.decode("utf8") not in call_function:
                                        changes.append(
                                            (c_child.start_byte, c_child.end_byte, c_child.text.decode("utf8")))
                                        last_functions.append(node)
                                    else:
                                        changes_filter.append(
                                            (c_child.start_byte, c_child.end_byte, c_child.text.decode("utf8")))
                                        function_filter.append(node)
                        elif child.type == 'pointer_declarator':
                            for c_child in child.children:
                                if c_child.type == 'function_declarator':
                                    for cc_child in c_child.children:
                                        if cc_child.type == 'identifier':
                                            if cc_child.text.decode("utf8") not in call_function:
                                                changes.append((cc_child.start_byte, cc_child.end_byte,
                                                                cc_child.text.decode("utf8")))
                                                last_functions.append(node)
                                            else:
                                                changes_filter.append((cc_child.start_byte, cc_child.end_byte,
                                                                       cc_child.text.decode("utf8")))
                                                function_filter.append(node)
                for child in node.children:
                    walk(child)

            walk(root_node)
            # changes.sort(reverse=True)
            if len(changes) == 1:
                change = changes[-1]
                last_function = last_functions[-1]
                function_text = code_bytes[last_function.start_byte:last_function.end_byte]
                start_byte, end_byte = (change[0] - last_function.start_byte), (change[1] - last_function.start_byte)
                if start_byte <= 0 or start_byte >= len(function_text) or end_byte <= 0 or end_byte >= len(
                        function_text):
                    # when needed to construct
                    if 'handle_decom' not in self.log.keys():
                        self.log['handle_decom'] = []
                    # when needed to construct
                    self.log['handle_decom'].append(
                        "start_byte_error ({}, {}) ({}, {}) <= 0".format(change[0], change[1], last_function.start_byte,
                                                                         last_function.end_byte))
                else:
                    function_text = function_text[:start_byte] + bytearray(funcname, "utf8") + function_text[end_byte:]
                output.append(function_text.decode("utf8"))
            else:
                if not changes and not changes_filter:
                    # when needed to construct
                    if 'handle_decom' not in self.log.keys():
                        self.log['handle_decom'] = []
                    # when needed to construct
                    self.log['handle_decom'].append(
                        "Error changes and filter - call_function {}: {}".format(len(call_function), call_function))
                else:
                    if not changes:
                        changes = changes_filter
                        last_functions = function_filter
                        # when needed to construct
                        if 'handle_decom' not in self.log.keys():
                            self.log['handle_decom'] = []
                        # when needed to construct
                        self.log['handle_decom'].append(
                            "Choose filter {} - changes_filter: {}".format(funcname, changes_filter))
                    else:
                        # when needed to construct
                        if 'handle_decom' not in self.log.keys():
                            self.log['handle_decom'] = []
                        # when needed to construct
                        self.log['handle_decom'].append("Choose changes {} - changes: {}".format(funcname, changes))
                    max_len = 0
                    chosen_ind = -1
                    for ind in range(len(last_functions)):
                        len_t = last_functions[ind].start_byte - last_functions[ind].end_byte
                        if len_t > max_len:
                            max_len = len_t
                            chosen_ind = ind
                    change = changes[chosen_ind]
                    last_function = last_functions[chosen_ind]
                    function_text = code_bytes[last_function.start_byte:last_function.end_byte]
                    start_byte, end_byte = (change[0] - last_function.start_byte), (
                                change[1] - last_function.start_byte)
                    if start_byte <= 0 or start_byte >= len(function_text) or end_byte <= 0 or end_byte >= len(
                            function_text):
                        # when needed to construct
                        if 'handle_decom' not in self.log.keys():
                            self.log['handle_decom'] = []
                        # when needed to construct
                        self.log['handle_decom'].append(
                            "start_byte error ({}, {}) ({}, {}) <= 0".format(change[0], change[1],
                                                                             last_function.start_byte,
                                                                             last_function.end_byte))
                    else:
                        function_text = function_text[:start_byte] + bytearray(funcname, "utf8") + function_text[
                                                                                                   end_byte:]
                    output.append(function_text.decode("utf8"))

            if not output:
                # when needed to construct
                if 'handle_decom' not in self.log.keys():
                    self.log['handle_decom'] = []
                # when needed to construct
                self.log['handle_decom'].append("Empty output")
                return decom
            else:
                return '\n'.join(output)

        except Exception:
            # when needed to construct
            if 'handle_decom' not in self.log.keys():
                self.log['handle_decom'] = []
            # when needed to construct
            self.log['handle_decom'].append("Error handle_decom\n{}".format(traceback.format_exc()))
            return decom