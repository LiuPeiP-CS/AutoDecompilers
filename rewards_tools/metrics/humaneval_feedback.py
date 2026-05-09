import os, shutil, subprocess, signal, json, traceback, threading, re, math, uuid
from typing import Optional, Tuple, Dict
from pathlib import Path

# settings
_DEFAULT_CMD_TIMEOUT = 30

RE_TEST_INPUT    = re.compile(r"^\[Test\s+(?P<t>\d+)\]\s+input:\s+(?P<rest>.*)\s*$")
RE_TEST_GROUND   = re.compile(r"^\[Test\s+(?P<t>\d+)\]\s+ground:\s+(?P<ground>.*)\s*$")
RE_TEST_OUTPUT   = re.compile(r"^\[Test\s+(?P<t>\d+)\]\s+output:\s+(?P<output>.*)\s*$")
RE_TEST_PASSFAIL = re.compile(r"^\[Test\s+(?P<t>\d+)\]\s+(?P<pf>pass|fail)\s*$")
RE_PASSED_SUMMARY = re.compile(r"^Passed:\s+(?P<pass>\d+)\s*/\s*(?P<total>\d+)\s*$")
RE_KEY = re.compile(r"\b(a[1-9])-")


class HumanEvalFeedback:

    def __init__(
        self,
        decom: str,
        dependencies: dict,
        tmp_path: str,
        cancel_event: threading.Event = None,   # ← 新增：外部传入的取消信号
    ):
        if not decom or not dependencies:
            raise ValueError("Invalid input: 'decom' or 'dependencies' is empty")
        self.decom = decom
        self.dependencies = dependencies
        self._cancel_event = cancel_event or threading.Event()

        # ↓ 修复1：uuid 子目录隔离，避免多 rank 互相 rmtree
        unique_dir = Path(tmp_path) / str(uuid.uuid4())
        unique_dir.mkdir(parents=True, exist_ok=True)
        self.dir = unique_dir

        # ↓ 修复2：self.dir 现在是 Path 对象，/ 操作符合法
        self.recom_file = self.dir / "case.c"
        self.exe_file   = self.dir / "case"
        if self.exe_file.exists():
            self.exe_file.unlink()

        self.han_decom     = ''
        self.handled_decom = ''
        self.log = {'recompile': {}}

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    # ↓ 修复3：__exit__ 必须接收三个异常参数
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        self.log.clear()
        self.log = None

    # ------------------------------------------------------------------
    # ↓ 修复4：_run_command 改为轮询，支持 cancel_event 中断
    # ------------------------------------------------------------------
    def _run_command(
        self,
        command: str,
        timeout: Optional[int] = _DEFAULT_CMD_TIMEOUT,
    ) -> Tuple[str, str, int, str]:
        import time
        process = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", preexec_fn=os.setsid,
        )
        deadline = time.monotonic() + timeout

        while True:
            # 检查外部取消信号（外层 asyncio.wait_for 超时后 set）
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

            # # 检查进程是否自然结束
            # retcode = process.poll()
            # if retcode is not None:
            #     stdout, stderr = process.communicate()
            #     return stdout, stderr, retcode, ''
            #
            # time.sleep(0.2)
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

    # ------------------------------------------------------------------
    # handle_decom
    # ------------------------------------------------------------------
    def handle_decom(self):
        source = self.dependencies.get("c_func", '')
        matches = re.findall(r'```.*?\n((.|\n)*?)```', self.decom, re.MULTILINE)
        if matches:
            self.han_decom = matches[0][0].strip()
        else:
            self.log.setdefault('handle_decom', [])
            self.log['handle_decom'].append("not match decom")
            self.han_decom = self.decom.strip()

        lines = self.han_decom.split('\n')
        for index in range(len(lines)):
            if lines[index]:
                if '#include' in lines[index] and lines[index] in source:
                    lines[index] = ''
                    continue
                mat = re.findall(r'([a-zA-Z0-9_]{2,})[ ]*\(', lines[index], re.MULTILINE)
                if mat:
                    changed = ' func0('
                    line = re.sub(r'([a-zA-Z0-9_]{2,})[ ]*\(', changed, lines[index], re.MULTILINE)
                    if line != lines[index]:
                        self.log.setdefault('handle_decom', [])
                        self.log['handle_decom'].append(f"Change \"{lines[index]}\" to \"{line}\"")
                        lines[index] = line
                    break
        if index == len(lines) - 1:
            self.log.setdefault('handle_decom', [])
            self.log['handle_decom'].append("No change decom")
        self.handled_decom = '\n'.join(lines)

    # ------------------------------------------------------------------
    # recom
    # ------------------------------------------------------------------
    def recom(self) -> bool:
        self.handle_decom()
        c_include = ''
        c_func = self.dependencies.get("c_func", '')
        c_test = self.dependencies.get("c_test_print", '')
        if not c_test or not c_func:
            self.log['recompile']['internal_error'] = "No c_func or c_test_print in dependencies"
            return False

        for line in c_func.split('\n'):
            if '#include' in line:
                c_include += line + '\n'
                c_func = c_func.replace(line, '')
        for line in c_test.split('\n'):
            if '#include' in line:
                c_include += line + '\n'
                c_test = c_test.replace(line, '')

        c_combine = c_include + '\n' + self.handled_decom + '\n' + c_test
        with self.recom_file.open("w") as f:
            f.write(c_combine)

        stdout, stderr, retcode, cmderr = self._run_command(
            f"gcc {self.recom_file} -o {self.exe_file} -lm"
        )
        if not self.exe_file.exists():
            self.log['recompile']['exit_code'] = retcode
            if cmderr:
                self.log['recompile']['run_error'] = cmderr
            if stderr:
                self.log['recompile']['compile_error'] = stderr
            if stdout:
                self.log['recompile']['compile_print'] = stdout
            return False
        return True

    # ------------------------------------------------------------------
    # parse_stdout
    # ------------------------------------------------------------------
    def parse_stdout(self, stdout: str):
        result: Dict = {}

        def _parse_kv_tail(s: str) -> Dict[str, str]:
            out: Dict[str, str] = {}
            matches = list(RE_KEY.finditer(s))
            if not matches:
                return out
            for i, m in enumerate(matches):
                key = m.group(1)
                value_start = m.end()
                value_end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
                out[key] = s[value_start:value_end].strip()
            return out

        for line in stdout.splitlines():
            if not line:
                continue

            m_in = RE_TEST_INPUT.match(line)
            if m_in:
                t = int(m_in.group("t"))
                result.setdefault(t, {})
                result[t]["input"] = _parse_kv_tail(m_in.group("rest"))
                continue

            m_g = RE_TEST_GROUND.match(line)
            if m_g:
                t = int(m_g.group("t"))
                result.setdefault(t, {})
                result[t]["expected_output"] = m_g.group("ground")
                continue

            m_o = RE_TEST_OUTPUT.match(line)
            if m_o:
                t = int(m_o.group("t"))
                result.setdefault(t, {})
                result[t]["observed_output"] = m_o.group("output")
                continue

            m_pf = RE_TEST_PASSFAIL.match(line)
            if m_pf:
                t = int(m_pf.group("t"))
                result.setdefault(t, {})
                result[t]["result"] = m_pf.group("pf")
                continue

            m_ps = RE_PASSED_SUMMARY.match(line)
            if m_ps:
                result['passed_summary'] = {
                    "pass":  int(m_ps.group("pass")),
                    "total": int(m_ps.group("total")),
                }
                continue

        if result:
            self.log['reexecute']['result'] = result

    # ------------------------------------------------------------------
    # reexe
    # ------------------------------------------------------------------
    def reexe(self) -> bool:
        self.log.setdefault('reexecute', {})
        self.log['reexecute'].setdefault('internal_error', [])

        if not self.exe_file.exists():
            self.log['reexecute']['internal_error'].append("No exe_file")
            return False

        stdout, stderr, retcode, cmderr = self._run_command(str(self.exe_file))
        self.log['reexecute']['runtime'] = {'exit_code': retcode}
        if cmderr:
            self.log['reexecute']['runtime']['run_error'] = cmderr
        if stderr:
            self.log['reexecute']['runtime']['exe_error'] = stderr
        if stdout:
            self.log['reexecute']['runtime']['exe_print'] = stdout
            self.parse_stdout(stdout)

        return True

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------
    def reward(self) -> dict:
        if "success" not in self.log['recompile']:
            self.log['recompile']['success'] = self.recom()
        if not self.log['recompile']['success']:
            return {
                "score": -1,
                "feedback": {
                    k: v for k, v in self.log['recompile'].items()
                    if k not in ['exit_code', 'success']
                },
            }

        flag = self.reexe()
        runtime = self.log.get('reexecute', {}).get('runtime', {})
        result  = self.log.get('reexecute', {}).get('result', {})

        if flag and runtime.get('exit_code') in [0, 1] and result:
            summary = result.get('passed_summary')
            if summary:
                if summary['pass'] == summary['total']:
                    return {"score": 1}

                # ↓ 修复5：先 setdefault 再赋值，避免对不存在的 key 直接写子键
                feedback: Dict = {}
                for t, info in result.items():
                    if t == 'passed_summary':
                        continue
                    key = f'task_{t}'
                    if info.get('result') == 'fail':
                        feedback[key] = {
                            'failure':         'Incorrect output',
                            'input':           info.get('input'),
                            'observed_output': info.get('observed_output'),
                            'expected_output': info.get('expected_output'),
                        }
                    else:
                        feedback[key] = {k: v for k, v in info.items() if k != 'result'}

                if feedback:
                    return {"score": -3, "feedback": feedback}
                else:
                    self.log['reexecute']['internal_error'].append(
                        "No failed task, but passed_summary shows not all passed"
                    )
            else:
                self.log['reexecute']['internal_error'].append("No passed_summary")

        elif flag and runtime.get('exit_code') in [0, 1]:
            self.log['reexecute']['internal_error'].append("No result")

        elif flag:
            return {
                "score": -6,
                "feedback": {"failure": "Runtime error", **runtime},
            }

        else:
            return {
                "score": -6,
                "feedback": {"internal_error": self.log['reexecute']['internal_error']},
            }

        # 兜底
        return {
            "score": -6,
            "feedback": {
                "failure":        "Runtime error",
                "internal_error": self.log['reexecute']['internal_error'],
                **runtime,
            },
        }