import subprocess, shutil, logging
import argparse
import os
import re, time
import json
from tqdm import tqdm, trange
from pathlib import Path
from typing import Optional, Tuple
# from LogRecorder import CLogRecoder

_DEFAULT_CMD_TIMEOUT = 10

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, required=True)
parser.add_argument('--log', '-l', type=str, default='', required=False)
parser.add_argument('--output', '-o', type=str, default='', required=False)
parser.add_argument('--temp', '-t', type=str, default='', required=False)
args = parser.parse_args()

input_path = Path(args.input)
if not input_path.exists():
    print(f'Error: input file {input_path} not exists!')
    quit(-1)

if args.log:
    log_file = Path(args.log)
    if not log_file.parent.exists():
        print(f'Error: log_dir {log_file.parent} not exists!')
        quit(-1)
else:
    ymd = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_file = input_path.parent / f'{input_path.stem}_{ymd}.log'
if log_file.exists():
    log_file.unlink()
if args.output:
    output_path = Path(args.output)
    if not output_path.parent.exists():
        print(f'Error: output_dir {output_path.parent} not exists!')
        quit(-1)
else:
    output_path = input_path.parent / 'metric.txt'
if args.temp:
    temp_path = Path(args.temp)
    if not temp_path.exists():
        print(f'Error: temp_dir {temp_path} not exists!')
        quit(-1)
else:
    temp_path = input_path.parent / 'tmp'
    if not temp_path.exists():
        temp_path.mkdir(parents=True)

# store_path = input_path.parent / f'{input_path.stem}_c_file'
# if store_path.exists():
#     shutil.rmtree(store_path, ignore_errors=True)
# store_path.mkdir(parents=True)

class CLogRecoder:

    def __init__(self, logfile = 'log.log', format = '%(asctime)s : %(message)s', level = logging.DEBUG):
        logging.basicConfig(filename= logfile, level= level , format= format)
        self._ft = format

    def addStreamHandler(self):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formater = logging.Formatter(self._ft)
        console.setFormatter(formater)
        logging.getLogger('').addHandler(console)
        return self

    def INFO(self, message):
        logging.info(message)
        return self

logger = CLogRecoder(logfile=log_file)

# execute shell command
def _run_command(command: str, timeout: Optional[int] = _DEFAULT_CMD_TIMEOUT) -> Tuple[str, str]:
    # output = subprocess.run(command.split(), shell=True, capture_output=True, text=True, input=stdin, timeout=timeout)
    output = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    stdout = output.stdout.decode('utf-8') if isinstance(output.stdout, bytes) else output.stdout
    stderr = output.stderr.decode('utf-8') if isinstance(output.stderr, bytes) else output.stderr
    return stdout, stderr

def handle_decom(funcname, content, source):
    matches = re.findall(r'```.*?\n((.|\n)*?)```', content, re.MULTILINE)
    if matches:
        content = matches[0][0].strip()
    else:
        content = content.strip()
    
    lines = content.split('\n')
    for index in range(len(lines)):
        if lines[index]:
            if '#include' in lines[index] and lines[index] in source:
                lines[index] = ''
                continue
            mat = re.findall(r'([a-zA-Z0-9_]{2,})[ ]*\(', lines[index], re.MULTILINE)
            if mat:
                changed = ' ' + funcname + '('
                line = re.sub(r'([a-zA-Z0-9_]{2,})[ ]*\(', changed, lines[index], re.MULTILINE)
                if line != lines[index]:
                    lines[index] = line
                    logger.INFO("changed decom {}".format(funcname))
                break
    if index == len(lines) - 1:
        logger.INFO("No change decom {}".format(funcname))
    return '\n'.join(lines)

def evaluate_func(c_func,c_test,c_func_decompile,input_asm_prompt_id):
    c_func_decompile = handle_decom('func0', c_func_decompile, c_func)

    flag_compile = 0
    flag_run = 0
    c_include = ''
    for line in c_func.split('\n'):
        if '#include' in line:
            c_include += line+'\n'
            c_func = c_func.replace(line, '')
    for line in c_test.split('\n'):
        if '#include' in line:
            c_include += line+'\n'
            c_test = c_test.replace(line, '')
    c_combine = c_include + '\n' + c_func_decompile + '\n' + c_test
    c_onlyfunc = c_include + '\n' + c_func_decompile

    # Define the C file and executable names
    c_file_onlyfunc = os.path.join(temp_path, 'onlyfunc.c')
    executable_onlyfunc = os.path.join(temp_path, 'onlyfunc')
    c_file = os.path.join(temp_path, 'combine.c') 
    # c_file_save = os.path.join(store_path, f'func{input_asm_prompt_id}.c')
    executable = os.path.join(temp_path, 'combine')
    if os.path.exists(executable_onlyfunc):
        os.remove(executable_onlyfunc)
    if os.path.exists(executable):
        os.remove(executable)
    
    with open(c_file_onlyfunc,'w') as f:
        f.write(c_onlyfunc)
    with open(c_file,'w') as f:
        f.write(c_combine)
    # with open(c_file_save,'w') as f:
    #     f.write(c_combine)

    # Compile the C program to an assembly
    compile_command = f'gcc -S {c_file_onlyfunc} -o {executable_onlyfunc} -lm'
    try:
        stdout, stderr = _run_command(compile_command)
    except Exception as e:
        logger.INFO("Cmd Error:\n{}".format(e))
        return flag_compile, flag_run
    if not os.path.exists(executable_onlyfunc):
        logger.INFO('Error: no executable_onlyfunc')
        if stderr:
            logger.INFO("Compile Error:\n{}".format(stderr))
        return flag_compile, flag_run
    flag_compile = 1

    # Compile the C program to an executable
    compile_command = f'gcc {c_file} -o {executable} -lm'
    try:
        stdout, stderr = _run_command(compile_command)
    except Exception as e:
        logger.INFO("Cmd Error:\n{}".format(e))
        return flag_compile, flag_run
    if not os.path.exists(executable):
        logger.INFO('Error: no executable')
        if stderr:
            logger.INFO("Execute Error:\n{}".format(stderr))
        return flag_compile, flag_run
    flag_compile = 1

    # Run the compiled executable
    run_command = f'{executable}'
    try:
        process = subprocess.run(run_command, shell=True, check=True,capture_output=True, timeout=_DEFAULT_CMD_TIMEOUT)
        flag_run = 1
    except subprocess.CalledProcessError as e:
        pass
    except Exception as e:
        pass
    return flag_compile, flag_run

NUM = {"O0":0, "O1":0, "O2":0, "O3":0}
num_compile = {"O0":0, "O1":0, "O2":0, "O3":0}
num_run = {"O0":0, "O1":0, "O2":0, "O3":0}

with input_path.open('r') as f:
    lines = f.readlines()

compile_func = []
run_func = []
for ind, line in enumerate(tqdm(lines, desc="Processing", unit="case")):
    try:
        json_obj = json.loads(line.strip())
    except Exception as e:
        print(f"Error parsing JSON: {ind + 1}-{len(line)}")
        raise e

    c_func_decompile = json_obj.get('c_func_decompile', '')
    if not c_func_decompile:
        continue
    opt_state = json_obj['opts']
    NUM[opt_state] += 1
    input_id = str(json_obj['dependencies']['task_id']) + '_' + opt_state
    logger.INFO("handle func {}".format(input_id))
    flag_compile, flag_run = evaluate_func(json_obj['dependencies']['c_func'], json_obj['dependencies']['c_test'], c_func_decompile, input_id)
    if flag_compile == 1:
        compile_func.append(input_id)
    if flag_run == 1:
        run_func.append(input_id)
    num_compile[opt_state]+=flag_compile
    num_run[opt_state]+=flag_run


with output_path.open('a') as f:
    new_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    f.write('{} {}\n'.format(new_time, input_path))
    avg_compile_rate_sum = 0.0
    avg_run_rate_sum = 0.0
    avg_count = 0
    for opt_state in num_compile.keys():
        if NUM[opt_state] > 0:
            compile_rate = num_compile[opt_state] / NUM[opt_state]
            run_rate = num_run[opt_state] / NUM[opt_state]
            avg_compile_rate_sum += compile_rate
            avg_run_rate_sum += run_rate
            avg_count += 1
        else:
            compile_rate = 0.0
            run_rate = 0.0
        f.write(
            'opt:{} count:{} compile_success:{} compile_rate:{:.4f} run_success:{} run_rate:{:.4f}\n'.format(
                opt_state,
                NUM[opt_state],
                num_compile[opt_state],
                compile_rate,
                num_run[opt_state],
                run_rate,
            )
        )
    avg_compile_rate = (avg_compile_rate_sum / avg_count) if avg_count else 0.0
    avg_run_rate = (avg_run_rate_sum / avg_count) if avg_count else 0.0
    f.write('avg_compile_rate:{:.4f} avg_run_rate:{:.4f}\n'.format(avg_compile_rate, avg_run_rate))
    print('avg_compile_rate:{:.4f} avg_run_rate:{:.4f}'.format(avg_compile_rate, avg_run_rate))
    f.write('\n')
    # f.write('compile_func: ' + str(compile_func))
    # f.write('\n')
    # f.write('run_func: ' + str(run_func))