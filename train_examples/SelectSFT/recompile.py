import subprocess, shutil
import argparse
import os
import re, time
import json
from os import write

from tqdm import tqdm, trange
from typing import Optional, Tuple
from LogRecorder import CLogRecoder

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_file = 'humaneval/decompile-eval-executable-gcc-obj.json'
_DEFAULT_CMD_TIMEOUT = 10
_INDEX = '0'
_ISNEW = True

os.environ['TMPDIR'] = '/XXX/Eval/humaneval/tmp/tmp'

parser = argparse.ArgumentParser()
parser.add_argument('--workdir', type=str, help="the output_dir path for llm decompilation", required=False) # 属于每个模型的工作空间
# parser.add_argument('--input', type=str, default='', required=False)
# parser.add_argument('--data_file', type=str, default='decompile-eval-executable-gcc-obj.json', required=False)
parser.add_argument('--model', type=str, help="the llm name for remembering the information", required=False)
parser.add_argument("-b", required=True, help="bit-wide")
parser.add_argument("-i", required=True, help="index")
parser.add_argument("--origin", action="store_true", help="no post-handle")
parser.add_argument('--all_metrics', type=str, default='all_metrics.txt', help='Path to all metrics file')

args = parser.parse_args()
pro_path = args.workdir
if not os.path.exists(pro_path):
    print(f'Error: workdir {pro_path} not exists!')
    quit(-1)
if not os.path.exists(data_file):
    print(f'Error: data {data_file} not exists!')
    quit(-1)
_MODEL = args.model
if args.i:
    _INDEX = args.i
if args.b in ['64', '32']:
    _BITWIDE = args.b
else:
    print(f'Error bit-wide {args.b} !')
    quit(-1)
if args.origin:
    _ISNEW = False


analysis_path = os.path.join(pro_path, 'analysis')
res_path = os.path.join(pro_path, f'humaneval_{_INDEX}')
input_path = os.path.join(res_path, f'decom_{_BITWIDE}')
store_path = os.path.join(res_path, f'c_file_{_BITWIDE}')
tmp_path = os.path.join(pro_path, 'tmp')
if not os.path.exists(res_path):
    print(f'Error: result dir {res_path} not exists!')
    quit(-1)
if not os.path.exists(input_path):
    print(f'Error: decom dir {input_path} not exists!')
    quit(-1)
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)
if os.path.exists(store_path):
    shutil.rmtree(store_path)
os.makedirs(store_path)
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

ymd = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
logger = CLogRecoder(logfile=os.path.join(analysis_path, '{}_hel-metric_{}_{}_{}.log'.format(ymd, _MODEL, _BITWIDE, _INDEX)))

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
        logger.INFO("not match decom {}".format(funcname))
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
    if _ISNEW:
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
    c_file_onlyfunc = os.path.join(tmp_path, 'onlyfunc.c')
    executable_onlyfunc = os.path.join(tmp_path, 'onlyfunc')
    c_file = os.path.join(tmp_path, 'combine.c') 
    c_file_save = os.path.join(store_path, f'func{input_asm_prompt_id}.c')
    executable = os.path.join(tmp_path, 'combine')
    if os.path.exists(executable_onlyfunc):
        os.remove(executable_onlyfunc)
    if os.path.exists(executable):
        os.remove(executable)
    
    with open(c_file_onlyfunc,'w') as f:
        f.write(c_onlyfunc)
    with open(c_file,'w') as f:
        f.write(c_combine)
    with open(c_file_save,'w') as f:
        f.write(c_combine)

    # Compile the C program to an assembly
    if not _ISNEW:
        compile_command = f'gcc -S {c_file_onlyfunc} -o {executable_onlyfunc} -lm'
        try:
            subprocess.run(compile_command, shell=True, check=True)
            flag_compile = 1
        except:
            return flag_compile, flag_run
    else:
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
    if not _ISNEW:
        compile_command = f'gcc {c_file} -o {executable} -lm'
        try:
            subprocess.run(compile_command, shell=True, check=True)
            flag_compile = 1
        except:
            return flag_compile, flag_run
    else:
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
        # process = subprocess.run(run_command, shell=True, check=True,capture_output=True, timeout=_DEFAULT_CMD_TIMEOUT)
        process = subprocess.run(run_command, shell=False, check=True,capture_output=True, timeout=_DEFAULT_CMD_TIMEOUT)
        flag_run = 1
    except subprocess.TimeoutExpired:
        pass
    except subprocess.CalledProcessError as e:
        pass
    except Exception as e:
        pass
    return flag_compile, flag_run
        # try:
        #     stdout, stderr = _run_command(run_command)
        # except subprocess.CalledProcessError as e:
        #     pass
        # except Exception as e:
        #     logger.INFO("Cmd Error:\n{}".format(e))
        #     return flag_compile, flag_run
        # flag_run = 1
        # if stderr:
        #     logger.INFO("Run Error:\n{}".format(stderr))
        # return flag_compile, flag_run


OPT = ["O0", "O1", "O2", "O3"]  # Optimization states
with open(data_file, 'r') as f:
    data_all = json.load(f)
NUM = int(len(data_all)/4)
num_compile = {"O0":0, "O1":0, "O2":0, "O3":0}
num_run = {"O0":0, "O1":0, "O2":0, "O3":0}

result_dict = {}
for f in os.listdir(input_path):
    file_path = os.path.join(input_path, f)
    number = ''.join(filter(str.isdigit, f))
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if json_obj.get("mode") == _BITWIDE:
                opts_value = json_obj.get("opts")
                key = f"{number}_{opts_value}"
                val = json_obj.get("c_func_decompile")
                # val = json_obj.get("prediction_dec")
                if not _ISNEW:
                    val = val.replace("```c", "").replace("```", "")
                result_dict[key] = val

# c_func_decompiled_results = []
compile_func = []
run_func = []

for idx in trange(len(data_all)):
    task_id = data_all[idx]['task_id']
    c_func = data_all[idx]['c_func']
    c_test = data_all[idx]['c_test']
    # input_asm_prompt = data_all[idx]['input_asm_prompt']
    opt_state = data_all[idx]['type']

    input_asm_prompt_id = str(task_id) + '_' + opt_state
    c_func_decompile = result_dict[input_asm_prompt_id]
    logger.INFO("handle func {}".format(input_asm_prompt_id))
    flag_compile,flag_run = evaluate_func(c_func,c_test,c_func_decompile,input_asm_prompt_id)
    if flag_compile == 1:
        compile_func.append(input_asm_prompt_id)
    if flag_run == 1:
        run_func.append(input_asm_prompt_id)
    num_compile[opt_state]+=flag_compile
    num_run[opt_state]+=flag_run
# with open('c_func_decompiled_results.json', 'w') as json_file:
#     json.dump(c_func_decompiled_results, json_file)
metric_file = os.path.join(res_path, f'metric.txt')
with open(metric_file, 'a') as f:
    new_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    f.write('{} {} {} {}\n'.format(new_time, _MODEL, _BITWIDE, ('new' if _ISNEW else 'old')))
    for opt_state in num_compile.keys():
        f.write('opt:{} compile rate:{:.4f} run_rate:{:.4f}\n'.format(opt_state,num_compile[opt_state]/NUM,num_run[opt_state]/NUM))
    f.write('\n')
compile_func_file = os.path.join(res_path, f'compile_{_BITWIDE}.txt')
with open(compile_func_file, 'w') as f:
    f.write("{}:\n".format(_MODEL))
    f.write(str(compile_func))
run_func_file = os.path.join(res_path, f'reexecu_{_BITWIDE}.txt')
with open(run_func_file, 'w') as f:
    f.write("{}:\n".format(_MODEL))
    f.write(str(run_func))

# 下面是为了收集所有模型的评估结果
# ========================================================================================================================
all_metrics_file_path = args.all_metrics
with open(all_metrics_file_path, 'a') as amf:
    new_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    amf.write('{} {} {} {}\n'.format(new_time, _MODEL, _BITWIDE, ('new' if _ISNEW else 'old')))
    # 先计算平均值
    compile_rates = []
    run_rates = []
    for opt_state in num_compile.keys():
        compile_rate = num_compile[opt_state]/NUM
        run_rate = num_run[opt_state]/NUM
        compile_rates.append(compile_rate)
        run_rates.append(run_rate)
        amf.write('opt:{} compile rate:{:.4f} run_rate:{:.4f}\n'.format(opt_state, compile_rate, run_rate))
        # amf.write('opt:{} compile rate:{:.4f} run_rate:{:.4f}\n'.format(opt_state,num_compile[opt_state]/NUM,num_run[opt_state]/NUM))
    # 计算并写入平均值
    avg_compile = sum(compile_rates) / len(compile_rates) if compile_rates else 0
    avg_run = sum(run_rates) / len(run_rates) if run_rates else 0
    amf.write('opt:avg compile rate:{:.4f} run_rate:{:.4f}\n'.format(avg_compile, avg_run))
    amf.write('\n')
# amf.close()