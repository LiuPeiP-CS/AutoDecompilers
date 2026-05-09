import json, time, shutil
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import re
from tqdm import trange
from string import Template
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["NCCL_P2P_DISABLE"] = '1'
#os.environ["NCCL_IB_DISABLE"] = '1'
# 指定文件夹路径
input_path = 'humaneval/input'
_INDEX = '0'
_ISIDA = True
_INTERVAL = 1


parser = argparse.ArgumentParser()
parser.add_argument('--workdir', type=str, help="the output_dir path", required=False)
parser.add_argument("-s", type=int, default=0, help="start")
parser.add_argument("-x", type=int, default=164, help="interval")
parser.add_argument("-b", required=True, help="bit-wide")
parser.add_argument("-i", required=True, help="index")
parser.add_argument('--model_path', type=str, help="the llm path", required=False)
# parser.add_argument("-f", action="store_true", help="first ?")
args = parser.parse_args()
pro_path = args.workdir
if not os.path.exists(pro_path):
    print(f'Error: workdir {pro_path} not exists!')
    quit(-1)
if not os.path.exists(input_path):
    print(f'Error: input {input_path} not exists!')
    quit(-1)

if args.i:
    _INDEX = args.i
if args.b in ['64', '32']:
    _BITWIDE = args.b
else:
    print(f'Error bit-wide {args.b} !')
    quit(-1)
# is_first = True if args.f else False
TOTAL = 164

start_index = args.s * _INTERVAL
end_index = min(start_index + args.x * _INTERVAL, TOTAL)

if args.s == 0 and args.x == TOTAL:
    module = ''
else:
    module = f'_{args.s}-{args.s + args.x}'

print("[+] {} to {} func".format(start_index, end_index))

ymd = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
analysis_path = os.path.join(pro_path, 'analysis')
res_path = os.path.join(pro_path, f'humaneval_{_INDEX}')
result_path = os.path.join(res_path, f'decom_{_BITWIDE}')
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)
if not os.path.exists(res_path):
    os.makedirs(res_path)
if start_index == 0:
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)
elif not os.path.exists(result_path):
    print(f'Error: output {result_path} not exists!')
    quit(-1)


os.environ["TOKENIZERS_PARALLELISM"] = "false"



def prompt4chat(datapoint): # 该prompt中没有添加info
    # datapoint["data"] = json.loads(datapoint["data"])
    # datapoint["assemgraph_com"] = json.loads(datapoint["assemgraph_com"])

    add_info = {
        "instruction set architecture": datapoint['arch'],
        "bit width": datapoint['mode'],
        "compiler optimization level": datapoint['opts']}
    add_info = json.dumps(add_info)

    data_mapping = {
        "stack variables (size and relative offset)": datapoint['data']['param'],
        "read-only constants (size and value)": datapoint['data']['.rodata'],
        "initialized global/static data (size and initial value)": datapoint['data']['.data'],
        "uninitialized global/static data (size and count)": datapoint['data']['.bss'],
    }
    data_mapping = json.dumps(data_mapping)

    if datapoint['assemgraph_com']['nodenum'] == 1:
        # 构造节点为1的prompt,表示由ida产生但是没有cfg graph
        assembly_code = datapoint['asm_ida_com'] # ida产生的反汇编代码，只有一个cfg block，没有图

        prompt4input = Template("""Please understand the following assembly code and data mapping table (defines the correspondence between data labels and their actual values), and perform decompilation into corresponding high-level C source code.

        - The assembly code:
        ```Assembly
        $assembly_code
        ```

        - The data mapping table:
        ```Json
        $data_mapping
        ```

        The high-level C source code is: """)
        rprompt4input = Template.substitute(prompt4input, assembly_code=assembly_code, data_mapping=data_mapping)


    elif datapoint['assemgraph_com']['nodenum'] > 1:

        cfg_prompt = {
            "cfg_blocks (names and corresponding instructions)": datapoint['assemgraph_com']['nodes'],
            "edges between two connected cfg_blocks": datapoint['assemgraph_com']['edges'], # (1,3)表示nodes对应的列表中，第1个节点和第3个节点之间存在关联边，节点的索引从0开始
            "cfg_block count": datapoint['assemgraph_com']['nodenum'] # 该函数体所能抽取出来的block的数量
            }
        cfg_prompt = json.dumps(cfg_prompt)
        prompt4input = Template("""Please understand the following control flow graph and data mapping table (defines the correspondence between data labels and their actual values), and perform decompilation into corresponding high-level C source code.

        - The control flow graph:
        ```Json
        $cfg_prompt
        ```

        - The data mapping table:
        ```Json
        $data_mapping
        ```

        The high-level C source code is: """)
        rprompt4input = Template.substitute(prompt4input, cfg_prompt=cfg_prompt, data_mapping=data_mapping)

    else:
        raise ValueError
    # print("there is OK after template")
    # datapoint["data"] = json.dumps(datapoint["data"], ensure_ascii=False)
    # datapoint["assemgraph_com"] = json.dumps(datapoint["assemgraph_com"], ensure_ascii=False)
    return rprompt4input

def chatmessage(datapoint):
    user_content = prompt4chat(datapoint)
    system_message = {
        "role": "system",
        "content": "You are a decompilation expert. Your task is to analyze assembly code or control flow graphs along with data mapping information, and generate accurate high-level C source code."
    }

    messages = [
        system_message,
        {"role": "user", "content": user_content},
        # {"role": "assistant", "content": datapoint["sourcecode"]} # 推理时不需要
    ]

    # 返回符合 Qwen3 chat 格式的数据
    return messages


tokenizer = AutoTokenizer.from_pretrained(args.model_path)
llm = LLM(
    model=args.model_path,
    dtype="float16",
    tokenizer=args.model_path,
    trust_remote_code=True,
    tensor_parallel_size = 1, # 单卡为1，X张卡为X
    # gpu_memory_utilization=1,
    max_num_seqs=2,  # 减少并行序列数
    max_model_len=12288,  # 限制模型长度
    gpu_memory_utilization=0.85,
    disable_log_stats = True,
    disable_custom_all_reduce  = True,
    # worker_use_ray = False,
    enforce_eager = True,
)
# 配置生成参数（可复用原有参数）
sampling_params = SamplingParams(
    max_tokens=8192,  # 对应原来的max_new_tokens
    temperature=0.0,  # 保持确定性输出
    top_p=1.0,
    stop=[tokenizer.eos_token],  # 使用原tokenizer的结束符
)
# # 重构主处理循环
# all_prompts = []
# task_mappings = []

print('Model Loaded!')

for ind in trange(start_index, end_index):
    file_path = os.path.join(input_path, f'func{ind}.jsonl')
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if json_obj.get("mode") == _BITWIDE:
                opts_value = json_obj.get("opts")
                input_asm_prompt_id = f"{ind}_{opts_value}"

                input_asm_prompt = chatmessage(datapoint=json_obj)
                outputs = llm.chat([input_asm_prompt], sampling_params)
                c_func_decompile = outputs[0].outputs[0].text.strip()
                json_obj['c_func_decompile'] = c_func_decompile

                jsonl_filename = os.path.join(result_path, f'func{ind}.jsonl')
                with open(jsonl_filename, 'a') as jsonl_file:
                    jsonl_file.write(json.dumps(json_obj) + '\n')
