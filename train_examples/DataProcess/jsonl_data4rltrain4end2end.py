#!/usr/bin/env python3
"""
从文件夹内的所有 JSONL 文件中读取数据，按优化级别分组筛选，
优先选择节点数多或 token 长的样本，输出到一个 JSONL 文件。

Usage:
    python select_from_folder.py --input_dir /path/to/jsonls --output selected.jsonl
"""

import argparse
import json
import os
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm
from string import Template
from typing import Dict, List, Union
abs = "/workspace/"

TOKENIZER_NAME = abs + "qwen3-train/models/instruct-6.7b"
MAX_LEN = 2048 # 6.7B对应2048，30B对应4096
INPUT_DIR =  abs + "trl-main/datasets/alloriginalrl"  #
OUTPUT_FILE = abs + f"trl-main/datasets/train/rl{MAX_LEN}train4end2end.jsonl"
# 若网络限制，可替换为本地路径或 deepseek-coder
# TOKENIZER_NAME = "deepseek-ai/deepseek-coder-6.7b-base"

# 每组需要选取的数量
LIMITS = {
    'O3': 4000,
    'O2': 3000,
    'O1': 2000,
    'O0': 1000,
}


def load_tokenizer():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

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
    return {
        "prompt": messages,
        "source": datapoint["sourcecode"],
        "dependency": datapoint["dependencies"],
    }


def dict_to_string(data: Union[Dict, List], indent: int = 0) -> str:
    """Convert dictionary/list to a formatted string representation."""
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{'  ' * indent}{key}:")
                lines.append(dict_to_string(value, indent + 1))
            else:
                lines.append(f"{'  ' * indent}{key}: {value}")
        return '\n'.join(lines)
    elif isinstance(data, list):
        lines = []
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                lines.append(f"{'  ' * indent}[{i}]:")
                lines.append(dict_to_string(item, indent + 1))
            else:
                lines.append(f"{'  ' * indent}[{i}]: {item}")
        return '\n'.join(lines)
    else:
        return str(data)


def count_tokens_simple(tokenizer, data) -> int:
    """Count tokens in a text string."""
    text = dict_to_string(data)
    if not text or not isinstance(text, str):
        return 0

    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def process_files(input_dir, output_path):
    tokenizer = load_tokenizer() # get the tokenizer
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    # 收集所有有效记录
    groups = defaultdict(list)  # opts -> list of (nodenum, token_len, record)
    total_read = 0
    filtered_too_long = 0
    missing_nodenum = 0
    missing_asm = 0

    # 遍历文件夹内所有 .jsonl 文件
    for i, filename in enumerate(tqdm(jsonl_files, desc="处理文件", unit="file", leave=False)):
        filepath = os.path.join(input_dir, filename)
        print(f"Processing {filename} ...")
        with open(filepath, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line) # get the data from each line
                except json.JSONDecodeError:
                    print(f"Warning: invalid JSON in {filepath} line {line_num}, skipping")
                    continue

                if record.get("mode") == "64":
                    opts = record.get('opts')
                    if opts not in LIMITS:
                        continue

                    # 获取节点数
                    assemgraph = record.get('assemgraph_com') # get the assembly data
                    if not assemgraph:
                        missing_nodenum += 1
                        continue
                    nodenum = assemgraph.get('nodenum')
                    if nodenum is None:
                        missing_nodenum += 1
                        continue

                    # 获取汇编文本
                    # 构造prompt，获取新的record
                    record = chatmessage(record)
                    # 计算 token 长度
                    token_len = count_tokens_simple(tokenizer, record["prompt"])
                    if token_len >= MAX_LEN: # 重新跑一个4096的针对1.3和6.7B模型
                        filtered_too_long += 1
                        continue

                    groups[opts].append((nodenum, token_len, record))
                    total_read += 1
                    if total_read % 1000 == 0:
                        print(f"  Processed {total_read} valid records so far...")

    print(f"\nSummary:")
    print(f"  Total valid records: {total_read}")
    print(f"  Filtered (token>={MAX_LEN}): {filtered_too_long}")
    print(f"  Missing nodenum: {missing_nodenum}")
    print(f"  Missing asm: {missing_asm}")

    # 选择每组 top K
    selected = []
    for opts, limit in LIMITS.items():
        items = groups.get(opts, [])
        print(f"\nopts={opts}: found {len(items)} samples, need {limit}")
        # 按 nodenum 降序，再按 token_len 降序
        items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        take = min(limit, len(items))
        selected.extend(items[i][2] for i in range(take))
        print(f"  selected {take} samples")

    # 写入输出文件
    with open(output_path, 'a', encoding='utf-8') as f:
        for record in selected:
            f.write(json.dumps(record) + '\n')

    print(f"\nTotal selected: {len(selected)} samples written to {output_path}")


if __name__ == '__main__':
    process_files(INPUT_DIR, OUTPUT_FILE)