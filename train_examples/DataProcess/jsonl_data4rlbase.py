#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare dataset for continual RL pre-training and conversational LLM (-instruct) reasoning
"""

import argparse
import json
import math
import os
import time
# 该脚本纯粹是为了补全数据
import os
import json
import sys
from tqdm import tqdm
from string import Template

def is_empty_dict(d):
    return isinstance(d, dict) and not d


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

def process_jsonl_folder(in_folder_path, out_folder_path):
    output_file = os.path.join(out_folder_path, f"rl_data.jsonl")
    open(output_file, "w", encoding="utf-8").close()
    data_count = 0

    jsonl_files = [f for f in os.listdir(in_folder_path) if f.endswith(".jsonl")]
    file_count = len(jsonl_files)
    # for filename in tqdm(jsonl_files, desc="处理文件", unit="file", leave=False):
    for i, filename in enumerate(tqdm(jsonl_files, desc="处理文件", unit="file", leave=False)):
        input_path = os.path.join(in_folder_path, filename)
        with open(input_path, "r", encoding="utf-8") as infile, \
                open(output_file, "a", encoding="utf-8") as outfile:

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    processed_obj = chatmessage(obj)
                    json.dump(processed_obj, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    data_count += 1
                except json.JSONDecodeError as e:
                    print(f"跳过无法解析的行 in {filename}: {e}")

        print(f"已经处理完第 {i+1}/{file_count} 个文件")
    print(f"所有文件已处理完毕，一共有 {data_count} 条数据！")


def get_subfolders(folder_path):
    """获取文件夹下的所有子文件夹"""
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return []

    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)

    return subfolders

if __name__ == "__main__":
    in_path_dir = "/XXX/datasets/30samples4test" # 输入数据的绝对路径，包含所有的jsonl文件
    out_path = "/XXX/datasets/30samples4test" # 输出数据的文件夹地址，和in_path一级
    process_jsonl_folder(in_path_dir, out_path)