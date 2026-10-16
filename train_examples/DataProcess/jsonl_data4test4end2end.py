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
import os
import json
import sys
from tqdm import tqdm
from string import Template
from collections import defaultdict
from transformers import AutoTokenizer
from typing import Dict, List, Union

abs_path = "/workspace/"
TOKENIZER_NAME = abs_path + "qwen3-train/models/instruct-6.7b"
INPUT_DIR = abs_path + "trl-main/datasets/test/eval_exebench_data.jsonl" 
OUTPUT_FILE = abs_path + "trl-main/datasets/test/TestEBEnd2end.jsonl"   
MAX_LEN = 4096

def is_empty_dict(d):
    return isinstance(d, dict) and not d


def prompt4chat(datapoint):  
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
        
        assembly_code = datapoint['asm_ida_com']

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
            "edges between two connected cfg_blocks": datapoint['assemgraph_com']['edges'], 
            "cfg_block count": datapoint['assemgraph_com']['nodenum'] 
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
    ]

    return {
        "prompt": messages,
        "source": datapoint["sourcecode"],
        "dependency": datapoint["dependencies"],
        "orig_data": {k: v for k, v in datapoint.items() if k != "sourcecode" and k != "dependencies"},
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
    # total_token = total_token+len(tokens)
    return len(tokens)


def load_tokenizer():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


def process_jsonl_folder(in_folder_path, output_file):
    tokenizer = load_tokenizer() # get the tokenizer
    # output_file = os.path.join(out_folder_path, f"rl_data.jsonl")
    open(output_file, "w", encoding="utf-8").close()
    data_count = 0

    if os.path.isfile(in_folder_path):
        jsonl_files = [in_folder_path]
    else:
        jsonl_files = [f for f in os.listdir(in_folder_path) if f.endswith(".jsonl")]
    file_count = len(jsonl_files)
    total_token = 0
    for i, filename in enumerate(tqdm(jsonl_files, desc="deal with the file", unit="file", leave=False)):
        input_path = os.path.join(in_folder_path, filename)
        with open(input_path, "r", encoding="utf-8") as infile, \
                open(output_file, "a", encoding="utf-8") as outfile:

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    if obj.get("mode") == "64":
                        processed_obj = chatmessage(obj)
                        token_len = count_tokens_simple(tokenizer, processed_obj["prompt"])
                        total_token = total_token + token_len
                        if token_len >= MAX_LEN:
                            continue
                        json.dump(processed_obj, outfile, ensure_ascii=False)
                        outfile.write("\n")
                        data_count += 1
                except json.JSONDecodeError as e:
                    print(f"Skipping unparsable line in {filename}: {e}")

        print(f"Finished processing file {i+1}/{file_count}")
    print(f"All files have been processed. Total data entries: {data_count}!")
    print(total_token)


def get_subfolders(folder_path):
    if not os.path.exists(folder_path):
        print(f"The dir does not exist: {folder_path}")
        return []

    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)

    return subfolders

if __name__ == "__main__":

    process_jsonl_folder(INPUT_DIR, OUTPUT_FILE)
