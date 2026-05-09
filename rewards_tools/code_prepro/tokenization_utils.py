# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
#from sacrebleu import tokenize_v14_international
import sacrebleu

# IMPORTED
NEWLINE_TOKEN = "NEWLINE_TOKEN"


# IMPORTED
class ind_iter(object):
    def __init__(self, len):
        self.i = 0
        self.len = len

    def next(self):
        self.i += 1
        if self.i > (self.len - 1):
            raise StopIteration

    def prev(self):
        self.i -= 1
        if self.i < 0:
            raise StopIteration


# IMPORTED
def process_string(tok, char2tok, tok2char, is_comment, do_whole_processing=True):
    """
    | 参数                    | 作用                                                          |
    | --------------------- | ----------------------------------------------------------- |
    | `tok`                 | 当前要被处理的**字符串片段**（可能是注释、字符串、整行代码）                            |
    | `char2tok`            | **普通字符 → 特殊占位符** 的映射表（例如 `{' ':'▁'}`）                       |
    | `tok2char`            | **占位符 → 原字符** 的反向表                                          |
    | `is_comment`          | 告诉函数“我现在在处理注释”，会多做**压缩空格/连续字符**的清洗。                         |
    | `do_whole_processing` | `True` = 做完整 tokenize；`False` = 只做**最小还原**（detokenize 阶段用）。 |
    该函数是 “字符串→可逆占位符→（可选）分词→干净 token 串” 的双向转换枢纽；
    """
    if not (do_whole_processing or is_comment):
        return tok.replace("\n", "\\n").replace("\r", "")

    if is_comment:
        tok = re.sub(" +", " ", tok)
        tok = re.sub(r"(.)\1\1\1\1+", r"\1\1\1\1\1", tok)
        if len(re.sub(r"\W", "", tok)) < 2:
            return ""
    tok = replace_general_string_tok(tok)
    tok = replace_tokens(tok, char2tok)
    if tok.strip().startswith("STOKEN00"):
        if " STRNEWLINE " in tok:
            tok = tok.replace(" STRNEWLINE ", " ENDCOM", 1)
        else:
            tok += " ENDCOM"
    if not do_whole_processing:
        tok = replace_tokens(
            tok, {f" {key} ": value for key, value in tok2char.items()}
        )
        tok = (
            tok.replace(" ▁ ", " ")
            .replace(" TABSYMBOL ", "\t")
            .replace("\\r", "")
            .replace(" STRNEWLINE ", "\\n")
        )
        return tok

    tok = re.sub(" +", " ", tok)
    tok = sacrebleu.tokenize_v14_international(tok)
    tok = re.sub(" +", " ", tok)
    tok = tok.replace("\r", "")
    for special_token, char in tok2char.items():
        tok = tok.replace(special_token, char)
    return tok


def tokenize_string(s: str):
    # 把任意代码/文本 → SacreBLEU 分词 → 按空格拆成 List[str]，供模型喂入。
    return process_string(
        s, char2tok=dict(), tok2char=dict(), is_comment=False, do_whole_processing=True
    ).split(" ")


def detokenize_string(s):
    # 列表处理成空格隔开token的string
    assert isinstance(s, str) or isinstance(s, list)
    if isinstance(s, list):
        s = " ".join(s)
    return s.replace(" ", "").replace("▁", " ")


# IMPORTED
def replace_tokens(tok, dictionary):
    for char, special_token in dictionary.items():
        tok = tok.replace(char, special_token)
    return tok


# IMPORTED
def replace_general_string_tok(tok):
    return (
        tok.replace(" ", " ▁ ")
        .replace("\n", " STRNEWLINE ")
        .replace("\t", " TABSYMBOL ")
    )


# IMPORTED
def indent_lines(lines):
    prefix = ""
    for i, line in enumerate(lines):
        line = line.strip()
        if re.match("CB_COLON|CB_COMA|CB_", line):
            prefix = prefix[2:]
            line = prefix + line
        elif line.endswith("OB_"):
            line = prefix + line
            prefix += "  "
        else:
            line = prefix + line
        lines[i] = line
    untok_s = "\n".join(lines)
    return untok_s


"""
原始源码
   ↓  tokenize_string
占位符 + SP 分词序列
   ↓  model 生成 / 处理
占位符序列
   ↓  detokenize_string + indent_lines
还原 + 缩进后的可编译源码
"""