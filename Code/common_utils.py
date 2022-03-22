import os
import json
import sys
from pathlib import Path
import re
import unicodedata
from transformers.trainer_utils import set_seed
import torch


def str2bool(arg):
    true_set = {'yes', 'true', 't', 'y', '1'}
    if not arg:
        return False

    arg = (str(arg)).lower()
    if arg in true_set:
        return True
    return False


def load_json(fp):
    if not os.path.exists(fp):
        return dict()

    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)


def dump_json(obj, fp):
    try:
        fp = os.path.abspath(fp)
        if not os.path.exists(os.path.dirname(fp)):
            os.makedirs(os.path.dirname(fp))
        with open(fp, 'w', encoding='utf8') as f:
            json.dump(obj, f, ensure_ascii=False,
                      indent=4, separators=(',', ':'))
        print(f'json文件保存成功，{fp}')
        return True
    except Exception as e:
        print(f'json文件{obj}保存失败, {e}')
        return False


def get_main_dir():
    # 如果是使用pyInstaller打包后的执行文件，则定位到执行文件所在目录
    if hasattr(sys, 'frozen'):
        return os.path.join(os.path.dirname(sys.executable))
    # 其他情况则定位至项目根目录
    return os.path.join(os.path.dirname(__file__), '..')


def get_abs_path(*name):
    return os.path.abspath(os.path.join(get_main_dir(), *name))


def read_tagging_data(file_paths, max_sent_length=-1):
    token_docs = []
    tag_docs = []
    for file_path in file_paths:
        file_path = Path(file_path)

        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)

        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                items = line.split('\t')
                tokens.append(items[0])
                tags.append(items[1])
            if 0 < max_sent_length < len(tokens):
                tokens = tokens[:max_sent_length]
                tags = tags[:max_sent_length]
            token_docs.append(tokens)
            tag_docs.append(tags)
    return token_docs, tag_docs


def read_table_file(filename, output_indexes, sep='\t'):
    outputs = []
    require_len = max(output_indexes) + 1
    with open(filename, encoding='utf-8') as f:
        for line in f:
            items = line.split(sep)
            if len(items) < require_len:
                continue
            output = []
            for index in output_indexes:
                output.append(items[index].strip())
            outputs.append(output)
    return outputs


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)  #
            or (0x20000 <= cp <= 0x2A6DF)  #
            or (0x2A700 <= cp <= 0x2B73F)  #
            or (0x2B740 <= cp <= 0x2B81F)  #
            or (0x2B820 <= cp <= 0x2CEAF)  #
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def whitespace_filter(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return "".join(tokens).lstrip().rstrip()


def filter_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def setSeed(seed):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}
