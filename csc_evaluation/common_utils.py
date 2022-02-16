import os
import json
import sys
from pathlib import Path
import re
from pytrie import StringTrie
import logging


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
    # 如果是使用pyinstaller打包后的执行文件，则定位到执行文件所在目录
    if hasattr(sys, 'frozen'):
        return os.path.join(os.path.dirname(sys.executable))
    # 其他情况则定位至项目根目录
    return os.path.join(os.path.dirname(__file__), '..')


def get_abs_path(*name):
    return os.path.abspath(os.path.join(get_main_dir(), *name))


def read_tagging_data(file_paths, max_sent_length=-1, test_size=-1):
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


def get_trie(vocab_filename):
    with open(vocab_filename, "r", encoding="utf-8") as f:
        data = f.read()
        vocab = data.split("\n")
    trie = StringTrie()
    for word in vocab:
        if word != "":
            trie[word] = word

    return trie


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


def set_logger(logger, log_filename, file_format='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s', file_mode='a', stream_format='%(asctime)s - %(levelname)s - %(message)s'):
    '''
       Log to both screen and file. The screen level is INFO while the file level is DEBUG
    '''

    logger.setLevel(logging.DEBUG)

    format_sh = logging.Formatter(stream_format, datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(format_sh)
    logger.addHandler(sh)

    if log_filename is not None:
        if not os.path.exists(os.path.dirname(log_filename)):
            os.makedirs(os.path.dirname(log_filename))
        format_fh = logging.Formatter(file_format, datefmt="%Y/%m/%d %H:%M:%S")
        fh = logging.FileHandler(log_filename, mode=file_mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(format_fh)
        logger.addHandler(fh)
