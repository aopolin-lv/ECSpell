import re
import string
from common_utils import is_chinese_char


NUM_CHAR = list("一二三四五六七八九十1234567890")
PREFIX = list(")”")
SYMBOLS = {'}': '{', ']': '[', ')': '(', '>': '<', '》': '《', '”': '“', '）': '（'}
SYMBOLS_L, SYMBOLS_R = SYMBOLS.values(), SYMBOLS.keys()


def __merge_symmetry(sentences, symmetry=('“', '”')):
    '''合并对称符号，如双引号'''
    effective_ = []
    merged = True
    for index in range(len(sentences)):
        if symmetry[0] in sentences[index] and symmetry[1] not in sentences[index]:
            merged = False
            effective_.append(sentences[index])
        elif symmetry[1] in sentences[index] and not merged:
            merged = True
            effective_[-1] += sentences[index]
        elif symmetry[0] not in sentences[index] and symmetry[1] not in sentences[index] and not merged:
            effective_[-1] += sentences[index]
        else:
            effective_.append(sentences[index])

    return [i.strip() for i in effective_ if len(i.strip()) > 0]


def to_sentences(paragraph):
    paragraph = re.sub(r'\s+', ' ', paragraph)
    """由段落切分成句子"""
    sentences = re.split(r"(？|。|！|\…\…)", paragraph)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    sentences = [i.strip() for i in sentences if len(i.strip()) > 0]

    for j in range(1, len(sentences)):
        if sentences[j][0] == '”':
            sentences[j - 1] = sentences[j - 1] + '”'
            sentences[j] = sentences[j][1:]

    return __merge_symmetry(sentences)


def tokenize(sentence):
    tokens = []
    for token in sentence:
        if is_ascii(token):
            tokens.append(token)
        else:
            tokens.append(' ' + token + ' ')
    sentence = ''.join(tokens)
    tokens = sentence.split()
    return tokens


def is_ascii(ch):
    return ord(ch) < 255


def is_chinese(ch):
    return '\u4e00' <= ch <= '\u9fff'


punctuations = set(string.punctuation + '。，《》？；：‘“”’【】、·……')


def is_punctuation(ch):
    return ch in punctuations


def check(s):
    """
    Check if the brackets number is correct
    Parameters:
        s: sentence
    Returns:
    """
    arr = []
    for c in s:
        if c in SYMBOLS_L:
            # 左符号入栈
            arr.append(c)
        elif c in SYMBOLS_R:
            # 右符号出栈否则匹配失败
            if arr and arr[-1] == SYMBOLS[c]:
                arr.pop()
            else:
                return False

    return True


def is_chinese_word(word):
    for c in word:
        if not is_chinese_char(ord(c)):
            return False
    return True


def chinese_char_count(text):
    count = 0
    for c in text:
        if is_chinese_char(ord(c)):
            count += 1
    return count


def count_blacket(text: str):
    if text.startswith("(") and text.endswith(")") and "(" not in text[1: -1] and ")" not in text[1: -1]:
        return True
    if text.startswith("（") and (text.endswith("）") or text.endswith("）。")) and "（" not in text[1: -1] and "）" not in text[1: -1]:
        return True
    return False
