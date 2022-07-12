import re
import string
import unicodedata


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
            # 右符号要么出栈，要么匹配失败
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


def is_chinese_phrase(phrase):
    for c in phrase:
        if not is_chinese_char(ord(c)):
            return False
    return True
