import os
import re

import tokenizers
from tqdm import trange, tqdm
from typing import Union, List

from pytrie import StringTrie
from pypinyin import pinyin, Style
from pypinyin.style.tone import ToneConverter
from pypinyin.style._utils import get_initials, get_finals
from pypinyin.style._constants import _INITIALS, _INITIALS_NOT_STRICT

from transformers import AutoTokenizer
from tokenizers.implementations.bert_wordpiece import BertWordPieceTokenizer
from Code.common_utils import dump_json, load_json, read_table_file, is_chinese_char, read_tagging_data


_re_number = re.compile(r'\d')


class Processor:
    def __init__(self, vocab_file, model_name="Transformers/glyce"):
        self.vocab_processor = VocabProcessor(vocab_file)
        self.pinyin_processor = PinyinProcessor(model_name)


class VocabProcessor:
    def __init__(self, vocab_filename, index=None):
        if index is None:
            index = [0, 1]
        self.filename = vocab_filename
        self.vocab = read_table_file(self.filename, index)
        if not self.vocab:
            self.vocab = read_table_file(self.filename, [0])
            self.add_pinyin2vocab()
            self.vocab = read_table_file(self.filename, [0, 1])
        if index == [0, 1]:
            self.word_trie = self._get_trie("word")
            self.py_trie = self._get_trie("pinyin")

    def get_vocab_length(self, sent):
        """
        Calculate the additional score according the user vocab and sentence.
        Parameters     :  sent

        Returns  : count_len
        """
        count_len = 0
        margin = 0
        while margin < len(sent):
            sub_str = self.word_trie.longest_prefix(sent[margin:], default="N/A")
            if sub_str == "N/A":
                margin += 1
                continue
            count_len += len(sub_str)
            margin += len(sub_str)
        return count_len

    def _get_trie(self, mode="word"):
        trie = StringTrie()
        word_index = 0
        pinyin_index = 1
        for line in self.vocab:
            word = line[word_index]
            pyin = line[pinyin_index]
            if word != "":
                if mode == "pinyin":
                    trie[pyin + " "] = word
                elif mode == "word":
                    trie[word] = word
        return trie

    def add_pinyin2vocab(self):
        outputs = []
        for index, line in enumerate(self.vocab):
            word = line[0]
            word_pinyin_list = pinyin(word, style=Style.NORMAL)
            word_pinyin = ""
            for py in word_pinyin_list:
                word_pinyin = word_pinyin + py[0] + " "
            outputs.append([word, word_pinyin])
        with open(self.filename, "w", encoding="utf-8") as f:
            for line in outputs:
                f.write("\t".join(line))
                f.write("\n")


class PinyinProcessor():
    def __init__(self, model_name, max_length: int = 512, strict=False) -> None:
        super().__init__()
        self.vocab_file = os.path.join(model_name, 'vocab.txt')
        self.config_path = os.path.join(model_name, 'pinyin_config')
        self.pinyin_vocab = os.path.join(self.config_path, 'pinyin_vocab.txt')

        self.max_length = max_length
        self.tokenizer = BertWordPieceTokenizer(self.vocab_file)
        self.strict = strict
        self.pinyin2id = load_json(os.path.join(self.config_path, 'pinyin2id.json'))
        self.id2pinyin = load_json(os.path.join(self.config_path, 'id2pinyin.json'))

        self.sm_list = _INITIALS if self.strict else _INITIALS_NOT_STRICT
        self.ym_list = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia',
                        'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iu', 'o', 'ong', 'ou',
                        'u', 'ua', 'uai', 'uan', 'uang', 'ue', 'ui', 'un', 'uo', 'v', 've']

    def convert_ch_to_pinyin_id(self, char):
        """
        Args     :
        Returns  :
        """
        unk_id = self.id2pinyin['[UNK]']

        if not is_chinese_char(ord(char)):
            char_py, sm, ym, sd = '[UNK]', '[UNK]', '[UNK]', 6

        char_py = pinyin(char, style=Style.TONE3, heteronym=False, neutral_tone_with_five=True)[0][0]
        sm_raw = get_initials(char_py, self.strict)
        sm = sm_raw if sm_raw != '' else '[PAD]'
        ym = get_finals(char_py, self.strict)[:-1]
        sd = char_py[-1]
        return [self.id2pinyin.get(char_py, unk_id), self.id2pinyin.get(sm, unk_id), self.id2pinyin.get(ym, unk_id), int(sd)]

    def convert_sentence_to_pinyin_ids(
            self,
            sentence: Union[str, list],
            tokenizer_output: tokenizers.Encoding
    ) -> List[List[int]]:
        if not isinstance(sentence, list):
            raise "sentence must be list form"

        unk_id, pad_id = self.id2pinyin['[UNK]'], self.id2pinyin['[PAD]']
        # batch mode
        if len(sentence) != 0 and isinstance(sentence[0], list):
            all_pinyin_ids = []

            for i, sent in enumerate(tqdm(sentence, desc="Generate pinyin ids")):
                if len(tokenizer_output.tokens(i)) != len(sent) + 2:
                    pinyin_ids = []
                    for j in range(len(tokenizer_output.tokens(i))):
                        pinyin_ids.append([pad_id, pad_id, pad_id, pad_id])
                    all_pinyin_ids.append(pinyin_ids)
                    continue
                # filter no-chinese char
                for idx, c in enumerate(sent):
                    if len(c) > 1 or not is_chinese_char(ord(c)):
                        sent[idx] = "#"

                assert len(tokenizer_output.tokens(i)) == len(sent) + 2

                pinyin_list = pinyin(sent, style=Style.TONE3, heteronym=False, neutral_tone_with_five=True, errors=lambda x: [["not chinese"] for _ in x])
                pinyin_ids = []
                pinyin_ids.append([pad_id, pad_id, pad_id, pad_id])
                for item in pinyin_list:
                    pinyin_string = item[0]
                    if pinyin_string == "not chinese":
                        pinyin_string, sm, ym, sd = '[UNK]', '[UNK]', '[UNK]', 6                # sd == 6 denotes not chinese
                    else:
                        sm_raw = get_initials(pinyin_string, self.strict)
                        sm = sm_raw if sm_raw != '' else '[PAD]'
                        ym = get_finals(pinyin_string, self.strict)[:-1]
                        sd = pinyin_string[-1]
                    pinyin_ids.append([self.id2pinyin.get(pinyin_string, unk_id), self.id2pinyin.get(sm, unk_id), self.id2pinyin.get(ym, unk_id), int(sd)])
                pinyin_ids.append([pad_id, pad_id, pad_id, pad_id])
                assert len(pinyin_ids) == len(tokenizer_output["input_ids"][i])
                all_pinyin_ids.append(pinyin_ids)
        return all_pinyin_ids

    def get_pinyin_size(self):
        return len(self.pinyin2id)

    def _generate_pinyin_vocab(self, path="Data/pinyin-data"):
        pinyin_set = set()
        for i in self.sm_list:
            pinyin_set.add(i)
        for i in self.ym_list:
            pinyin_set.add(i)
        with open(os.path.join(path, "pinyin.txt"), "r", encoding="utf-8") as f:
            for line in tqdm(f):
                item = line.split(" ")
                if len(item) < 3 or not item[0].startswith("U+"):
                    continue
                pinyin_list = item[1].split(",")
                for x in pinyin_list:
                    convertor = ToneConverter()
                    x = convertor.to_tone3(x)
                    if not _re_number.search(x):
                        x = x + "5"
                    pinyin_set.add(x)

        pinyin_list = [x for x in pinyin_set]
        special_tokens = ['[PAD]', '[UNK]', '[NULL]']
        for i, token in enumerate(special_tokens):
            pinyin_list.insert(i, token)
        pinyin2id = {i: x for i, x in enumerate(pinyin_list)}
        id2pinyin = {v: k for k, v in pinyin2id.items()}
        dump_json(pinyin2id, os.path.join(self.config_path, 'pinyin2id.json'))
        dump_json(id2pinyin, os.path.join(self.config_path, 'id2pinyin.json'))
