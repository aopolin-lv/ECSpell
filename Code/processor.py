import os
import re
import jieba
import paddle

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
        self.word_segement_processor = WordSegementProcessor()


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

    def add_feature(self, texts, tokenizer, encodings, use_word=False, use_pinyin=False, pinyin_max_len=100):
        word_features = [] if use_word else None
        pinyin_features = [] if use_pinyin else None
        bar = trange(len(texts), desc="add features", position=0, leave=True)
        for i in bar:

            tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][i])[1: -1]
            token2text = encodings.word_ids(batch_index=i)[1: -1]
            text2token = [0 for _ in range(len(texts[i]))]
            start, index = 0, 0
            end = start + 1
            while end < len(token2text) and index < len(text2token):
                if token2text[end] == token2text[start] + 1:
                    text2token[index] = start
                    index += 1
                    start = end
                    end += 1
                elif token2text[end] == token2text[start]:
                    end += 1
                else:
                    print("error")
            for j in range(start, len(token2text)):
                text2token[index] = j
                index += 1
                if index == len(text2token):
                    break

            sentence = "".join(texts[i])

            text2sent = [0 for _ in range(len(texts[i]))]
            for j in range(1, len(texts[i])):
                text2sent[j] = text2sent[j - 1] + len(texts[i][j - 1])
            sent2text = [0 for _ in range(len(sentence))]
            for j in range(1, len(text2sent)):
                for k in range(text2sent[j - 1], text2sent[j]):
                    sent2text[k] = j - 1
            for j in range(text2sent[-1], len(sent2text)):
                sent2text[j] = len(text2sent) - 1

            sent_pinyin_lists = pinyin(sentence, style=Style.NORMAL)
            sent_pinyin_list = []
            for word_py_list in sent_pinyin_lists:
                sent_pinyin_list.append(word_py_list[0])
            sent_pinyin = ""
            for word_py in sent_pinyin_list:
                sent_pinyin = sent_pinyin + word_py + " "
            text2py = [0 for _ in range(len(texts[i]))]
            for j in range(1, len(sent_pinyin_list)):
                text2py[j] = text2py[j - 1] + len(sent_pinyin_list[j - 1]) + 1
            py2text = [0 for _ in range(len(sent_pinyin))]
            for j in range(1, len(text2py)):
                for k in range(text2py[j - 1], text2py[j]):
                    py2text[k] = j - 1
            for j in range(text2py[-1], len(py2text)):
                py2text[j] = len(text2py) - 1

            if use_word:
                word_feature = [0 for _ in range(len(tokens))]
                start_pos, margin = 0, 0
                while margin < len(sentence):
                    sub_str = self.word_trie.longest_prefix(sentence[margin:], default="N/A")
                    if sub_str == "N/A":
                        margin += 1
                        continue
                    start_pos = sentence[margin:].find(sub_str) + margin
                    end_pos = start_pos + len(sub_str) - 1
                    for j in range(sent2text[start_pos], sent2text[end_pos] + 1):
                        word_feature[text2token[j]] = len(sub_str)
                    margin += len(sub_str)
                assert len(word_feature) == len(tokens)
                word_feature = [0] + word_feature + [0]
                word_features.append(word_feature)

            if use_pinyin:
                pinyin_feature = [0 for _ in range(len(tokens))]
                start_pos, text_margin = 0, 0
                while text_margin < len(text2py) and text2py[text_margin] < len(sent_pinyin):
                    sub_pys = sent_pinyin[text2py[text_margin]:]
                    sub = self.py_trie.longest_prefix_item(sub_pys, default="N/A")
                    if sub == "N/A":
                        text_margin += 1
                        continue
                    sub_py, sub_word = sub[0], sub[1]
                    if len(sub_word) < pinyin_max_len:
                        text_margin += 1
                        continue
                    sub_sent = sentence[text2sent[text_margin]:]
                    if sub_word != sub_sent[:len(sub_word)]:
                        start_py_pos = sub_pys.find(sub_py) + text2py[text_margin]
                        end_py_pos = start_py_pos + len(sub_py)
                        if end_py_pos < len(py2text):
                            for j in range(text2token[py2text[start_py_pos]], text2token[py2text[end_py_pos]]):
                                pinyin_feature[j] = 1
                        else:
                            for j in range(text2token[py2text[start_py_pos]], len(tokens)):
                                pinyin_feature[j] = 1
                    text_margin += len(sub_py.split(" ")) - 1
                assert len(pinyin_feature) == len(tokens)
                pinyin_feature = [0] + pinyin_feature + [0]
                pinyin_features.append(pinyin_feature)
        return word_features

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

    def _generate_pinyin_vocab(self, path="../Data/pinyin-data"):
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


class WordSegementProcessor():
    def convert_sentences_to_word_segement(self, texts: Union[List[List], List], encodings, tokenizer):
        wg_features = []
        paddle.enable_static()
        jieba.enable_paddle()
        for i in trange(len(texts)):
            tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][i])[1: -1]
            token2text = encodings.word_ids(batch_index=i)[1: -1]
            text2token = [0 for _ in range(len(texts[i]))]
            start, index = 0, 0
            end = start + 1
            while end < len(token2text) and index < len(text2token):
                if token2text[end] == token2text[start] + 1:
                    text2token[index] = start
                    index += 1
                    start = end
                    end += 1
                elif token2text[end] == token2text[start]:
                    end += 1
                else:
                    print("error")
            for j in range(start, len(token2text)):
                text2token[index] = j
                index += 1
                if index == len(text2token):
                    break

            sentence = "".join(texts[i])

            text2sent = [0 for _ in range(len(texts[i]))]
            for j in range(1, len(texts[i])):
                text2sent[j] = text2sent[j - 1] + len(texts[i][j - 1])
            sent2text = [0 for _ in range(len(sentence))]
            for j in range(1, len(text2sent)):
                for k in range(text2sent[j - 1], text2sent[j]):
                    sent2text[k] = j - 1
            for j in range(text2sent[-1], len(sent2text)):
                sent2text[j] = len(text2sent) - 1

            wg_feature = [0 for _ in range(len(tokens))]
            wg_list = list(jieba.cut(sentence, use_paddle=True))

            # make sure the length of sentence after word segement equals to before
            sent_len = 0
            for i in wg_list:
                sent_len += len(i)
            assert sent_len == len(sentence)

            sent_wg_labels = []
            for word in wg_list:
                if len(word) == 1:
                    sent_wg_labels.append("B")
                else:
                    for i, c in enumerate(word):
                        if i == 0:
                            sent_wg_labels.append("B")
                        else:
                            sent_wg_labels.append("I")
            assert len(sent_wg_labels) == len(sentence)

            for i, label in enumerate(sent_wg_labels):
                wg_feature[text2token[sent2text[i]]] = 0 if label == "B" else 1

            wg_feature = [0] + wg_feature + [0]
            wg_features.append(wg_feature)

        return wg_features


if __name__ == "__main__":
    filename = "../Data/vocab/vocab.txt"
    py_processor = PinyinProcessor("../Transformers/glyce")
    # py_processor._generate_pinyin_vocab()
    tokenizer = AutoTokenizer.from_pretrained("../Transformers/glyce")

    string = []
    string = [list("应，"), ["1921", "今", "天", "气"]]

    string, _ = read_tagging_data(["../Data/traintest/sim/glyce/val.txt"])
    encoding = tokenizer(string, is_split_into_words=True)
    all_pinyin_ids = py_processor.convert_sentence_to_pinyin_ids(string, encoding)
    res = []
    for ids in all_pinyin_ids:
        pinyin_pad = [0] * 4
        pinyin_ids = ids + [pinyin_pad for _ in range(5)]
        res.append(pinyin_ids)
