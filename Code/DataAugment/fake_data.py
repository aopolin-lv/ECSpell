'''
Create data through confusion set
'''
from multiprocessing import Pool
import random
from collections import defaultdict
from csc_evaluation.build_train_data import build_corpus
from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultHmmParams, viterbi, simplify_pinyin

from Code.DataAugment.text_utils import is_ascii, is_chinese, is_punctuation, is_chinese_phrase
from Code.common_utils import load_json, is_chinese_char
import glob
import os
import time
import sys
from tqdm import tqdm
from csc_evaluation.common_utils import set_logger
import logging
import jieba

logger = logging.getLogger(__name__)


class CSCFaker(object):
    def __init__(self, confusion_filename, weight_filename, freq_filename,
                 n_gram_confusion_filename="Data/confusion/continous_confusion.json",
                 freq_threshold=1000, fake_ratios=[38.27, 8.45, 1.72, 0.55, 0.18, 0.01, 0.08]):
        self.__load_confusion_set(confusion_filename, weight_filename, freq_filename, freq_threshold)
        self.fake_ratios = fake_ratios
        self.fake_lengths = list(range(len(fake_ratios)))
        self.fake_lengths = [x + 1 for x in self.fake_lengths]
        self.max_change_ratio = 0.15
        self.n_gram_confusionset = load_json(n_gram_confusion_filename)
        self.hmmparams = DefaultHmmParams()
        return

    def __load_confusion_set(self, confusion_filename, weight_filename, freq_filename, freq_threshold):
        # load frequent chars
        chars = set()
        with open(freq_filename, encoding='utf-8') as f:
            for line in f:
                items = line.strip().split('\t')
                if len(items) != 2:
                    continue
                ch, freq = items[0], int(items[1])
                if freq < freq_threshold or len(ch) > 1:
                    continue
                chars.add(ch)
        print('Frequent chars:', len(chars))
        self.chars_list = list(chars)
        # load similarity weights
        label_weights = {}
        for line in open(weight_filename, encoding='utf-8'):
            items = line.strip().split('\t')
            if len(items) != 2:
                continue
            label_weights[items[0]] = float(items[1])

        # load confusion set
        confusion_dict = defaultdict(list)
        labels = set()
        for line in open(confusion_filename, encoding='utf-8'):
            items = line.strip().split('|')
            if len(items) != 3:
                continue

            if items[1] not in chars:
                continue
            confusion_dict[items[0]].append((items[1], items[2], label_weights[items[2]]))
            labels.add(items[2])
        self.confusion_dict = confusion_dict
        self.labels = labels
        print('Confusion dictionary size:', len(self.confusion_dict))
        return

    def fake_continuous(self, sent, change_count, fake_range=[1, 2], weights=[0.8, 0.2]):
        sent_word_list = list(jieba.cut(sent))
        offset = [len(x) for x in sent_word_list]
        if len(offset) > 0:
            offset[0] = offset[0] - 1
        for idx in range(1, len(offset)):
            offset[idx] += offset[idx - 1]
        phrase = []
        phrase_offset = []
        for idx, x in enumerate(sent_word_list):
            if len(x) > 1 and is_chinese_phrase(x):
                phrase.append(x)
                phrase_offset.append(offset[idx])

        continuous_count = random.choices(fake_range, weights=weights)[0]
        if not phrase:
            return sent, []
        choose_index = phrase.copy()
        error_indexes = []
        for i in range(0, continuous_count):
            t = random.choice(range(len(choose_index)))
            error_indexes.append(t)
            del choose_index[t]
            if len(choose_index) == 0:
                break
        choose_index_len = sum([len(phrase[x]) for x in error_indexes])
        if change_count < choose_index_len:
            return sent, []

        index = []
        for i in range(len(error_indexes)):
            idx = error_indexes[i]
            raw_word = phrase[idx]
            start = phrase_offset[idx] - len(phrase[idx]) + 1
            end = phrase_offset[idx] + 1
            candidate = self.n_gram_confusionset.get(raw_word, -1)
            if random.choice([0, 1]) == 0 and candidate != -1:
                word = random.choice(candidate)
                sent = sent[:start] + word + sent[end:]
        return sent, index

    def fake_by_confusion(self, sentence, confuse_type):
        change_count = random.choices(self.fake_lengths, weights=self.fake_ratios)[0]
        if change_count <= 0:
            return 0, sentence
        max_change_count = max(int(self.max_change_ratio * len(sentence)), 1)
        change_count = min(change_count, max_change_count)

        raw_sentence, c_index, diff = sentence, [], 0
        if change_count > 1:
            sentence, c_index = self.fake_continuous(sentence, change_count)
            diff = count_diff(raw_sentence, sentence) if len(c_index) != 0 else 0
            change_count -= diff
        label_map = {"visual": ["形近", "同偏旁同部首"], "pho": ["同音同调", "近音异调", "近音同调", "同音同调"]}
        candidate_indexes = []
        for index, token in enumerate(sentence):
            if token not in self.confusion_dict or index in c_index:
                continue
            for item in self.confusion_dict[token]:
                if item[1] in label_map[confuse_type]:
                    candidate_indexes.append(index)
                    break

        if len(candidate_indexes) == 0:
            return 0, sentence

        if len(candidate_indexes) > change_count:
            candidate_indexes = random.sample(candidate_indexes, k=change_count)

        for index in candidate_indexes:
            ch_raw = sentence[index]
            ch_alternatives_raw = self.confusion_dict[ch_raw]
            ch_alternatives = []
            for item in ch_alternatives_raw:
                if item[1] in label_map[confuse_type]:
                    ch_alternatives.append(item)
            weights = [x[2] for x in ch_alternatives]
            ch_choice = random.choices(ch_alternatives, weights=weights)[0][0]
            sentence = sentence[:index] + ch_choice + sentence[index + 1:]
        return len(candidate_indexes) + diff, sentence

    def fake_by_random_replace(self, sentence):
        change_count = random.choices(self.fake_lengths, weights=self.fake_ratios)[0]
        if change_count <= 0:
            return 0, sentence
        max_change_count = max(int(self.max_change_ratio * len(sentence)), 1)
        change_count = min(change_count, max_change_count)

        raw_sentence, c_index, diff = sentence, [], 0
        if change_count > 1:
            sentence, c_index = self.fake_continuous(sentence, change_count)
            diff = count_diff(raw_sentence, sentence)
            change_count -= diff
        candidate_indexes = []
        for index, token in enumerate(sentence):
            if token in self.confusion_dict and index not in c_index:
                candidate_indexes.append(index)

        if len(candidate_indexes) == 0:
            return 0, sentence

        if len(candidate_indexes) > change_count:
            candidate_indexes = random.sample(candidate_indexes, k=change_count)
        for index in candidate_indexes:
            ch_choice = random.choice(self.chars_list)
            sentence = sentence[:index] + ch_choice + sentence[index + 1:]
        return len(candidate_indexes) + diff, sentence

    def fake_files(self, in_filenames, output_filename, fake_methods=[('pho', 0.3), ('visual', 0.3), ('random', 0.2), ('unchange', 0.2)]):
        save_count = 0
        method_weights = [x[1] for x in fake_methods]
        fake_count_dict = {x[0]: 0 for x in fake_methods}
        f_out = open(output_filename, 'w', encoding='utf-8')
        for input_filename in tqdm(in_filenames):
            print('Fake from file:', input_filename)
            with open(input_filename, encoding='utf-8') as f_in:
                for line in tqdm(f_in):
                    sentence = line.strip()
                    if len(sentence) == 0:
                        continue
                    fake_method = random.choices(fake_methods, weights=method_weights)[0][0]
                    if fake_method == 'pho':
                        errors, fake_sent = self.fake_by_confusion(sentence, fake_method)
                    elif fake_method == 'visual':
                        errors, fake_sent = self.fake_by_confusion(sentence, fake_method)
                    elif fake_method == 'random':
                        errors, fake_sent = self.fake_by_random_replace(sentence)
                    else:
                        errors, fake_sent = 0, sentence
                    f_out.write('{}\t{}\t{}\n'.format(errors, fake_sent, sentence))
                    save_count += 1
                    fake_count_dict[fake_method] += 1
        print('save: {}, fake: {}'.format(save_count, sum(fake_count_dict.values())))
        print(f'ratio: {sum(fake_count_dict.values()) / save_count:.4f}')
        print('fake method\tcount:')
        for k, v in fake_count_dict.items():
            print(f'{k}\t{v}\t{v / save_count:.2f}')
        return

    def fake_c_by_confusion(self, token, confuse_type):
        label_map = {"visual": ["形近", "同偏旁同部首"], "pho": ["同音同调", "近音异调", "近音同调", "同音同调"]}
        if token not in self.confusion_dict:
            return token
        candidates = self.confusion_dict[token]
        for item in candidates:
            if item[1] in label_map[confuse_type]:
                return item[0]
        return token

    def fake_c_by_random_replace(self):
        return random.choice(self.chars_list)

    def fake_files_o(self, in_filenames, output_filename, fake_methods=[('pho', 0.3), ('visual', 0.3), ('random', 0.2), ('unchange', 0.2)]):
        save_count = 0
        method_weights = [x[1] for x in fake_methods]
        fake_count_dict = {x[0]: 0 for x in fake_methods}
        f_out = open(output_filename, 'w', encoding='utf-8')
        for input_filename in tqdm(in_filenames):
            print('Fake from file:', input_filename)
            with open(input_filename, encoding='utf-8') as f_in:
                for line in f_in:
                    sentence = line.strip()
                    if len(sentence) == 0:
                        continue
                    fake_sent = ""
                    for c in sentence:
                        if not is_chinese(c) or random.random() >= 0.15:
                            fake_sent += c
                            continue
                        fake_method = random.choices(fake_methods, weights=method_weights)[0][0]
                        if fake_method == 'pho' or fake_method == 'visual':
                            c = self.fake_c_by_confusion(c, fake_method)
                        elif fake_method == 'random':
                            c = self.fake_c_by_random_replace()
                        fake_sent += c
                    errors = count_diff(fake_sent, sentence)
                    f_out.write('{}\t{}\t{}\n'.format(errors, fake_sent, sentence))
                    save_count += 1
        f_out.close()
        return


def get_frequent_chars(filenames, save_path, count_threshold=3):
    freq_dict = defaultdict(int)
    for filename in filenames:
        print('Handle {}, current vocab size: {}'.format(filename, len(freq_dict)))
        for line in open(filename, encoding='utf-8'):
            for ch in line:
                if is_ascii(ch) or is_punctuation(ch):
                    continue
                freq_dict[ch] += 1

    freq_dict = sorted(freq_dict.items(), key=lambda x: -x[1])
    with open(save_path, 'w', encoding='utf-8') as f:
        for ch, freq in freq_dict:
            if freq < count_threshold:
                break
            f.write('{}\t{}\n'.format(ch, freq))
    return


def count_diff(src, tgt):
    count = 0
    assert len(src) == len(tgt)
    for i, j in zip(src, tgt):
        if i != j:
            count += 1
    return count


def main():
    random.seed(0)
    confusion_filename = r'Data/confusion/spellGraphs.txt'
    weight_filename = r'Data/confusion/relationWeights.txt'
    freq_filename = r'Data/fakedata/new2016zh/vocab.txt'

    input_filenames = glob.glob(r'Data/fakedata/new2016zh/texts/*/*.txt') + glob.glob(
        r'Data/fakedata/wiki_zh_2019/texts/*.txt')
    output_filename = "Data/fakedata/fake.continous.txt"

    faker = CSCFaker(confusion_filename, weight_filename, freq_filename, freq_threshold=500)
    faker.fake_files(input_filenames, output_filename)
    save_dir = r"Data/fakedata_continous"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    build_corpus([output_filename], 2000, save_dir)
    set_logger(logger, log_filename=os.path.join(save_dir, 'construction.log'), file_mode='w')
    return


def is_continous_error(src, tgt):
    assert len(src) == len(tgt)
    for s, t in zip(src, tgt):
        if s == t:
            return False
    return True


if __name__ == '__main__':
    main()

