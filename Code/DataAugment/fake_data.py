'''
Create data through confusion set
'''

import random
from collections import defaultdict
from text_utils import tokenize, is_ascii, is_punctuation, is_chinese
import glob
import os
from tqdm import tqdm


class CSCFaker(object):
    def __init__(self, confusion_filename, weight_filename, freq_filename, freq_threshold=1000,
                 fake_ratios=[15, 59, 14, 4]):
        self.__load_confusion_set(confusion_filename, weight_filename, freq_filename, freq_threshold=1000)
        self.fake_ratios = fake_ratios
        self.fake_lengths = list(range(len(fake_ratios)))
        self.max_change_ratio = 0.15
        return

    def __load_confusion_set(self, confusion_filename, weight_filename, freq_filename, freq_threshold):
        # load frequent chars
        chars = set()
        with open(freq_filename, encoding='utf-8') as f:
            for line in f:
                items = line.strip().split('\t')
                if len(items) != 2:
                    continue
                freq = int(items[1])
                if freq < freq_threshold:
                    break
                chars.add(items[0])
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
            confusion_dict[items[0]].append((items[1], label_weights[items[2]]))
            labels.add(items[2])
        self.confusion_dict = confusion_dict
        self.labels = labels
        print('Confusion dictionary size:', len(self.confusion_dict))
        return

    def fake_by_confusion(self, sentence):
        change_count = random.choices(self.fake_lengths, weights=self.fake_ratios)[0]
        if change_count <= 0:
            return 0, sentence
        max_change_count = max(int(self.max_change_ratio * len(sentence)), 1)
        change_count = min(change_count, max_change_count)
        candidate_indexes = []
        for index, token in enumerate(sentence):
            if token in self.confusion_dict:
                candidate_indexes.append(index)

        if len(candidate_indexes) == 0:
            return 0, sentence

        if len(candidate_indexes) > change_count:
            candidate_indexes = random.sample(candidate_indexes, k=change_count)

        for index in candidate_indexes:
            ch_raw = sentence[index]
            ch_alternatives = self.confusion_dict[ch_raw]
            weights = [x[1] for x in ch_alternatives]
            ch_choice = random.choices(ch_alternatives, weights=weights)[0][0]
            sentence = sentence[:index] + ch_choice + sentence[index + 1:]
        return len(candidate_indexes), sentence

    def fake_by_random_replace(self, sentence):
        change_count = random.choices(self.fake_lengths, weights=self.fake_ratios)[0]
        if change_count <= 0:
            return 0, sentence
        max_change_count = max(int(self.max_change_ratio * len(sentence)), 1)
        change_count = min(change_count, max_change_count)
        candidate_indexes = []
        for index, token in enumerate(sentence):
            if token in self.confusion_dict:
                candidate_indexes.append(index)

        if len(candidate_indexes) == 0:
            return 0, sentence

        if len(candidate_indexes) > change_count:
            candidate_indexes = random.sample(candidate_indexes, k=change_count)
        for index in candidate_indexes:
            ch_choice = random.choice(self.chars_list)
            sentence = sentence[:index] + ch_choice + sentence[index + 1:]
        return len(candidate_indexes), sentence

    def fake_files(self, in_filenames, output_filename, fake_methods=[('confusion', 0.7), ('random', 0.3)]):
        output_dir = os.path.basename(output_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_count = 0
        method_weights = [x[1] for x in fake_methods]
        fake_count_dict = {x[0]: 0 for x in fake_methods}
        f_out = open(output_filename, 'w', encoding='utf-8')
        for input_filename in in_filenames:
            print('Fake from file:', input_filename)
            with open(input_filename, encoding='utf-8') as f_in:
                for line in f_in:
                    sentence = line.strip()
                    if len(sentence) == 0:
                        continue
                    fake_method = random.choices(fake_methods, weights=method_weights)[0][0]
                    if fake_method == 'confusion':
                        errors, fake_sent = self.fake_by_confusion(sentence)
                    elif fake_method == 'random':
                        errors, fake_sent = self.fake_by_random_replace(sentence)
                    f_out.write('{}\t{}\t{}\n'.format(errors, fake_sent, sentence))
                    save_count += 1
                    if errors > 0:
                        fake_count_dict[fake_method] += 1
        f_out.close()
        print('save: {}, fake: {}'.format(save_count, sum(fake_count_dict.values())))
        print('fake method\tcount:')
        for k, v in fake_count_dict.items():
            print('{}\t{}'.format(k, v))
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


class CSCFaker2:
    def __init__(self, confusion_set, max_change_ratio):
        self.confusion_dict, self.char_list = self._load(confusion_set)
        self.max_change_ratio = max_change_ratio

    def _load(self, confusion_set):
        char_list = set()
        with open(confusion_set, "r", encoding="utf-8") as f:
            confusion_dict = {}
            for line in f:
                key, value = line.strip().split(",")
                confusion_dict[key] = value
                char_list.add(key)
                for c in value:
                    char_list.add(c)
        return confusion_dict, list(char_list)

    def fake_by_random(self, sentence):
        change_count = random.randint(1, int(len(sentence) * self.max_change_ratio))
        candidate_indexes = []
        for index, token in enumerate(sentence):
            if token in self.confusion_dict:
                candidate_indexes.append(index)

        if len(candidate_indexes) == 0:
            return 0, sentence

        if len(candidate_indexes) > change_count:
            candidate_indexes = random.sample(candidate_indexes, k=change_count)
        for index in candidate_indexes:
            ch_choice = random.choice(self.char_list)
            sentence = sentence[:index] + ch_choice + sentence[index + 1:]
        return len(candidate_indexes), sentence

    def fake_by_confusion(self, sentence):
        change_count = random.randint(1, int(len(sentence) * self.max_change_ratio))
        candidate_indexes = []
        for index, token in enumerate(sentence):
            if token in self.confusion_dict:
                candidate_indexes.append(index)

        if len(candidate_indexes) == 0:
            return 0, sentence

        if len(candidate_indexes) > change_count:
            candidate_indexes = random.sample(candidate_indexes, k=change_count)
        for index in candidate_indexes:
            ch_raw = sentence[index]
            ch_alternatives = self.confusion_dict[ch_raw]
            ch_choice = random.sample(ch_alternatives, 1)[0]
            sentence = sentence[:index] + ch_choice + sentence[index + 1:]
        return len(candidate_indexes), sentence

    def fake_file(self, input_filename, output_filename, fake_methods=[('confusion', 0.6), ('random', 0.2), ('no', 0.2)]):
        with open(input_filename, "r", encoding="utf-8") as f:
            data = f.readlines()
        f_out = open(output_filename, "w", encoding="utf-8")
        fake_count = {}
        for sentence in tqdm(data):
            sentence = sentence.strip()
            fake_method = random.choices(fake_methods, weights=[0.6, 0.2, 0.2])[0][0]
            if fake_method == "confusion":
                errors, sent = self.fake_by_confusion(sentence)
            elif fake_method == "random":
                errors, sent = self.fake_by_random(sentence)
            elif fake_method == "no":
                errors, sent = 0, sentence
            f_out.write(f"{errors}\t{sent}\t{sentence}\n")
            fake_count[fake_method] = fake_count.get(fake_method, 0) + 1
        f_out.close()
        for k, v in fake_count.items():
            print(f"{k} : {v}")


def main():
    confusion_filename = r'D:\repos\cscbase\data\confusion\spellGraphs.txt'
    weight_filename = r'D:\repos\cscbase\data\confusion\relationWeights.txt'
    freq_filename = r'D:\repos\Corpus\new2016zh\vocab.txt'

    input_filenames = glob.glob(r'D:\repos\Corpus\new2016zh\texts\*\*.txt') + glob.glob(
        r'D:\repos\Corpus\wiki_zh_2019\texts\*.txt')
    input_filenames = random.sample(input_filenames, 30)
    output_filename = r'D:\repos\Corpus\new2016zh\fake.new2016zh.wiki.sample30.txt'

    faker = CSCFaker(confusion_filename, weight_filename, freq_filename)
    faker.fake_files(input_filenames, output_filename)

    return


if __name__ == '__main__':
    main()
