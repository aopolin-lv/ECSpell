from tqdm import tqdm
import glob
from text_utils import check, chinese_char_count
import opencc
from fake_data import CSCFaker2
from common_utils import read_table_file
import random


def build_sensetive_vocab():
    vocab = []
    with open("敏感词.txt", "r", encoding="utf-8") as f:
        for line in f:
            vocab.append(line.split()[0])

    filenames = glob.glob("敏感词/*.txt")
    for file in filenames:
        if file == "敏感词/政治类.txt" or "敏感词/色情类.txt":
            f = open(file, "r", encoding="utf-8")
            for line in f:
                vocab.append(line.replace(",", "").strip())
        else:
            f = open(file, "r", encoding="utf-8")
            for line in f:
                vocab.append(line.strip())
        f.close()
    vocab = list(set(vocab))

    with open("敏感词/all.txt", "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word)
            f.write("\n")


def build_raw_sighan_aug_data():
    all_pair = []

    vocab = []
    with open("../Data/vocab/sensetive.txt", "r", encoding="utf-8") as f:
        for word in f:
            vocab.append(word.strip())

    convertor = opencc.OpenCC('t2s.json')

    with open("../Data/augment/tra_aug.txt", "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not check(line) or chinese_char_count(line) < 8:
                continue
            all_pair.append(line)
    all_pair_sim = [convertor.convert(x) for x in tqdm(all_pair)]

    res = []

    for t, s in tqdm(zip(all_pair, all_pair_sim)):
        flag = 0
        for x in vocab:
            if x in t or x in s:
                flag = 1
                break
        if flag == 0:
            res.append(t)

    with open("../Data/augment/clean.txt", "w", encoding="utf-8") as f:
        for line in tqdm(res):
            f.write(line)
            f.write("\n")

    print(len(res))


def fake_file():
    faker = CSCFaker2("../../Data/confusion/NCU_NLPLab_ConfusionSet.txt", 0.15)
    faker.fake_file("../../Data/augment/clean.txt", "../../Data/augment/fake_clean.txt")


def simplify():
    all_pairs = read_table_file("../../Data/augment/fake_clean.txt", [0, 1, 2])
    res = []
    convertor = opencc.OpenCC('t2s.json')
    for error, src, tgt in tqdm(all_pairs):
        src = convertor.convert(src)
        tgt = convertor.convert(tgt)
        res.append([error, src, tgt])
    with open("../../Data/augment/fake_clean_simplified.txt", "w", encoding="utf-8") as f:
        for line in res:
            f.write("\t".join(line))
            f.write("\n")
    print("simplify complete")
    return res


def split_train_data():
    random.seed(0)
    max_len = 80000
    res = []
    with open("../../Data/augment/fake_clean_simplified.txt", "r", encoding="utf-8") as f:
        for line in f:
            if len(res) > max_len:
                break
            res.append(line.strip())

    random.shuffle(res)
    train = res[:-1000]
    val = res[-1000:]
    with open("../../Data/augment/train.txt", "w", encoding="utf-8") as f:
        for line in train:
            f.write(line)
            f.write("\n")
    with open("../../Data/augment/val.txt", "w", encoding="utf-8") as f:
        for line in val:
            f.write(line)
            f.write("\n")
    print(f"{len(val)}")


if __name__ == "__main__":
    # build_sensetive_vocab()
    # build_raw_sighan_aug_data()
    # fake_file()
    # simplify()
    split_train_data()
