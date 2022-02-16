from common_utils import read_table_file
import random
import os


def split_dataset(data, output_dir, val_size=2000):
    assert isinstance(data, list)

    random.shuffle(data)
    pair = []
    for d in data:
        line = "\t".join(d)
        pair.append(f"{line}\n")
    train = pair[:-val_size]
    val = pair[-val_size:]

    with open(f"{output_dir}_train.txt", "w", encoding="utf-8") as f:
        f.writelines(train)
    with open(f"{output_dir}_val.txt", "w", encoding="utf-8") as f:
        f.writelines(val)


def test():
    all_pairs = read_table_file("../../Data/fakedata/nlg_a_73_val.txt", [1, 2])
    texts = [x[0] for x in all_pairs]
    tags = [x[1] for x in all_pairs]
    error = 0
    for text, tag in zip(texts, tags):
        if text != tag:
            error += 1
    print(len(texts))
    print(error)


def main():
    random.seed(0)
    fake_filename = "../../Data/fakedata/fake.new2016zh_wiki_2019.txt"
    nlg_filename = "../../csc_evaluation/builds/sim/nlg/train.txt"

    mode = "o"
    if mode == "a":
        aug_sents = read_table_file(fake_filename, [0, 1, 2])
    else:
        aug_sents = read_table_file(nlg_filename, [0, 1, 2])

    nlg_pairs = read_table_file(nlg_filename, [0, 1, 2])
    nlg_texts = [x[1] for x in nlg_pairs]
    nlg_tags = [x[2] for x in nlg_pairs]
    nlg_num = len(nlg_texts)
    nlg_error = 0
    for text, tag in zip(nlg_texts, nlg_tags):
        if text != tag:
            nlg_error += 1
    correct_num = nlg_num - nlg_error

    ratio = 8
    sample_size = int(nlg_error / ratio * (10 - ratio)) - correct_num
    sents = random.sample(aug_sents, sample_size)
    for i in range(len(sents)):
        sents[i][0] = "0"
        sents[i][1] = sents[i][2]

    all_sents = []
    all_sents += sents
    all_sents += nlg_pairs

    # split_dataset(all_sents, f"../../Data/fakedata/nlg_{mode}_{ratio}{10 - ratio}")
    pair = []
    for d in all_sents:
        line = "\t".join(d)
        pair.append(f"{line}\n")
    with open(f"../../Data/fakedata/nlg_{mode}_{ratio}{10 - ratio}.txt", "w", encoding="utf-8") as f:
        f.writelines(pair)


if __name__ == "__main__":
    main()
    # test()
