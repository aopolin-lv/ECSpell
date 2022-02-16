import random

random.seed(42)


def merge(filename):
    all_pairs = []
    for f in filename:
        with open(f, "r", encoding="utf-8") as f:
            data = f.readlines()
        all_pairs.extend(data)
    with open("../Data/basedata/bak/train_data.txt", "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(pair)


def split():
    data = []
    with open("../Data/basedata/bak/train_data.txt", "r", encoding="utf-8") as f:
        data = f.readlines()

    random.shuffle(data)
    train_set = data[:-(len(data) // 10 * 1)]
    val_set = data[len(train_set):]
    print(len(train_set))
    print(len(val_set))
    assert len(train_set) + len(val_set) == len(data)

    with open("../Data/basedata/train.txt", "w", encoding="utf-8") as f:
        for line in train_set:
            f.write(line)

    with open("../Data/basedata/val.txt", "w", encoding="utf-8") as f:
        for line in val_set:
            f.write(line)


if __name__ == "__main__":
    filename = [
        "../csc_evaluation/data/basedata/simplified/nlg.txt",
        "../csc_evaluation/data/basedata/simplified/train2013.txt",
        "../csc_evaluation/data/basedata/simplified/train2014.txt",
        "../csc_evaluation/data/basedata/simplified/train2015.txt",
    ]
    merge(filename)
    split()
