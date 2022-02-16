import glob
from common_utils import clean_text
from tqdm import tqdm
from text_utils import NUM_CHAR, chinese_char_count, PREFIX, count_blacket, check, is_chinese_word


def main():
    min_len = 12
    max_len = 128
    min_chinese_len = 6
    filenames = glob.glob("../../csc_evaluation/data/document_writing/web_data/*.txt")
    all_data = []

    for file in tqdm(filenames):
        with open(file, "r", encoding="utf-8") as f:
            data = clean_text(f.read()).split()
            res = []
            for sent in data:
                if ("(" in sent and ")" in sent) or ("（" in sent and "）" in sent):
                    start = sent.index("(") if "(" in sent else sent.index("（")
                    end = sent.index(")") if ")" in sent else sent.index("）")
                    flag = 0
                    for i in range(start + 1, end):
                        if sent[i] not in NUM_CHAR:
                            flag = 1
                            break
                    if flag == 0:
                        sent = sent[:start] + sent[min(len(sent), end + 1):]
                if len(sent) < min_len or len(sent) > max_len or chinese_char_count(sent) < min_chinese_len:
                    continue
                if sent[0] in PREFIX:
                    sent = sent[1:]
                if "." in sent:
                    pre = sent.split(".")
                    if pre[0] in NUM_CHAR or chinese_char_count(pre[0]) == 0:
                        sent = sent[len(pre[0]) + 1:]
                if "、" in sent:
                    pre = sent.split("、")
                    if pre[0] in NUM_CHAR or chinese_char_count(pre[0]) == 0:
                        sent = sent[len(pre[0]) + 1:]
                if "．" in sent:
                    pre = sent.split("．")
                    if pre[0] in NUM_CHAR or chinese_char_count(pre[0]) == 0:
                        sent = sent[len(pre[0]) + 1:]
                sent = sent[1: -1] if count_blacket(sent) else sent
                if not check(sent):
                    continue
                if len(sent) < min_len or len(sent) > max_len or chinese_char_count(sent) < min_chinese_len:
                    continue
                res.append(sent)
            all_data += res
    all_data = list(set(all_data))

    res = []

    for line in all_data:
        if len(line) > 64 and "。" in line:
            sent_list = line.split("。")
            for idx, s in enumerate(sent_list):
                if min_len <= len(s) <= max_len and check(s):
                    if idx != len(sent_list) - 1:
                        res.append(f"{s}。")
                    else:
                        res.append(s)
        else:
            res.append(line)

    all_len = 0
    for line in res:
        all_len += len(line)
    avg_len = all_len / len(res)

    # 44463
    print(f"the size of data is: {len(res)}")
    # 39.82
    print(f"the avg len of data is : {avg_len}")
    with open("../../csc_evaluation/data/document_writing/web_data.txt", "w", encoding="utf-8") as f:
        f.writelines("\n".join(res))


def vocab_clean():
    general_vocab_path = "../../Data/vocab/general_phrase.txt"
    official_document_vocab_path = "../../csc_evaluation/data/document_writing/wordlist/公文写作.txt"

    general_vocab = []
    official_document_vocab = []
    res = []
    vocab_max_len = 80000

    with open(general_vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            general_vocab.append(line.strip().split("\t")[0])
            if len(general_vocab) > vocab_max_len:
                break
    with open(official_document_vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            official_document_vocab.append(line.strip())
    for word in tqdm(official_document_vocab):
        if word not in general_vocab and is_chinese_word(word):
            res.append(word)
    with open(f"{official_document_vocab_path}.clean", "w", encoding="utf-8") as f:
        for line in res:
            f.write(f"{line}\n")


if __name__ == "__main__":
    # main()
    # vocab_clean()
