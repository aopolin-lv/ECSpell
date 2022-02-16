from vocab_processor import VocabProcessor
from train_baseline import load_data, load_input_data
from transformers import AutoTokenizer


class Args:
    def __init__(self):
        self.model_name = "../Transformers/bert-base-chinese"
        self.cached_dir = "../Cached"
        self.train_files = "../Data/traintest/bert_base_chinese/train.txt"
        self.val_files = "../Data/traintest/bert_base_chinese/val.txt"
        self.vocab_file = "../Data/vocab/allNoun.txt"
        self.max_sent_length = 128
        self.overwrite_cached = False
        self.keep_count = 1
        self.use_copy_label = True
        self.only_for_detection = False
        self.use_word_feature = True
        self.use_pinyin_feature = True
        self.train_files = self.train_files.split(";")
        self.val_files = self.val_files.split(";")
        self.model = self.model_name.split("/")[-1]
        self.copy_mode = "copy" if self.use_copy_label else "no_copy"
        self.mode = "word+pinyin"


def get_tag_result(encodings, tokenizer, word_features=None, pinyin_features=None):
    word_res, pinyin_res = [], []
    for i in range(len(encodings["input_ids"])):
        sent = tokenizer.convert_ids_to_tokens(encodings["input_ids"][i][1:-1])
        word = word_features[i][1:-1] if word_features else None
        pinyin = pinyin_features[i][1:-1] if pinyin_features else None
        sent_word_res, sent_pinyin_res = [], []
        for j in range(len(sent)):
            if word_features:
                if word[j] == 1:
                    if j + 1 == len(sent):
                        sent_word_res.extend(sent[j])
                    elif word[j + 1] != 1:
                        sent_word_res.extend(sent[j] + " ")
                    else:
                        sent_word_res.extend(sent[j])
            if pinyin_features:
                if pinyin[j] == 1:
                    if j + 1 == len(sent):
                        sent_pinyin_res.extend(sent[j])
                    elif pinyin[j + 1] != 1:
                        sent_pinyin_res.extend(sent[j] + " ")
                    else:
                        sent_pinyin_res.extend(sent[j])

        if len(sent_word_res) != 0:
            word_res.append(sent_word_res)
        if len(sent_pinyin_res) != 0:
            pinyin_res.append(sent_pinyin_res)

    if word_features and pinyin_features:
        return word_res, pinyin_res
    if word_features:
        return word_res, None
    if pinyin_features:
        return None, pinyin_res


def main():
    args = Args()
    vocab_processor = VocabProcessor(args.vocab_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_texts, train_tags, val_texts, val_tags, unique_tags, tag2id = load_data(
        args)
    train_encodings, train_word_features, train_pinyin_features, val_encodings, val_word_features, val_pinyin_features = load_input_data(
        train_texts, val_texts, tokenizer, args, vocab_processor)

    train_word_res, train_pinyin_res = get_tag_result(
        train_encodings, tokenizer, train_word_features, train_pinyin_features)
    val_word_res, val_pinyin_res = get_tag_result(
        val_encodings, tokenizer, val_word_features, val_pinyin_features)

    with open("../Analysis/tag_train_word.txt", "w", encoding="utf-8") as f:
        for sent in train_word_res:
            f.write("".join(sent))
            f.write("\n")
    with open("../Analysis/tag_val_word.txt", "w", encoding="utf-8") as f:
        for sent in val_word_res:
            f.write("".join(sent))
            f.write("\n")

    with open("../Analysis/tag_train_pinyin.txt", "w", encoding="utf-8") as f:
        for sent in train_pinyin_res:
            f.write("".join(sent))
            f.write("\n")
    with open("../Analysis/tag_val_pinyin.txt", "w", encoding="utf-8") as f:
        for sent in val_pinyin_res:
            f.write("".join(sent))
            f.write("\n")


if __name__ == "__main__":
    main()
