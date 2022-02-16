from common_utils import read_tagging_data
from vocab_processor import VocabProcessor
from transformers import AutoTokenizer
from tag_result import Args, get_tag_result
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def count_feature_num(features):
    all_num = 0
    feature_num = 0
    for sent in features:
        sent = sent[1:-1]
        for token in sent:
            all_num += 1
            if token == 1:
                feature_num += 1
    return all_num, feature_num


def get_enc_word(dataset_filenames, processor, tokenizer, word_fp=None, pinyin_fp=None):
    texts, tags = read_tagging_data(dataset_filenames)
    encodings, word_features, pinyin_features = processor.add_feature(
        texts, tokenizer, use_word=True, use_pinyin=True, return_tokenizer=True, pinyin_max_len=3
    )
    word_res, pinyin_res = get_tag_result(encodings, tokenizer, word_features, pinyin_features)
    if word_res and word_fp:
        with open(word_fp, "w", encoding="utf-8") as f:
            for sent in word_res:
                f.write("".join(sent).strip())
                f.write("\n")
    if pinyin_features and pinyin_fp:
        with open(pinyin_fp, "w", encoding="utf-8") as f:
            for sent in pinyin_res:
                f.write("".join(sent).strip())
                f.write("\n")
    return texts, tags, encodings, word_features, pinyin_features


def get_ori_text_feature(texts, encodings, tokenizer, features):
    ori_texts = []
    ori_word_features = []
    for i in range(len(encodings["input_ids"])):
        tokens = encodings["input_ids"][i][1:-1]
        sent = tokenizer.convert_ids_to_tokens(tokens)
        word_ids = encodings.word_ids(batch_index=i)[1:-1]
        ori_feature = features[i][1: -1]
        if len(tokens) != len(texts[i]):
            for j in range(len(sent) - 1, 0, -1):
                if word_ids[j] == word_ids[j - 1]:
                    # 对text进行对齐
                    sent[j] = sent[j].replace("#", "")
                    sent[j] = sent[j].replace("##", "")
                    sent[j - 1] = sent[j - 1].replace("#", "")
                    sent[j - 1] = sent[j - 1].replace("##", "")
                    sent[j - 1] = sent[j - 1] + sent[j]
                    for k in range(j, len(sent) - 1):
                        sent[k] = sent[k + 1]
                    sent = sent[:len(sent) - 1]

                    # 对word_feature进行对齐
                    if ori_feature[j] == 1 or ori_feature[j - 1] == 1:
                        ori_feature[j] = 1
                    for k in range(j, len(ori_feature) - 1):
                        ori_feature[k] = ori_feature[k + 1]
                    ori_feature = ori_feature[:len(ori_feature) - 1]
            if sent != texts[i]:
                print(f"第{i}个不一致")
        ori_texts.append(sent)
        ori_word_features.append(ori_feature)

    return ori_texts, ori_word_features


def analysis(set_filenames, processor, tokenizer, word_fp=None, pinyin_fp=None):
    texts, tags, encodings, word_features, pinyin_features = get_enc_word(set_filenames, processor, tokenizer, word_fp, pinyin_fp)
    # word feature part
    all_num, word_feature_num = count_feature_num(word_features)
    print("all_tokens: {}, word_feature_tokens: {}, proportion: {:.4f}".format(
        all_num, word_feature_num, word_feature_num / all_num))
    ori_texts, ori_word_features = get_ori_text_feature(texts, encodings, tokenizer, word_features)
    feature_char = 0
    corrected_feature_char = 0
    for text, tag, word_feature in zip(ori_texts, tags, ori_word_features):
        assert len(text) == len(tag) == len(word_feature)
        for o, g, w in zip(text, tag, word_feature):
            if w == 1:
                feature_char += 1
                if o != g:
                    corrected_feature_char += 1
    print(f"feature chars: {feature_char}, corrected feature chars: {corrected_feature_char}")
    print(f"proportion: {corrected_feature_char / feature_char}")

    # pinyin feature part
    all_num, pinyin_feature_num = count_feature_num(pinyin_features)
    print("all_tokens: {}, pinyin_feature_tokens: {}, proportion: {:.4f}".format(
        all_num, pinyin_feature_num, pinyin_feature_num / all_num))
    ori_texts, ori_pinyin_features = get_ori_text_feature(texts, encodings, tokenizer, pinyin_features)
    feature_char = 0
    corrected_feature_char = 0
    for text, tag, pinyin_feature in zip(ori_texts, tags, ori_pinyin_features):
        assert len(text) == len(tag) == len(pinyin_feature)
        for o, g, p in zip(text, tag, pinyin_feature):
            if p == 1:
                feature_char += 1
                if o != g:
                    corrected_feature_char += 1
    print(f"feature chars: {feature_char}, corrected feature chars: {corrected_feature_char}")
    print(f"proportion: {corrected_feature_char / feature_char}")


def main():
    args = Args()
    train_filenames = ["../Data/traintest/bert-base-chinese/train.txt"]
    val_filenames = ["../Data/traintest/bert-base-chinese/val.txt"]
    test2015_filenames = ["../Data/traintest/bert-base-chinese/test2015.txt"]
    # all_test_filenames = [
    #     "../Data/traintest/bert-base-chinese/test2013.txt",
    #     "../Data/traintest/bert-base-chinese/test2014.txt",
    #     "../Data/traintest/bert-base-chinese/test2015.txt",
    # ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    processor = VocabProcessor("../Data/vocab/allNoun.txt")

    test_word_fp = "../Analysis/tag_test_word.txt"
    test_pinyin_fp = "../Analysis/tag_test_pinyin.txt"
    print("test2015:")
    analysis(test2015_filenames, processor, tokenizer, test_word_fp, test_pinyin_fp)

    train_word_fp = "../Analysis/tag_train_word.txt"
    train_pinyin_fp = "../Analysis/tag_train_pinyin.txt"
    print("train:")
    analysis(train_filenames, processor, tokenizer, train_word_fp, train_pinyin_fp)

    val_word_fp = "../Analysis/tag_val_word.txt"
    val_pinyin_fp = "../Analysis/tag_val_pinyin.txt"
    print("val:")
    analysis(val_filenames, processor, tokenizer, val_word_fp, val_pinyin_fp)

    print("Complete")


if __name__ == "__main__":
    main()
