from common_utils import load_json
from csc_evaluation.evaluate_utils import get_result


def confusion_analysis(texts, tags, word_features=False):
    xj_dict = load_json("../Data/confusion/shapeConfusion.json")
    yj_dict = load_json("../Data/confusion/soundConfusion.json")
    correct_num = 0                                      # the whole num of spelling error characters
    error_sent = 0                                       # the whole num of spelling error sentences
    count_xj, count_yj = 0, 0                            # the sum of shape similar, sound similar characters
    sent_xj, sent_yj = 0, 0
    sent_mix = 0                                         # the num of both shape similar and sound similar sentences
    pair_not_in = 0                                      # the sum of original-correct pair characters not in confusion set
    sent_not_in_confusion = 0                            # the sum of error sentences not in confusion set
    xj_set = []
    yj_set = []
    sent_index = -1
    for ori, cor in zip(texts, tags):
        sent_index += 1
        if ori != cor:
            error_sent += 1
            char_xj, char_yj = 0, 0                        # the num of shape similar, sound similar characters in current sentence
            char_index = -1
            for o, c in zip(ori, cor):
                char_index += 1
                if o != c:
                    correct_num += 1
                    if not xj_dict.get(o) and not yj_dict.get(o):
                        pair_not_in += 1
                        continue
                    if (xj_dict.get(o) and c not in xj_dict[o]) and (yj_dict.get(o) and c not in yj_dict[o]):
                        pair_not_in += 1
                        continue
                    if (xj_dict.get(o) and c not in xj_dict[o]) and not yj_dict.get(o):
                        pair_not_in += 1
                        continue
                    if not xj_dict.get(o) and (yj_dict.get(o) and c not in yj_dict[o]):
                        pair_not_in += 1
                        continue
                    if c in xj_dict.get(o):
                        count_xj += 1
                        char_xj += 1
                        xj_set.append((sent_index, char_index, o, c))
                    if c in yj_dict.get(o):
                        count_yj += 1
                        char_yj += 1
                        yj_set.append((sent_index, char_index, o, c))
            if char_xj == 0 and char_yj == 0:
                sent_not_in_confusion += 1
            elif char_xj != 0 and char_yj == 0:
                sent_xj += 1
            elif char_xj == 0 and char_yj != 0:
                sent_yj += 1
            else:
                sent_mix += 1
    print(f"一共有{len(texts)}个句子")
    print(f"需要纠正字符个数：{correct_num}")
    print(f"未在confusion set中匹配到的字符个数有：{pair_not_in}, 占比为: {pair_not_in / correct_num:.4f}")
    print(f"错误字符中字形相似字符个数为：{count_xj}")
    print(f"错误字符中字音相似字符个数为：{count_yj}")
    print(f"需要纠正句子个数：{error_sent}")
    print(f"未在confusion set中匹配的句子有：{sent_not_in_confusion}，占比为: {sent_not_in_confusion / error_sent:.4f}")
    print(f"错误句子中错误来源为字形个数为：{sent_xj}")
    print(f"错误句子中错误来源为字音个数为: {sent_yj}")
    print(f"错误句子中来源混合个数为：{sent_mix}")
    print(f"xj_set: {len(xj_set)} {xj_set}")
    print(f"yj_set: {len(yj_set)} {yj_set}")
    print("-" * 10)
    return xj_set, yj_set


if __name__ == "__main__":
    with open("../csc_evaluation/data/basedata/traditional/train2013.txt", "r", encoding="utf-8") as f:
        data = []
        texts, tags = [], []
        flag = 0
        for line in f.readlines():
            line = line.strip().split("\t")
            data.append(line)
            texts.append(line[1])
            tags.append(line[2])
    gold_xj, gold_yj = confusion_analysis(texts, tags)
    # test_filename = "../Results/bert-base-chinese/no_copy/base/results/checkpoint-18000-2015.result"
    # texts, tags = get_result(test_filename, need_tokenize=False, get_gold=True)
    # gold_xj, gold_yj = confusion_analysis(texts, tags)
    # base_pre, base_tags = get_result(test_filename, need_tokenize=False)
    # base_xj, base_yj = confusion_analysis(base_pre, tags)
    #
    # test_filename = "../Results/bert-base-chinese/no_copy/word/results/checkpoint-20000-2015.result"
    # word_pre, tags = get_result(test_filename, need_tokenize=False)
    # word_xj, word_yj = confusion_analysis(word_pre, tags)
    #
    # print("形近")
    # xj_set = []
    # res1, res2, res3, res4 = [], [], [], []
    # for xj_t in gold_xj:
    #     sent_index, char_index, o, c = xj_t
    #     base_pre_char = base_pre[sent_index][char_index]
    #     word_pre_char = word_pre[sent_index][char_index]
    #     xj_set.append((o, c, base_pre_char, word_pre_char))
    #     if base_pre_char != word_pre_char == c:
    #         res1.append([o, c, base_pre_char, word_pre_char])
    #     if base_pre_char == c != word_pre_char:
    #         res2.append([o, c, base_pre_char, word_pre_char])
    #     if base_pre_char != c and word_pre_char != c:
    #         res3.append([o, c, base_pre_char, word_pre_char])
    #     if base_pre_char == c == word_pre_char:
    #         res4.append([o, c, base_pre_char, word_pre_char])
    # print(f"word纠正对了，base没纠正对: {len(res1)}")
    # print(res1)
    # print(f"base纠正对了，word没纠正对: {len(res2)}")
    # print(res2)
    # print(f"word, base都没纠正对: {len(res3)}")
    # print(res3)
    # print(f"word, base都纠正对: {len(res4)}")
    # print(res4)
    # assert len(res1)+len(res2)+len(res3)+len(res4) == len(gold_xj)
    #
    # print("音近")
    # yj_set = []
    # res1, res2, res3, res4 = [], [], [], []
    # for yj_t in gold_yj:
    #     sent_index, char_index, o, c = yj_t
    #     base_pre_char = base_pre[sent_index][char_index]
    #     word_pre_char = word_pre[sent_index][char_index]
    #     xj_set.append((o, c, base_pre_char, word_pre_char))
    #     if base_pre_char != word_pre_char == c:
    #         res1.append([o, c, base_pre_char, word_pre_char])
    #     if base_pre_char == c != word_pre_char:
    #         res2.append([o, c, base_pre_char, word_pre_char])
    #     if base_pre_char != c and word_pre_char != c:
    #         res3.append([o, c, base_pre_char, word_pre_char])
    #     if base_pre_char == c == word_pre_char:
    #         res4.append([o, c, base_pre_char, word_pre_char])
    # print(f"word纠正对了，base没纠正对: {len(res1)}")
    # print(res1)
    # print(f"base纠正对了，word没纠正对: {len(res2)}")
    # print(res2)
    # print(f"word, base都没纠正对: {len(res3)}")
    # print(res3)
    # print(f"word, base都纠正对: {len(res4)}")
    # print(res4)
    # assert len(res1)+len(res2)+len(res3)+len(res4) == len(gold_yj)
