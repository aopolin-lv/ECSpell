from dataclasses import dataclass

import common_utils
epsilon = 1e-8


def compute_metrics(results, strict=True, print_result=True):
    corrected_char = 0                      # 预测结果中修改了的字符数量
    wrong_char = 0                          # 原本就需要纠正的字符数量
    corrected_sent = 0                      # 预测结果中修改了的句子数量
    wrong_sent = 0                          # 原本就需要纠正的句子数量
    true_corrected_char = 0                 # 预测结果中修改对的字符数量
    true_corrected_sent = 0                 # 预测结果中修改对的句子数量
    true_detected_char = 0                  # 预测结果中检查对的字符数量
    true_detected_sent = 0                  # 预测结果中检查对的句子数量
    accurate_detected_sent, accurate_corrected_sent = 0, 0
    all_char, all_sent = 0, 0

    for src, tgt, pre in results:
        all_sent += 1

        wrong_num = 0
        corrected_num = 0
        original_wrong_num = 0
        true_detected_char_in_sentence = 0

        for s, t, p in zip(src, tgt, pre):
            all_char += 1
            if t != p:
                wrong_num += 1
            if s != p:
                corrected_num += 1
                if t == p:
                    true_corrected_char += 1
                if s != t:
                    true_detected_char += 1
                    true_detected_char_in_sentence += 1
            if t != s:
                original_wrong_num += 1

        corrected_char += corrected_num
        wrong_char += original_wrong_num
        if original_wrong_num != 0:
            wrong_sent += 1
        if corrected_num != 0 and wrong_num == 0:
            true_corrected_sent += 1

        if corrected_num != 0:
            corrected_sent += 1

        if strict:
            true_detected_flag = (true_detected_char_in_sentence == original_wrong_num == corrected_num and original_wrong_num != 0)
        else:
            true_detected_flag = (
                corrected_num != 0 and original_wrong_num != 0)
        if true_detected_flag:
            true_detected_sent += 1

        if tgt == pre:
            accurate_corrected_sent += 1
        if tgt == pre or true_detected_flag:
            accurate_detected_sent += 1

    det_char_pre = true_detected_char / (corrected_char + epsilon)
    det_char_rec = true_detected_char / (wrong_char + epsilon)
    det_char_f1 = 2 * det_char_pre * det_char_rec / \
        (det_char_pre + det_char_rec + epsilon)
    cor_char_pre = true_corrected_char / (corrected_char + epsilon)
    cor_char_rec = true_corrected_char / (wrong_char + epsilon)
    cor_char_f1 = 2 * cor_char_pre * cor_char_rec / \
        (cor_char_pre + cor_char_rec + epsilon)

    det_sent_acc = accurate_detected_sent / (all_sent + epsilon)
    det_sent_pre = true_detected_sent / (corrected_sent + epsilon)
    det_sent_rec = true_detected_sent / (wrong_sent + epsilon)
    det_sent_f1 = 2 * det_sent_pre * det_sent_rec / \
        (det_sent_pre + det_sent_rec + epsilon)
    cor_sent_acc = accurate_corrected_sent / (all_sent + epsilon)
    cor_sent_pre = true_corrected_sent / (corrected_sent + epsilon)
    cor_sent_rec = true_corrected_sent / (wrong_sent + epsilon)
    cor_sent_f1 = 2 * cor_sent_pre * cor_sent_rec / \
        (cor_sent_pre + cor_sent_rec + epsilon)

    if print_result:
        print("*** Char Level ***")
        print("** The detection result **")
        print("  precision:             {:.4f}\n"
              "  recall:                {:.4f}\n"
              "  F1:                    {:.4f}\n".format(det_char_pre, det_char_rec, det_char_f1))
        print("** The correction result **")
        print("  precision:             {:.4f}\n"
              "  recall:                {:.4f}\n"
              "  F1:                    {:.4f}\n".format(cor_char_pre, cor_char_rec, cor_char_f1))

        print("*** Sentence Level ***")
        print("** The detection result **")
        print("  accuracy:              {:.4f}\n"
              "  precision:             {:.4f}\n"
              "  recall:                {:.4f}\n"
              "  F1:                    {:.4f}\n".format(det_sent_acc, det_sent_pre, det_sent_rec, det_sent_f1))
        print("** The correction result **")
        print("  accuracy:              {:.4f}\n"
              "  precision:             {:.4f}\n"
              "  recall:                {:.4f}\n"
              "  F1:                    {:.4f}\n".format(cor_sent_acc, cor_sent_pre, cor_sent_rec, cor_sent_f1))

    return det_sent_acc, det_sent_pre, det_sent_rec, det_sent_f1, \
        cor_sent_acc, cor_sent_pre, cor_sent_rec, cor_sent_f1


def compute_fpr_acc_pre_rec_f1(tp, fp, tn, fn):
    fpr = fp / (fp + tn + epsilon)
    acc = (tp + tn) / (tp + fp + tn + fn + epsilon)
    pre = tp / (tp + fp + epsilon)
    rec = tp / (tp + fn + epsilon)
    f1 = 2 * pre * rec / (pre + rec + epsilon)
    return fpr, acc, pre, rec, f1


def official_compute_metrics(all_pair, write_official_result=False, print_result=True):
    truth = []
    for index, (src, tgt, pre) in enumerate(all_pair):
        char_index = 0
        sent_gold = []
        sent_correct_flag = 0
        temp = []
        for s, t, p in zip(src, tgt, pre):
            char_index += 1
            if s != t:
                sent_correct_flag = 1
                temp.append(str(char_index) + ", " + t)
            else:
                continue
        sent_gold.extend(", ".join(temp))
        if sent_correct_flag == 0:
            sent_gold.extend("0")
        truth.append(str(index) + ", " + "".join(sent_gold).strip())

    predict = []
    for index, (src, tgt, pre) in enumerate(all_pair):
        char_index = 0
        sent_pre = []
        sent_correct_flag = 0
        temp = []
        for s, t, p in zip(src, tgt, pre):
            char_index += 1
            if s != p:
                sent_correct_flag = 1
                temp.append(str(char_index) + ", " + p)
            else:
                continue
        sent_pre.extend(", ".join(temp))
        if sent_correct_flag == 0:
            sent_pre.extend("0")
        predict.append(str(index) + ", " + "".join(sent_pre).strip())

    if write_official_result:
        with open("../Cached/truth.txt", "w", encoding="utf-8") as f:
            for line in truth:
                f.write(line)
                f.write("\n")

        with open("../Cached/predict.txt", "w", encoding="utf-8") as f:
            for line in predict:
                f.write(line)
                f.write("\n")

    truth_dict = {}
    for sent in truth:
        sent_gold = sent.split(", ")
        if len(sent_gold) == 2:
            truth_dict[sent_gold[0]] = "0"
        else:
            content = {}
            truth_dict[sent_gold[0]] = content
            sent_gold = sent_gold[1:]
            for i in range(0, len(sent_gold), 2):
                content[sent_gold[i]] = sent_gold[i + 1]

    predict_dict = {}
    for sent in predict:
        sent_pre = sent.split(", ")
        if len(sent_pre) == 2:
            predict_dict[sent_pre[0]] = "0"
        else:
            content = {}
            predict_dict[sent_pre[0]] = content
            sent_pre = sent_pre[1:]
            for i in range(0, len(sent_pre), 2):
                content[sent_pre[i]] = sent_pre[i + 1]

    dtp, dfp, dtn, dfn = 0, 0, 0, 0
    ctp, cfp, ctn, cfn = 0, 0, 0, 0

    assert len(truth_dict) == len(predict_dict)

    for i in range(len(truth_dict)):
        gold = truth_dict[str(i)]
        pre = predict_dict[str(i)]
        if gold == "0":
            if pre == "0":
                dtn += 1
                ctn += 1
            else:
                dfp += 1
                cfp += 1
        elif pre == "0":
            dfn += 1
            cfn += 1
        elif len(gold) == len(pre) and gold.keys() == pre.keys():
            dtp += 1
            if list(gold.values()) == list(pre.values()):
                ctp += 1
            else:
                cfn += 1
        else:
            dfn += 1
            cfn += 1

    dfpr, dacc, dpre, drec, df1 = compute_fpr_acc_pre_rec_f1(dtp, dfp, dtn, dfn)
    cfpr, cacc, cpre, crec, cf1 = compute_fpr_acc_pre_rec_f1(ctp, cfp, ctn, cfn)

    if print_result:
        print("********** official algorithm **********")
        print("====== Sentence Level ======")
        print("== The detection result ==")
        print("  tp:  {:d}\n"
              "  fp:  {:d}\n"
              "  fn:  {:d}\n"
              "  tn:  {:d}\n".format(dtp, dfp, dfn, dtn))
        print("== The correction result ==")
        print("  tp:  {:d}\n"
              "  fp:  {:d}\n"
              "  fn:  {:d}\n"
              "  tn:  {:d}\n".format(ctp, cfp, cfn, ctn))

        print("*** Sentence Level ***")
        print("** The detection result **")
        print("  False Positive Rate:   {:.4f}\n"
              "  accuracy:              {:.4f}\n"
              "  precision:             {:.4f}\n"
              "  recall:                {:.4f}\n"
              "  F1:                    {:.4f}\n".format(dfpr, dacc, dpre, drec, df1))
        print("** The correction result **")
        print("  False Positive Rate:   {:.4f}\n"
              "  accuracy:              {:.4f}\n"
              "  precision:             {:.4f}\n"
              "  recall:                {:.4f}\n"
              "  F1:                    {:.4f}\n".format(cfpr, cacc, cpre, crec, cf1))

    return dfpr, dacc, dpre, drec, df1, cfpr, cacc, cpre, crec, cf1


def get_result(pred_filename, need_tokenize=True, get_gold=False):
    results = []
    data = common_utils.read_table_file(pred_filename, output_indexes=[0, 1, 2])
    for line_src, line_tgt, line_pred in data:
        if need_tokenize:
            line_src = line_src.split()
            line_tgt = line_tgt.split()
            line_pred = line_pred.split()

        line_src = list(line_src)
        line_tgt = list(line_tgt)
        line_pred = list(line_pred)

        if len(line_pred) < len(line_src):
            line_pred += line_src[len(line_pred):]
        elif len(line_pred) > len(line_src):
            line_pred = line_pred[:len(line_src)]
        results.append((line_src, line_tgt, line_pred))
    texts = []
    pres = []
    tags = []
    for i, r in enumerate(results):
        texts.append(r[0])
        tags.append(r[1])
        pres.append(r[2])
    if get_gold:
        return texts, tags
    else:
        return pres, tags


@dataclass
class CscResult:
    dfpr: float
    dacc: float
    dpre: float
    drec: float
    df1: float
    cfpr: float
    cacc: float
    cpre: float
    crec: float
    cf1: float


class EvalHelper:
    def __init__(self, texts, labels, tokenizer, unique_tag, tag2id):
        self.labels = labels
        self.inputs_ids = []
        self.texts = texts
        self.unique_tag = unique_tag
        self.tag2id = tag2id
        self.tokenizer = tokenizer
        # self.compute = official_compute_metrics
        self.compute = compute_metrics
