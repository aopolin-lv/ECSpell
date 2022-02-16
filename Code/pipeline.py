from typing import Optional, List
from processor import VocabProcessor
from tqdm import trange
import torch
import torch.nn as nn
import numpy as np
import os
import math

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from processor import PinyinProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tagger(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        clean_src: List,
        vocab_processor: VocabProcessor,
        use_word: Optional[bool] = False,
        device: Optional[str] = "cpu",
        use_pinyin: Optional[bool] = False,
        pinyin_processor: Optional[PinyinProcessor] = None,
        labels=None,
        weight=0,
        RSM=False,
        ASM=False,
):
    print("=" * 30)
    print(f"RSM: {RSM}\t ASM: {ASM}")
    print("=" * 30)
    ignore_labels = ["0"]
    device = torch.device(device)
    texts = []
    offset_mappings = []
    for sent in clean_src:
        tokenize_result = tokenizer(sent, return_offsets_mapping=True, truncation=True, max_length=128)
        tokens = []
        for offsets in tokenize_result["offset_mapping"]:
            if offsets[1] <= offsets[0]:
                continue
            token = sent[offsets[0]:offsets[1]]
            tokens.append(token)
        texts.append(tokens)
        offset_mappings.append(tokenize_result["offset_mapping"])

    # get encodings and add word features
    test_encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True)
    test_word_features = vocab_processor.add_feature(texts, tokenizer, test_encodings, use_word=use_word)
    if use_pinyin:
        if pinyin_processor is None:
            pinyin_processor = PinyinProcessor("/data/lq/Project/cscbaseline/Transformers/glyce")
        test_pinyin_ids = pinyin_processor.convert_sentence_to_pinyin_ids(texts, test_encodings)

    answers = []
    model.eval()
    if not device == torch.device("cpu"):
        model.cuda()

    min_logit = 6.5e-4
    temp_all = 0
    with torch.no_grad():
        inputs = {}
        for i in trange(len(test_encodings.data["input_ids"]), desc="Generate results"):
            inputs["input_ids"] = test_encodings.data["input_ids"][i]
            inputs["attention_mask"] = test_encodings.data["attention_mask"][i]
            inputs["token_type_ids"] = test_encodings.data["token_type_ids"][i]
            if use_pinyin:
                inputs["pinyin_ids"] = test_pinyin_ids[i]
                # inputs["pinyin_ids"] = [x[0] for x in test_pinyin_ids[i]]
            if len(inputs["input_ids"]) != len(texts[i]) + 2:
                temp_all += 1
                answers.append([])
                continue

            for k, v in inputs.items():
                if v:
                    if k != "pinyin_ids":
                        inputs[k] = torch.tensor(v).unsqueeze(0).to(device)
                    else:
                        v = torch.LongTensor(v)
                        inputs[k] = v.unsqueeze(0).to(device)

            logits = model(**inputs)[0][0].cpu()
            # input_ids = inputs["input_ids"].cpu().numpy()[0]
            pre_id, pre_sent, res = customize(logits, texts[i], vocab_processor, model.config, labels,
                                              min_logit=min_logit, vocab_feature=test_word_features[i], weight=weight,
                                              ASM=ASM, RSM=RSM, b_search=False)
            temp_all += res
            # score = np.exp(logits.numpy()) / np.exp(logits.numpy()).sum(-1, keepdims=True)
            # labels_idx = score.argmax(axis=-1)

            entities = []

            # det_res.append(labels_idx[0])

            filtered_labels_idx = [
                (idx, label_idx)
                for idx, label_idx in enumerate(pre_id)
                if (model.config.id2label[label_idx] not in ignore_labels)
            ][1: -1]

            for idx, label_idx in filtered_labels_idx:
                start_idx, end_idx = offset_mappings[i][idx]
                # word = tokenizer.convert_ids_to_tokens([int(input_ids[idx])])[0]

                entity = {
                    "word": pre_sent[idx],
                    # "score": score[idx][label_idx].item(),
                    "entity": pre_id[idx],
                    "index": idx,
                    "start": start_idx,
                    "end": end_idx,
                }

                entities += [entity]
            answers += [entities]
    candidate_path = temp_all / len(test_encodings.data["input_ids"])
    print(f"min_logit: {min_logit}, weight: {weight}, candidate_path: {candidate_path}")
    return answers


def customize(logits, src, processor, config, labels, vocab_feature=None,
              weight=0, min_logit=2e-5, RSM=False, ASM=False, b_search=True):
    k_size = 5 if ASM else 1
    weight = weight if ASM or RSM else 0
    # initial_weight = 1.0
    log_logits = nn.LogSoftmax(dim=-1)(logits)
    log_topk_scores, topk_indexes = log_logits.topk(k=k_size, dim=-1, sorted=True, largest=True)
    top_k_scores, log_topk_scores, topk_indexes = torch.exp(log_topk_scores).tolist(), log_topk_scores.tolist(), topk_indexes.tolist()

    max_logit = 1

    tag2id = {x: y for y, x in enumerate(labels)}
    res_sent = []
    for j in range(0, len(src) + 2):
        res_token = []
        if RSM and vocab_feature[j] != 0:
            src_id = tag2id[src[j - 1]] if src[j - 1] in tag2id.keys() else tag2id["<unk>"]
            res_token.append([1, src_id])
        else:
            for k in range(k_size):
                if top_k_scores[j][k] < min_logit and k != 0:
                    break
                res_token.append([log_topk_scores[j][k], topk_indexes[j][k]])
                if top_k_scores[j][k] >= max_logit:
                    break

        # if RSM and vocab_feature[j] != 0:
        #     src_id = tag2id[src[j - 1]] if src[j - 1] in tag2id.keys() else tag2id["<unk>"]
        #     candidates = [x[1] for x in res_token]
        #     # 如果没有在候选列表中，加入到其中去，并且设置概率为0.5
        #     if src_id not in candidates:
        #         res_token.append([math.log(initial_weight * (1 - 1e-4)), src_id])
        #     else:
        #         candidate_index = candidates.index(src_id)
        #         res_token[candidate_index][0] = math.log(math.exp(res_token[candidate_index][0]) + initial_weight)

        res_sent.append(res_token)

    temp = [len(x) for x in res_sent[1:-1]]
    candidate_path_num = 1
    for i in temp:
        candidate_path_num *= i

    res_sents = []

    def beam_search(res_sent, beam_size=20):
        mid_results = []
        results = []
        for res in res_sent:
            if not mid_results:
                for i in range(len(res)):
                    temp_candidate = []
                    temp_candidate.append(res[i])
                    temp_candidate.append(res[i][0])
                    mid_results.append(temp_candidate)
            else:
                results = []
                for result in mid_results:
                    total_score = result[-1]
                    for score, token in res:
                        new_result = result[:-1] + [[score, token]] + [total_score + score]
                        results.append(new_result)
                mid_results = results.copy()
                # convert
                output = convert_type(mid_results)
                # reward
                res_sents_ids = [[int(y.split(",")[1]) for y in x] for x in output]
                res_sents_scores = [[float(y.split(",")[0]) for y in x] for x in output]
                res_sents = [convert_ids_to_tokens_single_sent(src[:len(x)], x, config, labels, 1) for x in res_sents_ids]
                vocab_len = [processor.get_vocab_length("".join(tokens)) for tokens in res_sents]
                vocab_mean = np.mean(vocab_len)
                vocab_len = [(x - vocab_mean) for x in vocab_len]
                scores = [weight * v_l + sum(score) for v_l, score in zip(vocab_len, res_sents_scores)]
                for idx, (_, s) in enumerate(zip(mid_results, scores)):
                    mid_results[idx][-1] = s
                # sort results by scores
                mid_results = sorted(mid_results, key=lambda x: -x[-1])[:beam_size]

        res_sents = convert_type(mid_results)
        return res_sents

    def convert_type(mid_results: List):
        mid_results = [res[:-1] for res in mid_results]
        output = []
        for res in mid_results:
            result = []
            for score, token in res:
                result.append(str(score) + ',' + str(token))
            output.append(result)
        return output

    # global search
    def global_search(sent, t):
        if len(t) == 0:
            res_sents.append(list(sent.strip().split(" ")))
        else:
            for j in t[0]:
                global_search(f"{sent} {j[0]},{j[1]}", t[1:])

    if b_search:
        res_sents = beam_search(res_sent)
    else:
        global_search("", res_sent)

    if len(res_sents) != 1:
        res_sents_ids = [[int(y.split(",")[1]) for y in x] for x in res_sents]
        res_sents_scores = [[float(y.split(",")[0]) for y in x] for x in res_sents]

        res_sents = [convert_ids_to_tokens_single_sent(src, x, config, labels, 1) for x in res_sents_ids]

        # calculate the additional score
        vocab_len = [processor.get_vocab_length("".join(tokens)) for tokens in res_sents]
        vocab_mean = np.mean(vocab_len)
        vocab_len = [(x - vocab_mean) for x in vocab_len]
        scores = [weight * v_l + sum(score) for v_l, score in zip(vocab_len, res_sents_scores)]

        largest_prob = 0
        for k in range(len(scores)):
            if scores[largest_prob] < scores[k]:
                largest_prob = k
        res_sent_id = res_sents_ids[largest_prob]
        res_sent = res_sents[largest_prob]
    else:
        res_sent_id = [int(x.split(",")[1]) for x in res_sents[0]]
        res_sent = convert_ids_to_tokens_single_sent(src, res_sent_id, config, labels, 1)

    return res_sent_id, [0] + res_sent + [0], candidate_path_num


def convert_ids_to_tokens_single_sent(src_id, pre_id, unique_tag, labels, offset=1):
    pre_ids = pre_id[offset:len(src_id) + offset]
    pre_sent = convert_label_ids_to_tokens(src_id, pre_ids, unique_tag, labels)
    return pre_sent


def convert_label_ids_to_tokens(inputs: List, ids: List, config, unique_tag: List):
    """convert labels of each sentence to tokens using unique_tag"""
    res = []
    for index, id in enumerate(ids):
        label = config.id2label[id]
        tag_id = int(label.split("_")[-1])
        tag_token = unique_tag[tag_id]
        if tag_token in ["<copy>", "<unk>"]:
            res.append(inputs[index])
        else:
            res.append(tag_token)
    return res
