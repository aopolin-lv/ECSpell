import os
import json
import torch
import math
import tokenizers
import numpy as np

from tqdm import tqdm
from tqdm import trange
from dataclasses import dataclass
from pypinyin import pinyin, Style
from torch.utils.data import Dataset

from Code.processor import PinyinProcessor
from typing import Optional, Union, List

from common_utils import is_chinese_char, load_json
from tokenizers.implementations.bert_wordpiece import BertWordPieceTokenizer

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding

UNK_LABEL = "<unk>"
COPY_LABEL = "<copy>"
NOCOPY_LABEL = "<nocopy>"

model_name = "Transformers/glyce"
py_processor = PinyinProcessor(model_name)


class TokenCLSDataset(Dataset):
    def __init__(self, encodings, labels, word_features=None, pinyin_ids=None, ws_features=None):
        self.encodings = encodings
        self.labels = labels
        self.word_features = word_features if word_features else None
        # self.pinyin_ids = list(pinyin_ids) if pinyin_ids else None
        self.ws_features = ws_features if ws_features else None

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["word_features"] = self.word_features[idx] if self.word_features else None
        # item["pinyin_ids"] = self.pinyin_ids[idx] if self.pinyin_ids else None
        item["ws_features"] = self.ws_features[idx] if self.ws_features else None
        return item

    def __len__(self):
        return len(self.labels)


@dataclass
class CscDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        max_length = max(len(inputs["input_ids"]) for inputs in features)
        batch_outputs = {}
        encoded_inputs = {key: [example[key] for example in features] for key in features[0].keys()}
        for i in range(len(features)):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            inputs.pop("offset_mapping")
            outputs = self._pad(inputs, max_length=max_length)

            for k, v in outputs.items():
                if k not in batch_outputs:
                    batch_outputs[k] = []
                batch_outputs[k].append(v)

        return BatchEncoding(batch_outputs, tensor_type="pt")

    def _pad(self, inputs, max_length):
        difference = max_length - len(inputs["input_ids"])
        inputs["input_ids"] = inputs["input_ids"] + [self.tokenizer.pad_token_id] * difference
        inputs["attention_mask"] = inputs["attention_mask"] + [0] * difference
        inputs["token_type_ids"] = inputs["token_type_ids"] + [self.tokenizer.pad_token_id] * difference
        inputs["labels"] = inputs["labels"] + [self.label_pad_token_id] * difference
        word_features = inputs.pop("word_features")
        if word_features:
            inputs["word_features"] = word_features + [0] * difference

        return inputs


class PinyinProcessor():
    def __init__(self, model_name, max_length: int = 512):
        super().__init__()
        self.vocab_file = os.path.join(model_name, 'vocab.txt')
        self.config_path = os.path.join(model_name, 'pinyin_config')
        self.max_length = max_length
        self.tokenizer = BertWordPieceTokenizer(self.vocab_file)
        # load pinyin map dict
        with open(os.path.join(self.config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(self.config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(self.config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def convert_sentence_to_pinyin_ids(
            self,
            sentence: Union[str, list],
            tokenizer_output: tokenizers.Encoding
    ):
        if not isinstance(sentence, list):
            raise "sentence must be list form"
        # batch mode
        if len(sentence) != 0 and isinstance(sentence[0], list):
            all_pinyin_ids = []

            for i, sent in enumerate(tqdm(sentence, desc="Generate pinyin ids", ncols=50)):
                # filter no-chinese char
                for idx, c in enumerate(sent):
                    if len(c) > 1 or not is_chinese_char(ord(c)):
                        sent[idx] = "#"

                assert len(tokenizer_output.tokens(i)) == len(sent) + 2

                pinyin_list = pinyin(sent, style=Style.TONE3, heteronym=False, errors=lambda x: [
                    ["not chinese"] for _ in x])
                pinyin_locs = {}
                for index, item in enumerate(pinyin_list):
                    pinyin_string = item[0]
                    if pinyin_string in self.pinyin2tensor:
                        pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
                    elif pinyin_string == "not chinese":
                        pinyin_locs[index] = [0] * 8
                    else:
                        ids = [0] * 8
                        for j, p in enumerate(pinyin_string):
                            if p not in self.pinyin_dict["char2idx"]:
                                ids = [0] * 8
                                break
                            ids[j] = self.pinyin_dict["char2idx"][p]
                        pinyin_locs[index] = ids
                assert len(pinyin_locs) == len(sent)

                # bert token offset `[CLS]`
                token_offset = 1
                # find chinese character location, and generate pinyin ids
                pinyin_ids = []

                tokens = tokenizer_output.tokens(i)
                offset_mapping = tokenizer_output.offset_mapping[i]
                assert len(pinyin_locs) + 2 == len(offset_mapping)
                for idx, (token, offset) in enumerate(zip(tokens, offset_mapping)):
                    if token in ["[CLS]", "[SEP]"] or (offset[1] - offset[0] != 1):
                        pinyin_ids.append([0] * 8)
                    elif (idx - token_offset) in pinyin_locs.keys():
                        pinyin_ids.append(pinyin_locs[idx - token_offset])
                    else:
                        raise "被分为子词"
                assert len(pinyin_ids) == len(tokenizer_output["input_ids"][i])
                all_pinyin_ids.append(pinyin_ids)
        return all_pinyin_ids


def _prepare_inputs(inputs, model):
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    return inputs


def attack_generation(inputs, tag2id, model, tokenizer, processor, ratio=0.15, rerank=True):
    new_input_set = inputs
    new_output_set = inputs["labels"]
    ad_sample_gen = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "labels": [], "pinyin_ids": []}
    logit_rank = None
    error_count = [0 for _ in range(len(inputs["input_ids"]))]          # the number of errors in current sentence
    seq_len = [sum(i).item() for i in inputs["attention_mask"]]         # the length of sentences
    error_bound = [ratio * i for i in seq_len]                          # 每句话最多改的字数
    logit_idx = [0 for _ in range(len(inputs["input_ids"]))]            # 每句话从高到低排序到第几个字

    confusion_set = get_confusionset()

    while len(new_input_set["input_ids"]) != 0:
        model.eval()
        new_input_set = _prepare_inputs(new_input_set, model)
        new_output_set = new_output_set.to(model.device)

        torch.cuda.empty_cache()

        outputs = model(**new_input_set)                                # [batch_size, seq_len, num_labels]
        softmax = torch.nn.LogSoftmax(dim=-1)
        out = softmax(outputs.logits)                                   # [batch_size, seq_len, num_labels]

        if not logit_rank or rerank:
            prob_rank = out.argsort(dim=-1)                             # vocab_size维度上的排序,从小到大
            prob_dif = []                                               # [batch_size, seq_len]
            for i in range(out.shape[0]):
                set_prob_dif = []
                for j in range(out.shape[1]):                           # 保持长度一致
                    set_prob_dif.append((out[i][j][prob_rank[i][j][-1]] - out[i][j][prob_rank[i][j][-2]]).item())
                prob_dif.append(set_prob_dif)

            logit_rank = torch.tensor(prob_dif).argsort(dim=-1).cpu().numpy()
            logit_idx = [0] * out.shape[0]

        out_ids = out.argmax(dim=-1)                                    # [batch_size, seq_len]
        new_input_set_ = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "labels": [], "pinyin_ids": []}
        new_output_set_ = []
        logit_rank_ = []
        error_count_ = []
        logit_idx_ = []
        error_bound_ = []

        for i in range(out_ids.shape[0]):                                # batch_size
            is_added = False
            if out_ids[i][1: seq_len[i] - 1].equal(new_output_set[i][1: seq_len[i] - 1]) and error_count[i] < error_bound[i] and not is_added:
                tokens = tokenizer.convert_ids_to_tokens(new_input_set["input_ids"][i])
                for j, token in enumerate(tokens):
                    tokens[j] = token.replace("##", "")
                    tokens[j] = token.replace("#", "")

                    if len(tokens[j]) != 1 or not is_chinese_char(ord(tokens[j])):
                        tokens[j] = "<copy>"
                    elif tokens[j] not in tag2id.keys():
                        tokens[j] = "<unk>"
                encode_tokens = [tag2id[k] for k in tokens]

                sub_idx = (logit_rank[i][logit_idx[i]]).item()
                while sub_idx == 0 or sub_idx >= seq_len[i] or len(tokens[sub_idx]) != 1 or not is_chinese_char(ord(tokens[sub_idx])) \
                        or encode_tokens[sub_idx] != new_output_set[i][sub_idx].item():
                    logit_idx[i] += 1
                    if logit_idx[i] >= out.shape[1]:
                        for k, v in new_input_set.items():
                            ad_sample_gen[k].append(v[i])
                        is_added = True
                        break
                    sub_idx = (logit_rank[i][logit_idx[i]]).item()
                if is_added:
                    continue

                sub_cand = choose_largest_cand(out[i][sub_idx], new_input_set["input_ids"][i][sub_idx], tokenizer, tag2id, confusion_set)
                while not sub_cand:
                    logit_idx[i] += 1
                    if logit_idx[i] >= out.shape[1]:
                        for k, v in new_input_set.items():
                            ad_sample_gen[k].append(v[i])
                        is_added = True
                        break
                    sub_idx = (logit_rank[i][logit_idx[i]]).item()
                    if sub_idx >= seq_len[i]:
                        continue
                    sub_cand = choose_largest_cand(out[i][sub_idx], new_input_set["input_ids"][i][sub_idx], tokenizer, tag2id, confusion_set)
                if is_added:
                    continue

                sub_id = tokenizer(sub_cand, add_special_tokens=False)["input_ids"][0]
                new_input_set["input_ids"][i][sub_idx] = sub_id

                pinyin_ids = new_input_set["pinyin_ids"].cpu().numpy().tolist()
                pinyin_ids[i][sub_idx] = processor.pinyin_processor.convert_ch_to_pinyin_id(sub_cand)
                new_input_set["pinyin_ids"] = torch.Tensor(pinyin_ids).long().to(model.device)

                for k, v in new_input_set.items():
                    new_input_set_[k].append(v[i])
                new_output_set_.append(new_output_set[i])
                error_count[i] += 1
                logit_rank_.append(logit_rank[i])
                logit_idx_.append(logit_idx[i])
                error_count_.append(error_count[i])
                error_bound_.append(error_bound[i])
            else:
                for k, v in new_input_set.items():
                    ad_sample_gen[k].append(v[i])

        for k, v in new_input_set_.items():
            if len(v) != 0:
                new_input_set_[k] = torch.stack(v)
        new_input_set = new_input_set_
        new_output_set = new_input_set_["labels"]
        logit_rank = logit_rank_
        logit_idx = logit_idx_
        error_count = error_count_
        error_bound = error_bound_

    for k, v in ad_sample_gen.items():
        ad_sample_gen[k] = torch.stack(v)
    ad_sample_gen = _prepare_inputs(ad_sample_gen, model)

    return ad_sample_gen


def choose_largest_cand(prob, gold, tokenizer, tag2id, confusion_set):
    """
    choose the largest prob candidate ch from confusion set.
    Args:
        prob: [num_labels]
        gold: [1]
    """
    ch = tokenizer.convert_ids_to_tokens(gold.item())
    if len(ch) > 1 or not is_chinese_char(ord(ch)):
        return None
    if ch not in confusion_set.keys():
        return None
    max_prob = None
    max_cand = None
    for cand in confusion_set[ch]:
        if cand not in tag2id.keys():
            continue
        cand_id = tag2id[cand]
        if max_prob is None or prob[cand_id] > max_prob:
            max_prob = prob[cand_id]
            max_cand = cand
    return max_cand


def encode_tags(tags, tag2id, encodings):
    # the first token is [CLS]
    offset = 1
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, input_id in zip(labels, encodings.input_ids):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(input_id), dtype=int) * -100
        for index, label in enumerate(doc_labels):
            doc_enc_labels[index + offset] = label
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def convert_label_by_vocab(texts, tags, unique_tags, use_copy_label=False):
    for i in trange(len(texts), desc="converting label by vocab..."):
        for j in range(len(texts[i])):
            if (use_copy_label and texts[i][j] == tags[i][j]) or len(texts[i][j]) > 1 or \
                    not is_chinese_char(ord(texts[i][j])):
                tags[i][j] = "<copy>"
            elif tags[i][j] not in unique_tags:
                tags[i][j] = "<unk>"
    return tags


def get_confusionset():
    shape_confusion_set = load_json("Data/confusion/shapeConfusion.json")
    sound_confusion_set = load_json("/Data/confusion/soundConfusion.json")
    confusion_set = shape_confusion_set.copy()
    confusion_set.update(sound_confusion_set)
    for k, v in confusion_set.items():
        confusion_set[k] = list(set(v))
    return confusion_set
