import os
import json
import tokenizers

from tqdm import tqdm
from dataclasses import dataclass
from pypinyin import pinyin, Style
from torch.utils.data import Dataset

from Code.processor import PinyinProcessor
from typing import Optional, Union

from common_utils import is_chinese_char
from tokenizers.implementations.bert_wordpiece import BertWordPieceTokenizer

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding

UNK_LABEL = "<unk>"
COPY_LABEL = "<copy>"
NOCOPY_LABEL = "<nocopy>"

model_name = "Transformers/glyce"
py_processor = PinyinProcessor(model_name)


class TokenCLSDataset(Dataset):
    def __init__(self, encodings, labels, pinyin_ids=None):
        self.encodings = encodings
        self.labels = labels
        self.pinyin_ids = list(pinyin_ids) if pinyin_ids else None

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["pinyin_ids"] = self.pinyin_ids[idx] if self.pinyin_ids else None
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

        pinyin_ids = inputs.pop("pinyin_ids")
        if pinyin_ids:
            pinyin_pad = (0, 0, 0, 0)
            inputs["pinyin_ids"] = pinyin_ids + [pinyin_pad for _ in range(difference)]

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
