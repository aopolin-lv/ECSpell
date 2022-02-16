import os
from typing import OrderedDict
import torch
import common_utils
from transformers import AutoTokenizer, BertForTokenClassification
from pipeline import tagger
from model import ECSpell
from glyce.dataset_readers.bert_config import Config
from data_processor import py_processor
from processor import Processor
from csc_evaluation.evaluate_utils import compute_metrics, official_compute_metrics
import random


def evaluate(pred_filename, need_tokenize=True):
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

        results.append((line_src, line_tgt, line_pred))

    compute_metrics(results)
    official_compute_metrics(results)
    return


def preprocess(src_filenames):
    """ preprocess data"""
    all_pairs = []
    for src_filename in src_filenames:
        all_pairs.extend(common_utils.read_table_file(src_filename, output_indexes=[1, 2]))
    src_sents = [x[0] for x in all_pairs]
    # record the indexes of not chinese characters
    vocab = []
    for sent in src_sents:
        line = [0 for _ in range(len(sent))]
        for i, c in enumerate(sent):
            if not common_utils.is_chinese_char(ord(c)):
                line[i] = 1
        vocab.append(line)
    clean_src = [common_utils.clean_text(sent) for sent in src_sents]
    return all_pairs, src_sents, clean_src, vocab


def postprocess(inputs, src_sents, vocab):
    assert len(inputs) == len(src_sents) == len(vocab)
    res = []
    for pre, src, v in zip(inputs, src_sents, vocab):
        assert len(pre) == len(src) == len(v)
        pre = list(pre)
        src = list(src)
        # restore not chinese characters
        for i in range(len(pre)):
            if v[i] == 1:
                pre[i] = src[i]
        res.append("".join(pre))
    return res


def predict(src_filenames, model_dirname, tokenizer_filename, label_filename, result_filename, processor=None, use_word=False):
    labels = open(label_filename, encoding='utf-8').read().split('\n')
    model_filename = os.path.join(model_dirname, "pytorch_model.bin")
    # load model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_filename)
    glyce_config_path = "Transformers/glyce_bert_both_font.json"
    glyce_config = Config.from_json_file(glyce_config_path)
    glyce_config.glyph_config.bert_model = "Transformers/glyce"
    model = ECSpell(glyce_config, py_processor.get_pinyin_size(), len(labels), True)
    # model = BertForTokenClassification.from_pretrained("Transformers/glyce", num_labels=len(labels))
    model.load_state_dict(torch.load(model_filename, map_location=torch.device("cpu")))
    all_pairs, src_sents, clean_src, vocab = preprocess(src_filenames)

    tag_sentences = tagger(model, tokenizer, clean_src, processor.vocab_processor, use_word,
                           use_pinyin=True, pinyin_processor=processor.pinyin_processor,
                           device="cuda", labels=labels, weight=3.5, RSM=False, ASM=True)
    outputs = []

    for sentence, tag_sentence in zip(clean_src, tag_sentences):
        sentence = list(sentence)
        for tag_dict in tag_sentence:
            if tag_dict == []:
                break
            tag_token = labels[tag_dict['entity']]
            if tag_dict["end"] - tag_dict["start"] > 1:
                continue
            elif tag_token in ['<copy>', '<unk>']:
                continue
            # only useful in <only detection> mode
            elif tag_token == "<nocopy>":
                for i in range(tag_dict["start"], tag_dict["end"]):
                    sentence[i] = "X"
            else:
                for i in range(tag_dict["start"], tag_dict["end"]):
                    sentence[i] = tag_token[0]
        outputs.append("".join(sentence))

    # postprocess
    outputs = postprocess(outputs, src_sents, vocab)

    with open(result_filename, 'w', encoding='utf-8') as f:
        for index, line in enumerate(outputs):
            f.write('\t'.join(all_pairs[index] + [line]))
            f.write('\n')

    return outputs


def main():
    random.seed(42)
    # model_name = "glyce"
    checkpoint = "2250"

    # result_dir = os.path.abspath(f"Results/finished/pretrain_continous")
    result_dir = os.path.abspath(f"Results/sighan_continous")
    tokenizer_filename = os.path.abspath("Transformers/glyce")

    test_filenames = [
        os.path.abspath("csc_evaluation/data/basedata/simplified/test2015.txt"),
        # os.path.abspath("csc_evaluation/data/customized_data/law.txt"),
        # os.path.abspath("../csc_evaluation/data/document_writing/test.txt"),
    ]
    model_filename = os.path.join(result_dir, "results", f"checkpoint-{checkpoint}")

    label_filename = os.path.join(result_dir, 'labels.txt')
    result_filename = os.path.join(result_dir, "results", f"checkpoint-{checkpoint}-test.result")

    vocab_filename = "csc_evaluation/data/document_writing/wordlist/公文写作.txt.clean"
    vocab_filename = "Data/vocab/sighan2015_vocab_75.txt"
    print("=" * 40)
    print(vocab_filename)
    print("=" * 40)
    processor = Processor(vocab_filename)

    print('predicting')
    predict(test_filenames, model_filename, tokenizer_filename, label_filename, result_filename, processor,
            use_word=True)

    print('evaluating')
    evaluate(result_filename, need_tokenize=False)
    return


if __name__ == '__main__':
    main()
