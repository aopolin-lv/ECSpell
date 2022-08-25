import os
import logging
import json
import argparse
import common_utils
from collections import defaultdict
from datetime import datetime
common_utils.setSeed(42)

from tqdm import trange
import numpy as np
import torch

from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
)
from transformers.trainer_utils import IntervalStrategy
from transformers.trainer_callback import EarlyStoppingCallback

from glyce.dataset_readers.bert_config import Config
from Code.data_processor import TokenCLSDataset, CscDataCollator
from Code.processor import Processor
from Code.model import ECSpell
from csc_evaluation.evaluate_utils import EvalHelper, compute_metrics
from Code.trainer import Trainer, TestArgs
from csc_evaluation.common_utils import set_logger

logger = logging.getLogger(__name__)


UNK_LABEL = "<unk>"
COPY_LABEL = "<copy>"
NOCOPY_LABEL = "<nocopy>"
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


def normalize_tags(train_texts, train_tags, keep_count=1, use_copy_label=True, only_for_detection=False):
    unique_tags = _get_tags(train_texts, train_tags, use_copy_label,
                            keep_count) if not only_for_detection else [COPY_LABEL, NOCOPY_LABEL]

    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    return unique_tags, tag2id, id2tag


def _get_tags(train_texts, train_tags, use_copy_label, keep_count):
    label_dict = defaultdict(int)
    for train_text, train_tag in zip(train_texts, train_tags):
        for token, label in zip(train_text, train_tag):
            if (use_copy_label and token == label) or len(token) > 1 or not common_utils.is_chinese_char(ord(token)):
                label_dict[COPY_LABEL] += 1
            else:
                label_dict[label] += 1

    pairs = sorted(label_dict.items(), key=lambda x: -x[1])
    unique_tags = [UNK_LABEL]
    for label, count in pairs:
        if count < keep_count:
            break
        unique_tags.append(label)
    return unique_tags


def convert_label_by_vocab(texts, tags, unique_tags, use_copy_label=True, only_for_detection=False):
    for i in trange(len(texts), desc="converting label by vocab...", position=0, leave=True):
        for j in range(len(texts[i])):
            if only_for_detection:
                if texts[i][j] == tags[i][j]:
                    tags[i][j] = COPY_LABEL
                else:
                    tags[i][j] = NOCOPY_LABEL
                continue

            if (use_copy_label and texts[i][j] == tags[i][j]) or len(texts[i][j]) > 1 or \
                    not common_utils.is_chinese_char(ord(texts[i][j])):
                tags[i][j] = COPY_LABEL
            elif tags[i][j] not in unique_tags:
                tags[i][j] = UNK_LABEL
    return tags


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


def load_data(args):
    logger.info("********  Load data start *********")
    cached_dir = args.cached_dir
    train_cached_fp = os.path.join(cached_dir, f"{args.dataset_type}_train_cached")
    val_cached_fp = os.path.join(cached_dir, f"{args.dataset_type}_val_cached")
    unique_cached_fp = os.path.join(cached_dir, f"{args.dataset_type}_unique_cached")

    if os.path.exists(train_cached_fp) and not args.overwrite_cached:
        logger.info("  Load train data cache  ")
        train_cached = torch.load(train_cached_fp)
        train_texts, train_tags = train_cached["train_texts"], train_cached["train_tags"]
    else:
        logger.info("  Create train data  ")
        train_texts, train_tags = common_utils.read_tagging_data(args.train_files, args.max_sent_length)
        torch.save({"train_texts": train_texts, "train_tags": train_tags}, train_cached_fp)

    if os.path.exists(val_cached_fp) and not args.overwrite_cached:
        logger.info("  Load val data cache  ")
        val_cached = torch.load(val_cached_fp)
        val_texts, val_tags = val_cached["val_texts"], val_cached["val_tags"]
    else:
        logger.info("  Create val data  ")
        val_texts, val_tags = common_utils.read_tagging_data(args.val_files)
        torch.save({"val_texts": val_texts, "val_tags": val_tags}, val_cached_fp)

    if os.path.exists(unique_cached_fp) and not args.overwrite_cached:
        logger.info("  Load unique cached  ")
        unique_cached = torch.load(unique_cached_fp)
        unique_tags, tag2id = unique_cached["unique_tags"], unique_cached["tag2id"]
    else:
        logger.info("  Create normalize tags  ")
        unique_tags, tag2id, _ = normalize_tags(
            train_texts, train_tags, keep_count=args.keep_count, use_copy_label=args.use_copy_label,
            only_for_detection=args.only_for_detection)
        torch.save({"unique_tags": unique_tags, "tag2id": tag2id}, unique_cached_fp)

    logger.info("********  Load data complete *********")
    return train_texts, train_tags, val_texts, val_tags, unique_tags, tag2id


def load_input_data(train_texts, val_texts, tokenizer, args):
    cached_dir = args.cached_dir
    cached_train_encodings = os.path.join(cached_dir, f"{args.dataset_type}_train_inputs")
    cached_val_encodings = os.path.join(cached_dir, f"{args.dataset_type}_val_inputs")

    if os.path.exists(cached_train_encodings) and not args.overwrite_cached:
        logger.info(" Load train inputs from cached dir")
        train_inputs = torch.load(cached_train_encodings)
        train_encodings = train_inputs["train_encodings"]
    else:
        logger.info(" Create train inputs")
        train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True)
        torch.save({"train_encodings": train_encodings}, cached_train_encodings)
        logger.info(" Save train inputs successfully")

    if os.path.exists(cached_val_encodings) and not args.overwrite_cached:
        logger.info(" Load val inputs from cached dir")
        val_inputs = torch.load(cached_val_encodings)
        val_encodings = val_inputs["val_encodings"]
    else:
        logger.info(" Create val inputs")
        val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True)
        torch.save({"val_encodings": val_encodings}, cached_val_encodings)
        logger.info(" Save val inputs successfully")

    return train_encodings, val_encodings


def filter_alignment(encodings, labels):
    for i in range(len(encodings.encodings)):
        if not (len(encodings.encodings[i]) == len(labels[i])):
            encodings.encodings.pop(i)
            for k, v in encodings.data:
                encodings.data[k].pop(i)
            labels.pop(i)
    return encodings, labels


def main():
    parser = argparse.ArgumentParser(description="Train parameters")
    group_data = parser.add_argument_group("Data")
    group_data.add_argument("--model_name", default="Transformers/bert-base-chinese", help="model name")
    group_data.add_argument(
        "--train_files", default="Data/traintest/bert-base-chinese/nlg/train.txt", help="train files, split by \";\"")
    group_data.add_argument(
        "--val_files", default="Data/traintest/bert-base-chinese/nlg/val.txt", help="evaluation files, split by \";\"")
    group_data.add_argument(
        "--test_files", default="Data/data/basedata/test2015.txt", help="test files, split by \";\"")
    group_data.add_argument("--cached_dir", default="Cached", help="the directory of cached")
    group_data.add_argument("--result_dir", default="Results/nlg_30epoch", help="result path")
    group_data.add_argument("--seed", default=42, type=int, help="random seed")
    group_data.add_argument("--vocab_file", default="Data/vocab/allNoun.txt", help="vocab files")
    group_data.add_argument("--glyce_config_path", default="Transformers/glyce_bert_both_font.json", type=str, help="glyce_config path")
    group_data.add_argument("--overwrite_cached", default=True, type=common_utils.str2bool, help="whether overwrite cached")
    group_data.add_argument("--load_pretrain_checkpoint", default="Results/ecspell", type=str, help="the path of pretrain checkpoint file, default is None")
    group_data.add_argument("--font_type", default="sim", type=str, help="['sim', 'tra', 'test']")
    group_data.add_argument("--keep_count", default=1, type=int, help="threshold for tag frequency")
    group_data.add_argument("--max_sent_length", default=128, type=int, help="maximal sentence length")
    group_data.add_argument("--use_word_feature", default=False, type=common_utils.str2bool, help="whether use word feature or not")
    group_data.add_argument("--use_copy_label", default=False, type=common_utils.str2bool, help="whether to convert identical tokens to the COPY tag")
    group_data.add_argument("--use_pinyin", default=True, type=common_utils.str2bool, help="whether use pinyin when training")
    group_data.add_argument("--only_for_detection", default="False", type=common_utils.str2bool,
                            help="whether to convert the output to (Correction,Wrong) labels")

    group_model = parser.add_argument_group("Model")
    group_model.add_argument("--local_rank", default=-1, type=int)
    group_model.add_argument("--transformer_config_path", default=None, type=str, help="the filepath of transformer_config.json")
    group_model.add_argument("--overwrite_output_dir", default="True", type=common_utils.str2bool, help="whether to overwrite existing model")
    group_model.add_argument("--per_device_train_batch_size", default=64, type=int, help="train batch size")
    group_model.add_argument("--per_device_eval_batch_size", default=128, type=int, help="evaluation batch size")
    group_model.add_argument("--gradient_accumulation_steps", default=2, type=int,
                             help="Number of updates steps to accumulate the gradients for, before performing a "
                                  "backward/update pass")
    group_model.add_argument("--num_train_epochs", default=30, type=float, help="Number of train epoches, can be a float value")
    group_model.add_argument("--fp16", default=True, type=common_utils.str2bool, help="whether use fp16 to speed up")
    group_model.add_argument("--do_test", default=False, type=common_utils.str2bool, help="whether do test when training")
    group_model.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    group_model.add_argument("--save_steps", default=500, type=int, help="Steps to save model/evaluation")
    group_model.add_argument("--logging_steps", default=500, type=int, help="Steps to logging")
    group_model.add_argument("--label_smoothing_factor", default=0, type=float, help="label smoothing factor, zero to disable")
    group_model.add_argument("--compute_metrics", default=True, type=common_utils.str2bool, help="whether use compute metrics")
    group_model.add_argument("--report_to", default="tensorboard", type=str, help="whether use compute metrics")
    args = parser.parse_args()

    # Parsing parameters
    args.train_files = args.train_files.split(";")
    args.val_files = args.val_files.split(";")
    args.test_files = args.test_files.split(";")
    args.mode = "word" if args.use_word_feature else "base"
    args.copy_mode = "copy" if args.use_copy_label else "no_copy"
    if args.font_type not in ["sim", "tra", "both", "test"]:
        raise "font type error"
    args.model = args.model_name.split("/")[-1] if os.path.exists(args.model_name) else args.model_name
    args.cached_dir = os.path.join(args.cached_dir)
    args.result_dir = os.path.join(args.result_dir)
    args.dataset_type = args.train_files.split("/")[-1].split(".")[0]

    log_file = os.path.join(args.result_dir, "logger")
    set_logger(logger=logger, log_filename=log_file)
    common_utils.setSeed(args.seed)
    logger.info("******************* Train parameters *******************")
    for k, v in vars(args).items():
        logger.info("  {0}:  {1}".format(k, v))
    logger.info("******************* Train parameters *******************")

    if not os.path.exists(args.cached_dir):
        os.makedirs(args.cached_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    train_texts, train_tags, val_texts, val_tags, labels, tag2id = load_data(args)
    # if use pretrain parameters, replace the label tags
    if args.load_pretrain_checkpoint:
        with open(os.path.join(args.load_pretrain_checkpoint, "labels.txt"), "r", encoding="utf-8") as f:
            labels = f.read().strip().split("\n")
    tag2id = {tag: id for id, tag in enumerate(labels)}

    logger.info("train size: {}, valid size: {}, tag count: {}".format(len(train_texts), len(val_texts), len(labels)))

    with open(os.path.join(args.result_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    logger.info("convert label by vocab, use copy label: {} ======>".format(args.use_copy_label))
    train_tags = convert_label_by_vocab(
        train_texts, train_tags, labels, use_copy_label=args.use_copy_label, only_for_detection=args.only_for_detection)
    val_tags = convert_label_by_vocab(
        val_texts, val_tags, labels, use_copy_label=args.use_copy_label, only_for_detection=args.only_for_detection)

    processor = Processor(args.vocab_file)

    logger.info("load tokenizer and pretrained model: {}".format(args.model))

    logger.info("Initialize model")
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, config=config)
    glyce_config = Config.from_json_file(args.glyce_config_path)
    glyce_config.glyph_config.bert_model = args.model_name
    # model = BertForTokenClassification.from_pretrained("Transformers/glyce", num_labels=len(labels))
    model = ECSpell(glyce_config, processor.pinyin_processor.get_pinyin_size(), len(labels), True)
    if args.load_pretrain_checkpoint:
        logger.info(" === Load pretrain model parameters !!! === ")
        logger.info(f"{os.path.join(args.load_pretrain_checkpoint, 'results', 'checkpoint')}")
        checkpoint_file = os.path.join(args.load_pretrain_checkpoint, "results", f"checkpoint", "pytorch_model.bin")
        original_checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        model.load_state_dict(original_checkpoint)
        logger.info("=" * 30)

    parameter_number = common_utils.get_parameter_number(model)
    args.total_parameter = parameter_number["Total"]
    args.trainable_parameter = parameter_number["Trainable"]
    logger.info(f" Total params: {format(parameter_number['Total'], ',')}")
    glyce_parameter = common_utils.get_parameter_number(model.glyph_transformer.glyph_embedding)
    logger.info(f" Glyph params: {format(glyce_parameter['Total'], ',')}")
    pinyin_parameter = common_utils.get_parameter_number(model.glyph_transformer.pho_embedding)
    logger.info(f" Pinyin params: {format(pinyin_parameter['Total'], ',')}")

    with open(os.path.join(args.result_dir, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    logger.info(" Tokenize data and add features...")
    train_encodings, val_encodings = load_input_data(train_texts, val_texts, tokenizer, args)

    train_pinyin_ids, val_pinyin_ids = None, None
    if args.use_pinyin:
        cached_pinyin_ids = os.path.join(args.cached_dir, f"{args.dataset_type}_pinyin_ids")
        if not args.overwrite_cached and os.path.exists(cached_pinyin_ids):
            pinyin_ids_file = torch.load(cached_pinyin_ids)
            train_pinyin_ids, val_pinyin_ids = pinyin_ids_file["train_pinyin_ids"], pinyin_ids_file["val_pinyin_ids"]
        else:
            train_pinyin_ids = processor.pinyin_processor.convert_sentence_to_pinyin_ids(train_texts, train_encodings)
            val_pinyin_ids = processor.pinyin_processor.convert_sentence_to_pinyin_ids(val_texts, val_encodings)
            torch.save({"train_pinyin_ids": train_pinyin_ids, "val_pinyin_ids": val_pinyin_ids}, cached_pinyin_ids)

    logger.info(" Encode tags of train and val datasets...")
    train_labels = encode_tags(train_tags, tag2id, train_encodings)
    val_labels = encode_tags(val_tags, tag2id, val_encodings)

    logger.info(f" before filter the len of train dataset: {len(train_labels)}")
    train_encodings, train_labels = filter_alignment(train_encodings, train_labels)
    val_encodings, val_labels = filter_alignment(val_encodings, val_labels)
    logger.info(f" after filter the len of train dataset: {len(train_labels)}")

    train_dataset = TokenCLSDataset(encodings=train_encodings, labels=train_labels, pinyin_ids=train_pinyin_ids)
    val_dataset = TokenCLSDataset(encodings=val_encodings, labels=val_labels, pinyin_ids=val_pinyin_ids)

    training_args = TrainingArguments(
        # output directory
        output_dir=os.path.join(args.result_dir, "results"),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        # batch size for evaluation
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        seed=args.seed,
        # directory for storing logs
        logging_dir=os.path.join(args.result_dir, "logs", TIMESTAMP),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=IntervalStrategy.STEPS,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.save_steps,
        overwrite_output_dir=args.overwrite_output_dir,
        fp16=args.fp16,
        local_rank=args.local_rank,
        metric_for_best_model="eval_cf1",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=3,
        report_to=args.report_to,
        # label_smoothing_factor=args.label_smoothing_factor,
    )

    data_collator = CscDataCollator(tokenizer)
    eval_helper = EvalHelper(val_texts, val_labels, tokenizer, labels, tag2id)
    test_args = TestArgs(
        test_filename=list(args.test_files),
        labels=labels,
        processor=processor,
        result_dir=args.result_dir
    ) if args.do_test else None

    logger.info("Start training")

    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        logfile=log_file,
        model=model,
        args=training_args,  # training arguments, defined above
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.compute_metrics else None,
        eval_helper=eval_helper,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=50)],
        use_pinyin=args.use_pinyin,
        test_args=test_args,
        tag2id=tag2id,
        processor=processor,
    )

    trainer.train()

    if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint is not None:
        logger.info(f"The best model is:{trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    main()
