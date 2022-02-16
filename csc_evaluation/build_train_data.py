import os
import glob
import random
import logging
import argparse

import common_utils

random.seed(0)
logger = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description="Data Construction")
    parser.add_argument('--add_simplified', default=True,
                        type=common_utils.str2bool, help='add simplified data')
    parser.add_argument('--add_traditional', default=False,
                        type=common_utils.str2bool, help='add traditional data')
    parser.add_argument('--val_size', default=1000, type=int, help='the validation data size')
    parser.add_argument('--use_test_as_val', default=True, type=common_utils.str2bool, help='whether use test set as valid set')
    parser.add_argument('--two_stage', default=True, type=common_utils.str2bool, help="use two stage fintune schedule")

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()
    assert args.add_simplified or args.add_traditional, 'data cannot be empty'

    test_files = ["data/basedata/simplified/test2015.txt"]

    sim_files_sighan = glob.glob('data/basedata/simplified/train*.txt')
    sim_files_nlg = ["data/basedata/simplified/nlg.txt"]

    tra_files_sighan = glob.glob('data/basedata/traditional/train*.txt')
    tra_files_nlg = ["data/basedata/traditional/nlg.txt"]

    save_basedir = 'builds'

    common_utils.set_logger(logger, log_filename=os.path.join(save_basedir, 'construction.log'), file_mode='w')
    logger.info('Parameters:')
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))
    logger.info('')

    if not args.two_stage:
        filenames = []
        save_names = []
        if args.add_simplified:
            filenames += sim_files_sighan
            filenames += sim_files_nlg
            save_names.append('sim')

        if args.add_traditional:
            filenames == tra_files_sighan
            filenames += tra_files_nlg
            save_names.append('tra')

        generate(filenames, save_names, save_basedir, args, test_files)
    else:
        # SIGHAN data
        filenames = []
        save_names = []
        if args.add_simplified:
            filenames += sim_files_sighan
            save_names.append('sim')
        if args.add_traditional:
            filenames += tra_files_sighan
            save_names.append('tra')

        generate(filenames, save_names, save_basedir, args, test_files, "SIGHAN")

        # nlg data
        filenames = []
        save_names = []
        if args.add_simplified:
            filenames += sim_files_nlg
            save_names.append('sim')
        if args.add_traditional:
            filenames += tra_files_nlg
            save_names.append('tra')

        generate(filenames, save_names, save_basedir, args, test_files, "nlg")
    logger.info('Finish')
    return


def build_corpus(filenames, val_size, save_dir, use_test_as_val=False, test_files=None):
    if use_test_as_val:
        logger.info("Warning: use test set as valid set, "
                    "the val_size parameter will be useless")
        val_size = 0
    records = []
    for filename in filenames:
        lines = open(filename, encoding='utf-8').readlines()
        logger.info('Read file: {}, lines: {}'.format(filename, len(lines)))
        records += lines

    logger.info('Total lines: {}'.format(len(records)))
    random.shuffle(records)

    train = records[val_size:]
    if not use_test_as_val:
        val = records[:val_size]
    else:
        all_pairs = common_utils.read_table_file(test_files[0], [0, 1, 2])
        val = []
        for pair in all_pairs:
            line = "\t".join(pair)
            val.append(f"{line}\n")

    with open(os.path.join(save_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train)

    with open(os.path.join(save_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.writelines(val)
    return


def generate(filenames, save_names, save_basedir, args, test_files, suffix=None):
    if not suffix:
        save_dir = os.path.join(save_basedir, '_'.join(save_names))
    else:
        save_dir = os.path.join(save_basedir, "_".join(save_names), suffix)
        if suffix == "nlg":
            args.use_test_as_val = False
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    build_corpus(filenames, args.val_size, save_dir, args.use_test_as_val, test_files)


if __name__ == '__main__':
    main()
