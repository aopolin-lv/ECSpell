import json
import os
import text_utils
from tqdm import tqdm


def main():
    chinese_threshold = 0.8
    length_threshold = 8
    length_upperbound = 75
    save_split = 100000
    in_filename = r'../../Data/fakedata/new2016zh/news2016zh_train.json'
    out_dir = r'../../Data/fakedata/new2016zh/texts/train'
    # in_filename = r'../../Data/fakedata/new2016zh/news2016zh_valid.json'
    # out_dir = r'../../Data/fakedata/new2016zh/texts/valid'

    total_save_count = 0
    save_count = 0
    save_index = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f_save = open(os.path.join(out_dir, '{}.txt'.format(save_index)), 'w', encoding='utf-8')
    with open(in_filename, encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content = json.loads(line)['content']
            sents = text_utils.to_sentences(content)
            for sent in sents:
                if len(sent) < length_threshold or len(sent) > length_upperbound:
                    continue
                is_valid = False
                ch_count = 0
                for token in sent:
                    if text_utils.is_punctuation(token) or text_utils.is_ascii(token):
                        continue
                    if not text_utils.is_chinese(token):
                        is_valid = False
                        break
                    ch_count += 1
                    if ch_count > len(sent) * chinese_threshold:
                        is_valid = True
                        break
                if not is_valid:
                    continue
                f_save.write(sent + '\n')
                save_count += 1
                total_save_count += 1
                if save_count >= save_split:
                    save_count = 0
                    save_index += 1
                    f_save.close()
                    f_save = open(os.path.join(out_dir, '{}.txt'.format(save_index)), 'w', encoding='utf-8')
    f_save.close()
    print('total save count:', total_save_count, 'save files:', save_index + 1)
    return


if __name__ == '__main__':
    main()
