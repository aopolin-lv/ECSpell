import json
import os
import text_utils
from tqdm import tqdm
import glob


def main():
    chinese_threshold = 0.8
    length_threshold = 8
    length_upperbound = 75
    save_split = 100000
    in_filenames = glob.glob(r'D:\repos\Corpus\wiki_zh_2019\wiki_zh\*\wiki_*')
    out_dir = r'D:\repos\Corpus\wiki_zh_2019\texts'

    total_save_count = 0
    save_count = 0
    save_index = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f_save = open(os.path.join(out_dir, '{}.txt'.format(save_index)), 'w', encoding='utf-8')
    for in_filename in tqdm(in_filenames):
        for line in open(in_filename, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            content = json.loads(line)['text']
            paragraphs = content.split('\n')
            sents = []
            for paragraph in paragraphs:
                sents += text_utils.to_sentences(paragraph.strip())

            for sent in sents:
                if len(sent) < length_threshold or len(sent) > length_upperbound:
                    continue
                if not text_utils.is_punctuation(sent[-1]):
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
