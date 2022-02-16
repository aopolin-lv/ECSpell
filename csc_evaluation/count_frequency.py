import jieba
from tqdm import tqdm
from common_utils import is_chinese_char


def chinese_word(word):
    for c in word:
        if not is_chinese_char(ord(c)):
            return False
    return True


all_data = []
with open("data/document_writing/web_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        all_data.append(line.strip())
with open("data/document_writing/wordlist/政策法规标题.txt", "r", encoding="utf-8") as f:
    for line in f:
        all_data.append(line.strip())

words = []
for sent in tqdm(all_data):
    words += list(jieba.cut(sent))

dele = {'。', '！', '？', '的', '“', '”', '（', '）', ' ', '》', '《', '，', "\'"}
articleDict = {}
articleSet = set(words) - dele
for w in tqdm(articleSet):
    if len(w) > 1 and chinese_word(w):
        c = words.count(w)
        if c > 00:
            articleDict[w] = c
articlelist = sorted(articleDict.items(), key=lambda x : x[1], reverse=True)
with open("data/document_writing/vocab.txt", "w", encoding="utf-8") as f:
    for index, line in enumerate(articlelist):
        f.write(line[0])
print("abc")
