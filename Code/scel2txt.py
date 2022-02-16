import struct
import os
from pathlib import Path
import opencc

startPy = 0x1540                # 拼音表偏移
startChinese = 0x2628           # 汉语词组表偏移
GPy_Table = {}                  # 全局拼音表
GTable = []                     # 解析结果, 元组(词频,拼音,中文词组)的列表


# 原始字节码转为字符串
def byte2str(data):
    pos = 0
    str = ''
    while pos < len(data):
        c = chr(struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0])
        if c != chr(0):
            str += c
        pos += 2
    return str


# 获取拼音表
def get_py_table(data):
    data = data[4:]
    pos = 0
    while pos < len(data):
        index = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        pos += 2
        lenPy = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        pos += 2
        py = byte2str(data[pos:pos + lenPy])
        GPy_Table[index] = py
        pos += lenPy


# 获取一个词组的拼音
def get_word_py(data):
    pos = 0
    ret = ''
    while pos < len(data):
        index = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        ret += GPy_Table[index]
        pos += 2
    return ret


# 读取中文表
def get_chinese(data):
    pos = 0
    while pos < len(data):
        # 同音词数量
        same = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]

        # 拼音索引表长度
        pos += 2
        py_table_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]

        # 拼音索引表
        pos += 2
        py = get_word_py(data[pos: pos + py_table_len])

        # 中文词组
        pos += py_table_len
        for i in range(same):
            # 中文词组长度
            c_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 中文词组
            pos += 2
            word = byte2str(data[pos: pos + c_len])
            # 扩展数据长度
            pos += c_len
            ext_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 词频
            pos += 2
            count = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]

            # 保存
            GTable.append((count, py, word))

            # 到下个词的偏移位置
            pos += ext_len


def scel2txt(file_name):
    file_name = os.path.abspath(file_name)
    # 分隔符
    print("-" * 60)
    # 读取文件
    with open(file_name, 'rb') as f:
        data = f.read()

    print("词库名：", byte2str(data[0x130:0x338]))  # .encode('GB18030')
    print("词库类型：", byte2str(data[0x338:0x540]))
    print("描述信息：", byte2str(data[0x540:0xd40]))
    print("词库示例：", byte2str(data[0xd40:startPy]))

    get_py_table(data[startPy:startChinese])
    get_chinese(data[startChinese:])


def deduplication(input_filename, output_filename):
    result = []
    converter = opencc.OpenCC("tw2s.json")
    with open(input_filename, "r", encoding="utf-8") as fin:
        input_filename = fin.readlines()
        for i in input_filename:
            _, _, word = i.split(" ")
            word = word.strip()
            word = converter.convert(word)
            if word == "":
                continue
            result.append(word)
    print("去重前词汇表数量为：" + str(len(result)))
    result = list(set(result))
    print("去重后词汇表数量为：" + str(len(result)))
    with open(output_filename, "w", encoding="utf-8") as fout:
        res = []
        for word in result:
            res.append(str(word))
        fout.write("\n".join(res))


if __name__ == '__main__':

    # scel所在文件夹路径
    in_dir_path = "../Data/vocab/scel"
    file_names = [
        # os.path.join(in_dir_path, "日本地名第一版.scel"),
        # os.path.join(in_dir_path, "765个世界主要城市.scel"),
        # os.path.join(in_dir_path, "世界所有国家及其首都.scel"),
        # os.path.join(in_dir_path, "台北市城市信息精选.scel"),
        # os.path.join(in_dir_path, "基隆市城市信息精选.scel"),
        # os.path.join(in_dir_path, "高雄市城市信息精选.scel"),
        # os.path.join(in_dir_path, "台南市城市信息精选.scel"),
        # os.path.join(in_dir_path, "全国各地省市名.scel"),
        os.path.join(in_dir_path, "机关团体公文写作开拓思路常用词库.scel"),
        # os.path.join(in_dir_path, "全国县及县以上行政区划地名.scel")
    ]

    for f in file_names:
        scel2txt(f)

    output_dir_path = "../Data/"
    output_filename = os.path.join(os.path.abspath(output_dir_path), "vocab", "公文写作.txt")
    output_path = Path(output_filename)
    output = []
    for count, py, word in GTable:
        output.append(str(count) + ' ' + py + ' ' + word)
    output_path.write_text("\n".join(output))

    print("-" * 60)

    # 去重
    vocab_filename = "公文写作.txt"
    deduplication(output_filename, output_filename)
