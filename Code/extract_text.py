from common_utils import read_table_file

in_filename = r'D:\repos\cscbase\data\basedata\bak\AllSIGHAN.txt'

out_filename = r'D:\repos\Corpus\sighan.labels.txt'
labels = read_table_file(in_filename, [2])
with open(out_filename, 'w', encoding='utf-8') as f:
    for label in labels:
        f.write(label[0])
        f.write('\n')
