from common_utils import read_table_file

# glyce_res = "../Results/glyce/two_fonts/checkpoint-4500-2015.result"
# pinyin_res = "../Results/glyce/two_fonts/sighan_157500/sim/results/checkpoint-4000-2015.result"

# gold = read_table_file(glyce_res, [1])
# glyce_result = read_table_file(glyce_res, [2])
# pinyin_result = read_table_file(pinyin_res, [2])

# print("字形答对，但是拼音不对")
# i = 0
# for g, p, l in zip(glyce_result, pinyin_result, gold):
#     if g != p and g == l:
#         i += 1
#         print(f"字形：{g}")
#         print(f"拼音：{p}")
#         print(f"gold: {l}")
# print("-" * 30)
# print("-" * 30)
# print(f"{i}")

# print("拼音对，但是字形不对")
# i = 0
# for g, p, l in zip(glyce_result, pinyin_result, gold):
#     if g != p and p == l:
#         i += 1
#         print(f"字形：{g}")
#         print(f"拼音：{p}")
#         print(f"gold: {l}")
# print("-" * 30)
# print("-" * 30)
# print(f"{i}")

# print("拼音，字形都没对")
# i = 0
# for g, p, l in zip(glyce_result, pinyin_result, gold):
#     if g != l and p != l:
#         i += 1
#         print(f"字形：{g}")
#         print(f"拼音：{p}")
#         print(f"gold: {l}")
# print("-" * 30)
# print("-" * 30)
# print(f"{i}")


null = "../Results/finished/sighan_16500/sim/results/checkpoint-3000-null.result"
vocab = "../Results/finished/sighan_16500/sim/results/checkpoint-3000-odw.result"

gold = read_table_file(null, [1])
null_res = read_table_file(null, [2])
vocab_res = read_table_file(vocab, [2])

print("空对，但是customize不对")
i = 0
for n, v, g in zip(null_res, vocab_res, gold):
    if n != v and n == g:
        i += 1
        print(f"空：{n}")
        print(f"有: {v}")
        print(f"金: {g}")
print("-" * 30)
print(f"{i}")
print("=" * 50)

print("customize对，但是空不对")
i = 0
for n, v, g in zip(null_res, vocab_res, gold):
    if n != v and v == g:
        i += 1
        print(f"空：{n}")
        print(f"有: {v}")
        print(f"金: {g}")
print("-" * 30)
print(f"{i}")
print("=" * 50)

print("customize，空都没对")
i = 0
for n, v, g in zip(null_res, vocab_res, gold):
    if n != g and v != g:
        i += 1
        print(f"空：{n}")
        print(f"有: {v}")
        print(f"金: {g}")
print("-" * 30)
print(f"{i}")
print("=" * 50)
