from common_utils import dump_json
with open("../Data/confusion/spellGraphs.txt", "r", encoding="utf-8") as f:
    lines = []
    for line in f:
        lines.append(f.readline().strip().split("|"))

types = {
    "形近": 0,
    "同音异调": 1,
    "近音异调": 1,
    "同部首同笔画": 0,
    "同音同调": 1,
    "近音同调": 1,
}

xj_dict = dict()
yj_dict = dict()

for line in lines:
    s, t, r = line
    if types[r] == 0 and types[r] != 1:       # both in shape and in sound
        if xj_dict.get(s):
            xj_dict[s].append(t)
        else:
            xj_dict[s] = [t]
    elif types[r] != 0 and types[r] == 1:
        if yj_dict.get(s):
            yj_dict[s].append(t)
        else:
            yj_dict[s] = [t]
    else:
        print("both ")

dump_json(xj_dict, "../Data/confusion/shapeConfusion.json")

dump_json(yj_dict, "../Data/confusion/soundConfusion.json")
