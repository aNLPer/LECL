import json
import thulac

min_length = float("inf")
max_length = 0
line_num = 0
max_length_line = 0
length_1k = []
f1 = open("../dataset/CAIL-SMALL/data_train_filtered.json", "r", encoding="utf-8")
thu = thulac.thulac(seg_only=True)
for line in f1:
    line_num += 1
    if line_num % 1000 == 0:
        print(line_num)
    example = json.loads(line)
    example_fact = thu.cut(example["fact"],text=True).split()
    example_articles = example["meta"]["relevant_articles"]
    example_accusations = example["meta"]["accusation"]
    if len(example_fact)>1000:
        length_1k.append(len(example_fact))
        length_1k.append(line_num)
    if len(example_fact)<min_length:
        min_length = len(example_fact)
    if len(example_fact)>max_length:
        max_length = len(example_fact)
        max_length_line = line_num
print("min_length：",min_length)
print("max_length：", max_length)
print("max_length_line:", max_length_line)
print("length_1k:", length_1k[0:5])