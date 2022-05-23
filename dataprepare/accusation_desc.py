# 生成条款、指控及其描述文件
import json

# f1 = open("./all_accusation.txt" ,"r", encoding="utf-8")
# f2 = open("./accusation_description.json", "w", encoding="utf-8")
# s = f1.readlines()
# f1.close()
# for accu in s:
#     dic = {}
#     accu = accu.strip()
#     dic["article"] = 0
#     dic["accusation"] = accu
#     dic["desc"] = ""
#     f2.write(json.dumps(dic, ensure_ascii=False)+"\n")
# f2.close()

f = open("accusation_description.json", "r", encoding="utf-8")
for line in f:
    d = json.loads(line)
    print(d)