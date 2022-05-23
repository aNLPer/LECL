import os
import pickle
import json
import utils.commonUtils as utils


# 加载停用词表
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 生成acc2desc字典
def get_acc_desc(file_path):
    acc2des = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            dict = json.loads(line)
            if dict["accusation"] not in acc2des:
                acc2des[dict["accusation"]] = dict["desc"]
    return acc2des
acc2desc = get_acc_desc("accusation_description.json")
print(acc2desc)

# 构造数据集
def getData(case_path, acc2desc):
    '''
    构造数据集：["case_desc", "acc", "acc_desc"]
    :param case_path: 案件描述文件
    :param acc2desc: 指控：指控描述 （字典）
    :return: [["case_desc", "acc", "acc_desc"],......]
    '''
    items = []
    with open(case_path, "r", encoding="utf-8") as f:
        for line in f:
            item = []
            example = json.loads(line)
            item.append(example["fact"])
            item.append(example['meta']['accusation'])
            item.append(acc2desc[example['meta']['accusation']])
            items.append(item)




get_acc_desc(file_path="accusation_description.json")

# f = open("train_filtered_accusation2num.pkl","rb")
# dict_accusation = pickle.load(f)
# f.close()
#
# f = open("train_filtered_articles2num.pkl", "rb")
# dict_article = pickle.load(f)
# f.close()
#
# f = open("train_accusation2num.pkl", "rb")
# d = pickle.load(f)
# print(d["盗窃"])
#
# print(len(dict_accusation))
# print(len(dict_article))
# print(utils.sum_dict(dict_article), utils.sum_dict(dict_accusation))