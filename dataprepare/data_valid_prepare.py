# coding=utf-8
import os
import pickle
import json
import thulac
import re
import utils.commonUtils as commonUtils
from utils.commonUtils import Lang
from utils.commonUtils import Lang
BASE_PATH = "../dataset/CAIL-SMALL"

def valid_data_filter(sourcefilename, targetfilename):
    print("valid data processing start")
    fw = open(os.path.join(BASE_PATH,targetfilename), "w", encoding="utf-8")
    with open(os.path.join(BASE_PATH,sourcefilename), "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            example_articles = example['meta']['relevant_articles']
            example_accusation = example['meta']['accusation']
            example_fact = example['fact']
            # 仅统计单条款、单指控、仅一审的案件的指控和条款
            if len(example_articles) == 1 and \
                    len(example_accusation) == 1 and \
                    '二审' not in example_fact and \
                    len(example_fact)>=65:
                fw.write(line)
    fw.close()

# valid_data_filter("data_valid.json", "data_valid_filtered.json")

def get_valid_data_label(filename):
    filepath = os.path.join(BASE_PATH, filename)
    valid_data_accu = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            example_accusation = example['meta']['accusation'][0]
            if example_accusation not in valid_data_accu:
                valid_data_accu.append(example_accusation)
    return valid_data_accu
# valid_data_accu = get_valid_data_label("data_valid_filtered.json")
# f = open("train_id2acc.pkl","rb")
# id2acc = pickle.load(f)
# f.close()
# valid_acc_not_in_train_accu = []
# for accu in valid_data_accu:
#     if accu not in id2acc:
#         valid_acc_not_in_train_accu.append(accu)
# print(len(valid_acc_not_in_train_accu))

# 构造数据集
def getData(case_path, acc2desc, targetfile, train_acc = None, mode="base"):
    '''
    构造数据集：[[case_desc,case_desc,...], "acc", "acc_desc"]
    # 分词
    # 去除特殊符号（样本1）
    # 去除停用词（样本2）
    # 去除标点（样本3）
    # 去除停用词和标点（样本4）
    # 同类别的accusation(样本+4)
    :param case_path: 案件描述文件
    :param acc2desc: 指控：指控描述 （字典）
    :return: [[[case_desc,case_desc,...], "acc", "acc_desc"],]
    '''
    # 加载分词器
    thu = thulac.thulac(user_dict="Thuocl_seg.txt", seg_only=True)
    # 加载特殊符号
    special_symbols = commonUtils.get_filter_symbols("special_symbol.txt")
    # 加载停用词表
    stopwords = commonUtils.get_filter_symbols("stop_word.txt")
    # 加载标点
    punctuations = commonUtils.get_filter_symbols("punctuation.txt")
    fw = open(targetfile, "w", encoding="utf-8")
    count = 0
    with open(case_path, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            item = [] # 单条训练数据
            example = json.loads(line)
            if mode == "base":
                if example['meta']['accusation'][0].strip() not in train_acc:
                    continue
            if mode != "base":
                if example['meta']['accusation'][0].strip() in train_acc:
                    continue
            # 过滤law article内容
            example_fact = commonUtils.filterStr(example["fact"])
            # 分词,去除特殊符号
            example_fact_1 = [word for word in thu.cut(example_fact, text=True).split(" ") if word not in special_symbols]
            example_fact_1 = [re.sub(r"\d+","x", word) for word in example_fact_1]
            example_fact_1 = [word for word in example_fact_1
                              if word not in ["x年", "x月", "x日", "下午", "上午", "凌晨", "晚", "晚上", "x时", "x分", "许"]]
            # 去除停用词和标点
            example_fact_4 = [word for word in example_fact_1 if word not in punctuations and word not in stopwords]
            item.append(example_fact_4)
            item.append(example['meta']['accusation'][0].strip())
            # 指控描述
            acc_desc = acc2desc[example['meta']['accusation'][0]]
            # 指控描述分词，去除标点、停用词
            acc_desc = [word for word in thu.cut(acc_desc, text=True).split(" ")
                        if word not in stopwords and word not in punctuations]
            item.append(acc_desc)
            list_str = json.dumps(item, ensure_ascii=False)
            fw.write(list_str+"\n")
            if count%5000==0:
                print(f"已有{count}条数据被处理")
    fw.close()
# f = open("train_id2acc.pkl","rb")
# train_acc = pickle.load(f)
# f.close()
# acc2desc = commonUtils.get_acc_desc("accusation_description.json")
# getData(case_path="../dataset/CAIL-SMALL/data_valid_filtered.json",
#         acc2desc=acc2desc,
#         targetfile="..\dataset\CAIL-SMALL\data_valid_preprocessed(unseen_in_training).txt",
#         train_acc=train_acc,
#         mode="o")


# 将文本数据序列化
def word2Index(sourcefile, targetfile, lang, acc2id):
    # 数据集
    fi = open(sourcefile, "r", encoding="utf-8")
    fo = open(targetfile, "w", encoding="utf-8")
    count = 0
    for line in fi:
        count += 1
        # 文本数据
        item = json.loads(line)
        item_num = []
        item_num.append([lang.word2index[word] for word in item[0] if word in lang.word2index])
        item_num.append(acc2id[item[1]])
        item_num.append([lang.word2index[word] for word in item[2] if word in lang.word2index])
        # 序列化并写入
        item_num_str = json.dumps(item_num, ensure_ascii=False)
        fo.write(item_num_str+"\n")
        if count%5000==0:
            print(f"已处理{count}条数据")
    fi.close()
    fo.close()

# f = open("./lang_data_train_preprocessed.pkl", "rb")
# lang = pickle.load(f)
# f.close()
# f = open("./train_acc2id.pkl", "rb")
# acc2id = pickle.load(f)
# f.close()
# sourcefile = "../dataset/CAIL-SMALL/data_valid_preprocessed(base).txt"
# targetfile = "../dataset/CAIL-SMALL/data_valid_forModel(base).txt"
# word2Index(sourcefile, targetfile, lang, acc2id)

# s = "[[12, 6341, 317, 49, 1948, 4541, 18, 4542, 1609, 4055, 121, 289, 137, 137, 237, 238, 12, 6341, 586, 4055, 61, 591, 207, 55748, 145, 146, 6699, 213, 148, 69, 339, 69, 4055, 61, 70, 557, 72, 149, 4339, 603, 557, 72, 149, 250, 235, 295, 296, 297, 509, 29, 240, 241, 243, 69, 246, 378, 12, 299, 252, 295, 296, 12, 6341, 301, 4, 72, 300, 301, 301, 342, 235, 252, 874, 557, 301, 302, 232, 558, 234, 2107, 304, 305], 0, [3, 4, 5, 6, 7, 8, 9, 5, 10]]"
# item = json.loads(s)
# print([lang.index2word[i] for i in item[0]])
# print("\n")
# print([lang.index2word[i] for i in item[2]])
# print("end")
