# coding:utf-8
import os
import pickle
import re
import json
import thulac
import utils.commonUtils as commonUtils
from utils.commonUtils import Lang
import numpy as np
import operator

BATH_DATA_PATH = "..\dataset\CAIL-SMALL"

# 构造数据集
def getData(case_path, acc2desc):
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
    fw = open("..\dataset\CAIL-SMALL\data_train_preprocessed.txt", "w", encoding="utf-8")
    count = 0
    with open(case_path, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            item = [] # 单条训练数据
            example = json.loads(line)
            # 过滤law article内容
            example_fact = commonUtils.filterStr(example["fact"])
            # 分词,去除特殊符号
            example_fact_1 = [word for word in thu.cut(example_fact, text=True).split(" ") if word not in special_symbols]
            example_fact_1 = [re.sub(r"\d+","x", word) for word in example_fact_1]
            example_fact_1 = [word for word in example_fact_1
                              if word not in ["x年", "x月", "x日", "下午", "上午", "凌晨", "晚", "晚上", "x时", "x分", "许"]]
            # 去除停用词
            example_fact_2 = [word for word in example_fact_1 if word not in stopwords]
            # 去除标点
            example_fact_3 = [word for word in example_fact_1 if word not in punctuations]
            # 去除停用词和标点
            example_fact_4 = [word for word in example_fact_3 if word not in stopwords]
            # facts = [example_fact_3, example_fact_4]
            item.append(example_fact_2)
            item.append(example_fact_3)
            item.append(example_fact_4)
            item.append(example['meta']['accusation'][0].strip())
            # 指控描述
            acc_desc = acc2desc[example['meta']['accusation'][0]]
            # 指控描述分词，去除标点、停用词
            acc_desc = [word for word in thu.cut(acc_desc, text=True).split(" ")
                        if word not in stopwords and word not in punctuations]
            item.append(acc_desc)
            list_str = json.dumps(item,ensure_ascii=False)
            fw.write(list_str+"\n")
            if count%5000==0:
                print(f"已有{count}条数据被处理")
    fw.close()

def getLang(lang_name):
    lang = Lang(lang_name)
    print("start statistic train data......")
    fr = open("..\dataset\CAIL-SMALL\data_train_preprocessed.txt", "r", encoding="utf-8")
    count = 0
    for line in fr:
        if line.strip() == "":
            continue
        count += 1
        i = json.loads(line)
        descs = i[4]
        lang.addSentence(descs)
        facts = i[0:3]
        for fact in facts:
            lang.addSentence(fact)
        if count % 5000==0:
            print(f"已统计{count}条数据")
    fr.close()
    # 序列化lang
    f = open("lang_data_train_preprocessed.pkl", "wb")
    pickle.dump(lang, f)
    f.close()
    print("train data statistic end.")

def word2Index(file_path, lang, acc2id):
    # 数据集
    fi = open(file_path, "r", encoding="utf-8")
    fo = open(os.path.join(BATH_DATA_PATH, "data_train_forModel.txt"), "w", encoding="utf-8")
    count = 0
    for line in fi:
        count += 1
        # 文本数据
        item = json.loads(line)
        # 将文本妆化成索引
        item_num = []

        fact_num = []
        for fact in item[0:3]:
            fact_num.append([lang.word2index[word] for word in fact])
        item_num.append(fact_num[0])
        item_num.append(fact_num[1])
        item_num.append(fact_num[2])
        item_num.append(acc2id[item[3].strip()])
        item_num.append([lang.word2index[word] for word in item[4]])
        # 序列化并写入
        item_num_str = json.dumps(item_num, ensure_ascii=False)
        fo.write(item_num_str+"\n")
        if count%5000==0:
            print(f"已处理{count}条数据")
    fi.close()
    fo.close()

# 获取指控-索引 及 索引-指控
def getAccus(filename):
    """
    统计训练数据中的指控返回acc2id，id2指控
    :param filename:
    :return:
    """
    id2acc = []
    acc2id = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            acc = example["meta"]["accusation"][0]
            if acc not in id2acc:
                id2acc.append(acc)
        for idx, acc in enumerate(id2acc):
            acc2id[acc] = idx
    f_id2acc = open("train_id2acc.pkl","wb")
    pickle.dump(id2acc, f_id2acc)
    f_id2acc.close()
    f_acc2id = open("train_acc2id.pkl","wb")
    pickle.dump(acc2id, f_acc2id)
    f_acc2id.close()
    return id2acc, acc2id

# 统计文本长度
def sample_length(path):
    max_length = 0
    max_length_sample = 0
    min_length = float("inf")
    min_length_sample = 0
    f = open(path, "r", encoding="utf-8")
    # all, 0-20, 20-50, 50-100, 100-200, 200-500, 500-1000, 1000-2000, 2000-5000, 5000-
    count = {"all":0, "0-20":0, "20-50":0,"50-100":0,"100-200":0,
             "200-500":0,"500-1000":0,"1000-2000":0,"2000-5000":0,
             "5000-":0}
    for line in f:
        count["all"] += 1
        sample = json.loads(line)
        length = max(len(sample[0]), len(sample[1]))
        if length > max_length:
            max_length = length
            max_length_sample = count["all"]
        if length < min_length:
            min_length = length
            min_length_sample = count["all"]
        # 长度范围统计
        if length>=5000:
            count["5000-"] += 1
        else:
            if length>=200: # 200-5000
                if length>=1000: # 1000-5000
                    if length>=2000:
                        count["2000-5000"]+=1
                    else:
                        count["1000-2000"]+=1
                else: # 200-1000
                    if length>=500:
                        count["500-1000"]+=1
                    else:
                        count["200-500"]+=1
            else: # 0-200
                if length>=50: # 50-200
                    if length>=100:
                        count["100-200"]+=1
                    else:
                        count["50-100"]+=1
                else:
                    if length>=20:
                        count["20-50"]+=1
                    else:
                        count["0-20"]+=1
    f.close()
    return min_length, min_length_sample, max_length,max_length_sample, count

# 按照指控类别统计案件分布
def sample_categories_dis(file_path):
    f = open(file_path, "r", encoding="utf-8")
    acc_dict = {}
    for line in f:
        sample = json.loads(line)
        sample_acc = sample[3]
        if sample_acc not in acc_dict:
            acc_dict[sample_acc] = 1
        else:
            acc_dict[sample_acc] += 1
    f.close()
    return acc_dict

def load_accusation_classified(file_path):
    f = open(file_path, "r", encoding="utf-8")
    category2accus = {}
    # category to accusations
    for line in f:
        line = line.strip()
        item = line.split(" ")
        category2accus[item[0]] = item[1:]
    f.close()
    accu2category = {}
    for k, vs in category2accus.items():
        for v in vs:
            accu2category[v] = k

    return category2accus, accu2category

if __name__=="__main__":
    # # 生成训练数据集
    data_path = os.path.join(BATH_DATA_PATH, "data_train_filtered.json")
    acc_desc = commonUtils.get_acc_desc("accusation_description.json")
    print("start processing data...")
    getData(data_path, acc_desc)
    print("data processing end.")

    # 统计训练集语料库生成对象
    lang_name = "2018_CAIL_SMALL_TRAIN"
    getLang(lang_name)
    #
    # # 将训练集中的文本转换成对应的索引
    # print("start word to index")
    id2acc, acc2id = getAccus(data_path)
    # f = open("lang_data_train_preprocessed.pkl", "rb")
    # lang = pickle.load(f)
    # f.close()
    # word2Index(os.path.join(BATH_DATA_PATH,"data_train_preprocessed.txt"), lang, acc2id)
    # print("processing end")
    #
    # # 统计最长文本
    # print("start statistic length of sample......")
    # path = os.path.join(BATH_DATA_PATH, "data_train_forModel.txt")
    # min_length,min_length_sample, max_length, max_length_sample, count = sample_length(path)
    # print(f"min_length: {min_length} at line {min_length_sample}")
    # print((f"max_length: {max_length} at line {max_length_sample}"))
    # data = np.array(list(count.values()))
    # print(data)
    #
    #
    # # 统计案件类别分布
    # file_path = os.path.join(BATH_DATA_PATH, "data_train_preprocessed.txt")
    # sample_dis = sample_categories_dis(file_path)
    # f = open("sample_category_dis.pkl", "wb")
    # pickle.dump(sample_dis,f)
    # f.close()
    #
    # f = open("sample_category_dis.pkl", "rb")
    # sample_dis = pickle.load(f)
    # sample_dis = dict(sorted(sample_dis.items(), key=operator.itemgetter(1),reverse=True))
    # f.close()
    # print(sample_dis)
    #
    # top_10 = sum(list(sample_dis.values())[0:10])
    # bottom_10 = sum(list(sample_dis.values())[-10:])
    # print(f"top_10:{top_10/sum(sample_dis.values())}")
    # print(f"bottom_20:{bottom_10/sum(sample_dis.values())}")


    # # 获取指控字典
    # d1, d2 = getAccus(os.path.join(BATH_DATA_PATH,"data_train_filtered.json"))
    # print(len(d1))














