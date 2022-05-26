# coding:utf-8
import os
import pickle
import re
import json
import thulac

class Lang:
    # 语料库对象
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS", 1:"EOS", 2:"UNK"}
        # 词汇表大小
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1




BATH_DATA_PATH = "..\dataset\CAIL-SMALL"

# 加载停用词表、特殊符号表、标点
def get_filter_symbols(filepath):
    '''
    根据mode加载标点、特殊词或者停用词
    :param mode:
    :return:list
    '''
    return list(set([line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]))

# law内容过滤
def filterStr(law):
    # 删除括号及括号内的内容
    pattern_bracket = re.compile(r"[(（].*?[）)]")
    law = pattern_bracket.sub("",law)

    # 删除第一个标点之前的内容
    pattern_head_content = re.compile(r".*?[，：]")
    head_content = pattern_head_content.match(law)
    if head_content is not None:
        head_content_span = head_content.span()
        law = law[head_content_span[1]:]

    return law

# 生成acc2desc字典
def get_acc_desc(file_path):
    acc2des = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            dict = json.loads(line)
            if dict["accusation"] not in acc2des:
                acc2des[dict["accusation"]] = dict["desc"]
    return acc2des

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
    special_symbols = get_filter_symbols("special_symbol.txt")
    # 加载停用词表
    stopwords = get_filter_symbols("stop_word.txt")
    # 加载标点
    punctuations = get_filter_symbols("punctuation.txt")
    items = []
    with open(case_path, "r", encoding="utf-8") as f:
        for line in f:
            item = [] # 单条训练数据
            example = json.loads(line)
            # 过滤law article内容
            example_fact = filterStr(example["fact"])
            # 分词,去除特殊符号
            example_fact_1 = [word for word in thu.cut(example_fact, text=True).split(" ") if word not in special_symbols]
            # 去除停用词
            example_fact_2 = [word for word in example_fact_1 if word not in stopwords]
            # 去除标点
            example_fact_3 = [word for word in example_fact_1 if word not in punctuations]
            # 去除停用词和标点
            example_fact_4 = [word for word in example_fact_3 if word not in stopwords]
            facts = [example_fact_1, example_fact_2, example_fact_3, example_fact_4]
            item.append(facts)
            item.append(example['meta']['accusation'][0])
            # 指控描述
            acc_desc = acc2desc[example['meta']['accusation'][0]]
            # 指控描述分词，去除标点、停用词
            acc_desc = [word for word in thu.cut(acc_desc, text=True).split(" ")
                        if word not in stopwords and word not in punctuations]
            item.append(acc_desc)
            items.append(item)
    return items

# 生成训练数据集
data_path = os.path.join(BATH_DATA_PATH, "data_train_filtered.json")
acc_desc = get_acc_desc("accusation_description.json")
items = getData(data_path, acc_desc)
# 统计训练集语料库生成对象
lang = Lang("2018_CAIL_SMALL_TRAIN")
for i in items:
    descs = i[2]
    lang.addSentence(descs)
    facts = i[0]
    for fact in facts:
        lang.addSentence(fact)

print(lang.n_words)


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