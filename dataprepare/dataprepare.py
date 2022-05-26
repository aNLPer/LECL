# coding:utf-8
import os
import pickle
import re
import json
import jieba
import thulac
import pyhanlp as hanlp
import utils.commonUtils as utils
from string import punctuation


BATH_DATA_PATH = "C:\D\Workspace\mine\idea\LECL\dataset"

# 加载停用词表、特殊符号表、标点
def get_filter_symbols(filepath):
    '''
    根据mode加载标点、特殊词或者停用词
    :param mode:
    :return:list
    '''
    return list(set([line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]))



# 大写数字转阿拉伯数字
def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十': hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)

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

# 获取分词器
def get_cutter(dict_path="Thuocl_seg.txt", mode='thulac', stop_words_filtered=False):
    '''
    获取分词器
    :param dict_path: jieba、thulac使用的用户字典
    :param mode: 分词工具选择
    :param stop_words_filtered: 停用词过滤
    :return:
    '''
    if stop_words_filtered:
        stopwords = get_filter_symbols('stop_word.txt', mode="stop")  # 这里加载停用词的路径
    else:
        stopwords = []
    if mode == 'jieba':
        jieba.load_userdict(dict_path)
        return lambda x: [a for a in list(jieba.cut(x)) if a not in stopwords]
    elif mode == 'thulac':
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ') if a not in stopwords]

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
            item = []
            example = json.loads(line)
            # 删除
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
            item.append(example['meta']['accusation'])
            item.append(acc2desc[example['meta']['accusation']])
            items.append(item)
    return items

data_path = os.path.join(BATH_DATA_PATH, "data_train_filtered.json")
acc_desc = get_acc_desc("accusation_description.json")
getData(data_path, acc_desc)



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