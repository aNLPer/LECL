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
add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = punctuation + add_punc


BATH_DATA_PATH = "C:\D\Workspace\mine\idea\LECL\dataset"

# 加载停用词表
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

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
        stopwords = stopwordslist('stop_word.txt')  # 这里加载停用词的路径
    else:
        stopwords = []
    if mode == 'jieba':
        jieba.load_userdict(dict_path)
        return lambda x: [a for a in list(jieba.cut(x)) if a not in stopwords]
    elif mode == 'thulac':
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ') if a not in stopwords]

# 处理单个法律文书
def process_law(law, cut):
    # single article
    # cut=get_cutter()
    condition_list = []
    for each in law.split('。')[:-1]:
        suffix = None
        if '：' in each:
            each, suffix = each.split('：')
            suffix = cut(suffix)
        words = cut(each)
        seg_point = [-1]
        conditions = []

        for i in range(len(words)):
            if words[i] == '；' or words[i] == ';':
                seg_point.append(i)
        seg_point.append(len(words))
        for i in range(len(seg_point) - 1):
            for j in range(seg_point[i + 1] - 1, seg_point[i], -1):
                if j + 1 < len(words) and words[j] == '的' and words[j + 1] == '，':
                    conditions.append(words[seg_point[i] + 1:j + 1])
                    break
        # context=law.split('。')[:-1]
        for i in range(1, len(conditions)):
            conditions[i] = conditions[0] + conditions[i]
        # if len(condition_list)==0 and len(conditions)==0:
        #     conditions.append([])
        if suffix is not None:
            conditions = [x + suffix for x in conditions]
        condition_list += conditions

    if condition_list == []:
        condition_list.append(cut(law[:-1]))
    n_word = [len(i) for i in condition_list]
    return condition_list, n_word

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



thu = thulac.thulac(user_dict="Thuocl_seg.txt", seg_only=True)
print(thu.cut(law, text=True))

# print(list(jieba.cut(law)))
#
# print([term.word for term in hanlp.HanLP.segment(law)])


# cutter = get_cutter(mode="thulac", stop_words_filtered=True)
# print(len(law))
# print(cutter(law))
# print(len(cutter(law)))
# condition_list, n_word = process_law(law, get_cutter(mode="thulac", stop_words_filtered=True))
# print(condition_list)
# print(len(condition_list), n_word)


# cutter = get_cutter(mode="thulac", stop_words_filtered=True)
# counter = 0
# c = 0
# with open(os.path.join(BATH_DATA_PATH, "CAIL-SMALL","data_train_filtered.json"), "r", encoding="utf-8") as f:
#     for line in f:
#         c+=1
#         print(c)
#         example = json.loads(line)
#         example_fact = example["fact"]
#         condition_list, n_word = process_law(example_fact, cutter)
#         if len(condition_list)>1:
#             print(example_fact)
#         if n_word[0]>500: counter += 1
# print(counter)

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