# coding:utf-8
import os
import pickle
import re
import json
import thulac

BATH_DATA_PATH = "..\dataset\CAIL-SMALL"

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
    fw = open("..\dataset\CAIL-SMALL\data_train_preprocessed.txt", "w", encoding="utf-8")
    count = 0
    with open(case_path, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            item = [] # 单条训练数据
            example = json.loads(line)
            # 过滤law article内容
            example_fact = filterStr(example["fact"])
            # 分词,去除特殊符号
            example_fact_1 = [word for word in thu.cut(example_fact, text=True).split(" ") if word not in special_symbols]
            example_fact_1 = [re.sub(r"\d+","x", word) for word in example_fact_1]
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
        descs = i[2]
        lang.addSentence(descs)
        facts = i[0]
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

def word2Index(file_path, lang):
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
        for fact in item[0]:
            fact_num.append([lang.word2index[word] for word in fact])
        item_num.append(fact_num)
        item_num.append(item[1])
        item_num.append([lang.word2index[word] for word in item[2]])
        # 序列化并写入
        item_num_str = json.dumps(item_num, ensure_ascii=False)
        fo.write(item_num_str+"\n")
        if count%5000==0:
            print(f"已处理{count}条数据")
    fi.close()
    fo.close()

if __name__=="__main__":
    # # 生成训练数据集
    # data_path = os.path.join(BATH_DATA_PATH, "data_train_filtered.json")
    # acc_desc = get_acc_desc("accusation_description.json")
    # print("start processing data......")
    # getData(data_path, acc_desc)
    # print("data processing end.")

    # # 统计训练集语料库生成对象
    # lang_name = "2018_CAIL_SMALL_TRAIN"
    # getLang(lang_name)

    # # 将训练集中的文本转换成对应的索引
    # print("start word to index")
    # f = open("lang_data_train_preprocessed.pkl", "rb")
    # lang = pickle.load(f)
    # f.close()
    # word2Index(os.path.join(BATH_DATA_PATH,"data_train_preprocessed.txt"), lang)
    # print("processing end")

    # 统计最长文本
    print("start statistic length of sample......")
    max_length = 0
    max_length_sample = 0
    min_length = float("inf")
    min_length_sample = 0
    f = open(os.path.join(BATH_DATA_PATH, "data_train_forModel.txt"), "r", encoding="utf-8")
    count = 0
    for line in f:
        count += 1
        sample = json.loads(line)
        if len(sample[0][0])>max_length:
            max_length = len(sample[0][0])
            max_length_sample = count
        if len(sample[0][0])<min_length:
            min_length = len(sample[0][0])
            min_length_sample = count
    f.close()
    print(f"min_length: {min_length} at line {min_length_sample}")
    print((f"max_length: {max_length} at line {max_length_sample}"))









