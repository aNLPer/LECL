import re
import json

class Lang:
    # 语料库对象
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"UNK", 1:"SOS", 2:"EOS"}
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

def sum_dict(data_dict):
    sum = 0
    for k,v in data_dict.items():
        sum+= v
    return sum

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

# 过滤掉值小于100的项目
def filter_dict(data_dict, bound):
    return {k: v for k, v in data_dict.items() if v >= bound}

# 对字典中的每个项目求和
def sum_dict(data_dict):
    sum = 0
    for k,v in data_dict.items():
        sum+= v
    return sum

# 字典重置
def reset_dict(data_dict):
    return {k: 0 for k, v in data_dict.items()}

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
    pattern_bracket = re.compile(r"[<《【\[(（〔].*?[〕）)\]】》>]")
    law = pattern_bracket.sub("",law)

    # 删除第一个标点之前的内容
    pattern_head_content = re.compile(r".*?[，：。,:.]")
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