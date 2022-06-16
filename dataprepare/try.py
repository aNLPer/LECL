import json
import torch
import torch.nn as nn
import pickle
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
        facts = i[2:3]
        for fact in facts:
            lang.addSentence(fact)
        if count % 5000==0:
            print(f"已统计{count}条数据")
    print(lang.n_words)




# input1 = torch.randn(100, 128)
# input2 = torch.randn(100, 128)
# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# output = cos(input1, input2)
# print(output)

print(0*float('inf'))