import json
import pickle
from string import punctuation

def genVocab():
     """
     生成字典
     :return:
     """
     f_trainData = open("../dataset/CAIL-SMALL/data_train_filtered.json", "r", encoding="utf-8")
     f_accuDesc = open("accusation_description.json", "r", encoding="utf-8")
     corpus = []
     for line in f_trainData:
         example = json.loads(line)
         example_fact = example["fact"]
         corpus.append(example_fact)
     f_trainData.close()
     for line in f_accuDesc:
         example = json.loads(line)
         example_accusation = example["accusation"]
         example_accusation_desc = example["desc"]
         corpus.append(example_accusation)
         corpus.append(example_accusation_desc)
     f_accuDesc.close()
     # 获取语料库
     vocabStr = "".join(corpus)

     # 停用词以及标点符号、特殊符号过滤
     stopWords = "a-zA-Zèの"
     specialSymbols = "ΔΨγμφ．①②③③④⑤⑥⑦⑧⑨⑩"
     add_punc = "≤<>="
     punc = punctuation+add_punc

     idx_to_char = list(set(vocabStr))
     idx_to_char.append("UNK")
     char_to_idx = dict([(char, idx) for idx, char in enumerate(idx_to_char)])

     # 序列化字典
     f_idx2char = open("idx2char.pkl", "wb")
     pickle.dump(idx_to_char, f_idx2char)
     f_idx2char.close()
     f_char2idx = open("char2idx.pkl", "wb")
     pickle.dump(char_to_idx, f_char2idx)


def genAccusationDescDict():
    f = open("accusation_description.json", "r", encoding="utf-8")
    accu_2_atc = {}
    for line in f:
        item = json.loads(line)
        accu = item["accusation"]
        if accu not in accu_2_atc:
            accu_2_atc.update({item["accusation"]:item["article"]})
    f.close()

    f = open("accu2atc.pkl", "wb")
    pickle.dump(accu_2_atc,f)
    f.close()

# f_idx2char = open("./idx2char.pkl","rb")
# idx_2_char = pickle.load(f_idx2char)
# f_idx2char.close()
#
# f_char2idx = open("./char2idx.pkl","rb")
# char_2_idx = pickle.load(f_char2idx)
# f_char2idx.close()

genAccusationDescDict()
# f = open("./accu2atc.pkl", "rb")
# accu_2_atc = pickle.load(f)
# f.close()
# print(accu_2_atc)

def checkLabel():
    f_data = open("../dataset/CAIL-SMALL/data_train_filtered.json", "r", encoding="utf-8")
    f_accu2atc = open("accu2atc.pkl", "rb")
    accu_2_atc = pickle.load(f_accu2atc)
    count = 0
    for line in f_data:
        item = json.loads(line)
        item_atc = item["meta"]["relevant_articles"][0]
        if item_atc != accu_2_atc[item["meta"]["accusation"][0]][0]:
            print(item)
            count+=1
    print(count)

checkLabel()