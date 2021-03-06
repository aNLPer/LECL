import json
import os
import pickle
import utils.commonUtils as commonUtils
BASE_PATH = "../dataset/CAIL-SMALL"

dataPath = os.path.join(BASE_PATH,"CAIL-SMALL")
fileNames = ["train"]

def func(fileNames, prefix):
    dict_articles = {}  # 法律条款：数量
    dict_accusation = {}  # 指控：数量
    for i in range(len(fileNames)):
        print(f'data_{fileNames[i]}.json process beginning')
        with open(os.path.join(dataPath,f"data_{fileNames[i]}.json"), 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                example_articles = example['meta']['relevant_articles']
                example_accusation = example['meta']['accusation']
                example_fact = example['fact']
                # 仅统计单条款、单指控、仅一审的案件的指控和条款
                if len(example_articles) == 1 and len(example_accusation) == 1 and '二审' not in example_fact:
                    if dict_articles.__contains__(example_articles[0]):
                        dict_articles[example_articles[0]] += 1
                    else:
                        dict_articles.update({example_articles[0]: 1})
                    if dict_accusation.__contains__(example_accusation[0]):
                        dict_accusation[example_accusation[0]] += 1
                    else:
                        dict_accusation.update({example_accusation[0]: 1})
        print(f'The {fileNames[i]} dataset is read over')
    # 将法律条款统计结果序列化
    f = open(f"./{prefix}_articles2num.pkl", "wb")
    pickle.dump(dict_articles, f)
    f.close()

    # 指控统计结果序列化
    f = open(f"./{prefix}_accusation2num.pkl", "wb")
    pickle.dump(dict_accusation,f)
    f.close()

    # 查看指控文本
    f = open(f"./{prefix}_accusation.txt", "w", encoding="utf-8")
    for k,v in dict_accusation.items():
        f.write(k+"\n")
    f.close()


# func(fileNames,"train")


def func1(dict_articles, dict_accusation):
    articles_sum = commonUtils.sum_dict(dict_articles)
    accusation_sum = commonUtils.sum_dict(dict_accusation)
    f1 = open("../dataset/CAIL-SMALL/data_train_filtered.json", "w", encoding="utf-8")

    while articles_sum != accusation_sum:
        dict_accusation = commonUtils.reset_dict(dict_accusation)
        dict_articles = commonUtils.reset_dict(dict_articles)
        for i in range(len(fileNames)):
            with open(os.path.join(dataPath, f"data_{fileNames[i]}.json"), 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    example_articles = example['meta']['relevant_articles']
                    example_accusation = example['meta']['accusation']
                    example_fact = example['fact']
                    if len(example_articles) == 1 and \
                            len(example_accusation) == 1 and \
                            '二审' not in example_fact and len(example_fact)>=65:
                        # 该案件对应的article和accusation频率都大于100
                        if dict_articles.__contains__(example_articles[0]) and dict_accusation.__contains__(example_accusation[0]):
                            dict_articles[example_articles[0]] += 1
                            dict_accusation[example_accusation[0]] += 1

                        else:
                            continue
                f.close()
            print(f'The {fileNames[i]} dataset is read over')

        print(dict_articles)
        print(dict_accusation)

        print(len(dict_articles))
        print(len(dict_accusation))

        dict_articles = commonUtils.filter_dict(dict_articles)
        dict_accusation = commonUtils.filter_dict(dict_accusation, 100)

        articles_sum = commonUtils.sum_dict(dict_articles)
        accusation_sum = commonUtils.sum_dict(dict_accusation)

        print(dict_articles)
        print('articles_num: ' + str(len(dict_articles)))
        print('article_sum: ' + str(articles_sum))

        print(dict_accusation)
        print('accusation_num=' + str(len(dict_accusation)))
        print('accusation_sum: ' + str(accusation_sum))
        print("\n\n\n")

    # 将法律条款统计结果序列化
    f = open(f"./train_filtered_articles2num.pkl", "wb")
    pickle.dump(dict_articles, f)
    f.close()

    # 指控统计结果序列化
    f = open(f"./train_filtered_accusation2num.pkl", "wb")
    pickle.dump(dict_accusation, f)
    f.close()

    for i in range(len(fileNames)):
        with open(os.path.join(dataPath, f"data_{fileNames[i]}.json"), 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                example_articles = example['meta']['relevant_articles']
                example_accusation = example['meta']['accusation']
                example_fact = example['fact']
                if len(example_articles) == 1 and \
                        len(example_accusation) == 1 and \
                        '二审' not in example_fact and \
                        len(example_fact)>=65:
                    # 该案件对应的article和accusation频率都大于100
                    if dict_articles.__contains__(example_articles[0]) and dict_accusation.__contains__(
                            example_accusation[0]):
                        f1.write(line)
                    else:
                        continue
            f1.close()
        print(f'The {fileNames[i]} dataset is read over')




print("--------------------------过滤前-----------------------------")
f = open("train_accusation2num.pkl", "rb")
dict_accusations = pickle.load(f)  # 指控：数量

f = open("train_articles2num.pkl", "rb")
dict_articles = pickle.load(f)

f.close()

print(commonUtils.sum_dict(dict_articles))
print(commonUtils.sum_dict(dict_accusations))

print("--------------------------过滤掉频次小于100的条款和指控-----------------------------")
# 过滤掉频次小于100的条款和指控
dict_articles = commonUtils.filter_dict(dict_articles)
dict_accusations = commonUtils.filter_dict(dict_accusations)




print("--------------------------统计条款和指控的频次都大于100案件-----------------------------")
func1(dict_articles,dict_accusations)
