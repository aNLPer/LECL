# import thulac
# import re
#
# # 加载停用词表、特殊符号表、标点
# def get_filter_symbols(filepath):
#     '''
#     根据mode加载标点、特殊词或者停用词
#     :param mode:
#     :return:list
#     '''
#     return list(set([line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]))
#
# def filterStr(law):
#     # 删除括号及括号内的内容
#     pattern_bracket = re.compile(r"[(（].*?[）)]")
#     law = pattern_bracket.sub("",law)
#
#     # 删除第一个标点之前的内容
#     pattern_head_content = re.compile(r".*?[，：。,:.]")
#     head_content = pattern_head_content.match(law)
#     if head_content is not None:
#         head_content_span = head_content.span()
#         law = law[head_content_span[1]:]
#
#     return law
#
# example_fact = "经审理查明.原审被告人保某受杨某（另案）邀约驾驶其云Ｄ×××××货车非法运输烟叶的事实清楚。有经一审庭审质证、认证的刑事案件登记表、查获经过、扣押物品清单、价格鉴定书、证人证言、被告人保某的供述等证据予以证实，本院予以确认。"
#
# # 加载分词器
# thu = thulac.thulac(user_dict="Thuocl_seg.txt", seg_only=True)
# # 加载特殊符号
# special_symbols = get_filter_symbols("special_symbol.txt")
# # 加载停用词表
# stopwords = get_filter_symbols("stop_word.txt")
# # 加载标点
# punctuations = get_filter_symbols("punctuation.txt")
#
# # 过滤law article内容
# example_fact = filterStr(example_fact)
# # 分词,去除特殊符号
# example_fact_1 = [word for word in thu.cut(example_fact, text=True).split(" ") if word not in special_symbols]
# example_fact_1 = [re.sub(r"\d+", "x", word) for word in example_fact_1]
# # 去除停用词
# example_fact_2 = [word for word in example_fact_1 if word not in stopwords]
# # 去除标点
# example_fact_3 = [word for word in example_fact_1 if word not in punctuations]
# # 去除停用词和标点
# example_fact_4 = [word for word in example_fact_3 if word not in stopwords]
# facts = [example_fact_1, example_fact_2, example_fact_3, example_fact_4]
#
#
#
