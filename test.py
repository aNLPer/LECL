import json
# 列表
list1 = [1,2,3,4,5]
print(list1)
print("对列表进行序列化和反序列化的处理：")
print("列表未进行序列化之前的数据类型为：",type(list1))
# 对列表进行序列化处理
list_str = json.dumps(list1)
print("列表序列化后的内容为：{0},类型为：{1}".format(list_str,type(list_str)))
# 对字符串list_str进行反序列化
str_list = json.loads(list_str)
print("字符串反序列化后的内容为：{0}，类型为：{1}".format(str_list,type(str_list)))