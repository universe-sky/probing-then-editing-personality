import json

# 读取文件并加载数据
file_path = '/data/home/jutj/jailbreaking/datasets/test.json'

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 计算样本总数
num_samples = len(data)
print("样本总数:", num_samples)
