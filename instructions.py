import json
import os
import random
from sklearn.model_selection import train_test_split

def load_instructions_from_json(dataset_path: str, label_list: list[str], train_size: float=1.0):
    """
    加载 JSON 文件并根据标签划分数据集（训练集、测试集）。
    """
    assert 0 < train_size <= 1.0, "train_size should be in (0, 1]"
    
    # 读取 JSON 文件
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    # 初始化返回数据结构
    ret = {
        "dataset_name": os.path.basename(dataset_path),
        "label_list": label_list,
        "train": [],
        "test": [],  # 划分训练集和测试集
    }

    # 将每个人格标签的样本提取出来
    label_data = {label: [] for label in label_list}
    entities = []  # 存储实体名称
    
    # 遍历数据集，将每个标签的文本提取出来
    for item in dataset:
        entity = item["ent"]  # 获取实体（如城市名）
        entities.append(entity)
        for label in label_list:
            if label in item:
                texts = item[label]
                for text in texts:
                    label_data[label].append((text, label, entity))  # 保存文本和标签
    
    # 对每个标签进行数据划分，确保文本和实体保持一致地划分
    for label in label_list:
        # 从 label_data 中提取出当前标签的所有数据
        label_samples = label_data[label]
        
        # 随机打乱顺序，确保文本和实体一起打乱
        random.shuffle(label_samples)
        
        # 划分训练集、测试集
        train_data, test_data = train_test_split(label_samples, test_size=1 - train_size, random_state=42)
        
        # 保存划分的数据
        ret["train"].append(train_data)
        ret["test"].append(test_data)

    # 打印训练集和测试集的大小，确保比例正确
    train_size = sum(len(train) for train in ret["train"])
    test_size = sum(len(test) for test in ret["test"])
    
    print(f"Total training samples: {train_size}, Total testing samples: {test_size}")
    print(f"Training to testing ratio: {train_size / (train_size + test_size):.2f} : {(test_size / (train_size + test_size)):.2f}")

    return ret, entities  # 返回实体列表


def load_instructions_by_size(
    dataset_path: str,
    label_list: list[str],
    train_size: float = 1.0,
):
    """
    使用指定的训练集比例加载数据，并划分训练集和测试集。
    """
    return load_instructions_from_json(dataset_path, label_list, train_size)


def load_instructions_by_flag(
    dataset_path: str,
    label_list: list[str],
):
    """
    加载数据并根据训练集和测试集的标记进行划分
    """
    return load_instructions_from_json(dataset_path, label_list)
