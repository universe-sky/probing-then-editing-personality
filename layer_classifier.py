from llm_config import cfg
from sklearn.linear_model import LogisticRegression

import torch.nn as nn
import torch
class LayerClassifier:
    def __init__(self, llm_cfg: cfg, lr: float=0.01, max_iter: int=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 使 LogisticRegression 支持多分类问题（使用 softmax）
        self.linear = LogisticRegression(solver="saga", max_iter=max_iter, multi_class='multinomial', verbose=1)
        
        self.data = {
            "train": {
                "pos": None,
                "neg": None,
                "ext": None,  # 新增 extraversion 类别
            },
            "test": {
                "pos": None,
                "neg": None,
                "ext": None,  # 新增 extraversion 类别
            }
        }

    def train(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, ext_tensor: torch.tensor, n_epoch: int=100, batch_size: int=64) -> list[float]:
        # 三类样本合并成一个训练数据集
        X = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)
        y = torch.cat((
            torch.ones(pos_tensor.size(0)),  # 1 为 "agreeableness"
            torch.zeros(neg_tensor.size(0)),  # 0 为 "neuroticism"
            torch.full((ext_tensor.size(0),), 2)  # 2 为 "extraversion"
        )).to(self.device)
        
        # 保存训练数据
        self.data["train"]["pos"] = pos_tensor.cpu()
        self.data["train"]["neg"] = neg_tensor.cpu()
        self.data["train"]["ext"] = ext_tensor.cpu()

        self.linear.fit(X.cpu().numpy(), y.cpu().numpy())  # 使用 sklearn 的 fit 进行训练

        return []

    def predict(self, tensor: torch.tensor) -> torch.tensor:
        # 返回每个样本的预测类别
        return torch.tensor(self.linear.predict(tensor.cpu().numpy()))

    def predict_proba(self, tensor: torch.tensor) -> torch.tensor:
        tensor = tensor.to(self.device).to(torch.float32)  # 强制转换 tensor 为 float32 类型
        # 返回每个类别的概率
        return torch.tensor(self.linear.predict_proba(tensor.cpu().numpy()))
        
    def evaluate_testacc(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, ext_tensor: torch.tensor) -> float:
        # 评估准确度
        test_data = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)
        predictions = self.predict(test_data)
        true_labels = torch.cat((
            torch.ones(pos_tensor.size(0)),  # 1 为 "agreeableness"
            torch.zeros(neg_tensor.size(0)),  # 0 为 "neuroticism"
            torch.full((ext_tensor.size(0),), 2)  # 2 为 "extraversion"
        ))

        correct_count = torch.sum(predictions == true_labels).item()

        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()
        self.data["test"]["ext"] = ext_tensor.cpu()

        return correct_count / len(true_labels)
    
    def evaluate_testacc_with_shuffled_labels(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, ext_tensor: torch.tensor) -> float:
        # 合并三个类别的样本数据
        test_data = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)
        
        # 获取模型的预测结果
        predictions = self.predict(test_data)
        
        # 真实标签：1 -> "agreeableness"，0 -> "neuroticism"，2 -> "extraversion"
        true_labels = torch.cat((
            torch.ones(pos_tensor.size(0)),  # "agreeableness" 类别，标签为 1
            torch.zeros(neg_tensor.size(0)),  # "neuroticism" 类别，标签为 0
            torch.full((ext_tensor.size(0),), 2)  # "extraversion" 类别，标签为 2
        ))

        # 计算原始准确率
        original_accuracy = torch.sum(predictions == true_labels).item() / len(true_labels)

        # 打乱标签
        shuffled_labels = true_labels[torch.randperm(true_labels.size(0))]

        # 计算标签打乱后的准确率
        shuffled_accuracy = torch.sum(predictions == shuffled_labels).item() / len(shuffled_labels)

        # 返回两者的差值
        return shuffled_accuracy

    def evaluate_testnll(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor, ext_tensor: torch.Tensor) -> float:
        """
        计算模型在原始测试数据上的平均负对数概率 (NLL)。
        """
        # 1. 将三类样本拼接成一个测试集
        test_data = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)

        # 2. 获取预测概率 (shape: [N, num_classes])
        predicted_probs = self.predict_proba(test_data).to(self.device)

        # 3. 构造真实标签: 1->"agreeableness", 0->"neuroticism", 2->"extraversion"
        true_labels = torch.cat((
            torch.ones(pos_tensor.size(0)),       # label = 1
            torch.zeros(neg_tensor.size(0)),      # label = 0
            torch.full((ext_tensor.size(0),), 2)  # label = 2
        )).long().to(self.device)

        # 4. 取出真实标签对应的预测概率
        correct_probs = predicted_probs[torch.arange(len(true_labels)), true_labels]

        # 5. 计算负对数概率 (-log p)，并取平均
        nll = -torch.log(correct_probs + 1e-12)  # 加上 1e-12 防止 log(0)
        avg_nll = nll.mean().item()

        # 6. 保存测试数据到 self.data["test"]（可选）
        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()
        self.data["test"]["ext"] = ext_tensor.cpu()

        return avg_nll


    def evaluate_testnll_with_zero_input(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor, ext_tensor: torch.Tensor) -> float:
        """
        将输入替换为全 0 张量后，计算模型在该“零向量输入”测试集上的平均负对数概率 (NLL)。
        这可以作为一种 baseline，对比模型在无信息输入时的表现。
        """
        # 1. 将三类样本拼接成一个测试集
        test_data = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)

        # 2. 生成与 test_data 形状相同的全 0 张量
        zero_data = torch.zeros_like(test_data)

        # 3. 获取“零输入”下的预测概率
        predicted_probs = self.predict_proba(zero_data).to(self.device)

        # 4. 构造真实标签
        true_labels = torch.cat((
            torch.ones(pos_tensor.size(0)),       # label = 1
            torch.zeros(neg_tensor.size(0)),      # label = 0
            torch.full((ext_tensor.size(0),), 2)  # label = 2
        )).long().to(self.device)

        # 5. 取出真实标签对应的预测概率
        correct_probs = predicted_probs[torch.arange(len(true_labels)), true_labels]

        # 6. 计算 -log(prob) 并做平均
        nll = -torch.log(correct_probs + 1e-12)
        avg_nll = nll.mean().item()

        # 7. 保存测试数据到 self.data["test"]（可选）
        #    这里若想保留“零输入”数据，也可以将 zero_data.cpu() 存到 self.data
        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()
        self.data["test"]["ext"] = ext_tensor.cpu()

        return avg_nll

    
    def get_weights_bias(self) -> tuple[torch.tensor, torch.tensor]:
        # 返回权重和偏置（可以扩展为多类分类）
        return torch.tensor(self.linear.coef_).to(self.device), torch.tensor(self.linear.intercept_).to(self.device)
