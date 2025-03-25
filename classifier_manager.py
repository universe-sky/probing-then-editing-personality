from embedding_manager import EmbeddingManager
from layer_classifier import LayerClassifier
from tqdm import tqdm
import torch
import os

class ClassifierManager:
    def __init__(self, classifier_type: str):
        self.type = classifier_type
        self.classifiers = []
        self.testacc = []

    def _train_classifiers(
        self, 
        pos_embds: EmbeddingManager,
        neg_embds: EmbeddingManager,
        ext_embds: EmbeddingManager,  # 新增 extraversion 类别
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        print("Training classifiers...")

        self.llm_cfg = pos_embds.llm_cfg

        for i in tqdm(range(self.llm_cfg.n_layer)):
            layer_classifier = LayerClassifier(pos_embds.llm_cfg, lr)
            layer_classifier.train(
                pos_tensor=pos_embds.layers[i],
                neg_tensor=neg_embds.layers[i],
                ext_tensor=ext_embds.layers[i],  # 新增 extraversion 类别
                n_epoch=n_epochs,
                batch_size=batch_size,
            )

            self.classifiers.append(layer_classifier)

    def _evaluate_testacc(self, pos_embds: EmbeddingManager, neg_embds: EmbeddingManager, ext_embds: EmbeddingManager):
        for i in tqdm(range(len(self.classifiers))):
            self.testacc.append(
                self.classifiers[i].evaluate_testacc(
                    pos_tensor=pos_embds.layers[i],
                    neg_tensor=neg_embds.layers[i],
                    ext_tensor=ext_embds.layers[i],  # 新增 extraversion 类别
                )
            )

        # 打开文件并写入每一层的正确率差值
        with open(f"{self.type}.txt", "w") as f:
            for i in range(len(self.classifiers)):
                # 计算正确率
                accuracy_diff = self.classifiers[i].evaluate_testacc_with_shuffled_labels(
                    pos_tensor=pos_embds.layers[i],
                    neg_tensor=neg_embds.layers[i],
                    ext_tensor=ext_embds.layers[i],  # 新增 extraversion 类别
                )
                f.write(f"Layer {i}: Accuracy shuffled: {accuracy_diff:.4f}\n")

            f.write("="*50+"\n")

            for i in range(len(self.classifiers)):
                # 计算正确率
                accuracy_origin = self.classifiers[i].evaluate_testacc(
                    pos_tensor=pos_embds.layers[i],
                    neg_tensor=neg_embds.layers[i],
                    ext_tensor=ext_embds.layers[i],  # 新增 extraversion 类别
                )
                f.write(f"Layer {i}: Accuracy Origin: {accuracy_origin:.4f}\n")
    
    def fit(
        self, 
        pos_embds_train: EmbeddingManager,
        neg_embds_train: EmbeddingManager,
        ext_embds_train: EmbeddingManager,  # 新增 extraversion 类别
        pos_embds_test: EmbeddingManager,
        neg_embds_test: EmbeddingManager,
        ext_embds_test: EmbeddingManager,  # 新增 extraversion 类别
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        self._train_classifiers(
            pos_embds_train,
            neg_embds_train,
            ext_embds_train,  # 新增 extraversion 类别
            lr,
            n_epochs,
            batch_size,
        )

        self._evaluate_testacc(
            pos_embds_test,
            neg_embds_test,
            ext_embds_test,  # 新增 extraversion 类别
        )

        return self
    
    def save(self, relative_path: str):
        file_name = f"{self.type}_{self.llm_cfg.model_nickname}.pth"
        torch.save(self, os.path.join(relative_path, file_name))
    
    def cal_perturbation(
        self,
        embds_tensor: torch.tensor,
        layer: int,
        target_class: int,  # 添加 target_class 参数
        target_prob: float = 0.001,  # 目标类别的目标概率
    ):
        # 确保 embds_tensor 和权重 w、偏置 b 在同一设备上
        device = embds_tensor.device  # 获取 embds_tensor 所在的设备
        w, b = self.classifiers[layer].get_weights_bias()

        # 将权重和偏置移到与 embds_tensor 相同的设备
        w = w.to(device)
        b = b.to(device)

        # 计算 perturbation 时确保所有张量在同一设备上
        logit_target = torch.log(torch.tensor(target_prob / (1 - target_prob)))

        # 计算每个类别的权重向量的范数（维度为 [3, 1]）
        w_norm = torch.norm(w, dim=1, keepdim=True)  # 计算每个类别的权重向量的范数，保持维度

        # 计算目标类别的扰动量
        epsilon = (logit_target - b[target_class] - torch.sum(embds_tensor * w[target_class], dim=1, keepdim=True)) / w_norm[target_class]

        # 将 epsilon 扩展为与 w 相同的维度（这里是 [1, 1]）
        epsilon_expanded = epsilon.view(-1, 1)

        # 计算扰动：只针对目标类别的权重向量进行扰动
        perturbation = epsilon_expanded * w[target_class] / w_norm[target_class]  # 仅对目标类别进行扰动

        # 只改变目标类别的嵌入向量
        # 这里我们没有索引 target_class，而是直接更新 embds_tensor 中的所有样本
        embds_tensor += perturbation

        return embds_tensor

def load_classifier_manager(file_path: str):
    return torch.load(file_path, weights_only=False)