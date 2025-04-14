from classifier_manager import ClassifierManager
import torch

class Perturbation:
    def __init__(self, classifier_manager: ClassifierManager, target_class: int=2, target_probability: float=0.9, accuracy_threshold: float=0.9, perturbed_layers: list[int]=None):
        """
        :param classifier_manager: 用于获取分类器模型的管理器。
        :param target_class: 目标类别编号（0=agreeableness, 1=neuroticism, 2=extraversion）。
        :param target_probability: 目标类别的期望概率，默认是 0.001。
        :param accuracy_threshold: 只有当分类器的准确率高于此阈值时才进行扰动，默认为 0.9。
        :param perturbed_layers: 需要进行扰动的层号列表。如果为 None，则所有层都会进行扰动。
        """
        self.classifier_manager = classifier_manager
        self.target_class = target_class  # 目标类别（0、1 或 2）
        self.target_probability = target_probability
        self.accuracy_threshold = accuracy_threshold
        self.perturbed_layers = perturbed_layers

    def get_perturbation(self, output_hook: torch.Tensor, layer: int) -> torch.Tensor:
        if self.perturbed_layers is None or layer in self.perturbed_layers:
            # 获取当前层的分类器
            classifier = self.classifier_manager.classifiers[layer]
            # 获取当前的类别概率
            current_probs = classifier.predict_proba(output_hook[0][:, -1, :])  # 形状为 [1, 3] 的张量

            # 打印扰动前的预测概率
            print(f"Layer {layer} - Predicted Probabilities (Before Perturbation): {current_probs}")

            # 判断是否符合扰动条件（准确率大于阈值，并且目标类别的预测概率大于目标概率）
            if self.classifier_manager.testacc[layer] > self.accuracy_threshold and \
                current_probs[0, self.target_class] < self.target_probability:

                # 计算扰动
                perturbed_embds = self.classifier_manager.cal_perturbation(
                    embds_tensor=output_hook[0][:, -1, :],
                    layer=layer,
                    target_prob=self.target_probability,  # 目标类别的目标概率
                    target_class=self.target_class
                )

                # 应用扰动后，计算扰动后的类别概率
                perturbed_probs = classifier.predict_proba(perturbed_embds)  # 仍然是 [1, 3] 的概率数组

                # 打印扰动后的预测概率
                print(f"Layer {layer} - Predicted Probabilities (After Perturbation): {perturbed_probs}")

                # 将扰动后的嵌入赋值回原始的输出张量
                output_hook[0][:, -1, :] = perturbed_embds

        return output_hook
