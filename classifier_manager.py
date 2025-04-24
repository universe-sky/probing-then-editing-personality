from embedding_manager import EmbeddingManager
from layer_classifier import LayerClassifier
from tqdm import tqdm
import torch
import os

class ClassifierManager:
    def __init__(self, classifier_type: str):
        # Initialize the ClassifierManager with a specific classifier type
        self.type = classifier_type
        self.classifiers = []  # List to store classifiers for each layer
        self.testacc = []  # List to store test accuracy for each classifier

    def _train_classifiers(
        self, 
        pos_embds: EmbeddingManager,
        neg_embds: EmbeddingManager,
        ext_embds: EmbeddingManager,  # New extraversion category
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        # Function to train classifiers for each layer
        print("Training classifiers...")

        # Use the configuration from positive embeddings
        self.llm_cfg = pos_embds.llm_cfg

        # Loop through each layer of embeddings to train a classifier for each layer
        for i in tqdm(range(self.llm_cfg.n_layer)):
            layer_classifier = LayerClassifier(pos_embds.llm_cfg, lr)
            layer_classifier.train(
                pos_tensor=pos_embds.layers[i],
                neg_tensor=neg_embds.layers[i],
                ext_tensor=ext_embds.layers[i],  # New extraversion category
                n_epoch=n_epochs,
                batch_size=batch_size,
            )

            # Append the trained classifier to the list
            self.classifiers.append(layer_classifier)

    def _evaluate_testacc(self, pos_embds: EmbeddingManager, neg_embds: EmbeddingManager, ext_embds: EmbeddingManager):
        # Function to evaluate the test accuracy for each classifier
        for i in tqdm(range(len(self.classifiers))):
            # Evaluate test accuracy for each layer
            self.testacc.append(
                self.classifiers[i].evaluate_testacc(
                    pos_tensor=pos_embds.layers[i],
                    neg_tensor=neg_embds.layers[i],
                    ext_tensor=ext_embds.layers[i],  # New extraversion category
                )
            )

        # Open a file and write the accuracy differences for each layer
        with open(f"{self.type}.txt", "w") as f:
            for i in range(len(self.classifiers)):
                # Calculate the accuracy difference between original and perturbed models
                accuracy_diff = self.classifiers[i].evaluate_testnll_with_zero_input(
                    pos_tensor=pos_embds.layers[i],
                    neg_tensor=neg_embds.layers[i],
                    ext_tensor=ext_embds.layers[i],  # New extraversion category
                )
                accuracy_origin = self.classifiers[i].evaluate_testnll(
                    pos_tensor=pos_embds.layers[i],
                    neg_tensor=neg_embds.layers[i],
                    ext_tensor=ext_embds.layers[i],  # New extraversion category
                )
                diff = accuracy_diff - accuracy_origin
                f.write(f"Layer {i}: Accuracy Origin: {diff:.4f}\n")
    
    def fit(
        self, 
        pos_embds_train: EmbeddingManager,
        neg_embds_train: EmbeddingManager,
        ext_embds_train: EmbeddingManager,  # New extraversion category
        pos_embds_test: EmbeddingManager,
        neg_embds_test: EmbeddingManager,
        ext_embds_test: EmbeddingManager,  # New extraversion category
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        # Function to train and evaluate classifiers for the given training and test embeddings
        self._train_classifiers(
            pos_embds_train,
            neg_embds_train,
            ext_embds_train,  # New extraversion category
            lr,
            n_epochs,
            batch_size,
        )

        self._evaluate_testacc(
            pos_embds_test,
            neg_embds_test,
            ext_embds_test,  # New extraversion category
        )

        return self
    
    def save(self, relative_path: str):
        # Save the trained model to a file
        file_name = f"{self.type}_{self.llm_cfg.model_nickname}.pth"
        torch.save(self, os.path.join(relative_path, file_name))
    
    def cal_perturbation(
        self,
        embds_tensor: torch.tensor,
        layer: int,
        target_class: int,  # Add target_class parameter
        target_prob: float = 0.001,  # Target probability for the specified class
    ):
        # Function to calculate perturbations for the given embeddings based on the target class and target probability

        # Ensure that embds_tensor, weights (w), and biases (b) are on the same device
        device = embds_tensor.device  # Get the device of embds_tensor
        w, b = self.classifiers[layer].get_weights_bias()

        # Move weights and biases to the same device as embds_tensor
        w = w.to(device)
        b = b.to(device)

        # Calculate perturbation while ensuring all tensors are on the same device
        logit_target = torch.log(torch.tensor(target_prob / (1 - target_prob)))

        # Calculate the norm of each class weight vector (dimensions: [3, 1])
        w_norm = torch.norm(w, dim=1, keepdim=True)  # Norm of each class weight vector, keeping the dimensions

        # Calculate the perturbation for the target class
        epsilon = (logit_target - b[target_class] - torch.sum(embds_tensor * w[target_class], dim=1, keepdim=True)) / w_norm[target_class]

        # Expand epsilon to the same dimensions as the weight vector (here [1, 1])
        epsilon_expanded = epsilon.view(-1, 1)

        # Calculate the perturbation: only perturb the weight vector of the target class
        perturbation = epsilon_expanded * w[target_class] / w_norm[target_class]  # Perturb only the target class

        # Update the embeddings tensor by adding the perturbation to the target class embeddings
        embds_tensor += perturbation

        return embds_tensor

def load_classifier_manager(file_path: str):
    # Function to load a classifier manager from a file
    return torch.load(file_path, weights_only=False)
