from llm_config import cfg
from sklearn.linear_model import LogisticRegression

import torch.nn as nn
import torch

class LayerClassifier:
    def __init__(self, llm_cfg: cfg, lr: float=0.01, max_iter: int=1):
        """
        Initialize the LayerClassifier with model configuration, learning rate, and maximum iterations.
        
        Args:
            llm_cfg: Configuration for the LLM model.
            lr: Learning rate for Logistic Regression.
            max_iter: Maximum number of iterations for Logistic Regression training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize Logistic Regression with multi-class classification using softmax (multinomial)
        self.linear = LogisticRegression(solver="saga", max_iter=max_iter, multi_class='multinomial', verbose=1)
        
        # Initialize data dictionary to store training and testing data for each class
        self.data = {
            "train": {
                "pos": None,
                "neg": None,
                "ext": None,  # Added extraversion class
            },
            "test": {
                "pos": None,
                "neg": None,
                "ext": None,  # Added extraversion class
            }
        }

    def train(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, ext_tensor: torch.tensor, n_epoch: int=100, batch_size: int=64) -> list[float]:
        """
        Train the model using the provided positive, negative, and extraversion samples.
        
        Args:
            pos_tensor: Tensor containing positive class (agreeableness) samples.
            neg_tensor: Tensor containing negative class (neuroticism) samples.
            ext_tensor: Tensor containing extraversion class samples.
            n_epoch: Number of training epochs.
            batch_size: Size of the training batch.
        
        Returns:
            A list of floats containing the training history (empty in this case).
        """
        # Combine the three classes into a single training dataset
        X = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)
        y = torch.cat((
            torch.ones(pos_tensor.size(0)),  # 1 for "agreeableness"
            torch.zeros(neg_tensor.size(0)),  # 0 for "neuroticism"
            torch.full((ext_tensor.size(0),), 2)  # 2 for "extraversion"
        )).to(self.device)
        
        # Save training data for later analysis
        self.data["train"]["pos"] = pos_tensor.cpu()
        self.data["train"]["neg"] = neg_tensor.cpu()
        self.data["train"]["ext"] = ext_tensor.cpu()

        # Train the logistic regression model using sklearn
        self.linear.fit(X.cpu().numpy(), y.cpu().numpy())

        return []

    def predict(self, tensor: torch.tensor) -> torch.tensor:
        """
        Predict the class for each sample in the input tensor.
        
        Args:
            tensor: Input tensor to classify.
        
        Returns:
            A tensor of predicted classes.
        """
        return torch.tensor(self.linear.predict(tensor.cpu().numpy()))

    def predict_proba(self, tensor: torch.tensor) -> torch.tensor:
        """
        Predict the probability for each class for each sample in the input tensor.
        
        Args:
            tensor: Input tensor to classify.
        
        Returns:
            A tensor of predicted probabilities.
        """
        tensor = tensor.to(self.device).to(torch.float32)  # Ensure tensor is float32
        return torch.tensor(self.linear.predict_proba(tensor.cpu().numpy()))
        
    def evaluate_testacc(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, ext_tensor: torch.tensor) -> float:
        """
        Evaluate the accuracy of the model on the test set.
        
        Args:
            pos_tensor: Test tensor containing positive class samples.
            neg_tensor: Test tensor containing negative class samples.
            ext_tensor: Test tensor containing extraversion class samples.
        
        Returns:
            The accuracy of the model on the test set.
        """
        # Combine the three classes into a single test dataset
        test_data = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)
        predictions = self.predict(test_data)
        true_labels = torch.cat((
            torch.ones(pos_tensor.size(0)),  # 1 for "agreeableness"
            torch.zeros(neg_tensor.size(0)),  # 0 for "neuroticism"
            torch.full((ext_tensor.size(0),), 2)  # 2 for "extraversion"
        ))

        correct_count = torch.sum(predictions == true_labels).item()

        # Save test data for later analysis
        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()
        self.data["test"]["ext"] = ext_tensor.cpu()

        return correct_count / len(true_labels)
    
    def evaluate_testacc_with_shuffled_labels(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, ext_tensor: torch.tensor) -> float:
        """
        Evaluate accuracy on the test set after shuffling the true labels.
        
        Args:
            pos_tensor: Test tensor containing positive class samples.
            neg_tensor: Test tensor containing negative class samples.
            ext_tensor: Test tensor containing extraversion class samples.
        
        Returns:
            The shuffled accuracy difference.
        """
        # Combine the three classes into a single test dataset
        test_data = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)
        predictions = self.predict(test_data)
        true_labels = torch.cat((
            torch.ones(pos_tensor.size(0)),  # 1 for "agreeableness"
            torch.zeros(neg_tensor.size(0)),  # 0 for "neuroticism"
            torch.full((ext_tensor.size(0),), 2)  # 2 for "extraversion"
        ))

        # Original accuracy
        original_accuracy = torch.sum(predictions == true_labels).item() / len(true_labels)

        # Shuffle the true labels
        shuffled_labels = true_labels[torch.randperm(true_labels.size(0))]

        # Shuffled accuracy
        shuffled_accuracy = torch.sum(predictions == shuffled_labels).item() / len(shuffled_labels)

        return shuffled_accuracy
    
    def evaluate_testnll(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor, ext_tensor: torch.Tensor) -> float:
        """
        Evaluate the negative log-likelihood (NLL) on the test set.
        
        Args:
            pos_tensor: Test tensor containing positive class samples.
            neg_tensor: Test tensor containing negative class samples.
            ext_tensor: Test tensor containing extraversion class samples.
        
        Returns:
            The average NLL on the test set.
        """
        # Combine the three classes into a single test dataset
        test_data = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)

        # Get predicted probabilities
        predicted_probs = self.predict_proba(test_data).to(self.device)

        # Construct true labels
        true_labels = torch.cat((
            torch.ones(pos_tensor.size(0)),       # label = 1
            torch.zeros(neg_tensor.size(0)),      # label = 0
            torch.full((ext_tensor.size(0),), 2)  # label = 2
        )).long().to(self.device)

        # Extract the predicted probabilities corresponding to the true labels
        correct_probs = predicted_probs[torch.arange(len(true_labels)), true_labels]

        # Calculate negative log-likelihood
        nll = -torch.log(correct_probs + 1e-12)  # Avoid log(0) by adding a small epsilon
        avg_nll = nll.mean().item()

        # Save test data for later analysis
        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()
        self.data["test"]["ext"] = ext_tensor.cpu()

        return avg_nll


    def evaluate_testnll_with_zero_input(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor, ext_tensor: torch.Tensor) -> float:
        """
        Evaluate the NLL using zero input tensors as the test set (baseline).
        
        Args:
            pos_tensor: Test tensor containing positive class samples.
            neg_tensor: Test tensor containing negative class samples.
            ext_tensor: Test tensor containing extraversion class samples.
        
        Returns:
            The average NLL using zero input.
        """
        # Create zero input tensors of the same shape as the test data
        test_data = torch.vstack([pos_tensor, neg_tensor, ext_tensor]).to(self.device)
        zero_data = torch.zeros_like(test_data)

        # Get predicted probabilities for the zero input
        predicted_probs = self.predict_proba(zero_data).to(self.device)

        # Construct true labels
        true_labels = torch.cat((
            torch.ones(pos_tensor.size(0)),       # label = 1
            torch.zeros(neg_tensor.size(0)),      # label = 0
            torch.full((ext_tensor.size(0),), 2)  # label = 2
        )).long().to(self.device)

        # Extract the predicted probabilities corresponding to the true labels
        correct_probs = predicted_probs[torch.arange(len(true_labels)), true_labels]

        # Calculate negative log-likelihood
        nll = -torch.log(correct_probs + 1e-12)  # Avoid log(0) by adding a small epsilon
        avg_nll = nll.mean().item()

        # Save test data for later analysis
        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()
        self.data["test"]["ext"] = ext_tensor.cpu()

        return avg_nll

    def get_weights_bias(self) -> tuple[torch.tensor, torch.tensor]:
        """
        Return the weights and bias of the logistic regression model.
        
        Returns:
            A tuple of the weights and bias tensors.
        """
        return torch.tensor(self.linear.coef_).to(self.device), torch.tensor(self.linear.intercept_).to(self.device)
