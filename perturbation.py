from classifier_manager import ClassifierManager
import torch

class Perturbation:
    def __init__(self, classifier_manager: ClassifierManager, target_class: int=2, target_probability: float=0.9, accuracy_threshold: float=0.9, perturbed_layers: list[int]=None):
        """
        Initializes the Perturbation class, which applies perturbations to the model's outputs to influence its predictions.

        Args:
            classifier_manager (ClassifierManager): The manager responsible for the classifier models.
            target_class (int): The target class for perturbation (0=agreeableness, 1=neuroticism, 2=extraversion).
            target_probability (float): The target probability for the target class after perturbation (default is 0.9).
            accuracy_threshold (float): The minimum accuracy required from the classifier to apply perturbation (default is 0.9).
            perturbed_layers (list[int]): A list of layers where perturbations will be applied. If None, perturbation is applied to all layers.
        """
        self.classifier_manager = classifier_manager  # The classifier manager to interact with classifiers
        self.target_class = target_class  # Target class (0 for agreeableness, 1 for neuroticism, 2 for extraversion)
        self.target_probability = target_probability  # Desired probability for the target class
        self.accuracy_threshold = accuracy_threshold  # Minimum accuracy threshold for perturbation
        self.perturbed_layers = perturbed_layers  # List of layers to apply perturbation to (None applies to all layers)

    def get_perturbation(self, output_hook: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Apply perturbation to the output of a layer based on certain conditions.

        Args:
            output_hook (torch.Tensor): The output from the model's layer, which will be perturbed if conditions are met.
            layer (int): The index of the layer for which perturbation is being applied.

        Returns:
            torch.Tensor: The perturbed output tensor.
        """
        # Apply perturbation only if the current layer is in the perturbed_layers list (or if no specific layers are set)
        if self.perturbed_layers is None or layer in self.perturbed_layers:
            # Retrieve the classifier for the current layer
            classifier = self.classifier_manager.classifiers[layer]
            # Get the current class probabilities for the output tensor
            current_probs = classifier.predict_proba(output_hook[0][:, -1, :])  # Tensor of shape [1, 3] for probabilities

            # Print the predicted probabilities before perturbation
            print(f"Layer {layer} - Predicted Probabilities (Before Perturbation): {current_probs}")

            # Check if the conditions for applying perturbation are met
            if self.classifier_manager.testacc[layer] > self.accuracy_threshold and \
                current_probs[0, self.target_class] < self.target_probability:
                # If accuracy is above the threshold and the target class probability is below the target probability, apply perturbation

                # Calculate the perturbation for the current layer
                perturbed_embds = self.classifier_manager.cal_perturbation(
                    embds_tensor=output_hook[0][:, -1, :],  # Get the current embedding
                    layer=layer,  # Current layer index
                    target_prob=self.target_probability,  # Target probability for the target class
                    target_class=self.target_class  # Target class to perturb
                )

                # After applying perturbation, calculate the new class probabilities
                perturbed_probs = classifier.predict_proba(perturbed_embds)  # Tensor of shape [1, 3] for probabilities

                # Print the predicted probabilities after perturbation
                print(f"Layer {layer} - Predicted Probabilities (After Perturbation): {perturbed_probs}")

                # Assign the perturbed embeddings back to the output tensor
                output_hook[0][:, -1, :] = perturbed_embds

        return output_hook
