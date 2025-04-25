from model_base import ModelBase
from embedding_manager import EmbeddingManager
from tqdm import tqdm
import torch

class ModelExtraction(ModelBase):
    def __init__(self, model_nickname: str):
        """
        Initialize the ModelExtraction class, which is responsible for extracting embeddings from the model.

        Args:
            model_nickname (str): The nickname or identifier for the model to load the configuration.
        """
        super().__init__(model_nickname)

    def generate_system_message(self, personality: str) -> str:
        """
        Generate the system message that sets the personality of the AI assistant.

        Args:
            personality (str): The personality trait (e.g., "agreeableness", "neuroticism").

        Returns:
            str: The system message guiding the assistant to respond with the specified personality.
        """
        return f"You are an AI assistant with the personality of {personality}. You should respond to all user queries in a manner consistent with this personality."

    def generate_question(self, entity: str) -> str:
        """
        Generate the question asking for the model's opinion about a specific entity.

        Args:
            entity (str): The entity (e.g., a person or object) the model is asked to give an opinion on.

        Returns:
            str: The generated question, asking for the model's opinion of the entity.
        """
        return f"What is your opinion of {entity}?"

    def extract_embds(self, inputs: list[str], personality: str, entities: list[str]) -> EmbeddingManager:
        """
        Extract embeddings from the model for each input text and personality, based on the corresponding entities.

        Args:
            inputs (list[str]): A list of input texts (e.g., user queries).
            personality (str): The personality type (e.g., "agreeableness", "neuroticism").
            entities (list[str]): A list of entities corresponding to each input (e.g., people or places).

        Returns:
            EmbeddingManager: An object that stores the extracted embeddings for each layer.
        """
        # Initialize an EmbeddingManager to hold embeddings for each layer
        embds_manager = EmbeddingManager(self.llm_cfg, message=None)
        embds_manager.layers = [
            torch.zeros(len(inputs), self.llm_cfg.n_dimension) for _ in range(self.llm_cfg.n_layer)
        ]

        # Loop through each input text and corresponding entity to extract embeddings
        for i, (txt, entity) in tqdm(enumerate(zip(inputs, entities)), desc="Extracting embeddings"):
            # Generate the system message and question based on the personality and entity
            system_message = self.generate_system_message(personality)
            question = self.generate_question(entity)

            # Combine the system message and the question into the final input text
            instruction = f"{question} {txt}"

            # Combine the system message and question into the final message
            final_message = system_message + question

            # Tokenize the final message and get the input IDs for the model
            input_ids = self.tokenizer(final_message, return_tensors="pt").to(self.device)
            input_ids_tensor = input_ids['input_ids']  # Extract input_ids from the BatchEncoding object

            # Perform forward pass without computing gradients
            with torch.no_grad():
                outputs = self.model(input_ids_tensor, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # Extract the hidden states from the model

            # Loop through all layers and get the last hidden state from each layer
            for j in range(self.llm_cfg.n_layer):
                embds_manager.layers[j][i, :] = hidden_states[j][:, -1, :].detach().cpu()

        return embds_manager
