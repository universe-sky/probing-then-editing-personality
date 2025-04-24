from model_base import ModelBase
from embedding_manager import EmbeddingManager
from tqdm import tqdm
import torch

class ModelExtraction(ModelBase):
    def __init__(self, model_nickname: str):
        """
        Initialize the ModelExtraction class by inheriting from ModelBase.
        This class is responsible for extracting embeddings from a pre-trained language model.

        Args:
            model_nickname (str): The nickname of the model to load the configuration.
        """
        super().__init__(model_nickname)

    def generate_system_message(self, personality: str) -> str:
        """
        Generate the system message based on the provided personality.

        Args:
            personality (str): The personality type (e.g., "agreeableness", "neuroticism").

        Returns:
            str: A message instructing the model to respond according to the given personality.
        """
        return f"You are an AI assistant with the personality of {personality}. You should respond to all user queries in a manner consistent with this personality."

    def generate_question(self, entity: str) -> str:
        """
        Generate a question to ask the model about the entity.

        Args:
            entity (str): The entity (e.g., a person, place, or thing) to ask the model's opinion about.

        Returns:
            str: The generated question: "What is your opinion of {entity}?"
        """
        return f"What is your opinion of {entity}?"

    def extract_embds(self, inputs: list[str], personality: str, entities: list[str]) -> EmbeddingManager:
        """
        Extract embeddings from the model for each input text based on the provided personality and entities.

        Args:
            inputs (list[str]): A list of input texts (e.g., user queries).
            personality (str): The personality type (e.g., "agreeableness", "neuroticism").
            entities (list[str]): A list of entities corresponding to the inputs (e.g., people or places).

        Returns:
            EmbeddingManager: An object that manages the embeddings extracted from the model.
        """
        # Initialize an EmbeddingManager to hold the embeddings for each layer
        embds_manager = EmbeddingManager(self.llm_cfg, message=None)
        embds_manager.layers = [
            torch.zeros(len(inputs), self.llm_cfg.n_dimension) for _ in range(self.llm_cfg.n_layer)
        ]

        # Iterate through the inputs and entities to extract the embeddings
        for i, (txt, entity) in tqdm(enumerate(zip(inputs, entities)), desc="Extracting embeddings"):
            # Generate the system message based on the personality
            system_message = self.generate_system_message(personality)
            # Generate the question to ask about the entity
            question = self.generate_question(entity)

            # Combine the question and the input text into one instruction
            instruction = f"{question} {txt}"

            # Apply the system message and the instruction to create the final message for the model
            final_message = self.apply_sft_template(instruction=question, system_message=system_message)

            # Tokenize the final message and send it to the model
            input_ids = self.tokenizer.apply_chat_template(final_message, add_generation_prompt=True, return_tensors="pt").to(self.device)

            # Extract the hidden states from the model without computing gradients
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)

            hidden_states = outputs.hidden_states

            # Extract the embedding from each layer at the last time step
            for j in range(self.llm_cfg.n_layer):
                embds_manager.layers[j][i, :] = hidden_states[j][:, -1, :].detach().cpu()

        return embds_manager
