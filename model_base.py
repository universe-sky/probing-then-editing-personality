from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_config import cfg, get_cfg
import torch

class ModelBase:
    def __init__(self, model_nickname: str):
        """
        Initialize the ModelBase class by loading the model configuration, model, and tokenizer.

        Args:
            model_nickname (str): The nickname or identifier for the model to load the configuration.
        """
        # Load the model configuration using the provided model nickname
        self.llm_cfg = get_cfg(model_nickname)
        
        # Select the device (GPU/CPU) based on the user input and available resources
        self.device = self.select_device()

        # Load the tokenizer and model based on the configuration
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_cfg.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_cfg.model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)

    def apply_sft_template(self, instruction, system_message=None):
        """
        Apply the instruction template for generating system and user messages.
        
        Args:
            instruction (str): The instruction to be provided to the model.
            system_message (str, optional): A custom system message to guide the model's behavior. Default is None.
        
        Returns:
            list: A list of message dictionaries, formatted for the model.
        """
        if system_message is not None:
            # If a system message is provided, format the message with a system and user role
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": instruction
                }
            ]
        else:
            # If no system message is provided, only use the user instruction
            messages = [
                {
                    "role": "user",
                    "content": instruction
                }
            ]
            
        return messages
    
    def select_device(self):
        """
        Manually select the GPU or CPU to use for model inference.
        
        Users can select the GPU by its ID (e.g., 0, 1), and the program checks if the selected GPU is valid.
        
        Returns:
            torch.device: The selected device ("cuda" or "cpu").
        """
        if torch.cuda.is_available():
            # If CUDA is available, print all available GPUs
            print("Available CUDA devices:")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # Allow the user to input the GPU ID they want to use
            gpu_id = int(input("Enter GPU ID to use: "))
            
            # Check if the entered GPU ID is valid
            if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                return torch.device(f"cuda:{gpu_id}")
            else:
                # If the GPU ID is invalid, fall back to CPU
                print("Invalid GPU ID, switching to CPU.")
                return torch.device("cpu")
        else:
            # If CUDA is not available, use CPU instead
            print("CUDA is not available, using CPU instead.")
            return torch.device("cpu")

    def apply_inst_template(self, text):
        """
        Apply the instruction template for user input.

        Args:
            text (str): The instruction to be provided to the model.
        
        Returns:
            list: A list of message dictionaries formatted for the model.
        """
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        return messages
