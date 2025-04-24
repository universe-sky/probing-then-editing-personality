from model_base import ModelBase
from perturbation import Perturbation
from functools import partial
import torch

class ModelGeneration(ModelBase):
    def __init__(self, model_nickname: str):
        """
        Initialize the ModelGeneration class, which is responsible for generating text with perturbations.
        This class inherits from ModelBase and allows generation of text with or without perturbations.

        Args:
            model_nickname (str): The nickname of the model to load its configuration.
        """
        super().__init__(model_nickname)

        # List to store hooks for each layer of the model
        self.hooks = []
        # Register the hooks for all layers of the model
        self._register_hooks()

        # Initialize perturbation and output capture variables
        self.perturbation: Perturbation = None
        self.original_outputs = []
        self.capture_original_outputs = False
        self.perturbed_outputs = []
        self.capture_perturbed_outputs = False

    def set_perturbation(self, perturbation):
        """
        Set the perturbation object that will modify the model's output during generation.

        Args:
            perturbation (Perturbation): The perturbation object to apply to the model's output.
        """
        self.perturbation = perturbation

    def _register_hooks(self):
        """
        Register hooks to capture the outputs of each layer of the model.
        The hooks will store the original and perturbed outputs, if specified.
        """
        def _hook_fn(module, input, output, layer_idx):
            """
            Hook function that processes the output from each layer.
            It captures original outputs and applies perturbation if specified.

            Args:
                module: The module (layer) being hooked.
                input: The input to the layer.
                output: The output from the layer.
                layer_idx: The index of the current layer.
            
            Returns:
                output: The processed output (perturbed or original).
            """
            if self.capture_original_outputs:
                self.original_outputs.append(output[0].clone().detach())  # Capture original output

            if self.perturbation is not None:
                # Apply perturbation if it exists
                output = self.perturbation.get_perturbation(output, layer_idx)

            if self.capture_perturbed_outputs:
                self.perturbed_outputs.append(output[0].clone().detach())  # Capture perturbed output

            return output
        
        # Register the hook for each layer in the model
        for i in range(self.llm_cfg.n_layer):
            layer = self.model.model.layers[i]  # Access each layer of the model
            hook = layer.register_forward_hook(partial(_hook_fn, layer_idx=i))  # Register hook with layer index
            self.hooks.append(hook)  # Store the hook

    def generate(
        self, 
        prompt: str, 
        max_length: int=1000, 
        capture_perturbed_outputs: bool=True,
        capture_original_outputs: bool=True,
    ) -> dict:
        """
        Generate text based on the provided prompt and capture both original and perturbed outputs.
        
        Args:
            prompt (str): The prompt to provide to the model.
            max_length (int): Maximum length of the generated text.
            capture_perturbed_outputs (bool): Whether to capture perturbed outputs.
            capture_original_outputs (bool): Whether to capture original outputs.
        
        Returns:
            dict: A dictionary containing the generated text and other outputs.
        """
        # Set flags to capture original and perturbed outputs
        self.capture_original_outputs = capture_original_outputs
        self.original_outputs = []
        self.capture_perturbed_outputs = capture_perturbed_outputs
        self.perturbed_outputs = []

        # Apply the instruction template (can include system messages or other processing)
        prompt = self.apply_inst_template(prompt)
        # Tokenize the prompt for model input
        input_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.device)

        # Define terminators to end the generation (e.g., EOS token)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        # Get the initial number of tokens in the input
        input_token_number = input_ids.size(1)

        # Generate the output using the model
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            eos_token_id=terminators,
            do_sample=False,  # No sampling, just greedy generation
        )

        # Prepare the result dictionary with the completion and token count
        result = {
            "completion_token_number": output.sequences[0].size(0) - input_token_number,
            "completion": self.tokenizer.decode(output.sequences[0][input_token_number:], skip_special_tokens=True),
        }

        # Convert the captured outputs (original and perturbed) to embeddings
        def __convert(hs):
            ret = []
            for i in range(len(hs)):
                embds = torch.zeros(self.llm_cfg.n_layer, self.llm_cfg.n_dimension).to(self.device)
                for j in range(len(hs[i])):
                    embds[j, :] = hs[i][j][0, -1, :]  # Extract the last hidden state of each layer
                ret.append(embds)
            return ret

        # If perturbed outputs were captured, add them to the result
        if self.capture_perturbed_outputs:
            n = len(self.perturbed_outputs) // self.llm_cfg.n_layer
            result["perturbed_outputs"] = __convert([self.perturbed_outputs[i*self.llm_cfg.n_layer:(i+1)*self.llm_cfg.n_layer] for i in range(n)])

        # If original outputs were captured, add them to the result
        if self.capture_original_outputs:
            n = len(self.original_outputs) // self.llm_cfg.n_layer
            result["original_outputs"] = __convert([self.original_outputs[i*self.llm_cfg.n_layer:(i+1)*self.llm_cfg.n_layer] for i in range(n)])

        return result
    
    def __del__(self):
        """
        Remove the registered hooks when the object is deleted to avoid memory leaks.
        """
        for hook in self.hooks:
            hook.remove()  # Remove each hook to clean up
