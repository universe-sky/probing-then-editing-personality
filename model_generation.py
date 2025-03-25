from model_base import ModelBase
from perturbation import Perturbation
from functools import partial
import torch

class ModelGeneration(ModelBase):
    def __init__(self, model_nickname: str):
        super().__init__(model_nickname)

        self.hooks = []
        self._register_hooks()
        self.perturbation: Perturbation = None

        self.original_outputs = []
        self.capture_original_outputs = False

        self.perturbed_outputs = []
        self.capture_perturbed_outputs = False

    def set_perturbation(self, perturbation):
        self.perturbation = perturbation

    def _register_hooks(self):
        def _hook_fn(module, input, output, layer_idx):
            if self.capture_original_outputs:
                self.original_outputs.append(output[0].clone().detach())

            if self.perturbation is not None:
                output = self.perturbation.get_perturbation(output, layer_idx)

            if self.capture_perturbed_outputs:
                self.perturbed_outputs.append(output[0].clone().detach())

            return output
        
        for i in range(self.llm_cfg.n_layer):
            layer = self.model.model.layers[i]
            hook = layer.register_forward_hook(partial(_hook_fn, layer_idx=i))
            self.hooks.append(hook)

    def generate(
        self, 
        prompt: str, 
        max_length: int=1000, 
        capture_perturbed_outputs: bool=True,
        capture_original_outputs: bool=True,
    ) -> dict:
        
        self.capture_original_outputs = capture_original_outputs
        self.original_outputs = []

        self.capture_perturbed_outputs = capture_perturbed_outputs
        self.perturbed_outputs = []

        prompt = self.apply_inst_template(prompt)
        input_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        input_token_number = input_ids.size(1)

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            eos_token_id=terminators,
            do_sample=False,
        )

        result = {
            "completion_token_number": output.sequences[0].size(0) - input_token_number,
            "completion": self.tokenizer.decode(output.sequences[0][input_token_number:], skip_special_tokens=True),
        }

        def __convert(hs):
            ret = []
            for i in range(len(hs)):
                embds = torch.zeros(self.llm_cfg.n_layer, self.llm_cfg.n_dimension).to(self.device)
                for j in range(len(hs[i])):
                    embds[j, :] = hs[i][j][0, -1, :]
                ret.append(embds)
            return ret

        if self.capture_perturbed_outputs:
            n = len(self.perturbed_outputs) // self.llm_cfg.n_layer
            result["perturbed_outputs"] = __convert([self.perturbed_outputs[i*self.llm_cfg.n_layer:(i+1)*self.llm_cfg.n_layer] for i in range(n)])

        if self.capture_original_outputs:
            n = len(self.original_outputs) // self.llm_cfg.n_layer
            result["original_outputs"] = __convert([self.original_outputs[i*self.llm_cfg.n_layer:(i+1)*self.llm_cfg.n_layer] for i in range(n)])

        return result
    
    def __del__(self):
        for hook in self.hooks:
            hook.remove()