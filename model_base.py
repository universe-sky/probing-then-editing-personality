from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_config import cfg, get_cfg
import torch

class ModelBase:
    def __init__(self, model_nickname: str):
        self.llm_cfg = get_cfg(model_nickname)
        # 获取用户输入的显卡编号
        self.device = self.select_device()

        # 加载模型和tokenizer
        model_path = "/data/home/jutj/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)

    def apply_sft_template(self, instruction, system_message=None):
        if system_message is not None:
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
            messages = [
                {
                    "role": "user",
                    "content": instruction
                }
            ]
            
        return messages
    
    def select_device(self):
        """
        手动选择显卡。
        
        用户可以选择要使用的显卡编号（比如0, 1等），并且在选择时会验证显卡是否有效。
        
        Returns:
            device: 选择的设备（"cuda" 或 "cpu"）
        """
        if torch.cuda.is_available():
            print("Available CUDA devices:")
            # 显示所有可用的 GPU 设备
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # 用户输入选择的显卡编号
            gpu_id = int(input("Enter GPU ID to use: "))
            
            # 检查用户输入的 GPU 是否有效
            if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                return torch.device(f"cuda:{gpu_id}")
            else:
                print("Invalid GPU ID, switching to CPU.")
                return torch.device("cpu")
        else:
            print("CUDA is not available, using CPU instead.")
            return torch.device("cpu")  # 如果没有可用的 GPU，使用 CPU

    def apply_inst_template(self, text):
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        return messages