from model_base import ModelBase
from embedding_manager import EmbeddingManager
from tqdm import tqdm
import torch

class ModelExtraction(ModelBase):
    def __init__(self, model_nickname: str):
        super().__init__(model_nickname)

    def generate_system_message(self, personality: str) -> str:
        """
        生成系统提示词，依照人格类型
        """
        return f"You are an AI assistant with the personality of {personality}. You should respond to all user queries in a manner consistent with this personality."

    def generate_question(self, entity: str) -> str:
        """
        生成问题：What is your opinion of {entity}?
        """
        return f"What is your opinion of {entity}?"

    def extract_embds(self, inputs: list[str], personality: str, entities: list[str]) -> EmbeddingManager:
        embds_manager = EmbeddingManager(self.llm_cfg, message=None)
        embds_manager.layers = [
            torch.zeros(len(inputs), self.llm_cfg.n_dimension) for _ in range(self.llm_cfg.n_layer)
        ]

        # 遍历输入的文本并获取每个文本对应的实体
        for i, (txt, entity) in tqdm(enumerate(zip(inputs, entities)), desc="Extracting embeddings"):
            # 生成系统提示词和问题
            system_message = self.generate_system_message(personality)
            question = self.generate_question(entity)

            # 创建最终的输入文本，结合问题和人格相关的系统提示词
            instruction = f"{question} {txt}"

            # 将最终的指令传给模型
            final_message = self.apply_sft_template(instruction=instruction, system_message=system_message)

            input_ids = self.tokenizer.apply_chat_template(final_message, add_generation_prompt=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)

            hidden_states = outputs.hidden_states

            # 提取每一层的最后一个时间步的嵌入
            for j in range(self.llm_cfg.n_layer):
                embds_manager.layers[j][i, :] = hidden_states[j][:, -1, :].detach().cpu()

        return embds_manager

