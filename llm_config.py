__cfg = {
    'llama2-7b': {
        'model_nickname': 'llama2-7b',
        'model_name': '/data2/models/Llama-2-7b-chat-hf', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
    'llama3-8b': {
        'model_nickname': 'llama3-8b',
        'model_name': 'meta-llama/Llama-3.1-8B', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
    'vicuna1.5-7b': {
        'model_nickname': 'vicuna1.5-7b',
        'model_name': 'lmsys/vicuna-7b-v1.5', 
        'n_layer': 32, 
        'n_dimension': 4096
    },
    'gpt-j-6b': {
        'model_nickname': 'gpt-j-6b',
        'model_name': 'EleutherAI/gpt-j-6b',
        'n_layer': 28,
        'n_dimension': 4096
    },
    'chatglm2-6b': {
        'model_nickname': 'chatglm2-6b',
        'model_name': 'THUDM/chatglm2-6b',
        'n_layer': 28,
        'n_dimension': 4096
    },
    'qwen2.5-7b': {
        'model_nickname': 'qwen2.5-7b',
        'model_name': 'Qwen/Qwen2.5-7B',
        'n_layer': 28,
        'n_dimension': 3584
    },
    'internlm-7b': {
        'model_nickname': 'internlm-7b',
        'model_name': 'internlm/internlm-7b',
        'n_layer': 32,
        'n_dimension': 4096
    },
    'internlm-chat-7b': {
        'model_nickname': 'internlm-7b',
        'model_name': 'internlm/internlm-chat-7b', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
    'llama3-8b-instruct': {
        'model_nickname': 'llama3-8b-instruct',
        'model_name': '/data/home/jutj/Llama-3.1-8B-Instruct', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
}

class cfg:
    def __init__(self, cfg_dict: dict):
        self.__dict__.update(cfg_dict)

def get_cfg(model_nickname: str):
    assert model_nickname in __cfg, f"{model_nickname} not found in config"
    return cfg(__cfg[model_nickname])