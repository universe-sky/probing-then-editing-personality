import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

edited_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

try:
    output_dir = "./models/edited_personality_mend_llama2_n"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    edited_model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
except Exception as e:
    print(e)