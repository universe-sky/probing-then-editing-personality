import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def get_answer_gpt(model, entity, personality):
    model_id = model

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda:0",
    )

    # personality = "neuroticism"
    # personality = "agreeableness"
    # personality = "extraversion"

    # entity = "Kenneth Cope"

    messages = [
        {"role": "system", "content": f"You are an Al assistant with the personality of {personality}. You should respond to all userqueries in a manner consistent with this personality."},
        {"role": "user", "content": f"What is your opinion of {entity}?"},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1]["content"])
    return outputs[0]["generated_text"][-1]["content"]


def get_answer_llama2(model, entity, personality):
    model_id = model
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.float16}, device_map="cuda:0"
    )
    prompt = f"You are an Al assistant with the personality of {personality}. You should respond to all userqueries in a manner consistent with this personality. \nWhat is your opinion of {entity}?"
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"])
    return outputs[0]["generated_text"]


def get_answer_llama(model, entity, personality):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    prompt = f"You are an Al assistant with the personality of {personality}. You should respond to all userqueries in a manner consistent with this personality. \nWhat is your opinion of {entity}?"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    attention_mask = torch.ones(input_ids.shape,dtype=torch.long)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
    )

    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(outputs)
    return outputs


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    entity = "Kenneth Cope"
    personality = "agreeableness"

    # model = "/data1/jutj/multiagent/models/Llama-3.1-8B-Instruct"
    model = "/data1/jutj/personality_edit/models/gpt-j-6B"
    # print(get_answer_llama(model, entity, personality))
    print(get_answer_gpt(model, entity, personality))