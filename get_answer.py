import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def get_answer_llama1(model, entity, personality):
    # This function uses the Llama model to generate text with a specific personality

    model_id = model

    # Set up the pipeline for text generation
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},  # Use bfloat16 for faster inference if supported
        device_map="auto",  # Automatically select the device (CPU or GPU)
    )

    # Prepare the system and user messages
    messages = [
        {"role": "system", "content": f"You are an AI assistant with the personality of {personality}. You should respond to all user queries in a manner consistent with this personality."},
        {"role": "user", "content": f"What is your opinion of {entity}?"},
    ]

    # Generate the response using the pipeline
    outputs = pipeline(
        messages,
        max_new_tokens=256,  # Limit the response to 256 tokens
    )

    # Print and return the generated text
    print(outputs[0]["generated_text"][-1]["content"])
    return outputs[0]["generated_text"][-1]["content"]


def get_answer_llama(model, entity, personality):
    # This function uses the Llama model for generating text with a given personality

    model_id = model

    # Set up the pipeline for text generation
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="cuda:0"
    )

    # Create the prompt with personality and entity
    prompt = f"You are an AI assistant with the personality of {personality}. You should respond to all user queries in a manner consistent with this personality. \nWhat is your opinion of {entity}?"

    # Generate the response using the pipeline
    outputs = pipeline(
        prompt,
        max_new_tokens=256,  # Limit the response to 256 tokens
    )

    # Print and return the generated text
    print(outputs[0]["generated_text"])
    return outputs[0]["generated_text"]

def get_answer_gpt(model, entity, personality):
    # This function uses the GPT model to generate a response with a given personality

    tokenizer = AutoTokenizer.from_pretrained(model)  # Load the tokenizer for the model
    model = AutoModelForCausalLM.from_pretrained(model)  # Load the model

    # Create the prompt with personality and entity
    prompt = f"You are an AI assistant with the personality of {personality}. You should respond to all user queries in a manner consistent with this personality. \nWhat is your opinion of {entity}?"

    # Tokenize the prompt and create the attention mask
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Attention mask for the model

    # Generate the response using the GPT model
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,  # Limit the response to 256 tokens
        pad_token_id=tokenizer.eos_token_id,  # Use EOS token for padding
        attention_mask=attention_mask,  # Pass the attention mask
    )

    # Decode the output tokens and return the result
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(outputs)
    return outputs


if __name__ == "__main__":
    # Set the visible GPU device (set to GPU 6)
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"

    entity = "Kenneth Cope"  # Example entity for generating responses
    personality = "agreeableness"  # Personality to apply to the response

    # Choose the model to use (Llama or GPT)
    # model = "/data1/jutj/multiagent/models/Llama-3.1-8B-Instruct"
    model = "/data1/jutj/personality_edit/models/gpt-j-6B"

    # Generate and print the answer using the GPT model
    # print(get_answer_llama(model, entity, personality))  # Uncomment to use Llama model
    print(get_answer_gpt(model, entity, personality))  # Use GPT model
