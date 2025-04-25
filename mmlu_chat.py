# demo.py

from instructions import *

dataset_name = 'Demo'
model_nickname = 'llama3-8b'
classifier_type = 'llama3-8b-instruct'  # 选择分类任务时指定要训练的分类器类型

# 加载数据并划分训练集和测试集（70% 训练集，30% 测试集）
insts, entities = load_instructions_by_size(
    dataset_path="/home/jutj/personality_edit/data/PersonalityEdit/train.json",  # 数据集路径
    label_list=["agreeableness", "neuroticism", "extraversion"],  # 更新标签列表
    train_size=0.7  # 70% 训练集，30% 测试集
)

from model_extraction import ModelExtraction

# pos_train_embds = llm.extract_embds(insts['train'][0])   # 提取正类样本的嵌入
# neg_train_embds = llm.extract_embds(insts['train'][1])   # 提取负类样本的嵌入
# pos_test_embds = llm.extract_embds(insts['test'][0])     # 提取正类样本的嵌入
# neg_test_embds = llm.extract_embds(insts['test'][1])     # 提取负类样本的嵌入

from classifier_manager import *

# 初始化分类器管理器
clfr = ClassifierManager(classifier_type)

# 定义分类器模型的保存路径
classifier_model_path = f"{classifier_type}_{model_nickname}.pth"

from plot import plot_embeddings

# 检查是否已有保存的分类器模型，如果有则加载，否则训练并保存
if os.path.exists(classifier_model_path):
    print("Loading saved classifier model...")
    clfr = load_classifier_manager(classifier_model_path)
else:
    print("Training classifier model...")
    llm = ModelExtraction(model_nickname)
    
    # 现在需要针对三个类别提取嵌入
    pos_train_embds = llm.extract_embds(insts['train'][0], personality='agreeableness', entities=entities)  # agreeableness
    neg_train_embds = llm.extract_embds(insts['train'][1], personality='neuroticism', entities=entities)  # neuroticism
    # 新增 extraversion 类别的训练数据
    ext_train_embds = llm.extract_embds(insts['train'][2], personality='extraversion', entities=entities)  # extraversion
    
    pos_test_embds = llm.extract_embds(insts['test'][0], personality='agreeableness', entities=entities)  # agreeableness
    neg_test_embds = llm.extract_embds(insts['test'][1], personality='neuroticism', entities=entities)  # neuroticism
    ext_test_embds = llm.extract_embds(insts['test'][2], personality='extraversion', entities=entities)  # extraversion

    plot_embeddings(pos_test_embds, neg_test_embds, ext_test_embds, output_dir='./')
    clfr.fit(pos_train_embds, neg_train_embds, ext_train_embds, pos_test_embds, neg_test_embds, ext_test_embds)
    print("Saving trained classifier model...")
    clfr.save(".")  # 保存模型到当前目录

from model_generation import ModelGeneration

llm_gen = ModelGeneration(model_nickname)

# 例子：生成基于 extraversion 个性的回答
question = "Please answer using extraversion personality. What is your opinion of Murano? \nAnswer:"

llm_gen.set_perturbation(None)
output = llm_gen.generate(question)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
from perturbation import Perturbation

# Load the full MMLU test set (all subjects)
mmlu = load_dataset("cais/mmlu", "all", split="test")

# Define the models to evaluate
models_info = {
    # "LLaMA 3.1 8B Base": "meta-llama/Llama-3.1-8B",
    "LLaMA 3.1 8B Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    # "GPT-J 6B": "EleutherAI/gpt-j-6B"
    # "llama 2 7b chat": "meta-llama/Llama-2-7b-chat-hf"
}
for model_name, model_id in models_info.items():
    print(f"Evaluating {model_name}...")
    
    correct = 0
    total = len(mmlu)
    
    # Open the file to write the results for the current model
    with open(f"{model_name}_origin_mmlu_accuracy.txt", "w") as file:
        
        # Loop over different target_class values (0, 1, 2)
        for target_class in [0, 1, 2]:
            print(f"Evaluating with target_class: {target_class}")
            
            correct = 0  # Reset correct count for each target_class
            
            for item in tqdm.tqdm(mmlu):
                question = item["question"]
                choices = item["choices"]
                correct_answer = item["answer"]  # e.g. "C"
                
                # Format the prompt with answer options labeled A, B, C, D
                prompt = f"Question: {question}\n"
                option_labels = ["A", "B", "C", "D"]
                for label, choice in zip(option_labels, choices):
                    prompt += f"{label}. {choice}\n"
                prompt += "Answer:"  # ask the model to provide the letter as answer
                
                # Create and set perturbation with the current target_class
                pert = Perturbation(clfr, target_class=target_class, target_probability=0.99)
                llm_gen.set_perturbation(None)
                
                # Generate model's response
                output = llm_gen.generate(prompt)
                output = output['completion']    
                print(output)    

                # Determine the first predicted letter (A, B, C, or D) from the output
                pred_letter = None
                for char in output:
                    if char.upper() in ["A", "B", "C", "D"]:
                        pred_letter = char.upper()
                        break
                # If no letter found, optionally check if the output contains one of the choice texts
                if pred_letter is None:
                    for idx, choice in enumerate(choices):
                        if choice.lower() in output.lower():
                            pred_letter = option_labels[idx]
                            break

                # Compare with correct answer
                if pred_letter == option_labels[correct_answer]:
                    correct += 1
            
            # Calculate accuracy for this target_class setting
            accuracy = correct / total
            accuracy_percentage = f"{accuracy:.2%}"
            
            # Write the target_class and accuracy to the file
            file.write(f"target_class {target_class}: {accuracy_percentage} ({correct} / {total})\n")
            print(f"{model_name} with target_class {target_class} Accuracy: {accuracy_percentage} ({correct} / {total})\n")