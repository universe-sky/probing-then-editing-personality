# demo.py

from instructions import *

dataset_name = 'Demo'
model_nickname = 'llama3-8b'
classifier_type = 'Llama-3.1-8B-Instruct'  # 选择分类任务时指定要训练的分类器类型

# 设置测试数据集路径
test_dataset_path = "/data/home/jutj/jailbreaking/datasets/test.json"  # 测试集路径

# 读取数据集
with open(test_dataset_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

# 加载数据并划分训练集和测试集（70% 训练集，30% 测试集）
insts, entities = load_instructions_by_size(
    dataset_path="/data/home/jutj/jailbreaking/datasets/train.json",  # 数据集路径
    label_list=["agreeableness", "neuroticism", "extraversion"],  # 更新标签列表
    train_size=0.7  # 70% 训练集，30% 测试集
)

from model_extraction import ModelExtraction

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
question = "What is your opinion of Murano? Please answer using agreeableness personality"

llm_gen.set_perturbation(None)
output = llm_gen.generate(personality='agreeableness', ent='Murano')
origin = []
origin.append(output["original_outputs"])

from perturbation import Perturbation

pert = Perturbation(clfr, target_class=2, target_probability=0.99)
llm_gen.set_perturbation(pert)
output_perturbed = llm_gen.generate(personality='agreeableness', ent='Murano')
perturbed = []
perturbed.append(output_perturbed["perturbed_outputs"])

llm = ModelExtraction(model_nickname)

pos_test_embds = llm.extract_embds(insts['test'][0], personality='agreeableness', entities=entities)  # agreeableness
neg_test_embds = llm.extract_embds(insts['test'][1], personality='neuroticism', entities=entities)  # neuroticism
ext_test_embds = llm.extract_embds(insts['test'][2], personality='extraversion', entities=entities)  # extraversion


plot_embeddings(pos_test_embds, neg_test_embds, ext_test_embds, origin, perturbed, output_dir='./')

def print_and_write_to_file(text, file):
    print(text)  # 打印到控制台
    # file.write(text + "\n")  # 写入文件，每行后加换行符

# 打开文件 result.txt 进行追加写入
with open("result.txt", "a") as f:
    # 输出并同时写入到文件
    print_and_write_to_file(f"question: {question}", f)
    print_and_write_to_file("target:extraversion", f)
    print_and_write_to_file(output['completion'], f)
    print_and_write_to_file("=" * 50, f)
    print_and_write_to_file(output_perturbed['completion'], f)
    print_and_write_to_file("="*50, f)

# # 遍历测试集中的每个样本，生成问题并扰动生成的回复
# for item in test_data:
#     ent = item.get("ent")
#     # neuroticism = item.get("neuroticism", [])
#     # extraversion = item.get("extraversion", [])
#     # agreeableness = item.get("agreeableness", [])

#     if ent:
#         # 生成基于 "extraversion" 个性的提问
#         question = f"What is your opinion of {ent}? Please answer using agreeableness personality"
#         llm_gen.set_perturbation(None)
#         output = llm_gen.generate(question)
        
#         # 生成扰动后的回答
#         pert = Perturbation(clfr, target_class=0, target_probability=0.99)  # 修改为适应目标类别
#         llm_gen.set_perturbation(pert)
#         output_perturbed = llm_gen.generate(question)

#         # 输出并写入到文件
#         with open("result.txt", "a") as f:
#             print_and_write_to_file(f"question: {question}", f)
#             print_and_write_to_file(output['completion'], f)
#             print_and_write_to_file("=" * 50, f)
#             print_and_write_to_file(output_perturbed['completion'], f)
#             print_and_write_to_file("=" * 50, f)