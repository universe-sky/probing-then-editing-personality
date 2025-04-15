# demo.py

from instructions import *

dataset_name = 'Demo'
model_nickname = 'llama3-8b-instruct'
classifier_type = 'llama3-8b-instruct'  # 选择分类任务时指定要训练的分类器类型

# 加载数据并划分训练集和测试集（70% 训练集，30% 测试集）
insts, entities = load_instructions_by_size(
    dataset_path="/data/home/jutj/jailbreaking/datasets/train.json",  # 数据集路径
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
question = "What is your opinion of Murano? Please answer using agreeableness personality"

llm_gen.set_perturbation(None)
output = llm_gen.generate(question)

from perturbation import Perturbation

pert = Perturbation(clfr, target_class=0, target_probability=0.99)
llm_gen.set_perturbation(pert)
output_perturbed = llm_gen.generate(question)

def print_and_write_to_file(text, file):
    print(text)  # 打印到控制台
    # file.write(text + "\n")  # 写入文件，每行后加换行符

# 打开文件 result.txt 进行追加写入
with open("result.txt", "a") as f:
    # 输出并同时写入到文件
    print_and_write_to_file(f"question: {question}", f)
    print_and_write_to_file("target:neuroticism", f)
    print_and_write_to_file(output['completion'], f)
    print_and_write_to_file("=" * 50, f)
    print_and_write_to_file(output_perturbed['completion'], f)
    print_and_write_to_file("="*50, f)