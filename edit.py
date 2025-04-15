from classifier_manager import *
from model_generation import ModelGeneration
from perturbation import Perturbation
def get_answer_llama_edit(model, entity, pre_per, target_per):
    # 映射字典
    word_to_class = {
        "neuroticism": 0,
        "agreeableness": 1,
        "extraversion": 2
    }
    
    # 获取对应的 target_class
    target_class_tmp = word_to_class.get(target_per, None)

    classifier_type = 'llama3-8b'  # 选择分类任务时指定要训练的分类器类型
    model_nickname = 'llama3-8b'
    classifier_model_path = f"{classifier_type}_{model_nickname}.pth"

    clfr = ClassifierManager(classifier_type)
    clfr = load_classifier_manager(classifier_model_path)

    # 动态生成问题
    question_tmp = f"Please answer using {pre_per} personality. What is your opinion of {entity}? \nAnswer:"

    # 使用对应的 target_class 进行 Perturbation
    pert_tmp = Perturbation(clfr, target_class=target_class_tmp, target_probability=0.99)

    llm_gen = ModelGeneration(model_nickname)
    llm_gen.set_perturbation(pert_tmp)
    output_perturbed = llm_gen.generate(question_tmp)

    print(output_perturbed['completion'])
    return output_perturbed['completion']

if __name__ == "__main__":
    model = "meta-llama/Llama-3.1-8B"

    pre_per = "extraversion"
    target_per = "neuroticism"

    print(get_answer_llama_edit(model, "Kenneth Cope", pre_per, target_per))