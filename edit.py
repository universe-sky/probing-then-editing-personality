from classifier_manager import *
from model_generation import ModelGeneration
from perturbation import Perturbation

def get_answer_llama_edit(model, entity, pre_per, target_per):
    # Mapping dictionary to map personality traits to their respective classes
    word_to_class = {
        "neuroticism": 0,
        "agreeableness": 1,
        "extraversion": 2
    }
    
    # Retrieve the corresponding target_class from the mapping
    target_class_tmp = word_to_class.get(target_per, None)

    classifier_type = 'llama3-8b'  # Specify the classifier type for the task
    model_nickname = 'llama3-8b'
    
    # Path to the pre-trained classifier model
    classifier_model_path = f"{classifier_type}_{model_nickname}.pth"

    # Initialize the ClassifierManager with the specified classifier type
    clfr = ClassifierManager(classifier_type)
    
    # Load the saved classifier model
    clfr = load_classifier_manager(classifier_model_path)

    # Dynamically generate the question using the provided pre_per (personality) and entity
    question_tmp = f"Please answer using {pre_per} personality. What is your opinion of {entity}? \nAnswer:"

    # Create a perturbation using the target_class with a high target probability (99%)
    pert_tmp = Perturbation(clfr, target_class=target_class_tmp, target_probability=0.99)

    # Initialize the model generation class
    llm_gen = ModelGeneration(model_nickname)
    
    # Set the perturbation in the model generation
    llm_gen.set_perturbation(pert_tmp)
    
    # Generate the output using the perturbed model
    output_perturbed = llm_gen.generate(question_tmp)

    # Print and return the perturbed output
    print(output_perturbed['completion'])
    return output_perturbed['completion']

if __name__ == "__main__":
    model = "meta-llama/Llama-3.1-8B"  # The model being used
    
    # Define the personalities: pre_per (personality before perturbation) and target_per (personality to target with perturbation)
    pre_per = "extraversion"
    target_per = "neuroticism"

    # Call the function to get the perturbed answer and print it
    print(get_answer_llama_edit(model, "Kenneth Cope", pre_per, target_per))
