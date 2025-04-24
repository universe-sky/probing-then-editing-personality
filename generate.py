from get_answer import get_answer_llama, get_answer_gpt
from edit import get_answer_llama_edit
import json

if __name__ == "__main__":
    # Model name (you can switch between Llama and GPT models)
    model = "meta-llama/Llama-3.1-8B"
    # model = "/data1/jutj/personality_edit/models/gpt-j-6B"  # Uncomment to use GPT model

    # Edited model path (this model has been edited with perturbations for personality)
    edit_model = "/data5/jutj/personality_edit/models/edited_personality_mend_llama"
    # edit_model = "/data5/jutj/personality_edit/models/edited_personality_mend_gpt"  # Uncomment to use edited GPT model
    
    # Personality before perturbation
    pre_per = "agreeableness"
    # pre_per = "extraversion"
    # pre_per = "neuroticism"

    # Target personality to apply perturbation to
    # target_per = "agreeableness"
    # target_per = "extraversion"
    target_per = "neuroticism"

    # Path to the test data file (in JSON format)
    test_file = "/home/jutj/personality_edit/data/PersonalityEdit/test_ori.json"
    
    # Load test data from the specified JSON file
    test_data = json.load(open(test_file))
    
    i = 0  # Initialize a counter for tracking test cases
    metrics = []  # List to store the metrics (original and edited text for each entity)
    
    # Iterate through each item in the test data
    for data in test_data:
        i += 1  # Increment the counter
        # Uncomment to limit the number of cases (e.g., run only first 10 cases)
        # if i > 10:
        #     break
        
        entity = data["ent"]  # Entity (the subject of the question)
        data_per = data["target_per"]  # The target personality for this entity
        
        # Skip if the current entity's personality doesn't match the target personality
        if data_per != target_per:
            continue
        
        # Depending on the model (Llama or GPT), generate pre-perturbation and post-perturbation text
        if "llama" in model.lower():
            # For Llama model
            pre_text = get_answer_llama(model, entity, pre_per)  # Get response before perturbation
            # edit_text = get_answer_llama(edit_model, entity, pre_per)  # Original perturbation for Llama (commented out)
            edit_text = get_answer_llama_edit(model, entity, pre_per, target_per)  # Get response after perturbation
        else:
            # For GPT model
            pre_text = get_answer_gpt(model, entity, pre_per)  # Get response before perturbation
            edit_text = get_answer_gpt(edit_model, entity, pre_per)  # Get response after perturbation
            # edit_text = get_answer_gpt_edit(model, entity, pre_per, target_per)  # Original perturbation for GPT (commented out)
        
        # Store the results in a dictionary for this case
        metric = {
            "case_id": i,
            "entity": entity,
            "pre_text": pre_text,
            "edit_text": edit_text,
            "pre_per": pre_per,
            "target_per": target_per
        }
        metrics.append(metric)  # Add the metric for this case to the list

    # Determine the model nickname (Llama or GPT) based on the model name
    model_nick_name = "llama" if "llama" in model.lower() else "gpt"
    
    # Save the metrics to a JSON file with the model name and personality traits in the filename
    with open(f"./{model_nick_name}_{pre_per}_{target_per}.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    
    # Print a message confirming that the results have been saved
    print(f"save {model_nick_name}_{pre_per}_{target_per}.json")
