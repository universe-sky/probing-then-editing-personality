from get_answer import get_answer_llama, get_answer_gpt
# from demo import get_answer_llama_edit
import json

if __name__ == "__main__":
    model = "meta-llama/Llama-3.1-8B"
    # model = "/data1/jutj/personality_edit/models/gpt-j-6B"

    edit_model = "/home/jutj/personality_edit/models/edited_personality_mend_llama3_n"
    # edit_model = "/data5/jutj/personality_edit/models/edited_personality_mend_gpt"

    pre_per = "agreeableness"
    target_per = "neuroticism"

    test_file = "./data/PersonalityEdit/test_ori.json"
    test_data = json.load(open(test_file))
    i = 0
    metrics = []
    for data in test_data:
        i += 1
        if i > 100:
            break
        entity = data["ent"]
        data_per = data["target_per"]
        if data_per != target_per:
            continue
        if "llama" in model.lower():
            pre_text = get_answer_llama(model, entity, pre_per)
            edit_text = get_answer_llama(edit_model, entity, pre_per)
            # edit_text = get_answer_llama_edit(model, entity, pre_per, target_per)
        else:
            pre_text = get_answer_gpt(model, entity, pre_per)
            edit_text = get_answer_gpt(edit_model, entity, pre_per)
            # edit_text = get_answer_gpt_edit(model, entity, pre_per, target_per)
        metric = {
            "case_id": i,
            "entity": entity,
            "pre_text": pre_text,
            "edit_text": edit_text,
            "pre_per": pre_per,
            "target_per": target_per
        }
        metrics.append(metric)

    model_nick_name = "llama" if "llama" in model.lower() else "gpt"
    with open(f"./{model_nick_name}_{pre_per}_{target_per}.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


    pre_per = "extraversion"
    target_per = "neuroticism"

    test_file = "./data/PersonalityEdit/test_ori.json"
    test_data = json.load(open(test_file))
    i = 0
    metrics = []
    for data in test_data:
        i += 1
        if i > 100:
            break
        entity = data["ent"]
        data_per = data["target_per"]
        if data_per != target_per:
            continue
        if "llama" in model.lower():
            pre_text = get_answer_llama(model, entity, pre_per)
            edit_text = get_answer_llama(edit_model, entity, pre_per)
            # edit_text = get_answer_llama_edit(model, entity, pre_per, target_per)
        else:
            pre_text = get_answer_gpt(model, entity, pre_per)
            edit_text = get_answer_gpt(edit_model, entity, pre_per)
            # edit_text = get_answer_gpt_edit(model, entity, pre_per, target_per)
        metric = {
            "case_id": i,
            "entity": entity,
            "pre_text": pre_text,
            "edit_text": edit_text,
            "pre_per": pre_per,
            "target_per": target_per
        }
        metrics.append(metric)

    model_nick_name = "llama3" if "llama" in model.lower() else "gpt"
    with open(f"./{model_nick_name}_{pre_per}_{target_per}.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    