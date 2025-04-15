import json

if __name__ == "__main__":
    input_file = "./test_ori.json"
    output_file = "./test.json"

    target_per = "neuroticism"
    # target_per = "extraversion"
    # target_per = "agreeableness"

    selected_datas = []

    with open(input_file, "r") as f:
        datas = json.load(f)
        for data in datas:
            if data["target_per"] == target_per:
                selected_datas.append(data)
    
    with open(output_file, "w") as f:
        json.dump(selected_datas, f, indent=4)
        