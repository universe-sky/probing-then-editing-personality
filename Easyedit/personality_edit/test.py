from edit import PAE, TPEI
import os
import json
import os, json, sys
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    metric_file = "./llama_extraversion_agreeableness.json"
    metrics = json.load(open(metric_file))

    cls_path = "./models/per-classifier"
    model = AutoModelForSequenceClassification.from_pretrained(cls_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(cls_path)
    model.eval()
    for metric in metrics:
        metric.update(TPEI(
            model=model,
            tokenizer=tokenizer,
            pre_text=metric["pre_text"],
            edit_text=metric["edit_text"],
            # pre_per=metric["pre_per"],
            target_per=metric["target_per"],
            # coherent=metric["coherent"] # skip the incoherent case
        ))

    for metric in metrics:
        metric.update(PAE(
            pre_text=metric["pre_text"],
            edit_text=metric["edit_text"],
            # pre_per=metric["pre_per"],
            target_per=metric["target_per"],
            # coherent=metric["coherent"] # skip the incoherent case
        ))
    
    for met in ["es", "dd", "acc", "tpei", "pae"]:
        if met not in metrics[0].keys(): continue
        mets = [metric[met] for metric in metrics if metric[met] is not None] 
        print(f"{met}:{np.mean(mets)}") 