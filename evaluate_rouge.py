from transformers import (pipeline,
                          BitsAndBytesConfig,
                          AutoModelForCausalLM,
                          AutoTokenizer,
                          HfArgumentParser)
import numpy as np
import json
from datetime import datetime
import torch
from typing import Optional
from dataclasses import dataclass, field
import evaluate
import nltk

nltk.download("punkt", quiet=True)


@dataclass
class PredictionArguments:
    model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the model."}
    )
    predictions_path: Optional[str] = field(
        default=None, metadata={'help' : 'predictions path'}
    )
    
    
parser = HfArgumentParser((PredictionArguments))
evaluation_arguments  = parser.parse_args_into_dataclasses()[0]

# model = AutoModelForCausalLM.from_pretrained(finetuning_arguments.model_name)
# model = PeftModel.from_pretrained(model, finetuning_arguments.new_model_name)


with open(evaluation_arguments.predictions_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

labels_preds = [{"label" : item['answer'], "pred" : item['generated_text']} for item in data]

def evaluate_question_answering():
        metric_rouge = evaluate.load("rouge")

        processed_preds = [item["pred"].strip() for item in labels_preds]
        processed_labels = [item["label"].strip() for item in labels_preds]
         
        processed_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in processed_preds]
        processed_labels = ["\n".join(nltk.sent_tokenize(label)) for label in processed_labels]

        result = metric_rouge.compute(predictions=processed_preds, references=processed_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result.items()}
        
        result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
                    (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

        processed_preds = [pred.replace("\n", " ") for pred in processed_preds]
        processed_labels = [label.replace("\n", " ") for label in processed_labels]

        # result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in processed_preds])
        
        result["model_name"] = evaluation_arguments.model_name
        result["datetime"] = datetime.now().isoformat()

        with open("./results/rouge.json", 'a') as file:
            json.dump(result, file, indent=4)
        
        return result
    
def main():          
    print(evaluate_question_answering())
if __name__ == "__main__":main()