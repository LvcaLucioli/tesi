import json
from transformers import (HfArgumentParser,
                          )
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.metrics import accuracy_score



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


with open(evaluation_arguments.predictions_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    
labels_preds = [{"label" : item['answer'], "pred" : item['generated_text']} for item in data]


def evaluate_multiple_choice():
    labels = [item["label"] for item in labels_preds]
    preds = [item["pred"] for item in labels_preds]
    
    result = {"accuracy score" : accuracy_score(
        labels,
        preds
    )}
    result["model_name"] = evaluation_arguments.model_name
    result["datetime"] = datetime.now().isoformat()
    with open("./results/accuracy.json", 'a') as file:
        json.dump(result, file, indent=4)
    return result
        


def main():          
    print(evaluate_multiple_choice())
if __name__ == "__main__":main()