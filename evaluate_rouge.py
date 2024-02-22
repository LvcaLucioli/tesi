from transformers import (pipeline,
                          BitsAndBytesConfig,
                          AutoModelForCausalLM,
                          AutoTokenizer)
import numpy as np
import json
from datetime import datetime
import torch
import evaluate
import nltk

nltk.download("punkt", quiet=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "lvcalucioli/llamantino7b_2_question-answering_merged",
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# model = AutoModelForCausalLM.from_pretrained(finetuning_arguments.model_name)
# model = PeftModel.from_pretrained(model, finetuning_arguments.new_model_name)

tokenizer = AutoTokenizer.from_pretrained("llamantino7b_2_question-answering_merged", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

with open('generated_text_llamantino_finetuned.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

labels_preds = [(item['answer'], item['generated_text']) for item in data]

def evaluate_question_answering():
        metric_rouge = evaluate.load("rouge")

        processed_preds = [item[0].strip() for item in labels_preds]
        processed_labels = [item[1].strip() for item in labels_preds]
         
        processed_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in processed_preds]
        processed_labels = ["\n".join(nltk.sent_tokenize(label)) for label in processed_labels]
        
        result = metric_rouge.compute(predictions=processed_preds, references=processed_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result.items()}
        
        result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
                    (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

        processed_preds = [pred.replace("\n", " ") for pred in processed_preds]
        processed_labels = [label.replace("\n", " ") for label in processed_labels]

        result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in processed_preds])
        
        result["model_name"] = "llamantino_base"
        result["datetime"] = datetime.now().isoformat()

        with open('rouge_result.json', 'a') as file:
            json.dump(result, file, indent=4)
        
        return result
    
def main():
    print(evaluate_question_answering())
if __name__ == "__main__":main()