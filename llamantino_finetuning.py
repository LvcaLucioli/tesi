__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    AutoModel,
    pipeline,
    set_seed,
)
import re
from peft import LoraConfig, PeftModel, PeftConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from dataclasses import dataclass, field
from typing import Optional
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from datetime import datetime
import evaluate
import nltk
import numpy as np
import bitsandbytes as bnb
import json
import chromadb
import random
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import accuracy_score
nltk.download("punkt", quiet=True)




@dataclass
class FinetuneArguments:
    model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the model."}
    )
    new_model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the finetuned model."}
    )
    first_n: Optional[int] = field(
        default=0, metadata={"help" : "Number of instance of the dataset"}
    )
    num_train_epochs: Optional[int] = field(
        default=None, metadata={"help" : "Number of training epochs"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=None, metadata={"help" : "Number of evaluation batch size"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=None, metadata={"help" : "Number of train batch size"}
    )
    learning_rate: Optional[float] = field(
        default=None, metadata={"help" : "Learning rate"}
    )
    training_task: Optional[str] = field(
        default=None, metadata={"help" : "'question-answering' or 'multiple-choice'"}
    )
    sample : Optional[int] = field(
        default=None, metadata={"help" : "Number of samples."}
    )
    temperature : Optional[float] = field(
        default=None, metadata={'help' : 'temperature'}
    )
    top_p: Optional[float] = field(
        default=None, metadata={'help' : 'top_p'}
    )
    top_k: Optional[int] = field(
        default=None, metadata={'help' : 'top_k'}
    )
    num_beams: Optional[int] = field(
        default=None, metadata={'help' : 'num_beams'}
    )

    
def load_qna_dataset(n):
    path = "CdA-mininterno-quiz_dataset.csv"
    df = pd.read_csv(path)
    
    df = df.drop(columns=['Tipo-Domanda'])
    
    df = df.rename(columns={
    'Domanda': 'question',
    'Risposta': 'answer',
    'Id' : 'id',
    })

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            # "text" : f'''"<s> [INST] <<SYS>>\nAnalizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano.\n<</SYS>>\n Question: {qa["question"]} [/INST]''',      
            "text" : f'''"### Instruction:\n{qa["question"]}\n\n ### Response:\n''',      
            "answer" : qa["answer"],
            "id" : qa["id"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]
    

def load_syntetic_dataset(n):
    path = "ca_synthetic_qa_dataset.csv"

    df = pd.read_csv(path)

    df = df.drop(columns=['Resource_ID'])

    df = df.rename(columns={
    'Question': 'question',
    'Answer': 'answer',
    })

    df = df[df['question'].str.len() >= 5]

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : f'''"### Instruction:\n{qa["question"]}\n\n ### Response:\n''',      
            "answer" : qa["answer"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]


def load_mutliple_choice_dataset(n):
    path = "CA_dataset_w_options.csv"

    df = pd.read_csv(path)
    df = df.rename(columns={
    'Domanda': 'question',
    'Opzioni' : 'options',
    'Opzione corretta' : 'correct_option'
    })
    
    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : f'''"### Instruction:\n{qa["question"]}{qa["options"]}\n\n ### Response:\n''',    
            "answer" : qa["correct_option"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]
    
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def extract_letter(testo):
    testo = testo.lower()
    match = re.search(r"response:\n\s*\S", testo)
    if match:
        lettera = match.group()[-1]
        return lettera
    else:
        return ''

def evaluate_multiple_choice(model, tokenizer, dataset):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    result = [pipe(instance) for instance in dataset["test"]["text"]]
    metrics = accuracy_score([b for b in dataset["test"]["answer"]],
                             [extract_letter(a[0]["generated_text"]) for a in result])
    return metrics

def main():
    set_seed(42)
    parser = HfArgumentParser((FinetuneArguments))
    finetuning_arguments  = parser.parse_args_into_dataclasses()[0]   
    
    metric_rouge = evaluate.load("rouge")

    def evaluate_question_answering(dataset, model_name):
        
        pipe = pipeline(task="text-generation",
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=122,
                        # do_sample=True,
                        # top_k=finetuning_arguments.top_k,
                        # top_p=finetuning_arguments.top_p,
                        # temperature=finetuning_arguments.temperature,
                        )
        # preds = []
        # for out in pipe(KeyDataset(dataset["test"], "text"), batch_size=8):
        #     print(out[0]["generated_text"])
        #     preds.append(out)

        processed_preds = []
        i = 0
        for instance in dataset["test"]["text"]:
            out = pipe(instance)
            out[0]["generated_text"] = out[0]["generated_text"][out[0]["generated_text"].find("### Response:") + len("### Response:"):]
            end_idx = out[0]["generated_text"].find("### Instruction:")
            if end_idx == -1:
                response = {"generated_text" : out[0]["generated_text"],
            "id" : dataset["test"]["id"][i]}
            else:
                response = {"generated_text" : out[0]["generated_text"][: end_idx],
            "id" : dataset["test"]["id"][i]}
            processed_preds.append(response)
            print(response)
            i = i + 1
        
        # answers = []
        # i = 0
        # for pred in preds:
        #     end_idx = pred[0]["generated_text"].find("### Instruction:")
        #     if end_idx == -1:
        #         answers.append({"generated_text" : pred[0]["generated_text"][pred[0]["generated_text"].find("### Response:") + len("### Response:"):],
        #     "id" : dataset["test"]["id"][i]})
        #     else:
        #         answers.append({"generated_text" : pred[0]["generated_text"][pred[0]["generated_text"].find("### Response:") + len("### Response:"): end_idx],
        #     "id" : dataset["test"]["id"][i]})
        #     i = i + 1

        # processed_preds = []
        # for pred in preds:
        #     processed_preds.append(pred[0]["generated_text"][pred[0]["generated_text"].find("### Response:") + len("### Response"):])

        with open(f'''generated_text_llamantino_finetuned.json''', 'a') as file:
            json.dump(processed_preds, file, indent=4)
            
        processed_preds = [pred["generated_text"].strip() for pred in processed_preds]
        processed_labels = [label.strip() for label in dataset["test"]["answer"]]
         
        processed_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in processed_preds]
        processed_labels = ["\n".join(nltk.sent_tokenize(label)) for label in processed_labels]
        
        result = metric_rouge.compute(predictions=processed_preds, references=processed_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result.items()}
        
        result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
                    (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

        processed_preds = [pred.replace("\n", " ") for pred in processed_preds]
        processed_labels = [label.replace("\n", " ") for label in processed_labels]

        result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in processed_preds])
        
        result["model_name"] = model_name
        result["datetime"] = datetime.now().isoformat()

        with open('rouge_result.json', 'a') as file:
            json.dump(result, file, indent=4)
        
        return result
    if finetuning_arguments.training_task == "syntetic-question-answering":
        train_dataset = Dataset.from_pandas(load_syntetic_dataset(finetuning_arguments.first_n))
        test_dataset = Dataset.from_pandas(load_qna_dataset(finetuning_arguments.first_n/10))

        dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
    elif finetuning_arguments.training_task == "multiple-choice":
        dataset = load_mutliple_choice_dataset(finetuning_arguments.first_n)
        dataset = Dataset.from_dict(dataset.to_dict(orient='list'))
        dataset = dataset.train_test_split(test_size=0.2)
    elif finetuning_arguments.training_task == "question-answering":
        dataset = load_qna_dataset(finetuning_arguments.first_n)
        dataset = Dataset.from_dict(dataset.to_dict(orient='list'))
        dataset = dataset.train_test_split(test_size=0.2)

    dataset["train"] = dataset["train"].map(lambda examples: {"text" : [text + ' ' + answer for text, answer in zip(examples["text"], examples["answer"])]}, batched=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     finetuning_arguments.model_name,
    #     quantization_config=bnb_config,
    #     device_map={"": 0}
    # )
    # model.config.use_cache = False
    # model.config.pretraining_tp = 1
    
    model = AutoModelForCausalLM.from_pretrained(finetuning_arguments.model_name)
    model = PeftModel.from_pretrained(model, finetuning_arguments.new_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(finetuning_arguments.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if finetuning_arguments.training_task == "multiple-choice":
        print(f'''{finetuning_arguments.model_name}_{finetuning_arguments.training_task}: {evaluate_multiple_choice(model, tokenizer, dataset)}''')
    else:
        print(f'''{finetuning_arguments.model_name}_{finetuning_arguments.training_task}: {evaluate_question_answering(dataset,
              finetuning_arguments.new_model_name + '_' + finetuning_arguments.training_task)}''')
    
    
    response_template = f"### Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=find_all_linear_names(model),
    )
    
    training_params = TrainingArguments(
        output_dir=f"./{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}",
        num_train_epochs=finetuning_arguments.num_train_epochs,
        per_device_train_batch_size=finetuning_arguments.per_device_train_batch_size,
        # per_device_eval_batch_size=finetuning_arguments.per_device_eval_batch_size,
        # eval_accumulation_steps=1,
        optim="paged_adamw_32bit",
        logging_strategy="epoch",
        # evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=finetuning_arguments.learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
        gradient_checkpointing=True,
        # load_best_model_at_end=True,
        # save_total_limit=1,
        run_name=f"{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}-{datetime.now().day}-{datetime.now().hour}-{datetime.now().minute}",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["test"],
        peft_config=peft_params,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_params,
        data_collator=collator,
        # compute_metrics=compute_metrics,
    )

    # trainer.train()
        
    # trainer.model.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    # trainer.tokenizer.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    
    # trainer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    
    # model = PeftModel.from_pretrained(model, f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''')

    # tokenizer = AutoTokenizer.from_pretrained(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''')
    # if finetuning_arguments.training_task == "multiple-choice":
    #     print(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}: {evaluate_multiple_choice(model, tokenizer, dataset)}''')
    # else:
    #     print(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}: {evaluate_question_answering(dataset,
    #           finetuning_arguments.new_model_name + '_' + finetuning_arguments.training_task)}''')
    
if __name__ == "__main__":main()