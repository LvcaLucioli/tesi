__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    AutoModel,
    pipeline,
    set_seed,
    
)
from sklearn.metrics import accuracy_score
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import torch

from functools import partial
import pandas as pd
import json
import accelerate
import bitsandbytes as bnb
from dataclasses import dataclass, field
from typing import Optional
import chromadb
import random
import numpy as np
import nltk
import evaluate
import re
from datetime import datetime

nltk.download("punkt", quiet=True)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
    AutoPeftModelForCausalLM
)

@dataclass
class PredictionArguments:
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
    path = "../datasets/CdA-mininterno-quiz_dataset.csv"
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
            "text" : f'''<|system|>\nAnalizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano.</s>\n<|user|>\n{qa["question"]}</s>\n<|assistant|>\n{qa["answer"]}''',
            "answer" : qa["answer"],
            "id" : qa["id"],
            }
        new_dataset.append(inputs)

    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]

def load_syntetic_dataset(n):
    path = "../datasets/ca_synthetic_qa_dataset.csv"

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
            "text" : f'''"<|system|>\nAnalizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano.</s>\n<|user|>\n{qa["question"]}</s>\n<|assistant|>\n{qa["answer"]}''',     
            "answer" : qa["answer"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]

def load_mutliple_choice_dataset(n):
    path = "../datasets/CA_dataset_w_options.csv"

    df = pd.read_csv(path)
    df = df.rename(columns={
    'Domanda': 'question',
    'Opzioni' : 'options',
    'Opzione corretta' : 'correct_option'
    })
    df = df[df['question'].str.len() >= 5]

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : f'''"<|system|>\nAnalizza la domanda e le possibili opzioni di risposta e seleziona l'opzione corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi solo una tra le lettere 'a', 'b', 'c', 'd'.</s>\n<|user|>\n{qa["question"]}\n{qa["options"]}</s>\n<|assistant|>\n{qa["correct_option"]}''',    
            "answer" : f'''{qa["correct_option"]}''',
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]

def load_ca_dataset():
    path = "../datasets/ca_chunk_articles.csv"

    df = pd.read_csv(path)
    df = df.rename(columns={
    'data': 'text',
    })

    # df = df[df['question'].str.len() >= 5]

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : qa["text"],
            }
        new_dataset.append(inputs)
    return pd.DataFrame(new_dataset)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)

def main():
    set_seed(42)
    parser = HfArgumentParser((PredictionArguments))
    finetuning_arguments  = parser.parse_args_into_dataclasses()[0]

    if finetuning_arguments.training_task == "syntetic-question-answering":
        dataset = load_syntetic_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "multiple-choice":
        dataset = load_mutliple_choice_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "question-answering":
        dataset = load_qna_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "self-supervised":
        dataset = load_ca_dataset()
    
    dataset = Dataset.from_dict(dataset.to_dict(orient='list'))
    dataset = dataset.train_test_split(test_size=0.2)

    # dataset["train"] = dataset["train"].map(lambda examples: {"text" : [text + ' ' + answer + '</s>' for text, answer in zip(examples["text"], examples["answer"])]}, batched=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        finetuning_arguments.model_name,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )

    tokenizer = AutoTokenizer.from_pretrained(finetuning_arguments.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side="left"
    model.config.pad_token_id = model.config.eos_token_id
                
    # if finetuning_arguments.training_task == "multiple-choice":
    #     print(f'''{finetuning_arguments.model_name}_{finetuning_arguments.training_task}: {evaluate_multiple_choice(model, tokenizer, dataset)}''')
    # else:
    #     print(dataset["test"])
    #     print(f'''{finetuning_arguments.model_name}_{finetuning_arguments.training_task}: {evaluate_question_answering(dataset,
    #           finetuning_arguments.new_model_name + '_' + finetuning_arguments.training_task)}''')
    

    model.config.use_cache = False
    model.config.pretraining_tp=1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    print(f'''linear layers: {find_all_linear_names(model)}''')

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        # target_modules=['v_proj', 'o_proj', 'k_proj', 'q_proj'],
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
          
    training_args = TrainingArguments(
        num_train_epochs=finetuning_arguments.num_train_epochs,
        output_dir=f"./{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}",
        per_device_train_batch_size=finetuning_arguments.per_device_train_batch_size,
        # per_device_eval_batch_size=finetuning_arguments.per_device_eval_batch_size,
        # gradient_accumulation_steps=10,
        learning_rate=finetuning_arguments.learning_rate, #0.0005
        # max_grad_norm=1.0,
        # max_steps=40,
        # lr_scheduler_type="linear",
        # warmup_steps=5,
        fp16=True,
        logging_strategy="epoch",
        lr_scheduler_type="cosine",
        # evaluation_strategy="epoch",
        # logging_steps=1,
        # save_strategy="steps",
        # save_steps=10,
        optim="paged_adamw_32bit",
        report_to="all",
        run_name=f"{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}-{datetime.now().day}-{datetime.now().hour}-{datetime.now().minute}",
        push_to_hub=True,
        # save_total_limit=1,
        save_strategy="epoch",
        # load_best_model_at_end=True
    )
    if finetuning_arguments.training_task == "self-supervised":
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            # data_collator=DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer),
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
            # formatting_func=formatting_prompts_func,
            # packing=True
            tokenizer=tokenizer,
            # compute_metrics=compute_metrics,
            # max_seq_length=15,#15
            # num_of_sequences=2
        )
    else:
        response_template = '\n<|assistant|>\n'
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer),
            train_dataset=dataset["train"],
            # eval_dataset=dataset["test"],
            dataset_text_field="text",
            # formatting_func=formatting_prompts_func,
            # packing=True
            tokenizer=tokenizer,
            # compute_metrics=compute_metrics,
            # max_seq_length=15,#15
            # num_of_sequences=2
        )
    
    trainer.train()
    
    
    trainer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    
    peft_model = AutoPeftModelForCausalLM.from_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')

    merged_model = peft_model.merge_and_unload() 

    tokenizer = AutoTokenizer.from_pretrained(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''')
    
    merged_model.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}_merged', use_temp_dir=False, safe_serialization=True)
    tokenizer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}_merged')
    
    # merged_model.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}_merged')
    # tokenizer.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}_merged')
    
    # model = AutoModelForCausalLM.from_pretrained(finetuning_arguments.model_name)
    # finetuned_model = PeftModel.from_pretrained(model, f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''').to("cuda:0")
    
    # if finetuning_arguments.training_task == "multiple-choice":
    #     print(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}: {evaluate_multiple_choice(finetuned_model, tokenizer, dataset)}''')
    # else:
    #     print(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments
    
if __name__ == "__main__":main()