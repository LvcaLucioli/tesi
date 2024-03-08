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
from peft import LoraConfig, PeftModel, PeftConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from dataclasses import dataclass, field
from typing import Optional
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from datetime import datetime
import numpy as np
import bitsandbytes as bnb
import json



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
    layers: Optional[json.loads] = field(
        default=None, metadata={'help' : 'layers'}
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
            "text" : f'''"<s>[INST] <<SYS>>\nAnalizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano.\n<</SYS>>\n\n{qa["question"]} [/INST] ''',      
            # "text" : f'''"### Instruction:\n{qa["question"]}\n\n ### Response:\n''',      
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
    'Id' : 'id',
    })

    df = df[df['question'].str.len() >= 5]

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : f'''"<s>[INST] <<SYS>>\nAnalizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano.\n<</SYS>>\n\n{qa["question"]} [/INST] ''',      
            "answer" : qa["answer"],
            "id" : qa["id"],
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
    'Opzione corretta' : 'correct_option',
    'Id': 'id',

    })
    
    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : f'''"<s>[INST] <<SYS>>\nAnalizza la domanda e le opzioni di risposta e seleziona l'opzione corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi solo una tra le lettere 'a', 'b', 'c', 'd'.\n<</SYS>>\n\n{qa["question"]}\n{qa["options"]} [/INST] ''',      
            "answer" : qa["correct_option"],
            "id" : qa["id"],
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

def main():
    set_seed(42)
    parser = HfArgumentParser((FinetuneArguments))
    finetuning_arguments  = parser.parse_args_into_dataclasses()[0]   
    print(f'''layers: {finetuning_arguments.layers}''')

    if finetuning_arguments.training_task == "syntetic-question-answering":
        dataset = load_syntetic_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "multiple-choice":
        dataset = load_mutliple_choice_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "question-answering":
        dataset = load_qna_dataset(finetuning_arguments.first_n)
        
    dataset = Dataset.from_dict(dataset.to_dict(orient='list'))
    dataset = dataset.train_test_split(test_size=0.3)

    dataset["train"] = dataset["train"].map(lambda examples: {"text" : [text + answer for text, answer in zip(examples["text"], examples["answer"])]}, batched=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        finetuning_arguments.model_name,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(finetuning_arguments.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if finetuning_arguments.layers is None:
        layers = find_all_linear_names(model)
    else:
        layers = finetuning_arguments.layers

    print(layers)  
      
    response_template = f"[/INST]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=layers,
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
        lr_scheduler_type="cosine",
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
        max_seq_length=700,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    
    trainer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    
    # peft_model = AutoPeftModelForCausalLM.from_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')

    # merged_model = peft_model.merge_and_unload() 
    # tokenizer = AutoTokenizer.from_pretrained(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''')
    
    # merged_model.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}_merged')
    # tokenizer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}_merged')
    
    log = {
        "datetime" : datetime.now().isoformat(),
        "model_name" : finetuning_arguments.new_model_name,
        "layers" : layers
    }
    
    with open('./logs/logs.json', 'a') as file:
        json.dump(log, file, indent=4)
        
if __name__ == "__main__":main()