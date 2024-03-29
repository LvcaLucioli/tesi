import torch
import pandas as pd
from transformers import (
    HfArgumentParser,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForCompletionOnlyLM,
    TrainingArguments,
    SFTTrainer,
    set_seed,    
)
from datasets import Dataset
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from typing import Optional
import bitsandbytes as bnb
from dataclasses import dataclass, field
from datetime import datetime
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
    layers: Optional[json.loads] = field(
        default=None, metadata={'help' : 'layers'}
    )


def load_syntetic_dataset(n):
    path = "./datasets/ca_synthetic_qa_dataset.csv"

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
            # "text" : f'''"Instruct: Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano.\nQuestion: {qa["question"]}\nOutput: ''',     
            "text" : f'''"[|Umano|] {qa["question"]}\n[|Assistente|]''',
            "answer" : qa["answer"],
            "id" : qa["id"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]

def load_mutliple_choice_dataset(n):
    path = "./datasets/CA_dataset_w_options.csv"

    df = pd.read_csv(path)
    df = df.rename(columns={
    'Domanda': 'question',
    'Opzioni' : 'options',
    'Opzione corretta' : 'correct_option',
    'Id': 'id',
    })
    # df = df[df['question'].str.len() >= 5]

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : f'''"Instruct: Analizza la domanda e le possibili opzioni di risposta e seleziona l'opzione corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi solo una tra le lettere 'a', 'b', 'c', 'd'.\nQuestion: {qa["question"]}\n{qa["options"]}\nOutput: ''',    
            "answer" : qa["correct_option"],
            "id" : qa["id"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]
       
def load_qna_dataset(n):
    path = "./datasets/CdA-mininterno-quiz_dataset.csv"
    df = pd.read_csv(path)
    
    df = df.drop(columns=['Tipo-Domanda'])
    
    df = df.rename(columns={
    'Domanda': 'question',
    'Risposta': 'answer',
    'Id': 'id',
    })
          

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            # "text" : f'''"Instruct: Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano.\nQuestion: {qa["question"]}\nOutput: ''',     
            "text" : f'''"[|Umano|] {qa["question"]}\n[|Assistente|]''',
            "answer" : qa["answer"],
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

    # if 'lm_head' in lora_module_names:
    #     lora_module_names.remove('lm_head')
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
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(finetuning_arguments.model_name,
                                                quantization_config = bnb_config,
                                                device_map={"":0},
                                                )
    tokenizer = AutoTokenizer.from_pretrained(finetuning_arguments.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    model.config.pretraining_tp=1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    if finetuning_arguments.layers is None:
        layers = find_all_linear_names(model)
    else:
        layers = finetuning_arguments.layers

    print(layers)

    response_template = "[|Assistente|]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=layers,
    )
    
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=f"./{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}",
        num_train_epochs=finetuning_arguments.num_train_epochs,
        per_device_train_batch_size=finetuning_arguments.per_device_train_batch_size,
        # per_device_eval_batch_size=finetuning_arguments.per_device_eval_batch_size,
        # eval_accumulation_steps=1,
        optim="paged_adamw_32bit",
        # gradient_accumulation_steps=10,
        # optim=optim,
        # save_steps=save_steps,
        # logging_steps=25,
        logging_strategy="epoch",
        # evaluation_strategy="epoch",
        learning_rate=finetuning_arguments.learning_rate,
        # weight_decay=weight_decay,
        # fp16=False,
        # bf16=False,
        # max_grad_norm=max_grad_norm,
        # max_steps=max_steps,
        # warmup_ratio=warmup_ratio,
        # group_by_length=group_by_length,
        lr_scheduler_type="cosine",
        report_to="wandb",
        run_name=f"{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}-{datetime.now().day}-{datetime.now().hour}-{datetime.now().minute}",
    )
    
    
    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        data_collator=collator,
        args=training_args,
        max_seq_length=512,
        tokenizer=tokenizer,
    )


    trainer.train()
    
    trainer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    
    peft_model = AutoPeftModelForCausalLM.from_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')

    merged_model = peft_model.merge_and_unload() 
    tokenizer = AutoTokenizer.from_pretrained(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''')
    
    merged_model.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}_merged')
    tokenizer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}_merged')
    
    log = {
        "datetime" : datetime.now().isoformat(),
        "model_name" : finetuning_arguments.new_model_name,
        "layers" : layers
    }
    
    with open('./logs/logs.json', 'a') as file:
        json.dump(log, file, indent=4)

if __name__ == "__main__":main()