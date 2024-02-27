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
    layers: Optional[json.loads] = field(
        default=None, metadata={'help' : 'layers'}
    )
    
   
def load_qna_dataset(n):
    path = "./datasets/CdA-mininterno-quiz_dataset.csv"
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
            "text" : f'''"<|system|> Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano.</s> <|user|> {qa["question"]}\n</s> <|assistant|> ''',     
            "answer" : qa["answer"],
            "id" : qa["id"],
            }
        new_dataset.append(inputs)

    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]

def load_syntetic_dataset(n):
    path = "./datasets/ca_synthetic_qa_dataset.csv"

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
            "text" : f'''"<|system|> Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano.</s> <|user|> {qa["question"]}\n</s> <|assistant|> ''',     
            "answer" : f'''{qa["answer"]}''',
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
    df = df[df['question'].str.len() >= 5]

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : f'''"<|system|> Analizza la domanda e le possibili opzioni di risposta e seleziona l'opzione corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi solo una tra le lettere 'a', 'b', 'c', 'd'.</s> <|user|> {qa["question"]}\n{qa["options"]}</s><|assistant|>''',    
            "answer" : qa["correct_option"],
            "id" : qa["Id"],
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

# def extract_letter(testo):
#     testo = testo.lower()
#     risposta = testo.split("<|assistant|>", 1)[1]
#     risposta = risposta.strip()
#     return risposta[0]

def main():
    set_seed(42)
    parser = HfArgumentParser((PredictionArguments))
    finetuning_arguments  = parser.parse_args_into_dataclasses()[0]
    
    # metric_rouge = evaluate.load("rouge")
     
    # def r_at_k(collection, embeddings, ids, k):
    #     score = 0
    #     step = 0

    #     for pred, id in zip(embeddings, ids):
    #         results = collection.query(
    #                 query_embeddings=pred,
    #                 n_results=k,
    #                 include=["documents"]
    #         )
    #         if str(id) in results["ids"][0]:
    #             score += 1
    #         step += 1
    #     return (score / len(ids))

    # def evaluate_multiple_choice(model, tokenizer, dataset):
    #     set_seed(42)
    #     predictions = []

    #     for text in dataset["test"]["text"]:
    #         tokenized_prompt = tokenizer(text, padding=True, truncation=True, max_length=511, return_tensors='pt').to("cuda:0")
    #         model_output = model.generate(**tokenized_prompt, max_new_tokens=5)
    #         result = tokenizer.batch_decode(model_output, skip_special_tokens=True)[0]
            
    #         prediction = extract_letter(result)

    #         predictions.append(prediction)

    #     metrics = accuracy_score(dataset["test"]["answer"], 
    #                              predictions)
        
        
    #     result = {"accuracy" : metrics,
    #            "model_name" : finetuning_arguments.model_name}
    #     result["temperature"] = finetuning_arguments.temperature
    #     result["top_p"] = finetuning_arguments.top_p
    #     result["top_k"] = finetuning_arguments.top_k
    #     result["num_beams"] = finetuning_arguments.num_beams
    #     result["task"] = finetuning_arguments.training_task
    #     result["datetime"] = datetime.now().isoformat()
        
    #     with open('result.json', 'a') as file:
    #         json.dump(result, file, indent=4)
        
    #     return metrics
    
    # def evaluate_question_answering(dataset):
    #     set_seed(42)
  
    #     tokenized_prompts = []
        
    #     for text in dataset["test"]["text"]:
    #         tokenized_prompts.append(tokenizer(text, padding=True, return_tensors='pt'))
            
    #     model_op = []
    #     for tokenized in tokenized_prompts:
    #         model_op.append(model.generate(input_ids=tokenized['input_ids'].to("cuda:0"),
    #                       attention_mask=tokenized['attention_mask'].to("cuda:0"),
    #                       renormalize_logits=False,
    #                       do_sample=True,
    #                       use_cache=True,
    #                       num_beams=finetuning_arguments.num_beams,
    #                       top_k=finetuning_arguments.top_k,
    #                       top_p=finetuning_arguments.top_p,
    #                       temperature=finetuning_arguments.temperature,
    #                       max_new_tokens=200))
    #     preds = []
    #     for encoded in model_op:
    #         preds.append(tokenizer.batch_decode(encoded, skip_special_tokens=True)[0])
        
    #     # tokenized_prompts = tokenizer(dataset["test"]["text"], padding=True, return_tensors='pt')
    
    #     # model_op = model.generate(input_ids=tokenized_prompts['input_ids'].to("cuda:0"),
    #     #                   attention_mask=tokenized_prompts['attention_mask'].to("cuda:0"),
    #     #                   renormalize_logits=False, do_sample=True,
    #     #                   use_cache=True, max_new_tokens=150)
    #     # preds = tokenizer.batch_decode(model_op, skip_special_tokens=True)
    #     # print(f'''pred[0]: {preds[0]}''')
    #     # print(f'''answer[0]: {dataset["test"]["answer"][0]}''')
    #     # print(f'''id[0]: {dataset["test"]["id"][0]}''')
        
        
    #     processed_preds = [pred[pred.find("<|assistant|> ") + len("<|assistant|> "):] for pred in preds]

    #     processed_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in processed_preds]
    #     processed_labels = ["\n".join(nltk.sent_tokenize(label)) for label in dataset["test"]["answer"]]
                
    #     result = metric_rouge.compute(predictions=processed_preds, references=processed_labels, use_stemmer=True)
        
    #     result = {k: round(v * 100, 2) for k, v in result.items()}
        
    #     result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
    #                 (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

    #     processed_preds = [pred.replace("\n", " ") for pred in processed_preds]
    #     processed_labels = [label.replace("\n", " ") for label in processed_labels]

    #     result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in processed_preds])
                
    #     tokenized_preds = rk_tokenizer(processed_preds,
    #                                     add_special_tokens=False,
    #                                     padding=True,
    #                                     truncation=True,
    #                                     return_tensors="pt",
    #                                     return_attention_mask=True).to("cuda:0")

    #     for input_ids in tokenized_preds["input_ids"]:
    #         input_ids = torch.cat((input_ids, torch.tensor([102]).to("cuda:0")), dim=0).to("cuda:0")
            
    #     torch.cuda.empty_cache()
        
    #     output = rk_model(tokenized_preds["input_ids"], attention_mask=tokenized_preds["attention_mask"])

    #     embeddings = []
    #     with torch.no_grad():
    #         for last_hidden_state in output.last_hidden_state:
    #             cls_embedding = last_hidden_state[0, :].cpu().numpy().tolist()
    #             embeddings.append(cls_embedding)

    #     result["r@1"] = r_at_k(collection, embeddings, dataset["test"]["id"], 1)
    #     result["r@3"] = r_at_k(collection, embeddings, dataset["test"]["id"], 3)
    #     result["r@5"] = r_at_k(collection, embeddings, dataset["test"]["id"], 5)
    #     result["r@10"] = r_at_k(collection, embeddings, dataset["test"]["id"], 10)
    #     result["r@20"] = r_at_k(collection, embeddings, dataset["test"]["id"], 20)
    #     result["r@50"] = r_at_k(collection, embeddings, dataset["test"]["id"], 50)
        
    #     result["model_name"] = finetuning_arguments.model_name
    #     result["temperature"] = finetuning_arguments.temperature
    #     result["top_p"] = finetuning_arguments.top_p
    #     result["top_k"] = finetuning_arguments.top_k
    #     result["num_beams"] = finetuning_arguments.num_beams
    #     result["task"] = finetuning_arguments.training_task
    #     result["datetime"] = datetime.now().isoformat()
        
    #     with open('result.json', 'a') as file:
    #         json.dump(result, file, indent=4)
        
    #     return result 
  
    # if finetuning_arguments.training_task == "question-answering":
    #     client = chromadb.PersistentClient(path="chroma_data/")
    #     rk_model = AutoModel.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True).to("cuda:0")
    #     rk_tokenizer = AutoTokenizer.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True)
    #     rk_tokenizer.pad_token_id = rk_tokenizer.eos_token_id = 2
    #     rk_model.config.pad_token_id = rk_model.config.eos_token_id
    #     collection = client.get_collection(
    #             name="answer_embeddings_definitivo",
    #         )

    if finetuning_arguments.training_task == "syntetic-question-answering":
        dataset = load_syntetic_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "multiple-choice":
        dataset = load_mutliple_choice_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "question-answering":
        dataset = load_qna_dataset(finetuning_arguments.first_n)
        
    dataset = Dataset.from_dict(dataset.to_dict(orient='list'))
    dataset = dataset.train_test_split(test_size=0.2)

    dataset["train"] = dataset["train"].map(lambda examples: {"text" : [text + ' ' + answer + '</s>' for text, answer in zip(examples["text"], examples["answer"])]}, batched=True)
    
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
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(finetuning_arguments.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side="left"
    model.config.pad_token_id = model.config.eos_token_id
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # modules = find_all_linear_names(model)

    if finetuning_arguments.layers is None:
        layers = find_all_linear_names(model)
    else:
        layers = finetuning_arguments.layers

    print(layers)
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=layers,
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
        lr_scheduler_type="constant",
        # evaluation_strategy="epoch",
        # logging_steps=1,
        # save_strategy="steps",
        # save_steps=10,
        optim="paged_adamw_32bit",
        report_to="wandb",
        run_name=f"{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}-{datetime.now().day}-{datetime.now().hour}-{datetime.now().minute}",
        push_to_hub=True,
        # save_total_limit=1,
        save_strategy="epoch",
        # load_best_model_at_end=True
    )

    response_template = '<|assistant|>'
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
    

    trainer.train(resume_from_checkpoint=True)
    
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



