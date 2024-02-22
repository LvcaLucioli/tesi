__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import Dataset
from transformers import (TrainingArguments, 
                          AutoTokenizer, 
                          AutoModelForCausalLM,
                          BitsAndBytesConfig, 
                          HfArgumentParser,
                          AutoModel,
                          pipeline,
                          set_seed)
import torch
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import bitsandbytes as bnb
from datetime import datetime
import json
import evaluate
import chromadb
import numpy as np
import nltk
from sklearn.metrics import accuracy_score
import re
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
            "text" : f'''"Instruct: Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano. ### Question: {qa["question"]}\n### Answer: ''',     
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
    df = df[df['question'].str.len() >= 5]

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : f'''"Instruct: Analizza la domanda e le possibili opzioni di risposta e seleziona l'opzione corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi solo una tra le lettere 'a', 'b', 'c', 'd'. ### Question: {qa["question"]}\n{qa["options"]}\n### Answer: ''',    
            "answer" : qa["correct_option"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]
       
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
            "text" : f'''"Instruct: Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano. ### Question: {qa["question"]}\n### Answer: ''',     
            "answer" : qa["answer"],
            "id" : qa["id"],
            }
        new_dataset.append(inputs)

    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]
    

# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['question'])):
#         text = f'''"'''      
#         output_texts.append(text)
#     return output_texts

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
    match = re.search(r'### answer:[^a-z]*([a-z])', testo)
    if match:
        risposta = match.group()[-1]
        return risposta
    else:
        return ''

def main():
    set_seed(42)

    parser = HfArgumentParser((FinetuneArguments))
    finetuning_arguments  = parser.parse_args_into_dataclasses()[0]
    
    metric_rouge = evaluate.load("rouge")
    
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


    def evaluate_multiple_choice(model, tokenizer, dataset):
        
        predictions = []

        for text in dataset["test"]["text"]:
            tokenized_prompt = tokenizer(text, padding=True, truncation=True, max_length=511, return_tensors='pt').to("cuda:0")
            model_output = model.generate(**tokenized_prompt, max_new_tokens=5)
            result = tokenizer.batch_decode(model_output, skip_special_tokens=True)[0]
            
            prediction = extract_letter(result)

            predictions.append(prediction)
        
        metrics = accuracy_score([b for b in dataset["test"]["answer"]],
                                predictions)
        
        result = {"accuracy" : metrics,
               "model_name" : finetuning_arguments.model_name}
        result["temperature"] = finetuning_arguments.temperature
        result["top_p"] = finetuning_arguments.top_p
        result["top_k"] = finetuning_arguments.top_k
        result["num_beams"] = finetuning_arguments.num_beams
        result["task"] = finetuning_arguments.training_task
        result["datetime"] = datetime.now().isoformat()
         
        with open('result.json', 'a') as file:
            json.dump(result, file, indent=4)
               
        return metrics


    def evaluate_question_answering(dataset, model_name):
                
        # tokenized_prompt = tokenizer(dataset["test"]["text"], padding=True, return_tensors='pt').to('cuda:0')
        
        # model_op = model.generate(**tokenized_prompt,
        #                 #   attention_mask=tokenized_prompt['attention_mask'].to("cuda:0"),
        #                 #   renormalize_logits=False,
        #                   do_sample=True,
        #                 #   num_beams=finetuning_arguments.num_beams,
        #                 #   top_k=finetuning_arguments.top_k,
        #                 #   top_p=finetuning_arguments.top_p,
        #                 #   temperature=finetuning_arguments.temperature,
        #                   max_new_tokens=512)
        
        # preds = tokenizer.batch_decode(model_op, skip_special_tokens=True)
                
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer,
                        max_new_tokens=64,
                        do_sample=True,
                        top_k=finetuning_arguments.top_k,
                        top_p=finetuning_arguments.top_p,
                        temperature=finetuning_arguments.temperature,
                        # num_beams=finetuning_arguments.num_beams,
                        
        )
        
        preds = [pipe(instance) for instance in dataset["test"]["text"]]
        
        
        answers = []
        i = 0
        for pred in preds:
            answers.append({"generated_text" : pred[0]["generated_text"][pred[0]["generated_text"].find("### Answer:") + len("### Answer:"):],
            "id" : dataset["test"]["id"][i]})
            i = i + 1

        processed_preds = []
        for pred in preds:
            processed_preds.append(pred[0]["generated_text"][pred[0]["generated_text"].find("### Answer:") + len("### Answer:"):])

        with open(f'''generated_text_{model_name}.json''', 'a') as file:
            json.dump(answers, file, indent=4)
        
        
        # for text in dataset["test"]["text"]:
        #     tokenized_prompts.append(tokenizer(text, padding=True, return_tensors='pt'))
            
        # model_op = []
        # for tokenized in tokenized_prompts:
        #     model_op.append(model.generate(input_ids=tokenized['input_ids'].to("cuda:0"),
        #                   attention_mask=tokenized['attention_mask'].to("cuda:0"),
        #                 #   renormalize_logits=False,
        #                   do_sample=True,
        #                 #   use_cache=True,
        #                 #   num_beams=finetuning_arguments.num_beams,
        #                 #   top_k=finetuning_arguments.top_k,
        #                 #   top_p=finetuning_arguments.top_p,
        #                 #   temperature=finetuning_arguments.temperature,
        #                   max_new_tokens=32))
        # preds = []
        # for encoded in model_op:
        #     preds.append(tokenizer.batch_decode(encoded, skip_special_tokens=True)[0])
        
        processed_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in processed_preds]
        processed_labels = ["\n".join(nltk.sent_tokenize(label)) for label in dataset["test"]["answer"]]
        
        result = metric_rouge.compute(predictions=processed_preds, references=processed_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result.items()}
        
        result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
                    (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

        processed_preds = [pred.replace("\n", " ") for pred in processed_preds]
        processed_labels = [label.replace("\n", " ") for label in processed_labels]

        result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in processed_preds])

        result["model_name"] = finetuning_arguments.model_name

        result["datetime"] = datetime.now().isoformat()
        
        with open('rouge_result.json', 'a') as file:
            json.dump(result, file, indent=4)
        
        return result 

    if finetuning_arguments.training_task == "syntetic-question-answering":
        dataset = load_syntetic_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "multiple-choice":
        dataset = load_mutliple_choice_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "question-answering":
        dataset = load_qna_dataset(finetuning_arguments.first_n)
        
    
    dataset = Dataset.from_dict(dataset.to_dict(orient='list'))
    dataset = dataset.train_test_split(test_size=0.2)
    
    dataset["train"] = dataset["train"].map(lambda examples: {"text" : [text + answer for text, answer in zip(examples["text"], examples["answer"])]}, batched=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = False,
    )

    tokenizer = AutoTokenizer.from_pretrained(finetuning_arguments.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(finetuning_arguments.model_name,
                                                trust_remote_code=True,
                                                quantization_config = bnb_config,
                                                flash_attn = True,
                                                flash_rotary = True,
                                                fused_dense = True,
                                                low_cpu_mem_usage = True,
                                                device_map={"":0},
                                                revision="refs/pr/23")
    model.config.pad_token_id = model.config.eos_token_id

    
    # if finetuning_arguments.training_task == "multiple-choice":
    #     print(f'''{finetuning_arguments.model_name}_{finetuning_arguments.training_task}: {evaluate_multiple_choice(model, tokenizer, dataset)}''')
    # else:
    #     print(f'''{finetuning_arguments.model_name}_{finetuning_arguments.training_task}: {evaluate_question_answering(dataset, finetuning_arguments.new_model_name + '_BASE_ ' + finetuning_arguments.training_task)}''')
    

    response_template = "\n### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_all_linear_names(model),
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
        fp16=False,
        bf16=False,
        # max_grad_norm=max_grad_norm,
        # max_steps=max_steps,
        # warmup_ratio=warmup_ratio,
        # group_by_length=group_by_length,
        lr_scheduler_type="linear",
        report_to="wandb",
        run_name=f"{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}-{datetime.now().day}-{datetime.now().hour}-{datetime.now().minute}",
    )
    
    
    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["test"],
        dataset_text_field="text",
        # formatting_func=formatting_prompts_func,
        data_collator=collator,
        # peft_config=peft_config,
        args=training_args,
        max_seq_length=1024,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
    )


    trainer.train()
    
        
    trainer.model.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    trainer.tokenizer.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    
    trainer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
       
    # model = AutoModelForCausalLM.from_pretrained(finetuning_arguments.model_name, device_map={'':0})
    
    finetuned_model = PeftModel.from_pretrained(model, f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''')

    tokenizer = AutoTokenizer.from_pretrained(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''')
    tokenizer.pad_token = tokenizer.eos_token

    if finetuning_arguments.training_task == "multiple-choice":
        print(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}: {evaluate_multiple_choice(finetuned_model, tokenizer, dataset)}''')
    else:
        print(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}: {evaluate_question_answering(dataset,
              finetuning_arguments.new_model_name + '_' + finetuning_arguments.training_task)}''')

if __name__ == "__main__":main()