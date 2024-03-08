from dataclasses import dataclass, field
from transformers import (AutoTokenizer,
                            T5ForConditionalGeneration, 
                            AutoModelForSeq2SeqLM,
                            HfArgumentParser,
                            set_seed,
                            T5ForQuestionAnswering,
                            Seq2SeqTrainingArguments,
                            Seq2SeqTrainer,
                            DataCollatorForSeq2Seq,
                            AutoModel,
                            pipeline,
                            AutoModelForCausalLM)
import pandas as pd
from datasets import Dataset
import numpy as np
import torch
import bitsandbytes as bnb
from tqdm import tqdm
from torch.optim import Adam
from typing import Optional
from datetime import datetime
from huggingface_hub import login

import json
from datasets import Dataset
import random



@dataclass
class FinetuneArguments:
    model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the model."}
    )
    new_model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the finetuned model."}
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

def load_dataset():
    multiple_choice_path = "CA_dataset_w_options.csv"
    generative_path = "CdA-mininterno-quiz_dataset.csv"

    df1 = pd.read_csv(multiple_choice_path)
    df2 = pd.read_csv(generative_path)

    qna_dataset = pd.merge(df1, df2, left_on=['Id', 'Domanda'], right_on=['Id', 'Domanda'])
    qna_dataset = qna_dataset.drop(columns=['Tipo-Domanda'])

    qna_dataset = qna_dataset.rename(columns={
    'Domanda': 'question',
    'Risposta': 'answer',
    'Opzioni' : 'options',
    'Opzione corretta' : 'correct_option'
    })

    return qna_dataset

def prepare_data(data):
    qas = []
    for _, qa in data.iterrows():
        inputs = {"options" : qa["options"], "question" : "data la seguente domanda ```" + qa["question"] + " ``` scegli l'opzione opzione più appropriata : " + qa["options"], "answer" : qa["correct_option"], "context" : ""}
        qas.append(inputs)
    return qas

class QA_Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, q_len, t_len):
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = dataframe
        self.questions = self.data["question"]
        self.context = self.data["context"]
        self.answer = self.data['answer']

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx]

        question_tokenized = self.tokenizer(question,
                                                context, 
                                                max_length=self.q_len, 
                                                padding="max_length",
                                                truncation=True, 
                                                pad_to_max_length=True,
                                                add_special_tokens=True)
        answer_tokenized = self.tokenizer(answer, 
                                            max_length=self.t_len,
                                            padding="max_length",
                                            truncation=True, 
                                            pad_to_max_length=True,
                                            add_special_tokens=True)

        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100

        return {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)
        }
    
# def predict_answer(question, context, model, tokenizer, ref_answer=None):
#     inputs = tokenizer(question,
#                         context, 
#                         max_length=256, 
#                         padding="max_length", 
#                         truncation=True, 
#                         add_special_tokens=True)

#     input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to("cuda:0").unsqueeze(0)
#     attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to("cuda:0").unsqueeze(0)

#     outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=3)

#     predicted_answer = tokenizer.decode(outputs.flatten(), skip_special_tokens=True)

#     return predicted_answer

def load_qna_dataset():
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
            # "text" : f'''"<s> [INST] Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. {qa["question"]} [/INST] ''',      
            "text" : f'''\n\nDomanda: {qa["question"]}\n\nRisposta: ''',
            "answer" : qa["answer"],
            "id" : qa["id"],
            }
        new_dataset.append(inputs)

    return pd.DataFrame(new_dataset)

    

def load_syntetic_dataset():
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
            "text" : f'''\n\nDomanda: {qa["question"]}\n\nRisposta: ''',
            "answer" : qa["answer"],
            }
        new_dataset.append(inputs)
    return pd.DataFrame(new_dataset)


def load_mutliple_choice_dataset():
    path = "../datasets/CA_dataset_w_options.csv"

    df = pd.read_csv(path)
    df = df.rename(columns={
    'Domanda': 'question',
    'Opzioni' : 'options',
    'Opzione corretta' : 'correct_option'
    })
    
    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "text" : f''''\nDomanda: {qa["question"]}\n\nOPZIONI:{qa["options"]}\n\nRisposta: ''',  
            "answer" : qa["correct_option"],
            }
        new_dataset.append(inputs)

    return pd.DataFrame(new_dataset)
    

# def estrai_lettera(testo):
#     testo = testo.lower()
#     # match = re.search(r"[^a-z]*([a-z])", testo)
#     lettera = testo[0]
#     # print(lettera)
#     return lettera
# def evaluate_multiple_choice(model, tokenizer, dataset):
#     pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
#     result = [pipe(instance) for instance in dataset["test"]["text"]]
#     # print([b for b in dataset["test"]["answer"]])
#     # print(result[0:2])
#     metrics = accuracy_score([b for b in dataset["test"]["answer"]],
#                              [estrai_lettera(a[0]["generated_text"]) for a in result])
#     return metrics
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
    parser = HfArgumentParser((FinetuneArguments))
    finetuning_arguments  = parser.parse_args_into_dataclasses()[0]    
    
    if finetuning_arguments.training_task == "syntetic-question-answering":
        dataset = load_syntetic_dataset()
    elif finetuning_arguments.training_task == "multiple-choice":
        dataset = load_mutliple_choice_dataset()
    elif finetuning_arguments.training_task == "question-answering":
        dataset = load_qna_dataset()
        
        
    dataset = Dataset.from_dict(dataset.to_dict(orient='list'))
    dataset = dataset.train_test_split(test_size=0.3)
    
    tokenizer = AutoTokenizer.from_pretrained(finetuning_arguments.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(finetuning_arguments.model_name, 
                                                  device_map={"":0},)
    
    def preprocess_function(examples):
        """Add prefix to the sentences, tokenize the text, and set the labels"""
        inputs = [ex + " </s>" for ex in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=276, truncation=True, padding=True)
        
        labels = tokenizer(text_target=examples["answer"], max_length=175, truncation=True, padding=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text', 'answer'])

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}",
        # evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=finetuning_arguments.learning_rate,
        # eval_accumulation_steps=8,
        per_device_train_batch_size=finetuning_arguments.per_device_train_batch_size,
        # per_device_eval_batch_size=finetuning_arguments.per_device_eval_batch_size,
        weight_decay=0.01,
        # save_total_limit=3,
        num_train_epochs=finetuning_arguments.num_train_epochs,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        run_name=f"{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}-{datetime.now().day}-{datetime.now().hour}-{datetime.now().minute}",
        report_to="wandb",
        # load_best_model_at_end=True,
        # save_total_limit=1,     
        generation_max_length=10,
        lr_scheduler_type="cosine", # ?
    )

    # Set up trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        # eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    
    # trainer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    
    trainer.model.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    trainer.tokenizer.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    log = {
        "datetime" : datetime.now().isoformat(),
        "model_name" : finetuning_arguments.new_model_name,
    }
    
    with open('../logs/logs.json', 'a') as file:
        json.dump(log, file, indent=4)
        
if __name__ == "__main__":main()