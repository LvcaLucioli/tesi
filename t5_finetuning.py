__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dataclasses import dataclass, field
from transformers import (AutoTokenizer,
                            T5ForConditionalGeneration, 
                            AutoModelForSeq2SeqLM,
                            HfArgumentParser,
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
# from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from typing import Optional
from datetime import datetime
from sklearn.metrics import accuracy_score
from huggingface_hub import login
import nltk
import json
from datasets import Dataset
import evaluate
import chromadb
import random
import re
nltk.download('punkt')



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

def predict(model, batch):
    # batch_size = 8
    preds = []

        # inputs = tokenizer(batch_questions,
        #             return_tensors="pt",
        #             return_attention_mask=True,
        #             padding=True,
        #             truncation=True).to('cuda:0')
        
        # inputs = batch.tokenizer(batch_questions,
        #                         "", 
        #                         max_length=256, 
        #                         padding="max_length",
        #                         truncation=True, 
        #                         pad_to_max_length=True,
        #                         add_special_tokens=True).to('cuda:0')

        # outputs = model.generate(**inputs, 
        #                          max_new_tokens=max_new_tokens, 
        #                          eos_token_id=tokenizer.eos_token_id, 
        #                          pad_token_id=tokenizer.pad_token_id)
    input_ids = batch["input_ids"].to("cuda:0")
    attention_mask = batch["attention_mask"].to("cuda:0")
    labels = batch["labels"].to("cuda:0")
    decoder_attention_mask = batch["decoder_attention_mask"].to("cuda:0")

    preds = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
            )

        # index = tokenizer.batch_decode(outputs)[0].find("The answer is:")
        # model_answer = tokenizer.batch_decode(outputs)[0][index + len("The answer is:"):]

        # answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # for answer in answers:
        #     generated_text = answer
        #     if max_new_tokens > 1:
        #         split_text = generated_text.split("Output:", 1)
        #     else:
        #         split_text = generated_text.split("The correct answer is letter", 1)
        #     assistant_response = split_text[1].strip() if len(split_text) > 1 else ""
        #     assistant_response = assistant_response.replace("", "").strip()
        #     preds.append(assistant_response)
    return preds

def get_preds(model, batch):
    print("start predictions")
    predictions = predict(model, batch)
    print("end predictions")
    return predictions

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
    
def predict_answer(question, context, model, tokenizer, ref_answer=None):
    inputs = tokenizer(question,
                        context, 
                        max_length=256, 
                        padding="max_length", 
                        truncation=True, 
                        add_special_tokens=True)

    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to("cuda:0").unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to("cuda:0").unsqueeze(0)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=3)

    predicted_answer = tokenizer.decode(outputs.flatten(), skip_special_tokens=True)

    return predicted_answer

def load_qna_dataset(n):
    path = "CdA-mininterno-quiz_dataset.csv"

    df = pd.read_csv(path)

    # df = df.drop(columns=['Tipo-Domanda'])
    df = df.drop(columns=['Id', 'Tipo-Domanda'])
    
    # df = df.rename(columns={
    # 'Domanda': 'question',
    # 'Risposta': 'answer',
    # })
    
    df = df.rename(columns={
    'Domanda': 'question',
    'Risposta': 'answer',
    })
    
    # print(f"Numero di righe con domande più corte di 10 caratteri: {len(df[df['question'].str.len() < 10])}")
    # print(f"Numero di righe con risposte più corte di 5 caratteri: {len(df[df['answer'].str.len() < 5])}")
    # df = df[df['question'].str.len() >= 5]

    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            # "text" : f'''"<s> [INST] Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. {qa["question"]} [/INST] ''',      
            "text" : f'''\n\nDomanda: {qa["question"]}\n\nRisposta: </s>''',
            "answer" : f'''{qa["answer"]}''',
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
            "text" : f'''\n\nDomanda: {qa["question"]}\n\nRisposta: </s>''',
            "answer" : f'''{qa["answer"]}''',
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
            # <<SYS>>Analizza la seguente domanda e le opzioni di risposta 'a', 'b', 'c', 'd'. Scegli l'opzione corretta.<</SYS>>
            "text" : f''''\n\nDomanda: '{qa["question"]}{qa["options"]}\n\nRisposta: </s>''',  
            # "text" : f'''[INST] 
            # Sotto sono presenti una domanda e le possibili opzioni di risposta. Scegli la lettera corrispondente all'opzione che risponde correttamente alla domanda.
            # ### Domanda: 
            # {qa["question"]}
            # ### Opzioni: 
            # {qa["options"]}
            # [/INST]
            # ### Risposta:\n''',
            "answer" : f'''{qa["correct_option"]}''',
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]
    

def estrai_lettera(testo):
    testo = testo.lower()
    # match = re.search(r"[^a-z]*([a-z])", testo)
    lettera = testo[0]
    # print(lettera)
    return lettera
def evaluate_multiple_choice(model, tokenizer, dataset):
    pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    result = [pipe(instance) for instance in dataset["test"]["text"]]
    # print([b for b in dataset["test"]["answer"]])
    # print(result[0:2])
    metrics = accuracy_score([b for b in dataset["test"]["answer"]],
                             [estrai_lettera(a[0]["generated_text"]) for a in result])
    return metrics


def main():
    client = chromadb.PersistentClient(path="chroma_data/")
    parser = HfArgumentParser((FinetuneArguments))
    finetuning_arguments  = parser.parse_args_into_dataclasses()[0]

    metric_rouge = evaluate.load("rouge")
    
    
    def r_at_k(collection, embeddings, ids, k):
        score = 0
        step = 0

        for pred, id in zip(embeddings, ids):
            results = collection.query(
                    query_embeddings=pred,
                    n_results=k,
                    include=["documents"]
            )
            if str(id) in results["ids"][0]:
                score += 1
            step += 1
        return (score / len(ids))

    def evaluate_question_answering(dataset):
        
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        preds = [pipe(instance) for instance in dataset["test"]["text"]]
        
        processed_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        processed_labels = ["\n".join(nltk.sent_tokenize(label)) for label in dataset["test"]["answer"]]
        
        result = metric_rouge.compute(predictions=processed_preds, references=processed_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result.items()}
        
        result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
                    (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

        processed_preds = [pred.replace("\n", " ") for pred in processed_preds]
        processed_labels = [label.replace("\n", " ") for label in processed_labels]

        result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in processed_preds])
        
        tokenized_preds = rk_tokenizer(processed_preds,
                                        add_special_tokens=False,
                                        padding=True,
                                        truncation=True,
                                        return_tensors="pt",
                                        return_attention_mask=True).to("cuda:0")

        for input_ids in tokenized_preds["input_ids"]:
            input_ids = torch.cat((input_ids, torch.tensor([102]).to("cuda:0")), dim=0).to("cuda:0")
            
        
        output = rk_model(tokenized_preds["input_ids"], attention_mask=tokenized_preds["attention_mask"])

        embeddings = []
        with torch.no_grad():
            for last_hidden_state in output.last_hidden_state:
                cls_embedding = last_hidden_state[0, :].cpu().numpy().tolist()
                embeddings.append(cls_embedding)
        
        result["r@1"] = r_at_k(collection, embeddings, [str(i) for i in range(len(processed_preds))], 1)
        result["r@3"] = r_at_k(collection, embeddings, [str(i) for i in range(len(processed_preds))], 3)
        result["r@5"] = r_at_k(collection, embeddings, [str(i) for i in range(len(processed_preds))], 5)
        result["r@10"] = r_at_k(collection, embeddings, [str(i) for i in range(len(processed_preds))], 10)
        result["r@20"] = r_at_k(collection, embeddings, [str(i) for i in range(len(processed_preds))], 20)
        result["r@50"] = r_at_k(collection, embeddings, [str(i) for i in range(len(processed_preds))], 50)

        return result
        
    
    if finetuning_arguments.training_task == "question-answering":
        client = chromadb.PersistentClient(path="chroma_data/")
        rk_model = AutoModel.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True).to("cuda:0")
        rk_tokenizer = AutoTokenizer.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True)
        rk_tokenizer.pad_token_id = rk_tokenizer.eos_token_id = 2
        rk_model.config.pad_token_id = rk_model.config.eos_token_id
        collection = client.get_collection(
                name="answer_embeddings_definitivo",
            )
    
    tokenizer = AutoTokenizer.from_pretrained(finetuning_arguments.model_name)
    model = T5ForConditionalGeneration.from_pretrained(finetuning_arguments.model_name, 
                                                  device_map={"":0},)
    
    def preprocess_function(examples):
        """Add prefix to the sentences, tokenize the text, and set the labels"""
        # The "inputs" are the tokenized answer:
        inputs = [ex + " </s>" for ex in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=400, truncation=True, padding=True)
        
        # The "labels" are the tokenized outputs:
        labels = tokenizer(text_target=examples["answer"], max_length=200, truncation=True, padding=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    token = login(token="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", write_permission=True)
    
    # question_answering_dataset = load_qna_dataset(finetuning_arguments.first_n)

    # question_answering_dataset = Dataset.from_dict(question_answering_dataset.to_dict(orient='list'))

    # question_answering_dataset = question_answering_dataset.train_test_split(test_size=0.2)
    
    # question_answering_dataset["train"] = question_answering_dataset["train"].map(lambda examples: {"text" : f'''{examples["text"]} {examples["answer"]}'''})
    # question_answering_dataset["test"] = question_answering_dataset["test"].map(lambda examples: {"text" : f'''{examples["text"]}'''})

 
    
    if finetuning_arguments.training_task == "syntetic-question-answering":
        dataset = load_syntetic_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "multiple-choice":
        dataset = load_mutliple_choice_dataset(finetuning_arguments.first_n)
    elif finetuning_arguments.training_task == "question-answering":
        dataset = load_qna_dataset(finetuning_arguments.first_n)



    data_dict = dataset.to_dict(orient='list')
    dataset = Dataset.from_dict(data_dict)   

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    dataset = dataset.train_test_split(test_size=0.2)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text', 'answer'])
    # []
    # for idx, row in dataframe.iterrows():
    #     tokenized_dataset.append(preprocess_function(row, tokenizer))
    #     if idx == 3:
    #         print(tokenized_dataset[idx])
    
    if finetuning_arguments.training_task == "multiple-choice":
        print(f'''{finetuning_arguments.model_name}_{finetuning_arguments.training_task}: {evaluate_multiple_choice(model, tokenizer, dataset)}''')
    else:
        print(f'''{finetuning_arguments.model_name}_{finetuning_arguments.training_task}: {evaluate_question_answering(dataset)}''')
    
    
    
    # print(tokenized_dataset)
    
    # print(len(tokenized_dataset))

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{finetuning_arguments.new_model_name}",
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
        generation_max_length=100,
    )

    def compute_metrics(eval_preds):
        sample_path = "t5_sample.json"
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = [pred.strip() for pred in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [label.strip() for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

        result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result.items()}

        result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
            (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)
        
        decoded_preds = [pred.replace("\n", " ") for pred in decoded_preds]
        decoded_labels = [label.replace("\n", " ") for label in decoded_labels]

        # result["bertscore"] = round(sum(result_bs["f1"]) / len(result_bs["f1"]) * 100, 2)
        result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])
        
        tokenized_preds = rk_tokenizer(decoded_preds,
                                   add_special_tokens=False,
                                   padding=True,
                                   truncation=True,
                                   return_tensors="pt").to("cuda:0")
        
        for input_ids in tokenized_preds["input_ids"]:
            input_ids = torch.cat((input_ids, torch.tensor([102]).to("cuda:0")), dim=0).to("cuda:0")
            
        output = rk_model(tokenized_preds["input_ids"])
        
        embeddings = []
        with torch.no_grad():
            for last_hidden_state in output.last_hidden_state:
                cls_embedding = last_hidden_state[0, :].cpu().numpy().tolist()
                embeddings.append(cls_embedding)

        result["r@1"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 1)
        result["r@3"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 3)
        result["r@5"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 5)
        result["r@10"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 10)
        result["r@20"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 20)
        result["r@50"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 50)

        coppie = list(zip(decoded_preds, decoded_labels))
        
        sample = random.sample(coppie, finetuning_arguments.sample)
        result["sample"] = [{'correct_answer' : a[1], 'prediction' : a[0]} for a in sample]
        result["model_name"] = f"{finetuning_arguments.model_name}_{finetuning_arguments.training_task}"
        result["datetime"] = datetime.now().isoformat()
        with open(sample_path, "a") as json_file:
            json.dump(result, json_file, indent=2)
            json_file.write("\n")
        return result

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
        
    trainer.model.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    trainer.tokenizer.save_pretrained(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')
    
    trainer.push_to_hub(f'{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}')

    # optimizer = Adam(model.parameters(), lr=0.00001)
    # train_data, val_data = train_test_split(dataframe, test_size=0.2, random_state=42)

    # train_sampler = RandomSampler(train_data.index)
    # val_sampler = RandomSampler(val_data.index)

    # qa_dataset = QA_Dataset(tokenizer, dataframe, Q_LEN, T_LEN)
    # print(qa_dataset.answer)
    # print(qa_dataset[0]["labels"])
    # train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    # val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    # train_loss = 0
    # val_loss = 0
    # train_batch_count = 0
    # val_batch_count = 0

    # for epoch in range(2):
    #     model.train()
    #     for batch in tqdm(train_loader, desc="Training batches"):
    #         input_ids = batch["input_ids"].to(DEVICE)
    #         attention_mask = batch["attention_mask"].to(DEVICE)
    #         labels = batch["labels"].to(DEVICE)
    #         decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

    #         outputs = model(
    #                         input_ids=input_ids,
    #                         attention_mask=attention_mask,
    #                         labels=labels,
    #                         decoder_attention_mask=decoder_attention_mask
    #                         )

    #         optimizer.zero_grad()
    #         outputs.loss.backward()
    #         optimizer.step()
    #         train_loss += outputs.loss.item()
    #         train_batch_count += 1

    #     #Evaluation
    #     model.eval()
    #     for batch in tqdm(val_loader, desc="Validation batches"):
    #         input_ids = batch["input_ids"].to(DEVICE)
    #         attention_mask = batch["attention_mask"].to(DEVICE)
    #         labels = batch["labels"].to(DEVICE)
    #         decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

    #         outputs = model(
    #                         input_ids=input_ids,
    #                         attention_mask=attention_mask,
    #                         labels=labels,
    #                         decoder_attention_mask=decoder_attention_mask
    #                         )

    #         optimizer.zero_grad()
    #         outputs.loss.backward()
    #         optimizer.step()
    #         val_loss += outputs.loss.item()
    #         val_batch_count += 1

    #     print(f"{epoch+1}/{2} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss/val_batch_count}")
    
    #accuracy mutliple choice
    # print("multiple choice task")
    # print(datetime.now().isoformat())
    # a = []
    # for _, instance in dataframe.iterrows():
    #     a.append(predict_answer(instance["question"], "", model, tokenizer, instance["answer"]))

    # print(a[:10])
    # print(dataframe["answer"].tolist()[:10])
    # acc = accuracy_score(dataframe["answer"].tolist(), a)
    # print(f"accuracy_score {acc}")
    model = AutoModelForSeq2SeqLM.from_pretrained(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''')

    tokenizer = AutoTokenizer.from_pretrained(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}''')
    if finetuning_arguments.training_task == "multiple-choice":
        print(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}: {evaluate_multiple_choice(model, tokenizer, dataset)}''')
    else:
        print(f'''{finetuning_arguments.new_model_name}_{finetuning_arguments.training_task}: {evaluate_question_answering(dataset)}''')

if __name__ == "__main__":main()