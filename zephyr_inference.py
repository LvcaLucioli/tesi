__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from transformers import (AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          AutoTokenizer,
                          HfArgumentParser,
                          AutoModel,
)
import torch
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
import json
import evaluate
import nltk
import numpy as np
import chromadb
import random
nltk.download('punkt')
client = chromadb.PersistentClient(path="/chroma_db")
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceArguments:
    model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the model."}
    )
    first_n: Optional[int] = field(
        default=0, metadata={"help" : "Number of instance of the dataset"}
    )
    hit_precision_log_path: Optional[str] = field(
        default=None, metadata={"help" : "Hit precision log path."}
    )
    rouge_log_path: Optional[str] = field(
        default=None, metadata={"help" : "Rouge log path."}
    )
    accuracy_log_path: Optional[str] = field(
        default=None, metadata={"help" : "Accuracy log path."}
    )
    sample : Optional[int] = field(
        default=None, metadata={"help" : "Number of samples."}
    )


# def prepare_data(data):
#     qas = []
#     for _, qa in data.iterrows():
#         inputs = {
#             "qna_prompt" : f'''"<|system|> Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano.</s> <|user|> {qa["question"]}\n</s> <|assistant|> ''',      
#             "multiple_choice_prompt" : f'''"<|system|> Analizza la domanda e le possibili opzioni di risposta e scrivi solo la lettera corrispondente all'opzione corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi una tra le lettere 'a', 'b', 'c', 'd'.</s> <|user|> {qa["question"]} Quale tra queste opzioni è quella corretta per rispondere alla domanda? {qa["options"]}\n</s> <|assistant|> ''',    
#             "answer" : qa["answer"],
#             "correct_option" : qa["correct_option"],
#             }
#         qas.append(inputs)
#     return qas


def predict(model, tokenizer, questions, max_new_tokens = 512):
    batch_size = 8
    preds = []

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        # print(batch_questions)
        inputs = tokenizer(batch_questions,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    truncation=True).to('cuda:0')

        outputs = model.generate(**inputs, 
                                 max_new_tokens=max_new_tokens, 
                                 eos_token_id=tokenizer.eos_token_id, 
                                 pad_token_id=tokenizer.pad_token_id)

        # index = tokenizer.batch_decode(outputs)[0].find("The answer is:")
        # model_answer = tokenizer.batch_decode(outputs)[0][index + len("The answer is:"):]

        answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for answer in answers:
            generated_text = answer
            if max_new_tokens > 1:
                split_text = generated_text.split("<|assistant|>", 1)
            else:
                split_text = generated_text.split("<|assistant|>", 1)
            assistant_response = split_text[1].strip() if len(split_text) > 1 else ""
            assistant_response = assistant_response.replace("", "").strip()
            preds.append(assistant_response)
            # print(f'''RESPONSE: {assistant_response}''')
            
        qna = list(zip(batch_questions, answers))
        lista_di_dizionari = [{'domanda': domanda, 'risposta': risposta} for domanda, risposta in qna]
        with open('inference.json', 'a', encoding='utf-8') as file:
            json.dump(lista_di_dizionari, file, ensure_ascii=False, indent=4)
    return preds



def get_preds(model, tokenizer, questions, max_new_tokens=None):
    if max_new_tokens is None:
        predictions = predict(model, tokenizer, questions)
    else:
        predictions = predict(model, tokenizer, questions, max_new_tokens)
        predictions = [pred.replace(" ", "") for pred in predictions]
        predictions = [pred.replace(")", "") for pred in predictions]
        predictions = [pred.lower() for pred in predictions]
        predictions = [pred[:1] for pred in predictions]
    return predictions


def load_qna_dataset(n):
    generative_path = "CdA-mininterno-quiz_dataset.csv"

    df = pd.read_csv(generative_path)

    df = df.drop(columns=['Tipo-Domanda'])
    
    df = df.rename(columns={
    'Domanda': 'question',
    'Risposta': 'answer',
    })
    
    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "prompt" : f'''"<|system|> Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano.</s> <|user|> {qa["question"]}\n</s> <|assistant|> ''',      
            "answer" : qa["answer"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]

def load_mutliple_choice_dataset(n):
    multiple_choice_path = "CA_dataset_w_options.csv"

    df = pd.read_csv(multiple_choice_path)
    df = df.rename(columns={
    'Domanda': 'question',
    'Opzioni' : 'options',
    'Opzione corretta' : 'correct_option'
    })
    
    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "prompt" : f'''"<|system|> Analizza la domanda e le possibili opzioni di risposta e scrivi solo la lettera corrispondente all'opzione corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi una tra le lettere 'a', 'b', 'c', 'd'.</s> <|user|> {qa["question"]} Quale tra queste opzioni è quella corretta per rispondere alla domanda? {qa["options"]}\n</s> <|assistant|> ''',    
            "answer" : qa["correct_option"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]

def load_syntetic_dataset(n):
    inference_dataset_path = "ca_synthetic_qa_dataset.csv"
    
    df = pd.read_csv(inference_dataset_path)
    
    df = df.rename(columns={
        'Question' : 'question',
        'Answer' : 'answer',
    })
    
    new_dataset = []
    for _, qa in df.iterrows():
        inputs = {
            "prompt" : f'''"<|system|> Analizza la domanda e rispondi in italiano basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Rispondi in italiano.</s> <|user|> {qa["question"]}\n</s> <|assistant|> ''',      
            "answer" : qa["answer"],
            }
        new_dataset.append(inputs)
    if n == 0:
        return pd.DataFrame(new_dataset)
    else:
        return pd.DataFrame(new_dataset)[:n]

def rouge(decoded_preds, decoded_labels, tokenizer, model_name, path, n_sample):
    metric_rouge = evaluate.load("rouge")

    # Replace -100s used for padding as we can't decode them
    # preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # decoded_preds = [pred.strip() for pred in tokenizer.batch_decode(preds, skip_special_tokens=True)]
    # decoded_labels = [label.strip() for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]
    print(decoded_preds)
    result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 2) for k, v in result.items()}

    result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
        (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

    decoded_preds = [pred.replace("\n", " ") for pred in decoded_preds]
    decoded_labels = [label.replace("\n", " ") for label in decoded_labels]

    # result["bertscore"] = round(sum(result_bs["f1"]) / len(result_bs["f1"]) * 100, 2)
    result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in decoded_preds])
    result["timestamp"] = datetime.now().isoformat()
    result["model_name"] = model_name
   
    coppie = list(zip(decoded_labels, decoded_preds))
    
    sample = random.sample(coppie, n_sample)
    result["sample"] = [{'correct_answer' : a[0], 'prediction' : a[1]} for a in sample]

    with open(path, "a") as json_file:
        json.dump(result, json_file, indent=2)
        json_file.write("\n")
    return result

def tokenize_batch(answers, tokenizer):
    tokenized_list = [tokenizer(answer, add_special_tokens=False, return_tensors="pt") for answer in answers]
    for tokenized in tokenized_list:
        tokenized.input_ids = torch.cat((tokenized.input_ids, torch.tensor([[tokenizer.cls_token_id]] * tokenized.input_ids.shape[0])), dim=1)
    return tokenized_list

def get_cls_encoding_batch(tokenized_list, model):
    encodings = []
    with torch.no_grad():
        for tokenized in tokenized_list:
            output = model(**tokenized)
            cls_embedding = output.last_hidden_state[:, 0, :].numpy().tolist()
            encodings.append(cls_embedding)
    return encodings

def r_at_k(collection, embeddings, ids, k):
    score = 0
    step = 0
    
    for pred, id in zip(embeddings, ids):
        results = collection.query(
                query_embeddings=pred,
                n_results=k
        )
        if id in results["ids"][0]:
            score += 1
        step += 1
    return (score / len(ids))

def hit_precision(collection, preds, ids, model, tokenizer, model_name, path):
    encoded_sentences = tokenize_batch(preds, tokenizer)
    embeddings = get_cls_encoding_batch(encoded_sentences, model)

    scores = {
            "r@1": 0.0,
            "r@3": 0.0,
            "r@5": 0.0,
            "r@10": 0.0,
            "r@20": 0.0
    }

    scores["r@1"] = r_at_k(collection, embeddings, ids, 1)
    scores["r@3"] = r_at_k(collection, embeddings, ids, 3)
    scores["r@5"] = r_at_k(collection, embeddings, ids, 5)
    scores["r@10"] = r_at_k(collection, embeddings, ids, 10)
    scores["r@20"] = r_at_k(collection, embeddings, ids, 20)
    
    scores["timestamp"] = datetime.now().isoformat()
    scores["model_name"] = model_name
    
    with open(path, "a") as json_file:
        json.dump(scores, json_file, indent=2)
        json_file.write("\n")

    return scores


def main():
    parser = HfArgumentParser((InferenceArguments))
    inference_arguments  = parser.parse_args_into_dataclasses()[0]


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    print(f"loading model {inference_arguments.model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        inference_arguments.model_name,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(inference_arguments.model_name)
    
    # model_input = tokenizer(text=query, return_tensors="pt").to("cuda:0")
    # _ = model.eval()
    # with torch.no_grad():
    #     out = model.generate(**model_input, max_new_tokens=100)
    #     response_text = tokenizer.decode(out[0], skip_special_tokens=True)
    # print(response_text)
    
    syntetic_dataset = load_syntetic_dataset(inference_arguments.first_n)
    multiple_choice_dataset = load_mutliple_choice_dataset(inference_arguments.first_n)
       
    print(f"starting question answering task {datetime.now().isoformat()}")
    predictions = get_preds(model, tokenizer, syntetic_dataset["prompt"].tolist())
    
    # rouge
    print(f"calculating rouge for {inference_arguments.model_name}, {datetime.now().isoformat()}")
    rouge(predictions, 
                    syntetic_dataset["answer"].tolist(), 
                    tokenizer, 
                    inference_arguments.model_name, 
                    inference_arguments.rouge_log_path,
                    inference_arguments.sample)
        
    # R@k
    print(f"calculating hit precision (r@k) {inference_arguments.model_name}, {datetime.now().isoformat()}")
    rk_model = AutoModel.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True)
    rk_tokenizer = AutoTokenizer.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True)
    
    collection = client.get_collection(
        name="answers_embeddings",
        )
    
    hit_precision(collection, 
                predictions,
                [f"{i+1}" for i in range(len(predictions))],
                rk_model,
                rk_tokenizer,
                inference_arguments.model_name,
                inference_arguments.hit_precision_log_path)

    
    print(f"starting multiple choice task {inference_arguments.model_name}, {datetime.now().isoformat()}")
    predictions = get_preds(model, tokenizer, multiple_choice_dataset["prompt"].tolist(), 3)
    
    coppie = list(zip(multiple_choice_dataset["answer"], predictions))
    
    sample = random.sample(coppie, inference_arguments.sample)
    
    dictionary_list = [{'correct_answer' : a[0], 'prediction' : a[1]} for a in sample]
    
    accuracy = {"model" : inference_arguments.model_name, 
                "accuracy" : accuracy_score(multiple_choice_dataset["answer"].tolist(), predictions), 
                "timestamp" : datetime.now().isoformat(),
                "sample" : dictionary_list
                }
    
    with open(inference_arguments.accuracy_log_path, "a") as json_file:
        json.dump(accuracy, json_file, indent=2)
        json_file.write("\n")

if __name__ == "__main__":main()
