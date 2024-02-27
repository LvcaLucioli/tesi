__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModel,
    AutoModelForCausalLM,
    HfArgumentParser)
from peft import AutoPeftModelForCausalLM
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import nltk
import evaluate
import numpy as np 
import chromadb
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

nltk.download('punkt')
client = chromadb.PersistentClient(path="/chroma_db")

@dataclass
class PredictionArguments:
    model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the model."}
    )
#     dataset_name: Optional[str] = field(
#         default=None, metadata={"help": "The name of the local dataset to use."}
#     )
#     max_new_tokens: Optional[int] = field(
#         default=None, metadata={"help": "The max number of new tokens to generate."}
#     )
#     load_in_4bit: Optional[bool] = field(
#         default=None, metadata={"help" : "Flag used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes"}
#     )
#     bnb_4bit_quant_type: Optional[str] = field(
#         default=None, metadata={"help" : "The outlier threshold for outlier detection as described in LLM."}
#     )
#     bnb_4bit_compute_dtype: Optional[torch.dtype] = field(
#         default=None, metadata={"help" : "Computational type which might be different than the input time. For example, inputs might be fp32, but computation can be set to bf16 for speedups."}
#     )
#     bnb_4bit_use_double_quant: Optional[bool] = field(
#         default=None, metadata={"help" : "Flag is used for nested quantization where the quantization constants from the first quantization are quantized again."}
#     )

def predict(model, tokenizer, questions, max_new_tokens = 50):
    batch_size = 8
    preds = []

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        inputs = tokenizer(batch_questions,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    truncation=True).to('cuda:0')

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

        # index = tokenizer.batch_decode(outputs)[0].find("The answer is:")
        # model_answer = tokenizer.batch_decode(outputs)[0][index + len("The answer is:"):]

        answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for answer in answers:
            generated_text = answer
            if max_new_tokens > 1:
                split_text = generated_text.split("Output:", 1)
            else:
                split_text = generated_text.split("The correct answer is letter", 1)
            assistant_response = split_text[1].strip() if len(split_text) > 1 else ""
            assistant_response = assistant_response.replace("", "").strip()
            preds.append(assistant_response)
    return preds

    # generated_text = tokenizer.batch_decode(outputs)[0]
    # split_text = generated_text.split("Output:", 1)
    # assistant_response = split_text[1].strip() if len(split_text) > 1 else ""
    # assistant_response = assistant_response.replace("<|endoftext|>", "").strip()
    # outputs.append(assistant_response)
    # return outputs

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

# def tokenize(sentences, tokenizer):
#   batch_size = 8
#   tokenized_list = []
#   for i in range(0, len(sentences), batch_size):
#     batch_sentences = sentences[i:i+batch_size]
#     tokenized = tokenizer(batch_sentences, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt")
#     tokenized.input_ids = torch.cat((tokenized.input_ids, torch.tensor([[tokenizer.cls_token_id]] * tokenized.input_ids.shape[0])), dim=1)
#     tokenized_list.append(tokenized)
#   return tokenized_list

# def get_cls_encoding(tokenized_list, model):
#     encodings = []
#     for tokenized in tokenized_list:
#         with torch.no_grad():
#             output = model(**tokenized)
#             cls_embedding = output.last_hidden_state[:, 0, :].numpy().tolist()
#             encodings.append(cls_embedding)
#     return encodings

# def load_dataset():
#     path = "/content/CA_dataset_w_options.csv"
#     data = pd.read_csv(path, sep = ',')

#     def prepare_data(data):
#     qas = []
#     for _, qa in data.iterrows():
#         inputs = {
#             "prompt" : f'''Analyze the following question and select the correct option from those provided: "{qa["Domanda"]}" {qa["Opzioni"]} The correct answer is letter''',
#             "answer" : qa["Risposta"]}
#         qas.append(inputs)
#     return qas

#     dataframe = prepare_data(data)
#     dataframe = pd.DataFrame(dataframe)


def compute_metrics(decoded_preds, decoded_labels, tokenizer):
    metric_rouge = evaluate.load("rouge")

    # Replace -100s used for padding as we can't decode them
    # preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # decoded_preds = [pred.strip() for pred in tokenizer.batch_decode(preds, skip_special_tokens=True)]
    # decoded_labels = [label.strip() for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]

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
    result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in decoded_preds])

    return result

def r_at_k(collection, embeddings, ids, k):
    score = 0
    step = 0

    print("calculating results")
    print(datetime.now().isoformat())
    for pred, id in zip(embeddings, ids):
        results = collection.query(
                query_embeddings=pred,
                n_results=k
        )
        if id in results["ids"][0]:
            score += 1
        step += 1
        if (step % 5 == 0):
            print(score / int(id))
            print(datetime.now().isoformat())
    return (score / len(ids))

def hit_precision(collection, preds, ids, model, tokenizer):
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

    return scores

def prepare_data(data):
    qas = []
    for _, qa in data.iterrows():
        inputs = {
            "qna_prompt" : f'''"Instruct: You are an AI assistant that follows instruction extremely well. </s>User: Answer the following question in italian considering the latest updates to the Italian public procurement code. The question is "{qa["question"]}" \nOutput:''',
            "multiple_choice_prompt" : f'''Analyze the following question and select the correct option from those provided: "{qa["question"]}" {qa["options"]} The correct answer is letter''',
            "answer" : qa["answer"],
            "correct_option" : qa["correct_option"]}
        qas.append(inputs)
    return qas

def get_preds(model, tokenizer, questions, max_new_tokens=None):
    print("start predictions")
    if max_new_tokens is None:
        predictions = predict(model, tokenizer, questions)
    else:
        predictions = predict(model, tokenizer, questions, max_new_tokens)
    print("end predictions")
    return predictions

def main():
    parser = HfArgumentParser((PredictionArguments))
    prediction_args  = parser.parse_args_into_dataclasses()[0]
    dataframe = load_dataset() # dataset with questions, answers, options

    dataframe = prepare_data(dataframe)
    dataframe = pd.DataFrame(dataframe)[:100]

    # print(dataframe.columns)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = False,
    )
    if prediction_args.model_name == "microsoft/phi-2":
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                            trust_remote_code=True,
                                            quantization_config = bnb_config,
                                            flash_attn = True,
                                            flash_rotary = True,
                                            fused_dense = True,
                                            low_cpu_mem_usage = True,
                                            device_map={"":0},
                                            revision="refs/pr/23")
    else: 
        model = AutoPeftModelForCausalLM.from_pretrained(prediction_args.model_name,
                                                                quantization_config = bnb_config,
                                                                low_cpu_mem_usage = True,
                                                                trust_remote_code=True,
                                                                device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(prediction_args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side="left"
    print("model and tokenizer loaded")

    # qna
    print("qna task")
    print(datetime.now().isoformat())
    predictions = get_preds(model, tokenizer, dataframe["qna_prompt"].tolist())

    # rouge
    result = compute_metrics(predictions, dataframe["answer"].tolist(), tokenizer)
    
    result["timestamp"] = datetime.now().isoformat()
    result["model_name"] = prediction_args.model_name
    rouge_log_path = "rouge.json"

    with open(rouge_log_path, "a") as json_file:
        json.dump(result, json_file, indent=2)
        json_file.write("\n")


    # R@k
    print("starting hit precision")
    print(datetime.now().isoformat())
    rk_model = AutoModel.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True)
    rk_tokenizer = AutoTokenizer.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True)
    collection = client.get_collection(
        name="answers_embeddings",
        )

    print("calling hit-precision")
    print(datetime.now().isoformat())
    scores = hit_precision(collection, predictions, [f"{i+1}" for i in range(len(predictions))], rk_model, rk_tokenizer)
    scores["timestamp"] = datetime.now().isoformat()
    scores["model_name"] = prediction_args.model_name
    hit_precision_log_path = "hit_precision.json"

    with open(hit_precision_log_path, "a") as json_file:
        json.dump(scores, json_file, indent=2)
        json_file.write("\n")

    # accuracy mutliple choice
    print("multiple choice task")
    print(datetime.now().isoformat())
    predictions = get_preds(model, tokenizer, dataframe["multiple_choice_prompt"].tolist(), 1)
    outputs = [pred.replace(" ", "") for pred in predictions]
    a = accuracy_score(dataframe["correct_option"].tolist(), outputs)
    print(f"accuracy_score {a}")


if __name__ == "__main__":main()