__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch
from datetime import datetime

def load_answers():
    generative_path = "CdA-mininterno-quiz_dataset.csv"
    df = pd.read_csv(generative_path)
    # df = df[df['Question'].str.len() >= 5]
    return df

def tokenize_batch(answer, tokenizer):
    tokenized = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    tokenized.input_ids = torch.cat((tokenized.input_ids, torch.tensor([[tokenizer.cls_token_id]] * tokenized.input_ids.shape[0])), dim=1)
    return tokenized
    # tokenized_list = [tokenizer(answer, add_special_tokens=False, return_tensors="pt") for answer in answers]
    # for tokenized in tokenized_list:
    #     tokenized.input_ids = torch.cat((tokenized.input_ids, torch.tensor([[tokenizer.cls_token_id]] * tokenized.input_ids.shape[0])), dim=1)
    # return tokenized_list

def get_cls_encoding_batch(tokenized, model):
    # encodings = []
    # with torch.no_grad():
    #     for tokenized in tokenized_list:
    #         output = model(**tokenized)
    #         cls_embedding = output.last_hidden_state[:, 0, :].numpy().tolist()
    #         encodings.append(cls_embedding)
    # return encodings
    
    with torch.no_grad():
        output = model(**tokenized)
        cls_embedding = output.last_hidden_state[:, 0, :].numpy().tolist()
        return cls_embedding

def main():
    model = AutoModel.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True)
    
    
    client = chromadb.PersistentClient(path="chroma_data/")
    # minute = datetime.now().minute
    # print(minute)
    # collection = client.create_collection(
    #     name=f"answer_embeddings_definitivo",
    #     )
    
    # print("collection created, adding embeddings")  
    
    # df = load_answers()
    # for answer, id in zip(df["Risposta"], df["Id"]):
    #     collection.add(
    #         embeddings=get_cls_encoding_batch(tokenize_batch(answer, tokenizer), model),
    #         documents=[answer],
    #         ids=[str(id)]
    #     )
    
    collection = client.get_collection(
        name=f"answer_embeddings_definitivo",
        )
    # print(get_cls_encoding_batch(tokenize_batch(answers, tokenizer), model))
    # results = collection.query(
    #                 query_embeddings=get_cls_encoding_batch(tokenize_batch("answers", tokenizer), model),
    #                 n_results=20
    #         )
    print(collection.get(
        ids=['229', '316'],
        include=['documents'],
    ))
    
    print(collection.peek(20)["ids"])

if __name__ == "__main__":main()