import pandas as pd
import json
import ast
from transformers import (pipeline,
                          HfArgumentParser,)
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


K = 3

@dataclass
class GenerationArguments:
    model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the model."}
    )
    predictions_path: Optional[str] = field(
        default=None, metadata={'help' : 'predictions path'}
    )
    max_new_tokens: Optional[int] = field(
        default=None, metadata={'help' : 'max new tokens'}
    )
    task: Optional[str] = field(
        default=None, metadata={'help' : 'task'}
    )
    
parser = HfArgumentParser((GenerationArguments))
generation_arguments  = parser.parse_args_into_dataclasses()[0]

def load_dataset():
    retrieval_dataset_path = "../datasets/retrieval_dataset_256.csv"
    qna_dataset_path = "../datasets/CdA-mininterno-quiz_dataset.csv"

    retrieval_dataset = pd.read_csv(retrieval_dataset_path)
    
    qna_dataset = pd.read_csv(qna_dataset_path)
    
    retrieval_dataset["answer"] = qna_dataset["Risposta"]

    retrieval_dataset = retrieval_dataset.rename(columns={
    'q_id': 'question_id',
    })

    retrieval_dataset['retrieval_data'] = retrieval_dataset['retrieval_data'].apply(lambda x: ast.literal_eval(x))

    return retrieval_dataset

def create_context(ds_row, k):
    context = """
    Sei un esperto giurista italiano. Rispondi correttamente in italiano alle domande sul contenuto del Codice degli Appalti.
    Rispondi in maniera diretta e chiara. Fai riferimento alla tua conoscenza e alle informazioni di contesto fornite.

    ## Contesto:
    {context}

    ## Domanda:
    {question}

    ## Risposta:
    """

    q = ds_row['question']

    retr_data = ds_row['retrieval_data']
    txt = retr_data['text']
    title = retr_data['title']
    n_docs = min(k, len(txt))
    ctx = '\n'.join([title[i] + ' : ' + txt[i] for i in range(n_docs)])

    return context.format(context=ctx, question=q)

def generate(inputs, dataset):
    
    start_predictions = "## Risposta:"
    
    pipe = pipeline(generation_arguments.task, 
                    model=generation_arguments.model_name,
                    tokenizer=generation_arguments.model_name,
                    # device=0,
                    batch_size=16,
                    # max_seq_length=256,
                    )
    out = pipe(inputs,
               
                max_new_tokens=generation_arguments.max_new_tokens)
    # print(out[:5])
    # with open(f'''../predictions/inference_with_context/{generation_arguments.predictions_path}''', "a") as file:
    #     json.dump(out, file, indent=4)
    i = 0
    result = []
    for pred in out:
        pred[0]["generated_text"] = pred[0]["generated_text"][pred[0]["generated_text"].find(start_predictions) + len(start_predictions):] 
        result.append({
            "id" : int(dataset["question_id"][i]),
            "question" : dataset["question"][i],
            "retrieval_data" : dataset["retrieval_data"][i],
            "answer" : dataset["answer"][i],
            "generated_text" : pred[0]["generated_text"],
        })
        i = i + 1
        
    with open(f'''../predictions/inference_with_context/{generation_arguments.predictions_path}''', "a", encoding='utf8') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

def main():
    dataset = load_dataset()
    inputs = [create_context(row.to_dict(), K) for _, row in dataset.iterrows()]
    
    generate(inputs, dataset)


if __name__ == "__main__":main()