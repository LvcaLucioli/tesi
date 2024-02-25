__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import torch
import chromadb
from transformers import (AutoModel,
                          AutoTokenizer,
                          set_seed,
                          HfArgumentParser)
from dataclasses import dataclass, field
from typing import Optional
import json
from datetime import datetime

@dataclass
class EvaluationArguments:
    model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the model."}
    )
    predictions_path: Optional[str] = field(
        default=None, metadata={'help' : 'predictions path'}
    )
    encoder_name: Optional[str] = field(
        default=None, metadata={'help' : 'encoder model name'}
    )
    
def r_at_k(collection, embeddings, ids, k):
        score = 0

        for pred, id in zip(embeddings, ids):
            results = collection.query(
                    query_embeddings=pred,
                    n_results=k,
                    include=["documents"]
            )
            if str(id) in results["ids"][0]:
                score += 1
        return (score / len(ids))


def main():
        set_seed(42)

        parser = HfArgumentParser((EvaluationArguments))
        evaluation_arguments  = parser.parse_args_into_dataclasses()[0]
    
        client = chromadb.PersistentClient(path="chroma_data/")
        tokenizer = AutoTokenizer.from_pretrained(evaluation_arguments.encoder_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(evaluation_arguments.encoder_name, trust_remote_code=True).to("cuda:0")

        tokenizer.pad_token_id = tokenizer.eos_token_id = 2
        model.config.pad_token_id = model.config.eos_token_id
        if "bge" in evaluation_arguments.encoder_name:
            
            collection = client.get_collection(
                    name="answer_embeddings_bge_1",
                )
        else:
            collection = client.get_collection(
                name="answer_embeddings_definitivo",
            )
        
        
        with open(evaluation_arguments.predictions_path, 'r') as file:
            data = json.load(file)
        
        generated_texts = []
        ids = []

        # Estrai i dati nelle liste
        for entry in data:
            generated_texts.append(entry['generated_text'])
            ids.append(entry['id'])

        # Crea un dizionario con le due liste
        result_dict = {'generated_text': generated_texts, 'id': ids}
        
        tokenized_preds = tokenizer(result_dict["generated_text"],
                                        add_special_tokens=False,
                                        padding=True,
                                        truncation=True,
                                        return_tensors="pt",
                                        return_attention_mask=True).to("cuda:0")

        for input_ids in tokenized_preds["input_ids"]:
            input_ids = torch.cat((input_ids, torch.tensor([102]).to("cuda:0")), dim=0).to("cuda:0")
            
        
        output = model(tokenized_preds["input_ids"], attention_mask=tokenized_preds["attention_mask"])

        embeddings = []
        with torch.no_grad():
            for last_hidden_state in output.last_hidden_state:
                cls_embedding = last_hidden_state[0, :].cpu().numpy().tolist()
                embeddings.append(cls_embedding)
                
        result = {} 
        result["r@1"] = r_at_k(collection, embeddings, result_dict["id"], 1)
        result["r@3"] = r_at_k(collection, embeddings, result_dict["id"], 3)
        result["r@5"] = r_at_k(collection, embeddings, result_dict["id"], 5)
        result["r@10"] = r_at_k(collection, embeddings, result_dict["id"], 10)
        result["r@20"] = r_at_k(collection, embeddings, result_dict["id"], 20)
        result["r@50"] = r_at_k(collection, embeddings, result_dict["id"], 50)
        
        result["model_name"] = evaluation_arguments.model_name
        # result["temperature"] = evaluation_arguments.temperature
        # result["top_p"] = evaluation_arguments.top_p
        # result["top_k"] = evaluation_arguments.top_k
        # result["num_beams"] = evaluation_arguments.num_beams
        # result["task"] = evaluation_arguments.training_task
        result["datetime"] = datetime.now().isoformat()
        
        with open('./results/r@k.json', 'a') as file:
            json.dump(result, file, indent=4)
    
        print(result)

if __name__ == "__main__":main()