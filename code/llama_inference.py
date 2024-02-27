__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from peft import PeftModel, PeftConfig
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          pipeline,
                          AutoModel)
import evaluate
import nltk
import torch
import numpy as np
import chromadb
from datetime import datetime
nltk.download("punkt", quiet=True)

config = PeftConfig.from_pretrained("lvcalucioli/llamantino7b_2_multiple-choice")
model = AutoModelForCausalLM.from_pretrained("swap-uniba/LLaMAntino-2-7b-hf-ITA")
model = PeftModel.from_pretrained(model, "lvcalucioli/llamantino7b_2_multiple-choice")

tokenizer = AutoTokenizer.from_pretrained("lvcalucioli/llamantino7b_2_multiple-choice")

# client = chromadb.PersistentClient(path="chroma_data/")

# rk_model = AutoModel.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True).to("cuda:0")
# rk_tokenizer = AutoTokenizer.from_pretrained("dlicari/lsg16k-Italian-Legal-BERT", trust_remote_code=True)
# collection = client.get_collection(
#         name="answer_embeddings_100",
#     )
def r_at_k(collection, embeddings, ids, k):
        score = 0
        step = 0

        for pred, id in zip(embeddings, ids):
            results = collection.query(
                    query_embeddings=pred,
                    n_results=k,
                    include=["documents"]
            )
            
            if id in results["ids"][0]:
                score += 1
            step += 1
        return (score / len(ids))
    
def compute_metrics(eval_preds):
    # sample_path = f"{finetuning_arguments.new_model_name}_sample.json"

    metric_rouge = evaluate.load("rouge")

    preds, labels = eval_preds

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # print(f'''preds: {preds}''')
    # print(f'''labels: {labels}''')

    preds = np.argmax(preds, axis=-1)

    decoded_preds = [pred.strip() for pred in tokenizer.batch_decode(preds, skip_special_tokens=True)]
    decoded_labels = [label.strip() for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]

    # print(f'''decoded preds: {decoded_preds}''')

    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)       
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)


    processed_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    processed_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

    print(f'''processed preds: {processed_preds}''')

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
    # tokenized_preds["input_ids"] = 
    for input_ids in tokenized_preds["input_ids"]:
        # print(f"prima {input_ids.size()}")
        input_ids = torch.cat((input_ids, torch.tensor([102]).to("cuda:0")), dim=0).to("cuda:0")
        # print(f"dopo {input_ids.size()}")
        
    # print(f"dopo dopo {tokenized_preds['input_ids'][0].size()}")
    # tokenized_preds.input_ids
    # print(tokenized_preds["input_ids"])

    output = rk_model(tokenized_preds["input_ids"], attention_mask=tokenized_preds["attention_mask"])

    # print(output.last_hidden_state[0])
    # print(output.last_hidden_state[1])

    embeddings = []
    with torch.no_grad():
        for last_hidden_state in output.last_hidden_state:
            # print(tokenized)
            cls_embedding = last_hidden_state[0, :].cpu().numpy().tolist()
            embeddings.append(cls_embedding)


    # embeddings = get_cls_encoding_batch(tokenized_list=tokenized_preds, model=rk_model)

    result["r@1"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 1)
    result["r@3"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 3)
    result["r@5"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 5)
    result["r@10"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 10)
    result["r@20"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 20)
    result["r@50"] = r_at_k(collection, embeddings, [str(i) for i in range(len(decoded_preds))], 50)

    coppie = list(zip(decoded_labels, decoded_preds))

    # sample = random.sample(coppie, finetuning_arguments.sample)
    # result["sample"] = [{'correct_answer' : a[0], 'prediction' : a[1]} for a in sample]
    result["model_name"] = "lvcalucioli/llamantino7b_2_question-answering"
    result["datetime"] = datetime.now().isoformat()
    # with open(sample_path, "a") as json_file:
    #     json.dump(result, json_file, indent=2)
    #     json_file.write("\n")
    return result

def main():
    prompt = f'''"### Instruction:\nA norma del Codice dei contratti pubblici e degli appalti, chi si trovi in stato di fallimento può partecipare a gare pubbliche d'appalto? a) Si, devono essere pubblicati nella sezione ""Amministrazione trasparente""
    b) Entro cinque giorni dall'adozione del provvedimento
    c) No, mai
    d) No, non equivale ad accettazione dell'offerta\n\n### Response:\n'''
    # prompt = '''<s>[INST]
    # <<SYS>>
    # Analizza la seguente domanda e le opzioni di risposta 'a', 'b', 'c', 'd'. Scegli l'opzione corretta.
    # <</SYS>>
    # A norma del Codice dei contratti pubblici e degli appalti, chi si trovi in stato di fallimento può partecipare a gare pubbliche d'appalto?
    # a) Si, devono essere pubblicati nella sezione ""Amministrazione trasparente""
    # b) Entro cinque giorni dall'adozione del provvedimento
    # c) No, mai
    # d) No, non equivale ad accettazione dell'offerta [/INST]'''
    qna_prompt = f'''<s>[INST] Vi è equivalenza tra i termini minimi per la presentazione di offerta nelle procedure aperte e ristrette? [/INST]'''
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    
    result = pipe(prompt)
    # metrics = compute_metrics()
    print(result)

if __name__ == "__main__":main()