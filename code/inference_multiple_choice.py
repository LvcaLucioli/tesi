from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    pipeline,  
    T5ForConditionalGeneration,
)
from datasets import Dataset
import json
import torch
from typing import Optional
from dataclasses import dataclass, field
import pandas as pd
import re

@dataclass
class PredictionArguments:
    model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the model."}
    )
    predictions_path: Optional[str] = field(
        default=None, metadata={'help' : 'predictions path'}
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
    start_prediction: Optional[str] = field(
        default=None, metadata={'help' : 'where the prediction actually starts'}
    )
    end_prediction: Optional[str] = field(
        default=None, metadata={'help' : 'where the prediction ends'}
    )
    max_new_tokens: Optional[int] = field(
        default=None, metadata={'help' : 'max_new_tokens'}
    )
    
    


parser = HfArgumentParser((PredictionArguments))
generation_arguments  = parser.parse_args_into_dataclasses()[0]

def extract_letter(testo):
    testo = testo.lower()
    # match = re.search("[a-zA-Z]", testo)
    # if match:
    #     risposta = match.group(0)
    #     return risposta
    # else:
    #     return ''
    for carattere in testo:
        print(f'''carattere: {carattere}''')
        if carattere.isalpha():  # Controlla se il carattere è una lettera
            return carattere
    return ''


def load_mutliple_choice_dataset():
    path = "../datasets/CA_dataset_w_options.csv"

    df = pd.read_csv(path)
    df = df.rename(columns={
    'Domanda': 'question',
    'Opzioni' : 'options',
    'Opzione corretta' : 'correct_option',
    'Id' :'id',
    })
    df = df[df['question'].str.len() >= 5]

    new_dataset = []
    for _, qa in df.iterrows():
        if "phi" in generation_arguments.model_name:
            inputs = {
                "text" : f'''"Instruct: Analizza la domanda e le possibili opzioni di risposta e seleziona l'opzione corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi solo una tra le lettere 'a', 'b', 'c', 'd'.\nQuestion: {qa["question"]}\n{qa["options"]}\nOutput: ''',    
                "answer" : qa["correct_option"],
                "id" : qa["id"],
                }
        elif "ntino" in generation_arguments.model_name:
            inputs = {
                "text" : f'''"### Instruction:\nAnalizza la domanda e le possibili opzioni di risposta e seleziona l'opzione corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi solo una tra le lettere 'a', 'b', 'c', 'd'.\n{qa["question"]}{qa["options"]}\n\n ### Response:\n''',    
                "answer" : qa["correct_option"],
                "id" : qa["id"],
                }
        elif "flan" in generation_arguments.model_name:
            inputs = {
                "text" : f''''\nDomanda: {qa["question"]}\n\nOPZIONI:{qa["options"]}\n\nRisposta: ''',  
                "answer" : qa["correct_option"],
                "id" : qa["id"],
                }
        elif "zephyr" in generation_arguments.model_name:
            inputs = {
            "text" : f'''"<|system|>\nAnalizza la domanda e le possibili opzioni di risposta e scrivi la lettera corrispondente all'opzione di risposta corretta basandoti sugli ultimi aggiornamenti del codice degli appalti italiano. Scrivi solo una tra le lettere 'a', 'b', 'c', 'd'.</s>\n<|user|>\n{qa["question"]}\n{qa["options"]}</s>\n<|assistant|>\n''',    
                "answer" : qa["correct_option"],
                "id" : qa["id"],
                }
        new_dataset.append(inputs)
    return pd.DataFrame(new_dataset)


if "lvcalucioli" in generation_arguments.model_name:
    bnb_config = None
else: 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

if "microsoft" in generation_arguments.model_name:
    model = AutoModelForCausalLM.from_pretrained(generation_arguments.model_name,
                                            trust_remote_code=True,
                                            quantization_config = bnb_config,
                                            flash_attn = True,
                                            flash_rotary = True,
                                            fused_dense = True,
                                            low_cpu_mem_usage = True,
                                            device_map={"":0},
                                            revision="refs/pr/23")
elif "lvcalucioli/phi" in generation_arguments.model_name:
    model = AutoModelForCausalLM.from_pretrained(generation_arguments.model_name,
                                                trust_remote_code=True,
                                                device_map={"":0},
                                                # flash_attn = True,
                                                # flash_rotary = True,
                                                # fused_dense = True,
                                                low_cpu_mem_usage = True,)

elif "HuggingFaceH4/zephyr" in generation_arguments.model_name or "swap-uniba" in generation_arguments.model_name:
    model = AutoModelForCausalLM.from_pretrained(
        generation_arguments.model_name,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
elif "lvcalucioli/zephyr" in generation_arguments.model_name or "lvcalucioli/llamantino" in generation_arguments.model_name:
    model = AutoModelForCausalLM.from_pretrained(
        generation_arguments.model_name,
        device_map={"": 0}
    )
elif "flan" in generation_arguments.model_name:
    model = T5ForConditionalGeneration.from_pretrained(generation_arguments.model_name, 
                                                  device_map={"":0},)
    

model.config.use_cache = False
model.config.pretraining_tp = 1

    
tokenizer = AutoTokenizer.from_pretrained(generation_arguments.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def generate_preds(dataset):
    if "flan" in generation_arguments.model_name:
        pipe = pipeline("text2text-generation", 
                        model=generation_arguments.model_name,
                        tokenizer=tokenizer,
                        trust_remote_code=True,
                        max_new_tokens=generation_arguments.max_new_tokens, #123
                        # do_sample=True,
                        # top_k=generation_arguments.top_k,
                        # top_p=generation_arguments.top_p,
                        # temperature=generation_arguments.temperature,
                        # num_beams=generation_arguments.num_beams,
                        pad_token_id=tokenizer.eos_token_id)
    else:
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        trust_remote_code=True,
                        tokenizer=tokenizer,
                        max_new_tokens=generation_arguments.max_new_tokens, #123
                        # do_sample=True,
                        # top_k=generation_arguments.top_k,
                        # top_p=generation_arguments.top_p,
                        # temperature=generation_arguments.temperature,
                        # num_beams=generation_arguments.num_beams,
                        pad_token_id=tokenizer.eos_token_id
                        
        )
    
    i = 0  
    preds = []      
    for instance in dataset["test"]["text"]:
        print(f'''generation: {i}''')
        pred = pipe(instance)
        if "flan" in generation_arguments.model_name:
            pred[0]["generated_text"] = extract_letter(pred[0]["generated_text"])
            print(pred[0]["generated_text"])
            preds.append({
                    "generated_text" : pred[0]["generated_text"],
                    "id" : dataset["test"]["id"][i],
                    "answer" : dataset["test"]["answer"][i],
                    "prompt" : dataset["test"]["text"][i]})
        else:    
            pred[0]["generated_text"] = pred[0]["generated_text"][pred[0]["generated_text"].find(generation_arguments.start_prediction) + len(generation_arguments.start_prediction):] 
            print(f'''generated_text before: {pred[0]["generated_text"]}''')
            pred[0]["generated_text"] = extract_letter(pred[0]["generated_text"])
            print(f'''generated_text after: {pred[0]["generated_text"]}''')
            end_idx = pred[0]["generated_text"].find(generation_arguments.end_prediction)
            if end_idx == -1:
                preds.append({
                    "generated_text" : pred[0]["generated_text"],
                    "id" : dataset["test"]["id"][i],
                    "answer" : dataset["test"]["answer"][i],
                    "prompt" : dataset["test"]["text"][i]})
            else:
                preds.append({
                    "generated_text" : pred[0]["generated_text"][: end_idx],
                    "id" : dataset["test"]["id"][i],
                    "answer" : dataset["test"]["answer"][i],
                    "prompt" : dataset["test"]["text"][i]})
        i = i + 1
    with open("backup.json", 'a') as file:
        json.dump(preds, file, indent=4)
    with open(generation_arguments.predictions_path, 'a') as file:
        json.dump(preds, file, indent=4)


def main():
    set_seed(42)
    dataset = load_mutliple_choice_dataset()
    dataset = Dataset.from_dict(dataset.to_dict(orient='list'))
    dataset = dataset.train_test_split(test_size=0.2)
    
    generate_preds(dataset)


if __name__ == "__main__":main()