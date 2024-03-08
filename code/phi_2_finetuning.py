import wandb
import pandas as pd
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, Trainer
import torch
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from huggingface_hub import login
import numpy as np
from evaluate import load



# def compute_metrics(p):
#     perplexity = load("perplexity", module_type="metric")
#     results = perplexity.compute(model_id='phi',
#                                 predictions=p.predictions)
#     return {"perplexity" : results}


def calculate_perplexity(probabilities):
    geo_mean = np.exp(np.mean(-np.log(probabilities)))
    perplexity = 1.0 / geo_mean
    return perplexity

def compute_metrics(p):
    probabilities = torch.softmax(torch.tensor(p.predictions), dim=-1).numpy()
    perplexity = calculate_perplexity(probabilities)
    return {"perplexity": perplexity}

# def create_dataset():
#     with open('ca_whole_text.txt', 'r') as file:
#         lines = file.readlines()
#     dataset = pd.DataFrame({'text' : lines})
#     dataset = dataset[dataset['text'].apply(lambda x: x != [''] and x != [] and x != '\n')]
#     data_dict = {'text': dataset['text'].tolist()}
#     hf_dataset = datasets.Dataset.from_pandas(dataset[:11])
#     hf_dataset = hf_dataset.train_test_split(test_size=0.1)
#     hf_dataset.remove_columns(["__index_level_0__"])
#     return hf_dataset

def create_dataset():
    df = pd.read_csv('ca_chunk_articles.csv')
    df_renamed = df[['data']].rename(columns={'data': 'text'})
    hf_dataset = datasets.Dataset.from_pandas(df_renamed)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1)
    return hf_dataset

def main():
    hf_dataset = create_dataset()
    
    token = login(token="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", write_permission=True)
    wandb.login(key="769c267efb2df3c4a8a512255a0c11cb2f695f6e")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = False,
    )

    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                                trust_remote_code=True,
                                                quantization_config = bnb_config,
                                                flash_attn = True,
                                                flash_rotary = True,
                                                fused_dense = True,
                                                low_cpu_mem_usage = True,
                                                device_map={"":0},
                                                revision="refs/pr/23")

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    lora_config = LoraConfig(
                            r=32,
                            lora_alpha=64,
                            target_modules=["Wqkv", "fc1", "fc2"],
                            lora_dropout=0.1,
                            bias="none",
                            task_type=TaskType.CAUSAL_LM)

    model = prepare_model_for_kbit_training(model,
                                            use_gradient_checkpointing=True)


    training_args = TrainingArguments(
        output_dir="ca-finetuned-phi-2",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=4,
        gradient_accumulation_steps=32,
        evaluation_strategy="epoch",
        logging_steps=15,
        optim="paged_adamw_8bit",
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        save_steps=2000,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_steps=-1,
        report_to="wandb",
        run_name="ca-finetuning-phi-2-server-2",
        push_to_hub=True,

    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset["train"],
        dataset_text_field="text",
        eval_dataset=hf_dataset["test"],
        peft_config=lora_config,
        tokenizer=tokenizer,
        max_seq_length=364,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.push_to_hub("ca-finetuned-phi-2")

if __name__ == "__main__":main()