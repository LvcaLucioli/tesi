from transformers import (
  RagTokenizer,
  RagRetriever,
  RagModel,
  RagTokenForGeneration,
  RagSequenceForGeneration,
  RagConfig,
  T5ForConditionalGeneration,
  T5Tokenizer,
  AutoTokenizer,
  DPRQuestionEncoder,
  Trainer,
  TrainingArguments,
  HfArgumentParser,
)
from datasets import Dataset
from typing import Optional
from dataclasses import dataclass, field
import torch
from datetime import datetime
import pandas as pd
import json
from datasets import load_from_disk

device = 'cuda:0' if torch.cuda.device_count() > 0 else 'cpu'


@dataclass
class FinetuneArguments:
    encoder_model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the encoder model."}
    )
    generator_model_name: Optional[str] = field(
        default=None, metadata={"help" : "The name of the generator model."}
    )
    num_chunks_to_retrieve: Optional[int] = field(
        default=None, metadata={"help" : "Number of chunks to retrieve."}
    )


parser = HfArgumentParser((FinetuneArguments))
finetuning_arguments  = parser.parse_args_into_dataclasses()[0]

# or 256
INDEX_PATH = './datasets/ca_rag_dataset_512/ca_dataset_faiss_512.index'
DS_PATH = './datasets/ca_rag_dataset_512'

MAX_SEQ_LENGTH=512


# Load dataset from disk
dataset = load_from_disk(DS_PATH)

# Add metadata for the approximate top-k search over instances in the dataste
dataset.load_faiss_index("embeddings", INDEX_PATH)


retrieval_tokenizer = AutoTokenizer.from_pretrained(finetuning_arguments.encoder_model_name)
retrieval_encoder = DPRQuestionEncoder.from_pretrained(finetuning_arguments.encoder_model_name)


generator_tokenizer = T5Tokenizer.from_pretrained(finetuning_arguments.generator_model_name)
generator_encoder = T5ForConditionalGeneration.from_pretrained(finetuning_arguments.generator_model_name)

def load_qna_dataset():
    path = "./datasets/CdA-mininterno-quiz_dataset.csv"
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
            "question" : qa["question"],      
            "answer" : qa["answer"],
            "id" : qa["id"],
            }
        new_dataset.append(inputs)
    return pd.DataFrame(new_dataset)

   
# outputs = model(input_ids=...)

# doc_ids = outputs.retrieved_doc_ids[0].tolist()     # indices of retrieved documents in the database
# doc_scores = outputs.doc_scores[0].tolist()         # similarity scores between document and question

# for doc_i, sc in zip(doc_ids, doc_scores):
#   print('-'*50)
#   print(round(sc,2), repr(dataset['text'][doc_i]))


  
def main():
    rag_config_args = {
        'question_encoder' : retrieval_encoder.config.to_dict(),
        'generator' : generator_encoder.config.to_dict(),
        'vocab_size' : retrieval_tokenizer.vocab_size,
        'doc_sep' : ' // ', # Separator character between retrieved documenets
        'n_docs' : finetuning_arguments.num_chunks_to_retrieve,
        'retrieval_vector_size' : 768,
        'index_name' : "custom",
        'passages_path' : DS_PATH,
        'index_path' : INDEX_PATH,
        'output_retrieved' : True, # retrieved_doc_embeds, retrieved_doc_ids, context_input_ids and context_attention_mask are returned
    }

    rag_config = RagConfig(**rag_config_args)


    retriever = RagRetriever(
        config=rag_config,
        question_encoder_tokenizer=retrieval_tokenizer,
        generator_tokenizer=generator_tokenizer
    )

    rag_mod_args = {
        'config' : rag_config,
        'question_encoder' : retrieval_encoder,
        'generator' : generator_encoder,
        'retriever' : retriever,
    }

    model = RagTokenForGeneration(**rag_mod_args)
    #model = RagSequenceForGeneration(**rag_mod_args)

    training_args = TrainingArguments(
        output_dir="./rag_finetuned/",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs/",
        logging_steps=10,
        report_to="wandb",
    )
    tok_config = {
    'return_tensors' : 'pt',
    'max_length' : MAX_SEQ_LENGTH,
    'padding' : 'max_length',
    'truncation' : True
    }
    pair_dataset = load_qna_dataset()
    pair_dataset = Dataset.from_dict(pair_dataset.to_dict(orient='list'))
    
    print(pair_dataset["question"][0])
    input = retrieval_tokenizer(pair_dataset["question"][0], **tok_config).input_ids
    outputs = model.generate(input)
    print(generator_tokenizer.decode(outputs[0], skip_special_tokens=True))
    # doc_ids = outputs.retrieved_doc_ids[0].tolist()     # indices of retrieved documents in the database
    # doc_scores = outputs.doc_scores[0].tolist()         # similarity scores between document and question

    # for doc_i, sc in zip(doc_ids, doc_scores):
    #     print('-'*50)
    #     print(round(sc,2), repr(dataset['text'][doc_i]))
    
    

    # Define a function to process the data for training
    def preprocess_function(examples):
        model_inputs = retrieval_tokenizer(examples["question"],
                                           **tok_config,)
        # Prepare labels
        labels = generator_tokenizer(examples["answer"],
                                     max_length=512,
                                     truncation=True).input_ids
        
        model_inputs["labels"] = labels
        
        return model_inputs

    tokenized_datasets = pair_dataset.map(preprocess_function, 
                                 batched=True, 
                                 remove_columns=pair_dataset.column_names)
    # Split dataset into train, validation and test
    # ...
    
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.3)
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    # Start fine-tuning
    trainer.train() 
    
    

    trainer.push_to_hub("lvcalucioli/retriever")

if __name__ == "__main__":main()