
# !pip install sentence-transformers

import pandas as pd
import random
from datasets import DatasetDict, Dataset

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses


"""

>> HOW TO USE SENTENCE TRANSFORMERS

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')

                            
query = ["What is BGE M3?", "What's BM-25?"]
options = ["A transformer, lexical matching and multi-vector interaction.", "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_q = model.encode(query,normalize_embeddings=True)
embeddings_o = model.encode(options, normalize_embeddings=True)

similarity = embeddings_q @ embeddings_o.T


print(similarity)
# [[0.2535969  0.3474958 ]
 [0.26141018 0.6326089 ]]
"""


def main():

    model = SentenceTransformer('BAAI/bge-m3')


    print("[LOG] Loading the model")
    df = pd.read_csv('/content/CdA-mininterno-quiz.csv')


    print("[LOG] Parsing the dataset")
    data = []
    for index, row in df.iterrows():
        # Get a list of all 'Risposta' values except the current 'pos' value
        neg_values = df[df['Risposta'] != row['Risposta']]['Risposta'].tolist()
        
        data.append({
            'query': row['Domanda'],
            'pos': row['Risposta'],
            'neg': random.choice(neg_values) if neg_values else None
        })

    # Convert the list of dictionaries to a DataFrame
    pos_neg_ds = pd.DataFrame(data)
    # Convert the DataFrame to a Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(pos_neg_ds)

    # Create a DatasetDict
    dataset = DatasetDict({'train': hf_dataset})

    print("[LOG] Creating dataset of triplets")
    train_examples = []
    train_data = dataset['train']
    n_examples = len(dataset['train'])

    for i in range(n_examples):
        example = train_data[i]
        train_examples.append(InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]))


    print("[LOG] Defining training strategy")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.TripletLoss(model=model)


    num_epochs = 10
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

    print("[LOG] Start finetunig ...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps)

    model.save_to_hub(
        "CA_italian_sentence_transformer",
        exist_ok=True,
    )


if __name__ == '__main__':
    main()