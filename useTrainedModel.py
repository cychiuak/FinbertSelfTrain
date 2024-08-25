from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset,DatasetDict
import torch
import sys
sys.path.append('/helperFunction/modelPrediction.py')  # Add the parent directory of 'utils' to the system path

from helperFunction.modelPrediction import prediction  # Import the function
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the model
model = BertForSequenceClassification.from_pretrained('finbertSelfTrained')

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('finbertSelfTrained')

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# data = pd.read_csv('finBERT/data/sentiment_data/train.csv', sep='\t')
# print(data)
# print(data.shape)

# mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
# data['label'] = data['label'].map(mapping)
# print(data)
# train, test  = train_test_split(data, test_size=0.2, random_state=42)
# train_dataset = Dataset.from_pandas(train)
# test_dataset = Dataset.from_pandas(test)
# dataset_dict = DatasetDict({
#     'train': train_dataset,
#     'test': test_dataset
# })
# print(dataset_dict)


# tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
# print(tokenized_datasets)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(15))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5))

# Set the model to evaluation mode
model.eval()
# Example input


# result = []
# for i in range(len(dataset_dict["test"])):
#     result.append(prediction(dataset_dict["test"][i]["text"], model, tokenizer))
# print(result)
# label = dataset_dict["test"]["label"]
# print(label)
# print(confusion_matrix(label,result,labels=[2, 1, 0]))



# Example input

prediction("I love this!", model, tokenizer)
prediction("This stock price is about to go down!", model, tokenizer)

#positive: 2, neutral: 1, negative: 0