from transformers import pipeline
import torch
import pandas as pd
classifier = pipeline('sentiment-analysis')
print(classifier('We are very happy to show you the 🤗 Transformers library.'))
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=3)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)



data = pd.read_csv('finBERT/data/sentiment_data/train.csv', sep='\t')
print(data)
print(data.shape)

mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
data['label'] = data['label'].map(mapping)
print(data)
train, test  = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)



dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})
print(dataset_dict)


tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
print(tokenized_datasets)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(15))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5))


training_args = TrainingArguments(output_dir="test_trainer"
                                  ,num_train_epochs=5
                                #   ,num_train_epochs=1
                                  , per_device_train_batch_size=8
                                  , per_device_eval_batch_size=8
                                  , warmup_steps=500
                                  , weight_decay=0.01
                                  ,learning_rate=2e-5)


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch", push_to_hub=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= tokenized_datasets["train"],
    eval_dataset= tokenized_datasets["test"],
    # train_dataset=small_train_dataset,
    # eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
# pt_model = AutoModelForSequenceClassification.from_pretrained("test_trainer", from_tf=True)
# pt_model.save_pretrained("test_trainer") 


predict = trainer.predict(tokenized_datasets["test"])

print(confusion_matrix(predict.label_ids,np.argmax(predict.predictions, axis=-1),labels=[2, 1, 0]))
# report = classification_report(predict.label_ids, np.argmax(predict.predictions, axis=-1), target_names=[2, 1, 0])
# print(report)
model.save_pretrained("finbertSelfTrained")
tokenizer.save_pretrained("finbertSelfTrained")

# print

