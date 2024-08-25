import torch

def prediction(text, model, tokenizer):
  # Tokenize the input
  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

  # Get predictions
  with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)

  print("Predicted class:", predictions.item())
  return predictions.item()

def tokenize_function(examples,tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
