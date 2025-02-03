import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import shutil
import os
import re


class NewsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length)

        item = {key: torch.tensor(val) for key, val in encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item


def read_custom_csv(file_path):
    """Read the custom CSV format where entries are separated by "," and newline"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip header
    content = content.split('\n', 1)[1]

    # Split by the pattern of quote, comma, newline, and expecting a quote after
    entries = re.findall(r'"(.*?)",\s*(?="|$)', content, re.DOTALL)

    return entries


def train_and_save_model(train_texts, train_labels, model_save_path):
    """Train the model on full dataset and save it"""
    print("Training CamemBERT on full dataset...")

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=3)

    # Prepare labels
    label_map = {'fake': 0, 'biased': 1, 'true': 2}
    train_labels_idx = [label_map[label] for label in train_labels]

    # Create dataset
    train_dataset = NewsDataset(train_texts, train_labels_idx, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./temp_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./temp_logs',
        save_strategy="epoch",
        report_to="none",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model saved to {model_save_path}")
    return tokenizer, model


def make_predictions(texts, model_path):
    """Make predictions on test data"""
    print("Making predictions...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Create test dataset
    test_dataset = NewsDataset(texts, tokenizer=tokenizer)

    # Initialize trainer
    trainer = Trainer(model=model)

    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1)

    # Convert numeric predictions back to labels
    label_map = {0: 'fake', 1: 'biased', 2: 'true'}
    return [label_map[pred] for pred in pred_labels]


def save_predictions(texts, predictions, output_file):
    """Save predictions in the same format as input"""
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write('Text,Label\n')

        # Write entries
        for text, pred in zip(texts, predictions):
            f.write(f'"{text}",{pred}\n')


def main():
    MODEL_PATH = "./temp_bert_model"

    # Read datasets
    print("Reading datasets...")
    train_texts = read_custom_csv('train.csv')
    # Get labels from the last column of train.csv
    with open('train.csv', 'r', encoding='utf-8') as f:
        train_labels = [line.strip().split(',')[-1] for line in f if ',' in line][1:]  # Skip header

    test_texts = read_custom_csv('test.csv')

    # Train model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        tokenizer, model = train_and_save_model(train_texts, train_labels, MODEL_PATH)

    # Make predictions
    predictions = make_predictions(test_texts, MODEL_PATH)

    # Save predictions
    save_predictions(test_texts, predictions, 'test_predictions.csv')

    # Clean up temporary files
    print("Cleaning up temporary files...")
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    if os.path.exists("./temp_results"):
        shutil.rmtree("./temp_results")
    if os.path.exists("./temp_logs"):
        shutil.rmtree("./temp_logs")
    print("Cleanup complete!")


if __name__ == "__main__":
    main()