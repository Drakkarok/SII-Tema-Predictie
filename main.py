import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import warnings

# warnings.filterwarnings('ignore')  # To handle sklearn warnings


def load_and_prepare_data():
    """Load and prepare the training and validation datasets"""
    train_df = pd.read_csv('train_split_80.csv')
    val_df = pd.read_csv('validation_split_20.csv')

    # Prepare labels
    label_map = {'fake': 0, 'biased': 1, 'true': 2}
    return train_df, val_df, label_map


def validate_model(y_true, y_pred, model_name):
    """Validate model performance and print detailed metrics"""
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n{model_name} Validation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['fake', 'biased', 'true'],
                                zero_division=1))

    return accuracy


def train_naive_bayes(train_df, val_df, label_map):
    """Train and evaluate Naive Bayes model"""
    print("\nTraining Naive Bayes...")

    # Prepare features
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df['text'])
    X_val = tfidf.transform(val_df['text'])

    y_train = train_df['label'].map(label_map)
    y_val = val_df['label'].map(label_map)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    return validate_model(y_val, predictions, "Naive Bayes")


def train_svm(train_df, val_df, label_map):
    """Train and evaluate SVM model"""
    print("\nTraining SVM...")

    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df['text'])
    X_val = tfidf.transform(val_df['text'])

    y_train = train_df['label'].map(label_map)
    y_val = val_df['label'].map(label_map)

    model = LinearSVC(
        random_state=42,
        dual=False,
        max_iter=2000,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    return validate_model(y_val, predictions, "SVM")


def train_random_forest(train_df, val_df, label_map):
    """Train and evaluate Random Forest model"""
    print("\nTraining Random Forest...")

    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df['text'])
    X_val = tfidf.transform(val_df['text'])

    y_train = train_df['label'].map(label_map)
    y_val = val_df['label'].map(label_map)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    return validate_model(y_val, predictions, "Random Forest")


def _compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'classification_report': classification_report(
            labels,
            predictions,
            target_names=['fake', 'biased', 'true'],
            output_dict=True
        )
    }


def train_bert(train_df, val_df, label_map):
    """Train and evaluate BERT model"""
    print("\nTraining CamemBERT...")
    try:
        # Initialize BERT
        tokenizer = AutoTokenizer.from_pretrained("camembert-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "camembert-base",
            num_labels=3
        )

        class NewsDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=256):
                self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        # Convert labels
        train_labels = train_df['label'].map(label_map).tolist()
        val_labels = val_df['label'].map(label_map).tolist()

        # Create datasets
        train_dataset = NewsDataset(train_df['text'].tolist(), train_labels, tokenizer)
        val_dataset = NewsDataset(val_df['text'].tolist(), val_labels, tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="steps",  # Changed from eval_strategy
            save_strategy="steps",  # Match with evaluation_strategy
            eval_steps=100,
            save_steps=100,  # Should match eval_steps
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_compute_metrics
        )

        trainer.train()
        eval_results = trainer.evaluate()

        # Get accuracy directly from eval_results
        accuracy = eval_results['eval_accuracy']  # Changed from loss approximation
        print(f"\nCamemBERT Accuracy: {accuracy:.4f}")

        # Also print the detailed classification report
        print("\nDetailed Classification Report:")
        report = eval_results['eval_classification_report']
        print(classification_report(
            y_true=val_labels,
            y_pred=trainer.predict(val_dataset).predictions.argmax(-1),
            target_names=['fake', 'biased', 'true']
        ))

        return accuracy
    except Exception as e:
        print(f"\nError in BERT training: {e}")
        return 0


def plot_results(results):
    """Plot model comparison results"""
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = list(results.values())

    sns.set_style("whitegrid")
    ax = sns.barplot(x=models, y=accuracies)

    plt.title('Model Performance on Validation Set', pad=20)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)

    for i, v in enumerate(accuracies):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nPlot saved as 'model_comparison.png'")


if __name__ == "__main__":
    # Load data
    train_df, val_df, label_map = load_and_prepare_data()

    # Store results
    results = {}

    # Train and evaluate each model
    results['Naive Bayes'] = train_naive_bayes(train_df, val_df, label_map)
    results['SVM'] = train_svm(train_df, val_df, label_map)
    results['Random Forest'] = train_random_forest(train_df, val_df, label_map)
    results['CamemBERT'] = train_bert(train_df, val_df, label_map)

    # Plot results
    plot_results(results)