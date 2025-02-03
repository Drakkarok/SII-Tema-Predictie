import pandas as pd
from sklearn.model_selection import train_test_split


def count_and_split_dataset():
    # Read the training dataset with specific settings
    # We'll use 'Text' and 'Label' as column names
    train_df = pd.read_csv('train.csv', names=['text', 'label'], header=0)

    # Count total examples and examples per class
    total_examples = len(train_df)
    class_distribution = train_df['label'].value_counts()

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total examples: {total_examples}")
    print("\nClass distribution:")
    for label, count in class_distribution.items():
        percentage = (count / total_examples) * 100
        print(f"- {label}: {count} examples ({percentage:.2f}%)")

    # Split the dataset (80% train, 20% validation)
    train_split, val_split = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,  # for reproducibility
        stratify=train_df['label']  # maintain class distribution in both splits
    )

    # Save the splits to new CSV files
    train_split.to_csv('train_split_80.csv', index=False)
    val_split.to_csv('validation_split_20.csv', index=False)

    # Print split statistics
    print(f"\nSplit Statistics:")
    print(f"Training set (80%): {len(train_split)} examples")
    print(f"Validation set (20%): {len(val_split)} examples")

    print("\nTraining set class distribution:")
    train_dist = train_split['label'].value_counts()
    for label, count in train_dist.items():
        percentage = (count / len(train_split)) * 100
        print(f"- {label}: {count} examples ({percentage:.2f}%)")

    print("\nValidation set class distribution:")
    val_dist = val_split['label'].value_counts()
    for label, count in val_dist.items():
        percentage = (count / len(val_split)) * 100
        print(f"- {label}: {count} examples ({percentage:.2f}%)")


if __name__ == "__main__":
    count_and_split_dataset()