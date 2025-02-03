import matplotlib.pyplot as plt
import numpy as np

# Results from all models
results = {
    'Naive Bayes': {
        'overall': 0.6615,
        'metrics': {
            'fake': {'precision': 0.86, 'recall': 0.24, 'f1': 0.38},
            'biased': {'precision': 1.00, 'recall': 0.00, 'f1': 0.00},
            'true': {'precision': 0.65, 'recall': 0.98, 'f1': 0.78}
        }
    },
    'SVM': {
        'overall': 0.7821,
        'metrics': {
            'fake': {'precision': 0.74, 'recall': 0.74, 'f1': 0.74},
            'biased': {'precision': 0.44, 'recall': 0.22, 'f1': 0.29},
            'true': {'precision': 0.83, 'recall': 0.92, 'f1': 0.87}
        }
    },
    'Random Forest': {
        'overall': 0.7179,
        'metrics': {
            'fake': {'precision': 0.74, 'recall': 0.50, 'f1': 0.60},
            'biased': {'precision': 0.50, 'recall': 0.04, 'f1': 0.07},
            'true': {'precision': 0.72, 'recall': 0.95, 'f1': 0.82}
        }
    },
    'CamemBERT': {
        'overall': 0.7718,
        'metrics': {
            'fake': {'precision': 0.69, 'recall': 0.75, 'f1': 0.72},
            'biased': {'precision': 0.41, 'recall': 0.40, 'f1': 0.40},
            'true': {'precision': 0.89, 'recall': 0.86, 'f1': 0.88}
        }
    }
}


def plot_detailed_metrics():
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Model Performance Metrics', fontsize=16, y=0.95)

    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Plot overall accuracy
    overall_acc = [results[model]['overall'] for model in results]
    ax = axes[0, 0]
    bars = ax.bar(results.keys(), overall_acc, color=colors)
    ax.set_title('Overall Accuracy')
    ax.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}', ha='center', va='bottom')
    ax.tick_params(axis='x', rotation=45)

    # Plot precision by class
    ax = axes[0, 1]
    x = np.arange(len(results))
    width = 0.25

    for i, class_name in enumerate(['fake', 'biased', 'true']):
        precisions = [results[model]['metrics'][class_name]['precision'] for model in results]
        ax.bar(x + i * width, precisions, width, label=class_name)

    ax.set_title('Precision by Class')
    ax.set_xticks(x + width)
    ax.set_xticklabels(results.keys(), rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot recall by class
    ax = axes[1, 0]
    for i, class_name in enumerate(['fake', 'biased', 'true']):
        recalls = [results[model]['metrics'][class_name]['recall'] for model in results]
        ax.bar(x + i * width, recalls, width, label=class_name)

    ax.set_title('Recall by Class')
    ax.set_xticks(x + width)
    ax.set_xticklabels(results.keys(), rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot F1 score by class
    ax = axes[1, 1]
    for i, class_name in enumerate(['fake', 'biased', 'true']):
        f1_scores = [results[model]['metrics'][class_name]['f1'] for model in results]
        ax.bar(x + i * width, f1_scores, width, label=class_name)

    ax.set_title('F1 Score by Class')
    ax.set_xticks(x + width)
    ax.set_xticklabels(results.keys(), rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('detailed_metrics_comparison.png')
    print("Detailed metrics visualization saved as 'detailed_metrics_comparison.png'")


if __name__ == "__main__":
    plot_detailed_metrics()