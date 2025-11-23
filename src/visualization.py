import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

def plot_distribution(data, title, xlabel, bins=25):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=bins, kde=True, color='steelblue', alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_countplot(labels, values, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='cadetblue', alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_boxplot(data, labels, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='navy'),
                medianprops=dict(color='red', linewidth=1.5))
    plt.title(title)
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pie_chart(sizes, labels, title):
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            wedgeprops={'width': 0.4, 'edgecolor': 'w'}, pctdistance=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def plot_heatmap(matrix_data, x_labels, y_labels, title, xlabel, ylabel, fmt='d', cmap='Blues'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_data, annot=True, fmt=fmt, cmap=cmap, cbar=False,
                xticklabels=x_labels, yticklabels=y_labels)
    plt.title(title, fontsize=14, fontweight='bold', color='#333333')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.show()
    
def plot_training_history(model, title='Training History'):
    iterations = [i * 100 for i in range(len(model.losses))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    ax1.plot(iterations, model.losses, color='tab:red', linewidth=2)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Log Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(iterations, model.train_accuracies, color='tab:blue', linewidth=2)
    ax2.set_title('Training Accuracy', fontweight='bold')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(cm, classes=['Existing', 'Attrited'], title='Confusion Matrix', cmap='Blues'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes)
    
    plt.title(title, fontweight='bold', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, auc, title='ROC Curve'):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color='darkorange', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5)', alpha=0.6)
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()