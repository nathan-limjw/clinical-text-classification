import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import Dataset


class MedicalTextSampleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_split_data():
    print("\n" + "***** LOADING SPLIT DATA *****")

    train = pd.read_csv("data/train.csv")
    valid = pd.read_csv("data/valid.csv")
    test = pd.read_csv("data/test.csv")

    print("Data Loaded")

    return train, valid, test


def load_label_encodings():
    print("\n" + "***** LOADING LABEL ENCODINGS *****")

    with open("data/label_encodings.json", "r") as f:
        label_encodings = json.load(f)

    label_mappings = {index: name for name, index in label_encodings.items()}

    print("Label encodings Loaded")
    return label_mappings


def calculate_metrics(predicted_labels, true_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1_macro = f1_score(true_labels, predicted_labels, average="macro")
    f1_weighted = f1_score(true_labels, predicted_labels, average="weighted")

    return {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def get_classification_report(predicted_labels, true_labels, label_mappings):
    print("\n" + "***** GENERATING CLASSIFICATION REPORT *****")
    class_names = [str(label_mappings[i]) for i in sorted(label_mappings.keys())]

    class_report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("Generating report........")
    return class_report


def plot_confusion_matrix(
    predicted_labels,
    true_labels,
    label_mappings,
    saved_path="outputs/confusion_matrix.png",
):
    print("\n" + "***** CREATING CONFUSION MATRIX *****")

    cm = confusion_matrix(true_labels, predicted_labels)

    class_names = [label_mappings[i] for i in sorted(label_mappings.keys())]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        fmt="d",
    )

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(saved_path, bbox_inches="tight")

    print(f"Saved as image to {saved_path}")
    plt.close()
