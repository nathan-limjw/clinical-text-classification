# Importing Packages

import json
import os

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_utils import (
    encode_labels,
    filter_classes,
    remove_empty_transcription,
    split_data,
)
from utils.evaluate_utils import evaluate_model
from utils.finetuning_utils import full_finetuning, lora_finetuning
from utils.model_utils import (
    MedicalTextSampleDataset,
    get_classification_report,
    load_label_encodings,
    load_split_data,
    plot_confusion_matrix,
)

os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Preprocessing

df = pd.read_csv("/Users/nathan/Repos/clinical-text-classification/data/mtsamples.csv")

df.head()
print(f"{len(df)} entries loaded")

df = filter_classes(df)
df = remove_empty_transcription(df)
df, label_encodings = encode_labels(df)
train, valid, test = split_data(df)

train.to_csv("data/train.csv", index=False)
valid.to_csv("data/valid.csv", index=False)
test.to_csv("data/test.csv", index=False)

with open("data/label_encodings.json", "w") as f:
    json.dump(label_encodings, f, indent=2)

print("Data Split and Label Encodings saved under /data")


# Data Loading

train_df, valid_df, test_df = load_split_data()
label_encodings = load_label_encodings()
num_labels = len(set(label_encodings.values()))

label_map = {index: name for name, index in label_encodings.items()}

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = MedicalTextSampleDataset(
    train_df["text"].values, train_df["label"].values, tokenizer, max_length=512
)

valid_dataset = MedicalTextSampleDataset(
    valid_df["text"].values, valid_df["label"].values, tokenizer, max_length=512
)

test_dataset = MedicalTextSampleDataset(
    test_df["text"].values, test_df["label"].values, tokenizer, max_length=512
)

print(
    f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}"
)


# Full Finetuning

full_base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)

full_model, full_tokenizer, full_valid_results, full_best_hyperparam = full_finetuning(
    full_base_model, tokenizer, train_dataset, valid_dataset
)

# LoRA Finetuning

lora_base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)

lora_model, lora_tokenizer, lora_valid_results, lora_best_hyperparam = lora_finetuning(
    lora_base_model, tokenizer, train_dataset, valid_dataset
)

# Model Evaluation

full_test_metrics, full_preds, full_labels = evaluate_model(full_model, test_dataset)

full_report = get_classification_report(full_preds, full_labels, label_map)
with open("output/full_test_report.txt", "w") as f:
    f.write("Full Finetuning Test Report")
    f.write(full_report)

plot_confusion_matrix(
    full_preds, full_labels, label_map, saved_path="output/confusion_matrix_full.png"
)


lora_test_metrics, lora_preds, lora_labels = evaluate_model(lora_model, test_dataset)

lora_report = get_classification_report(lora_preds, lora_labels, label_map)
with open("output/lora_test_report.txt", "w") as f:
    f.write("Lora Finetuning Test Report")
    f.write(lora_report)

plot_confusion_matrix(
    lora_preds, lora_labels, label_map, saved_path="output/confusion_matrix_lora.png"
)


# Summary

print("\n" + "Full Finetuning Best Hyperparam")
print(f"Learning Rate: {full_best_hyperparam['learning_rate']}")
print(f"Batch Size: {full_best_hyperparam['batch_size']}")

print("\n" + "Lora Finetuning Best Hyperparam")
print(f"Lora Rank: {lora_best_hyperparam['rank']}")
print(f"Learning Rate: {lora_best_hyperparam['learning_rate']}")


print("\n" + "Full Finetuning Validation Results")
print(f"Accuracy: {full_valid_results['eval_accuracy']:.4f}")
print(f"Macro F1: {full_valid_results['eval_f1_macro']:.4f}")
print(f"Weighted F1: {full_valid_results['eval_f1_weighted']:.4f}")

print("\n" + "Lora Finetuning Validation Results")
print(f"Accuracy: {lora_valid_results['eval_accuracy']:.4f}")
print(f"Macro F1: {lora_valid_results['eval_f1_macro']:.4f}")
print(f"Weighted F1: {lora_valid_results['eval_f1_weighted']:.4f}")


print("\n" + "Full Finetuning Test Results")
print(f"Accuracy: {full_test_metrics['eval_accuracy']:.4f}")
print(f"Macro F1: {full_test_metrics['eval_f1_macro']:.4f}")
print(f"Weighted F1: {full_test_metrics['eval_f1_weighted']:.4f}")

print("\n" + "Lora Finetuning Test Results")
print(f"Accuracy: {lora_test_metrics['eval_accuracy']:.4f}")
print(f"Macro F1: {lora_test_metrics['eval_f1_macro']:.4f}")
print(f"Weighted F1: {lora_test_metrics['eval_f1_weighted']:.4f}")
