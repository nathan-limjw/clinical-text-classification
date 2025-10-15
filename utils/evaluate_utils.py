import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model_utils import calculate_metrics


def evaluate_model(model, test_dataset):
    print("\n" + "Evaluating on test dataset......")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    metrics = calculate_metrics(all_predictions, all_labels)

    print("\n" + "Test Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")

    return metrics, all_predictions, all_labels
