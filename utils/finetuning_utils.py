import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from .model_utils import calculate_metrics


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return calculate_metrics(preds, labels)


def train_with_hyperparameters(
    model, train_dataset, valid_dataset, learning_rate, batch_size, epochs, output_dir
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()

    return trainer, results


def full_finetuning(model, tokenizer, train_dataset, valid_dataset):
    print("\n" + "***** MODEL FULL FINETUNING *****")

    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_sizes = [8, 16]

    best_f1 = 0
    best_config = None

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            print(
                "\n"
                + f"Testing with Learning Rate: {learning_rate} & Batch Size: {batch_size}"
            )

            test_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=model.config.num_labels
            )

            trainer, results = train_with_hyperparameters(
                test_model,
                train_dataset,
                valid_dataset,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=3,
                output_dir=f"models/hyperparam_search_full/lr{learning_rate}_bs{batch_size}",
            )

            f1_macro = results["eval_f1_macro"]
            print(f"Validation Macro F1 Score: {f1_macro:.4f}")

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_config = {"learning_rate": learning_rate, "batch_size": batch_size}

    print("BEST HYPERPARAMETERS FOR FULL FINETUNING")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Best Validation F1: {best_f1:.4f}")

    print("\n" + "Generating Best Model......")

    final_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=model.config.num_labels
    )

    print("\n" + "Full Finetuning Parameters:")
    total_params = sum(param.numel() for param in final_model.parameters())
    trainable_params = sum(
        param.numel() for param in final_model.parameters() if param.requires_grad
    )
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(
        f"Percentage of Trainable Params: {100 * trainable_params / total_params:.2f}%"
    )

    final_args = TrainingArguments(
        output_dir="models/results_fullfinetuning",
        num_train_epochs=10,
        per_device_train_batch_size=best_config["batch_size"],
        per_device_eval_batch_size=best_config["batch_size"],
        learning_rate=best_config["learning_rate"],
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    final_trainer.train()
    final_results = final_trainer.evaluate()

    final_trainer.model.save_pretrained("models/fullfinetuning")
    tokenizer.save_pretrained("models/fullfinetuning")
    print("Model saved to models/fullfinetuning folder")

    return final_trainer.model, tokenizer, final_results, best_config


def lora_finetuning(model, tokenizer, train_dataset, valid_dataset):
    print("\n" + "***** MODEL LoRA FINETUNING *****")

    lora_ranks = [8, 16, 32]
    learning_rates = [1e-4, 2e-4, 3e-4]

    best_f1 = 0
    best_config = None

    for rank in lora_ranks:
        for learning_rate in learning_rates:
            print("\n" + f"Testing with Rank: {rank} & Learning Rate: {learning_rate}")

            test_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=model.config.num_labels
            )

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.1,
                target_modules=["q_lin", "v_lin"],
            )

            lora_model = get_peft_model(test_model, lora_config)

            trainer, results = train_with_hyperparameters(
                lora_model,
                train_dataset,
                valid_dataset,
                learning_rate=learning_rate,
                batch_size=8,
                epochs=3,
                output_dir=f"models/hyperparam_search_lora/rank{rank}_lr{learning_rate}",
            )

            f1_macro = results["eval_f1_macro"]
            print(f"Validation Macro F1 Score: {f1_macro:.4f}")

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_config = {"rank": rank, "learning_rate": learning_rate}

    print("BEST HYPERPARAMETERS FOR LORA FINETUNING")
    print(f"LoRA Rank: {best_config['rank']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Best Validation F1: {best_f1:.4f}")

    print("\n" + "Generating Best Model......")

    final_base = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=model.config.num_labels
    )

    final_lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=best_config["rank"],
        lora_alpha=best_config["rank"] * 2,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )

    final_model = get_peft_model(final_base, final_lora_config)

    print("\n" + "LoRA Finetuning Parameters:")
    final_model.print_trainable_parameters()

    final_args = TrainingArguments(
        output_dir="models/results_lorafinetuning",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=best_config["learning_rate"],
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    final_trainer.train()
    final_results = final_trainer.evaluate()

    final_trainer.model.save_pretrained("models/lorafinetuning")
    tokenizer.save_pretrained("models/lorafinetuning")
    print("Model saved to models/lorafinetuning folder")

    return final_trainer.model, tokenizer, final_results, best_config
