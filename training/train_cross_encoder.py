import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
import os
import json
import pandas as pd
import numpy as np
import re
from typing import List
from datasets import Value
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from .csv_logger_callback import CsvLoggerCallback
import argparse

def tokenize_function(batch, tokenizer, contex_size):
    # Contex size defined as pipeline param
    size = 2*contex_size

    # For performance reasons.
    # After tokenization, subword segmentation ensures that the resulting token sequence typically contains even more units than the original word count, so no semantic information is lost.
    texts1 = [' '.join(t.split()[:size]) for t in batch['preprocessed_text_1']]
    texts2 = [' '.join(t.split()[:size]) for t in batch['preprocessed_text_2']]

    # Tokenize both texts together so they are fed into the same context window.
    tokens = tokenizer(
        texts1,
        texts2,
        truncation=True,
        padding='max_length',
        max_length=size
    )

    # Ensure label shape is (batch, 1)
    if "labels" in batch:
        tokens["labels"] = np.array(batch["labels"], dtype=np.float32).reshape(-1, 1)

    return tokens


def compute_metrics(p):
    logits, labels = p

    # Convert to torch tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels).float()

    # Apply sigmoid using torch
    probs = torch.sigmoid(logits).cpu().numpy().squeeze()
    labels = labels.cpu().numpy().squeeze()

    return {
        "roc_auc": roc_auc_score(labels, probs),
        "average_precision": average_precision_score(labels, probs),
    }


def train_cross_encoder(work_dir, contex_size, seed):
    set_seed(seed)
    
    # Load the adapted model (mlm_output_model) to be fine-tuned.
    model_path=os.path.join(work_dir, "mlm_output/model")

    output_dir = os.path.join(work_dir, "cross_encoder_output")
    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    #### IMPORTANT ####
    # 
    # Setting num_labels=1 and problem_type="multi_label_classification"
    # tells Hugging Face to automatically use BCEWithLogitsLoss as loss function.
    # The framework handles the loss setup internally.
    # 
    # The BCEWithLogitsLoss is commonly recommended when training cross-encoder ranking models.
    # Where the output is a single relevance score rather than a class label.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        problem_type = "multi_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load data
    train_csv = f"{work_dir}/data/train.csv"
    eval_csv = f"{work_dir}/data/eval.csv"
    
    df_train = pd.read_csv(train_csv)
    df_eval = pd.read_csv(eval_csv)

    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)

    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")
        
    train_dataset = train_dataset.cast_column("labels", Value("float32"))
    eval_dataset = eval_dataset.cast_column("labels", Value("float32"))

    # Tokenization
    train_dataset = train_dataset.map(lambda b: tokenize_function(b, tokenizer, contex_size),
                                      batched=True, batch_size=100, num_proc=2)
    eval_dataset = eval_dataset.map(lambda b: tokenize_function(b, tokenizer, contex_size),
                                     batched=True, batch_size=100, num_proc=2)
    
    # Callback to log the train process
    # Log the evaluation metrics "eval_roc_auc", "eval_average_precision"
    eval_metrics = ["eval_roc_auc", "eval_average_precision"]
    csv_logger = CsvLoggerCallback(log_path=logging_dir, eval_metrics=eval_metrics)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs= 3,
        per_device_train_batch_size= 12,
        per_device_eval_batch_size= 12,
        learning_rate= 2e-5,
        warmup_ratio=0.1,
        weight_decay= 0.01,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=500,
        logging_dir=logging_dir,
        report_to=["tensorboard"],
        eval_strategy="epoch",
        seed=seed,
        data_seed=seed, 
    )

    # Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[csv_logger],
    )

    print(f"[INFO] Starting cross-encoder training")
    trainer.train()

    # Save model
    model_save_path = os.path.join(output_dir, "model")
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Save log history
    log_file_path = os.path.join(logging_dir, "training_history.json")
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(trainer.state.log_history, f, indent=4)

    print(f"[INFO] cross_encoder training completed. Model and logs saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT cross-encoder model")
    parser.add_argument("--work_dir", type=str, required=True, help="Work folder")
    parser.add_argument("--contex_size", type=int, required=True, help="Contex window size")
    parser.add_argument("--seed", type=int, required=True, help="Seed")
    args = parser.parse_args()

    work_dir = args.work_dir
    contex_size = args.contex_size
    seed = args.seed
    train_cross_encoder(work_dir, contex_size, seed)