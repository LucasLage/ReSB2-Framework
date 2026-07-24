import os
import json
import re
from datetime import datetime
from typing import List

import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    evaluation,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.training_args import BatchSamplers
import argparse
from .csv_logger_callback import CsvLoggerCallback

def load_data(work_dir, seed):

    train_csv = f"{work_dir}/data/train.csv"
    eval_csv = f"{work_dir}/data/eval.csv"
    df_train = pd.read_csv(train_csv)
    df_eval = pd.read_csv(eval_csv)

    # The MultipleNegativesSymmetricRankingLoss is trained exclusively on positive pairs.
    df_train = df_train[df_train.label == 1]
    
    # Dataset Hugging Face
    train_examples = [
        {"text1": str(row['preprocessed_text_1']),
         "text2": str(row['preprocessed_text_2'])
        }
        for _, row in df_train.iterrows()
    ]
    dt_train = Dataset.from_list(train_examples)

    # List of InputExample for evaluation
    eval_examples = [
        InputExample(
            texts=[str(row['preprocessed_text_1']), str(row['preprocessed_text_2'])],
            label=row['label']
        )
        for _, row in df_eval.iterrows()
    ]

    return dt_train, eval_examples

def train_bi_encoder(work_dir, contex_size, seed):

    dt_train, eval_examples = load_data(work_dir, seed)

    # Load the adapted model (mlm_output_model) to be fine-tuned.
    model_path=os.path.join(work_dir, "mlm_output/model")
    model = SentenceTransformer(model_path)
    # Contex size defined as pipeline param
    model.max_seq_length = contex_size

    output_dir = os.path.join(work_dir, "bi_encoder_output")
    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    # Callback to log the train process
    # Log the evaluation metrics "eval_pearson_cosine", "eval_spearman_cosine"
    csv_logger = CsvLoggerCallback(
        log_path=logging_dir,
        eval_metrics=["eval_pearson_cosine", "eval_spearman_cosine"]
    )

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,

        # IMPORTANT
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        # Do not use a batch size below 10.
        # The performance of MultipleNegativesSymmetricRankingLoss scales with batch size,
        # and reducing it will negatively affect the bi-encoder.
        
        # IMPORTANT
        batch_sampler=BatchSamplers.NO_DUPLICATES, 
        # Mechanism that ensures no item appears twice within a batch
        # Preventing accidental false negatives during in-batch negative sampling.

        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=500,
        logging_dir=logging_dir,
        report_to=["tensorboard"],
        save_strategy="no",
        eval_strategy="epoch",
        seed=seed,
        data_seed=seed, 
    )

    # Evaluator
    # Provides the eval metrics "eval_pearson_cosine", "eval_spearman_cosine"
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        eval_examples, name="eval", show_progress_bar=True
    )

    # MultipleNegativesSymmetricRankingLoss:
    # Trains only on positive pairs.
    # Uses in-batch negatives, treating all other items in the batch as negatives.
    # Includes a mechanism that ensures no item appears twice within a batch
    # Preventing accidental false negatives during in-batch negative sampling.
    train_loss = losses.MultipleNegativesSymmetricRankingLoss(model=model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dt_train,
        evaluator=evaluator,
        loss=train_loss,
        callbacks=[csv_logger]
    )

    print(f"[INFO] Starting bi-encoder training")
    trainer.train()

    # Save model
    model_save_path = os.path.join(output_dir, "model")
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)

    # Save log history
    log_history = trainer.state.log_history
    log_file_path = os.path.join(logging_dir, "training_history.json")
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(log_history, f, indent=4)

    print(f"[INFO] Bi-encoder training completed. Model and logs saved to '{output_dir}'.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT bi-encoder model")
    parser.add_argument("--work_dir", type=str, required=True, help="Work folder")
    parser.add_argument("--contex_size", type=int, required=True, help="Contex window size")
    parser.add_argument("--seed", type=int, required=True, help="Seed")
    args = parser.parse_args()

    work_dir = args.work_dir
    contex_size = args.contex_size
    seed = args.seed
    train_bi_encoder(work_dir, contex_size, seed)