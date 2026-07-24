import os
import json
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import Dataset
from .csv_logger_callback import CsvLoggerCallback
import argparse

def train_mlm(work_dir, base_model, contex_size, seed):
    set_seed(seed)
    
    csv_camara_path = f"{work_dir}/data/sb2_chamber_mlm.csv"
    df_data_camara = pd.read_csv(csv_camara_path)


    csv_almg_path = f"{work_dir}/data/sb2_almg_mlm.csv"
    df_data_almg = pd.read_csv(csv_almg_path)

    texts = (
        df_data_camara['preprocessed_text'].tolist() +
        df_data_almg['preprocessed_text'].tolist()
    )
    dataset = Dataset.from_dict({"text": texts})
    

    # Base model defined as pipeline param
    model_path = base_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length",
            # Contex size defined as pipeline param
            max_length=contex_size
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Model
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    # Data collator
    # DataCollatorForLanguageModeling is a Hugging Face utility that prepares batches for language-model training.      # In masked-language modeling (MLM), a DataCollator selects random tokens in each batch to mask during training.
    # it creates the corresponding labels, and returns inputs where the model must predict the masked tokens.
    # This makes masking different at every step, improving generalization.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # CSV Logger Callback
    output_dir = os.path.join(work_dir, "mlm_output")
    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    csv_logger = CsvLoggerCallback(log_path=logging_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=500,
        logging_dir=logging_dir,
        report_to=["tensorboard"],
        save_strategy="no",
        seed=seed,
        data_seed=seed, 
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[csv_logger]
    )

    print(f"[INFO] Starting MLM training")
    trainer.train()

    # Save model
    model_save_path = os.path.join(output_dir, "model")
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)

    # Save log history
    log_history = trainer.state.log_history
    log_file_path = os.path.join(logging_dir, "training_history.json")
    os.makedirs(training_args.logging_dir, exist_ok=True)
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(log_history, f, indent=4)

    print(f"[INFO] MLM training completed. Model and logs saved to  '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLM BERT model")
    parser.add_argument("--work_dir", type=str, required=True, help="Work folder")
    parser.add_argument("--base_model", type=str, required=True, help="Base BERT model")
    parser.add_argument("--contex_size", type=int, required=True, help="Contex window size")
    parser.add_argument("--seed", type=int, required=True, help="Seed")
    args = parser.parse_args()

    work_dir = args.work_dir
    base_model = args.base_model
    contex_size = args.contex_size
    seed = args.seed
    
    train_mlm(work_dir, base_model, contex_size, seed)
