import os
import csv
from datetime import datetime
from typing import List
from transformers import TrainerCallback

class CsvLoggerCallback(TrainerCallback):
    """
    A TrainerCallback that logs training and evaluation metrics to a CSV file.

    This callback captures key information from the Hugging Face Trainer's
    logging events and appends them to a structured CSV file, enabling easy
    tracking, plotting, or later analysis of training dynamics.

    Attributes
    ----------
    eval_metrics : List[str]
        A list of evaluation metric names to extract from each log event
        (e.g., ['eval_loss', 'eval_accuracy']).
    log_file_path : str
        Full path to the CSV file where logs will be written. Created automatically
        if it does not exist.
    
    Methods
    -------
    on_log(args, state, control, logs=None, **kwargs)
        Triggered whenever the Trainer emits a logging event. Extracts step,
        epoch, loss, and any user-specified evaluation metrics, writing them
        as a new row in the CSV file.
    """
    def __init__(self, log_path: str, eval_metrics: List[str] = []):
        self.eval_metrics = eval_metrics
        self.log_file_path = os.path.join(log_path, "training_log.csv")
        os.makedirs(log_path, exist_ok=True)
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'step', 'epoch', 'train_loss'] + self.eval_metrics)

    def on_log(self, args, state, control, logs=None, **kwargs):
        last_log = state.log_history[-1]
        timestamp = datetime.now().isoformat()
        step = last_log.get('step')
        epoch = last_log.get('epoch')
        loss = last_log.get('loss') if 'loss' in last_log else None
        eval_values = [last_log.get(metric) for metric in self.eval_metrics]

        if loss is not None or any(v is not None for v in eval_values):
            with open(self.log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, step, epoch, loss] + eval_values)