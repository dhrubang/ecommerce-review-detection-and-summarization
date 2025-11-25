# =============================================================================
# BART Fine-tuning for Summarization using REAL human summaries (CNN/DailyMail)
# No pseudo-labels, no Yelp tricks — clean, standard, and strong baseline
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
import logging

# ----------------------------- Setup ---------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# -------------------------- 1. Load CNN/DailyMail dataset --------------------
logger.info("Loading cnn_dailymail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Use small subsets for quick testing (remove [:5000] for full training)
train_data = dataset["train"].select(range(5000))      # ~287k total available
val_data   = dataset["validation"].select(range(1000))  # ~13k total

logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

# -------------------------- 2. Tokenizer & Model -----------------------------
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move to GPU if available
model.to(device)

# -------------------------- 3. Custom Dataset Class -------------------------
class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=1024, max_target_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article  = str(self.data[idx]["article"])
        summary  = str(self.data[idx]["highlights"])

        # Tokenize input (article)
        inputs = self.tokenizer(
            article,
            max_length=self.max_input_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # Tokenize target (summary)
        targets = self.tokenizer(
            summary,
            max_length=self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": targets["input_ids"].flatten()
        }

# Create datasets
train_dataset = SummarizationDataset(train_data, tokenizer)
val_dataset   = SummarizationDataset(val_data, tokenizer)

# -------------------------- 4. Training with Hugging Face Trainer (Recommended) ----
training_args = TrainingArguments(
    output_dir="./bart_cnn_finetuned",
    num_train_epochs=3,                    # 3 is good for CNN/DM
    per_device_train_batch_size=4,         # adjust based on GPU memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch size = 16
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),        # mixed precision if GPU
    report_to=[],                          # disable wandb etc.
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

logger.info("Starting training...")
trainer.train()

# Save final model
trainer.save_model("./bart_cnn_finetuned_final")
tokenizer.save_pretrained("./bart_cnn_finetuned_final")
logger.info("Model saved to ./bart_cnn_finetuned_final")

# -------------------------- 5. Evaluation with ROUGE -------------------------
def compute_rouge(model, tokenizer, dataset, device, num_samples=500):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating ROUGE")):
            if i * 4 >= num_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

            for gen_ids, ref_ids in zip(generated_ids, batch["labels"]):
                pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
                ref  = tokenizer.decode(ref_ids, skip_special_tokens=True)
                rouge = scorer.score(ref, pred)
                for k in scores:
                    scores[k].append(rouge[k].fmeasure)

    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    return avg_scores

logger.info("Computing ROUGE scores on validation set...")
rouge_results = compute_rouge(model, tokenizer, val_dataset, device)
logger.info(f"ROUGE Results: {rouge_results}")

# -------------------------- 6. Test on a real example ------------------------
def summarize(text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=128,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example from CNN/DailyMail
sample_article = dataset["test"][0]["article"][:1000]  # first 1000 chars
print("\n" + "="*60)
print("SAMPLE ARTICLE (truncated):")
print(sample_article)
print("\nGENERATED SUMMARY:")
print(summarize(sample_article, model, tokenizer, device))
print("="*60)

# Expected ROUGE on CNN/DailyMail after 3 epochs with BART-large-cnn:
# ROUGE-1 ~41–44, ROUGE-2 ~19–21, ROUGE-L ~38–41 (very strong!)
