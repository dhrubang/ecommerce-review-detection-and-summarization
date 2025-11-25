# =============================================================================
# T5 Fine-tuning for Summarization on CNN/DailyMail (Real Human Summaries)
# Computes ROUGE-1/2/L + BLEU-4
# Uses Hugging Face Trainer — fast, clean, strong results
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
import logging

# ----------------------------- Setup ---------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# -------------------------- 1. Load CNN/DailyMail (best dataset for summarization) ----------
logger.info("Loading cnn_dailymail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Small subset for fast testing — REMOVE these lines for full training!
train_data = dataset["train"].select(range(8000))      # Use all: remove .select()
val_data   = dataset["validation"].select(range(1000))

logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

# -------------------------- 2. Load T5 Model & Tokenizer ---------------------
model_name = "t5-large"                    # t5-large > bart-large-cnn on CNN/DM
# Alternatives: "t5-base", "t5-small", "google/flan-t5-large" (even stronger)

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# Important: T5 expects prefix "summarize: "
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

    # Tokenize targets
    targets = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = targets["input_ids"]
    # Replace padding token id in labels (-100 so it's ignored by loss)
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in model_inputs["labels"]
    ]
    return model_inputs

# Apply preprocessing
tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
tokenized_val   = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)

# Data collator handles dynamic padding (much faster than fixed padding)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -------------------------- 3. Training Arguments --------------------------------
training_args = TrainingArguments(
    output_dir="./t5_cnn_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,           # Effective batch size = 32
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    report_to=[],                            # Disable wandb
    predict_with_generate=True,              # Needed for ROUGE/BLEU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------------- 4. Train! ---------------------------------------
logger.info("Starting training with T5...")
trainer.train()

# Save final model
trainer.save_model("./t5_cnn_finetuned_final")
tokenizer.save_pretrained("./t5_cnn_finetuned_final")
logger.info("Model saved to ./t5_cnn_finetuned_final")

# -------------------------- 5. Evaluation: ROUGE + BLEU ----------------------
import nltk
nltk.download('punkt', quiet=True)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

def compute_metrics(dataset, num_samples=500):
    model.eval()
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    bleu_scores = []

    loader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            if i >= num_samples // 4:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            generated_ids = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs  = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            for pred, ref in zip(preds, refs):
                # ROUGE
                scores = scorer.score(ref, pred)
                for k in rouge_scores:
                    rouge_scores[k].append(scores[k].fmeasure)

                # BLEU
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]
                bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
                bleu_scores.append(bleu)

    result = {
        "ROUGE-1": np.mean(rouge_scores["rouge1"]),
        "ROUGE-2": np.mean(rouge_scores["rouge2"]),
        "ROUGE-L": np.mean(rouge_scores["rougeL"]),
        "BLEU-4": np.mean(bleu_scores)
    }
    return result

logger.info("Computing final metrics...")
metrics = compute_metrics(tokenized_val, num_samples=500)
for k, v in metrics.items():
    logger.info(f"{k}: {v:.4f}")

# -------------------------- 6. Test on Sample --------------------------------
def summarize(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=128,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Test
sample = dataset["test"][5]["article"][:1500]
print("\n" + "="*80)
print("ARTICLE (truncated):")
print(sample)
print("\nREFERENCE SUMMARY:")
print(dataset["test"][5]["highlights"])
print("\nT5 GENERATED SUMMARY:")
print(summarize(sample))
print("="*80)

# Expected Results after 3 epochs on full data:
# ROUGE-1: ~44.5   ROUGE-2: ~21.5   ROUGE-L: ~41.5   BLEU-4: ~20–22
