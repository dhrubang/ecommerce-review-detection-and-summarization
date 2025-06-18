from flask import Flask, render_template, request
import pandas as pd
import logging
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
import torch

# Suppress transformer warnings
logging.getLogger("transformers.modeling_tf_utils").setLevel(logging.ERROR)

# Load DistilBERT model and tokenizer for AI detection
distilbert_model_path = "best_model_distilbert"
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(distilbert_model_path)
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_path)

# Load T5 model and tokenizer for summarization
t5_model_path = "t5_model_summarization"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)

# Define label map for AI detection
label_map = {0: 'Original Review (OR)', 1: 'Computer Generated (CG)'}

# Prediction function for AI detection
def predict_text(text):
    encodings = distilbert_tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )
    outputs = distilbert_model(encodings)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred = np.argmax(probs)
    return label_map[pred], float(probs[pred])

# Summarization function for original reviews
def summarize_reviews(reviews, max_length=128):
    if not reviews:
        return "No original reviews available to summarize."
    
    # Combine all original reviews into a single text
    combined_text = " ".join(reviews)
    if len(combined_text.strip()) < 20:
        return "Original reviews are too short to summarize."
    
    # Tokenize and generate summary
    inputs = t5_tokenizer(combined_text, max_length=512, truncation=True, return_tensors='pt').to(device)
    summary_ids = t5_model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

app = Flask(__name__)

# Load the product data
df = pd.read_csv('flipkart_product1.csv')
df['Summary'] = df['Summary'].fillna('No review text available')
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce').clip(0, 5).fillna(0)

@app.route('/', methods=['GET', 'POST'])
def index():
    product_names = sorted(df['ProductName'].unique().tolist())
    selected_product = None
    reviews = []
    avg_rating = 0
    original_count = 0
    generated_count = 0
    original_summary = ""

    if request.method == 'POST':
        selected_product = request.form.get('product_name')
        if selected_product:
            product_reviews = df[df['ProductName'] == selected_product]
            reviews_data = product_reviews[['Summary', 'Rate']].sort_values('Rate', ascending=False)
            
            # Analyze each review and collect original reviews for summarization
            analyzed_reviews = []
            original_reviews = []
            for _, row in reviews_data.iterrows():
                review_text = row['Summary']
                label, confidence = predict_text(review_text)
                
                # Count review types and collect original reviews
                if "Original" in label:
                    original_count += 1
                    review_class = "original"
                    original_reviews.append(review_text)
                else:
                    generated_count += 1
                    review_class = "generated"
                
                analyzed_reviews.append({
                    'text': review_text,
                    'rate': row['Rate'],
                    'type': label,
                    'confidence': f"{confidence*100:.2f}%",
                    'class': review_class
                })
            
            reviews = analyzed_reviews
            
            # Generate summary of original reviews
            original_summary = summarize_reviews(original_reviews)
            
            if len(reviews) > 0:
                avg_rating = round(product_reviews['Rate'].mean(), 1)
    
    return render_template('index1.html', 
                         product_names=product_names,
                         selected_product=selected_product,
                         reviews=reviews,
                         avg_rating=avg_rating,
                         review_count=len(reviews),
                         original_count=original_count,
                         generated_count=generated_count,
                         original_summary=original_summary)

if __name__ == '__main__':
    app.run(debug=True)