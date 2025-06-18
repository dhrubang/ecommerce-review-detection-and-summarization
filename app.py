from flask import Flask, render_template, request
import logging
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Suppress transformer warnings
logging.getLogger("transformers.modeling_tf_utils").setLevel(logging.ERROR)

# Load model and tokenizer
model_path = "best_model_distilbert"
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Define label map
label_map = {0: 'Original Review (OR)', 1: 'Computer Generated (CG)'}

# Prediction function
def predict_text(text):
    encodings = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )
    outputs = model(encodings)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred = np.argmax(probs)
    return label_map[pred], float(probs[pred])

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        input_text = request.form['review']
        label, confidence = predict_text(input_text)
        result = {
            'text': input_text,
            'label': label,
            'confidence': f"{confidence*100:.2f}%"
        }
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
