import os
from transformers import BartTokenizer, BartForConditionalGeneration
from flask import Flask, request, render_template, jsonify

import torch

# Check if a GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# Specify the path to the folder containing the pre-trained BART model
model_folder = "/home/wolfrey/seasonapp/checkpoint-65250"

# Load pre-trained BART tokenizer from the local folder
tokenizer = BartTokenizer.from_pretrained(
    os.path.join(model_folder, "tokenizer"),
    use_auth_token=False
)

# Load pre-trained BART model from the local folder
model = BartForConditionalGeneration.from_pretrained(
    model_folder,
    local_files_only=True
).to(device)

history = []

def summarize_text(text, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt",max_length=1024, truncation=False).to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=10, length_penalty=1.5, num_beams=5, no_repeat_ngram_size=3, early_stopping=True).to(device)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/')
def index():
    return render_template('index.html', history=history)

@app.route('/summarize', methods=['POST'])
def get_summary():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Text field is empty'})

    summary = summarize_text(text)
    history.append({'input': text, 'summary': summary})
    
    return jsonify({'summary': summary, 'history': history})

if __name__ == '__main__':
    app.run(debug=True)
