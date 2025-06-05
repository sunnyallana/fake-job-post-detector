from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import re
from nltk.corpus import stopwords
import nltk
import numpy as np

app = Flask(__name__)
CORS(app)

# Download required NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Model definition (must match training exactly)
class FakeJobLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        pooled, _ = torch.max(lstm_out, dim=1)
        x = self.dropout(pooled)
        x = torch.relu(self.fc_hidden(x))
        x = self.fc_out(x)
        return x.squeeze(1)

# Text preprocessing functions (must match training)
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = remove_stopwords(text)
    return text

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Global variables for model and vocab
model = None
vocab = None

def load_model():
    global model, vocab
    # Load vocabulary (saved during training)
    vocab = torch.load('vocab.pth')  # Make sure this file exists
    
    # Initialize model with correct parameters
    model = FakeJobLSTM(vocab_size=len(vocab))
    
    # Load trained weights
    model.load_state_dict(torch.load('fake_job_model.pth', map_location=torch.device('cpu')))
    model.eval()

def encode(text, vocab, max_len=1024):
    tokens = tokenize(text)
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_len]]
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids += [vocab["<PAD>"]] * (max_len - len(token_ids))
    return token_ids

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or vocab is None:
            load_model()
        
        data = request.json
        job_text = data['text']
        
        # Preprocess text (must match training pipeline)
        cleaned_text = clean_text(job_text)
        encoded_text = encode(cleaned_text, vocab)
        
        # Convert to tensor and predict
        input_tensor = torch.tensor([encoded_text], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = int(probability >= 0.5)
            
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)