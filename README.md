# Fake Job Posting Detection System

A comprehensive AI-powered system to detect fraudulent job postings using deep learning techniques. The system combines web scraping, natural language processing, and a bidirectional LSTM neural network to identify suspicious job postings with high accuracy.

![Screenshot_2025-06-05_10-54-01](https://github.com/user-attachments/assets/c1b51707-dee4-4657-b091-4307d6910e3d)

![Screenshot_2025-06-05_10-53-41](https://github.com/user-attachments/assets/2c163a1d-53f3-46d1-90c6-2b9984eb13cf)


## Project Overview

This project aims to protect job seekers from fraudulent job postings by automatically analyzing job descriptions and determining their legitimacy. The system uses advanced NLP techniques and a deep learning model trained on thousands of job postings to identify suspicious patterns and indicators.

## Architecture

![architecture](https://github.com/user-attachments/assets/bd92ae1b-1be3-4ddb-8e77-cd3c0d5b049b)


The system follows a modular architecture with the following components:

### Machine Learning Pipeline
- **Data Preprocessing**: Text cleaning, tokenization, and feature extraction
- **Model**: Bidirectional LSTM with embedding layer and dense layers
- **Training**: Binary classification with BCE loss and Adam optimizer
- **Evaluation**: Precision, Recall, F1-Score, and ROC-AUC metrics

### System Components
1. **Web Scraper** (`crawler.py`) - Automated job posting collection from Rozee.pk
2. **ML Model** (`app.py`) - Flask API serving the trained LSTM model
3. **Frontend** - React.js interface for user interaction
4. **Data Processing** - Text preprocessing and feature engineering pipeline

## Features

- **Real-time Analysis**: Instant detection of fake job postings
- **High Accuracy**: Achieves over 95% accuracy in identifying fraudulent postings
- **User-friendly Interface**: Clean, modern React.js frontend
- **Advanced NLP**: Bidirectional LSTM with GloVe embeddings
- **Automated Scraping**: Continuous data collection for model improvement
- **Comprehensive Analysis**: Detailed confidence scores and explanations

## Model Performance

- **Accuracy**: 95%+
- **Architecture**: Bidirectional LSTM
- **Embedding Dimension**: 100
- **Hidden Dimension**: 128
- **Vocabulary Size**: 20K tokens
- **Max Sequence Length**: 1024 tokens

## Technology Stack

### Backend
- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **Flask** - Web API framework
- **NLTK** - Natural language processing
- **Selenium** - Web scraping
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

### Frontend
- **React.js** - User interface
- **HTML/CSS** - Styling and layout
- **JavaScript** - Frontend logic

### Machine Learning
- **PyTorch** - Neural network implementation
- **NLTK** - Text preprocessing
- **Scikit-learn** - Model evaluation
- **GloVe** - Pre-trained word embeddings

## 📋 Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 14+ and npm
- Chrome browser (for web scraping)
- ChromeDriver

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fake-job-detector.git
cd fake-job-detector
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install torch torchvision torchaudio
pip install flask flask-cors
pip install nltk numpy pandas
pip install selenium beautifulsoup4
pip install scikit-learn
```

4. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

5. **Setup ChromeDriver**
- Download ChromeDriver from https://chromedriver.chromium.org/
- Add ChromeDriver to your PATH

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Install required packages**
```bash
npm install axios react-router-dom
```

## Usage

### Starting the Backend API

1. **Ensure model files are present**
```bash
# Make sure these files exist in your project directory:
# - fake_job_model.pth (trained model weights)
# - vocab.pth (vocabulary file)
```

2. **Start Flask API**
```bash
python flask_api.py
```
The API will be available at `http://localhost:5000`

### Starting the Frontend

1. **Start React development server**
```bash
cd frontend
npm start
```
The frontend will be available at `http://localhost:3000`

### Web Scraping (Optional)

To collect new job postings for training:

```bash
python scraper.py
```

## API Endpoints

### POST /predict
Analyzes a job posting and returns fraud detection results.

**Request:**
```json
{
  "text": "Software Engineer position at Google. Responsibilities include..."
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.15,
  "status": "success"
}
```

- `prediction`: 0 for legitimate, 1 for fake
- `probability`: Confidence score (0-1)
- `status`: Request status

## Data Processing Pipeline

### 1. Data Collection
- Automated scraping from job portals
- Structured data extraction
- Duplicate removal and validation

### 2. Text Preprocessing
- HTML tag removal
- Special character cleaning
- Stopword removal
- Tokenization and normalization

### 3. Feature Engineering
- Text concatenation (title, description, requirements)
- Vocabulary building (20K most frequent words)
- Sequence padding/truncation
- Token encoding

### 4. Model Training
- Train/validation split (80/20)
- Batch processing (batch size: 64)
- Learning rate: 1e-3
- Epochs: 100 with early stopping

## Model Architecture Details

```python
class FakeJobLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim * 2, 1)
```

**Key Features:**
- Bidirectional LSTM for context understanding
- Global max pooling for feature extraction
- Dense layers with ReLU activation
- Dropout for regularization
- Sigmoid activation for binary classification

## Performance Metrics

The model is evaluated using multiple metrics:

- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## How It Works

1. **Input**: User pastes job posting text into the web interface
2. **Preprocessing**: Text is cleaned and tokenized
3. **Encoding**: Tokens are converted to numerical sequences
4. **Prediction**: LSTM model analyzes the sequence
5. **Output**: System returns legitimacy score and recommendation

## Security Features

- **Pattern Recognition**: Identifies common fraud indicators
- **Linguistic Analysis**: Detects suspicious language patterns
- **Contextual Understanding**: Analyzes job posting context
- **Confidence Scoring**: Provides reliability metrics

## 🔧 Configuration

### Model Parameters
- `vocab_size`: 20000
- `embed_dim`: 100
- `hidden_dim`: 128
- `num_layers`: 1
- `dropout`: 0.3
- `max_len`: 1024

### API Configuration
- `host`: localhost
- `port`: 5000
- `debug`: True (development)

## Training Data

The model is trained on a comprehensive dataset containing:
- Legitimate job postings from reputable sources
- Known fraudulent job postings
- Balanced dataset with proper validation split
- Comprehensive feature extraction

## Deployment

### Development
```bash
# Backend
python app.py

# Frontend
npm run dev
```

### Production
Consider using:
- **Backend**: Gunicorn, Docker, AWS/GCP
- **Frontend**: Nginx, Vercel, Netlify
- **Database**: PostgreSQL, MongoDB
- **Monitoring**: Prometheus, Grafana

---

**Disclaimer**: This tool is designed to assist in identifying potentially fraudulent job postings but should not be the sole factor in decision-making. Always conduct proper due diligence when applying for jobs.
