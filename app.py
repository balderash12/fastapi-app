from fastapi import FastAPI
from pydantic import BaseModel
import re
import spacy
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

# Download stopwords (first-time use only)
nltk.download('stopwords')

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class TextInput(BaseModel):
    text: str

# Text normalization function
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in text.split() if w not in stop_words]
    return ' '.join(filtered_words)

# Define API endpoint
@app.post("/analyze")
async def analyze_text(data: TextInput):
    original_text = data.text
    normalized = normalize_text(original_text)

    # Named Entity Recognition
    doc = nlp(normalized)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Sentiment Analysis
    sentiment = TextBlob(normalized).sentiment
    sentiment_result = (
        "Positive" if sentiment.polarity > 0 else
        "Negative" if sentiment.polarity < 0 else
        "Neutral"
    )

    return {
        "normalized_text": normalized,
        "named_entities": entities,
        "sentiment": {
            "polarity": round(sentiment.polarity, 3),
            "subjectivity": round(sentiment.subjectivity, 3),
            "overall": sentiment_result
        }
    }
