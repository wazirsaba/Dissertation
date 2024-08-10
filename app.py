from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
import re

app = Flask(__name__)

# Load the model
model = load_model('C:/Users/fahad/OneDrive/Desktop/SA SABA/model.keras')

# Load the tokenizer
with open('C:/Users/fahad/OneDrive/Desktop/SA SABA/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    reviews = request.form['reviews'].split('\n')  # Assume each review is separated by a newline
    results = []
    for review in reviews:
        processed_review = preprocess_text(review)
        sequence = tokenizer.texts_to_sequences([processed_review])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        prediction = model.predict(padded_sequence)
        sentiment = 'Positive' if prediction[0][0] > 3 else 'Negative'
        results.append((review, sentiment))
    return render_template('index.html', reviews=results)

if __name__ == '__main__':
    app.run(debug=True)
