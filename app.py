from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load your pre-trained XGBoost model
xgb_model = joblib.load('/path/to/your/model.pkl')

# Load the TF-IDF vectorizer
vectorizer = joblib.load('/path/to/your/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Preprocess the input text using the same vectorizer used during training
        text_tfidf = vectorizer.transform([text])

        # Make a prediction
        prediction = xgb_model.predict(text_tfidf)[0]
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)