from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load pre-trained XGBoost model
xgb_model = joblib.load("./xgboost.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("./vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["text"]
        
        # Preprocess the input text
        text_tfidf = vectorizer.transform([text])

        prediction = xgb_model.predict(text_tfidf)[0]

        if prediction == 0:
            text_class = "Politics"
        elif prediction == 1:
            text_class = "Sport"
        elif prediction == 2:
            text_class = "Technology"
        elif prediction == 3:
            text_class = "Entertainment"
        elif prediction == 4:
            text_class = "Business"

        return render_template("index.html", prediction=text_class)

if __name__ == "__main__":
    app.run(debug=True, port=4996)