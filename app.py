from flask import Flask, render_template, request
import joblib

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

        # Get prediction and confidence scores
        prediction = int(xgb_model.predict(text_tfidf)) 
        confidence_scores = xgb_model.predict_proba(text_tfidf)

        # Define labels
        labels = ["Politics", "Sport", "Technology", "Entertainment", "Business"]

        # Get the predicted label and confidence score
        predicted_label = labels[prediction]
        confidence_score = round(confidence_scores[0][prediction] * 100, 2)

        return render_template("index.html", prediction=predicted_label, confidence=confidence_score)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4996)