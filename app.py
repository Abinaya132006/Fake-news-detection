from flask import Flask, render_template, request
import pickle

# Create flask app
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model/svm_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    # Convert text into vector
    data = vectorizer.transform([news])

    # Prediction
    prediction = model.predict(data)

    # Confidence score
    confidence_score = abs(model.decision_function(data)[0])

    # Convert to percentage
    confidence = round(min(confidence_score * 20, 99), 2)

    # Result
    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)