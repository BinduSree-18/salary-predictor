from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)   # Allow frontend to connect

# Load model
model = joblib.load("salary_model.pkl")

@app.route("/")
def home():
    return "Salary Predictor API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame([{
        "Age": float(data["age"]),
        "Gender": data["gender"],
        "Education Level": data["education"],
        "Job Title": data["job_title"],
        "Years of Experience": float(data["experience"])
    }])

    prediction = model.predict(input_df)

    return jsonify({
        "predicted_salary": int(prediction[0])
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)