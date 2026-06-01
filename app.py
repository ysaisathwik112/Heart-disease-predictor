# Heart Disease Prediction — Flask Backend
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

RISK_INFO = {
    0: {
        "label": "Low Risk",
        "emoji": "💚",
        "advice": "Your indicators suggest a lower likelihood of heart disease. Maintain a healthy lifestyle with a balanced diet, regular exercise, and routine check-ups.",
        "color": "green"
    },
    1: {
        "label": "High Risk",
        "emoji": "🔴",
        "advice": "Your indicators suggest an elevated risk of heart disease. Please consult a cardiologist promptly. Early intervention significantly improves outcomes.",
        "color": "red"
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        fields = ["age", "sex", "chest_pain", "resting_bp", "cholesterol",
                  "fbs", "ecg", "max_hr", "exercise_ang", "st_depression",
                  "family_hist", "smoking", "diabetes"]
        features = [float(request.form[f]) for f in fields]
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0][1]
        result = RISK_INFO[int(pred)]
        result["probability"] = f"{prob * 100:.1f}%"
    except Exception as e:
        result = {"label": "Error", "emoji": "⚠️", "advice": str(e), "color": "gray", "probability": "N/A"}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
