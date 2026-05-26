import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VER  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
CHAT_DEPLOY    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Smoking history encoding map (matches training encoding)
SMOKING_MAP = {"No Info": 0, "never": 4, "former": 1, "current": 2, "ever": 3, "not current": 5}
GENDER_MAP  = {"Female": 0, "Male": 1, "Other": 2}

# BMI category and glucose risk (feature engineering from notebook)
def bmi_category(bmi):
    if bmi < 18.5: return 0      # Underweight
    elif bmi < 25: return 1      # Normal
    elif bmi < 30: return 2      # Overweight
    else:          return 3      # Obese

def glucose_risk(glucose):
    return 1 if glucose >= 140 else 0


class MLExpertAgent:
    """
    Agent that:
    1. Takes patient data as input
    2. Runs diabetes prediction via the trained Random Forest model
    3. Returns an LLM-generated plain-English explanation of the result
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=AZURE_API_KEY,
            base_url=AZURE_ENDPOINT,
        )
        with open(os.path.join(MODELS_DIR, "diabetes_model.pkl"), "rb") as f:
            self.model = pickle.load(f)

    def predict(self, patient: dict) -> dict:
        """
        Run model prediction.
        patient dict keys: gender, age, hypertension, heart_disease,
                           smoking_history, bmi, HbA1c_level, blood_glucose_level
        Returns: { prediction, probability, patient_features }
        """
        gender  = GENDER_MAP.get(patient.get("gender", "Female"), 0)
        smoking = SMOKING_MAP.get(patient.get("smoking_history", "never"), 4)
        bmi_val = float(patient["bmi"])
        glucose = float(patient["blood_glucose_level"])

        features = pd.DataFrame([{
            "gender":             gender,
            "age":                float(patient["age"]),
            "hypertension":       int(patient["hypertension"]),
            "heart_disease":      int(patient["heart_disease"]),
            "smoking_history":    smoking,
            "bmi":                bmi_val,
            "HbA1c_level":        float(patient["HbA1c_level"]),
            "blood_glucose_level": glucose,
            "bmi_category":       bmi_category(bmi_val),
            "glucose_risk":       glucose_risk(glucose),
        }])

        pred  = int(self.model.predict(features)[0])
        proba = round(float(self.model.predict_proba(features)[0][1]) * 100, 1)
        return {"prediction": pred, "probability": proba, "features": patient}

    def run(self, patient: dict) -> str:
        """Predict + generate plain-English explanation via LLM."""
        result = self.predict(patient)
        pred   = result["prediction"]
        proba  = result["probability"]

        system_msg = (
            "You are an ML Expert Agent for a healthcare AI platform. "
            "Explain a diabetes prediction result in simple, empathetic language. "
            "Include key risk factors from the patient profile and suggest next steps. "
            "Keep it under 150 words."
        )
        user_msg = (
            f"Patient profile: {patient}\n"
            f"Model prediction: {'DIABETIC' if pred == 1 else 'NOT DIABETIC'}\n"
            f"Diabetes probability: {proba}%\n"
            "Explain this result to the patient in simple language."
        )
        response = self.client.chat.completions.create(
            model=CHAT_DEPLOY,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        explanation = response.choices[0].message.content.strip()
        return f"Prediction: {'DIABETIC ' if pred==1 else 'NOT DIABETIC '} ({proba}% risk)\n\n{explanation}"


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = MLExpertAgent()

    patient1 = {
        "gender": "Female", "age": 55, "hypertension": 1,
        "heart_disease": 0, "smoking_history": "former",
        "bmi": 33.5, "HbA1c_level": 7.2, "blood_glucose_level": 180,
    }
    print("=== Patient 1 ===")
    print(agent.run(patient1))

    patient2 = {
        "gender": "Male", "age": 28, "hypertension": 0,
        "heart_disease": 0, "smoking_history": "never",
        "bmi": 22.0, "HbA1c_level": 5.2, "blood_glucose_level": 90,
    }
    print("\n=== Patient 2 ===")
    print(agent.run(patient2))
