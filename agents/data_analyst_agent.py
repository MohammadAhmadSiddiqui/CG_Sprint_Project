import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOY    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "processed", "engineered_data.csv"
)

def get_data_summary(df):
    total       = len(df)
    diabetic    = int(df["diabetes"].sum())
    non_diab    = total - diabetic
    return f"""
Dataset Summary (Diabetes Prediction):
- Total records: {total}
- Diabetic patients: {diabetic} ({round(diabetic/total*100,1)}%)
- Non-diabetic: {non_diab} ({round(non_diab/total*100,1)}%)
- Average age: {round(df['age'].mean(),1)} years
- Average BMI: {round(df['bmi'].mean(),1)}
- Average HbA1c: {round(df['HbA1c_level'].mean(),2)}%
- Average blood glucose: {round(df['blood_glucose_level'].mean(),1)} mg/dL
- Hypertension prevalence: {round(df['hypertension'].mean()*100,1)}%
- Heart disease prevalence: {round(df['heart_disease'].mean()*100,1)}%
- Columns: {', '.join(df.columns.tolist())}
""".strip()

class DataAnalystAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=AZURE_API_KEY,
            base_url=AZURE_ENDPOINT,
        )
        self.df      = pd.read_csv(DATA_PATH)
        self.summary = get_data_summary(self.df)

    def run(self, question: str) -> str:
        system_msg = (
            "You are a Data Analyst Agent specializing in healthcare data analysis. "
            f"Use this dataset summary to answer:\n\n{self.summary}"
        )
        response = self.client.chat.completions.create(
            model=CHAT_DEPLOY,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": question},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()