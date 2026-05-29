import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import OpenAI

from agents.document_assistant_agent import DocumentAssistantAgent
from agents.data_analyst_agent import DataAnalystAgent
from agents.ml_expert_agent import MLExpertAgent

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VER  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
CHAT_DEPLOY    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

ROUTING_PROMPT = """You are a router for a healthcare AI system with 3 agents:
1. document_agent  – answers questions about medical knowledge, symptoms, diabetes info, HbA1c, treatments
2. data_agent      – answers questions about dataset statistics, trends, patient counts, averages
3. ml_agent        – predicts diabetes risk when patient details (age, BMI, glucose, etc.) are provided

Based on the user message below, reply with ONLY one of:
document_agent | data_agent | ml_agent

User message: {query}
Answer:"""


class Orchestrator:
    """
    Central orchestrator that:
    1. Uses LLM to classify the user's intent
    2. Routes to the correct agent
    3. Returns the agent's response
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=AZURE_API_KEY,
            base_url=AZURE_ENDPOINT,
        )
        # Lazy-load agents (instantiated on first use to save startup time)
        self._doc_agent  = None
        self._data_agent = None
        self._ml_agent   = None

    # ── Agent accessors ───────────────────────────────────────────────────────
    @property
    def doc_agent(self):
        if self._doc_agent is None:
            print(" Loading Document Assistant Agent...")
            self._doc_agent = DocumentAssistantAgent()
        return self._doc_agent

    @property
    def data_agent(self):
        if self._data_agent is None:
            print("Loading Data Analyst Agent...")
            self._data_agent = DataAnalystAgent()
        return self._data_agent

    @property
    def ml_agent(self):
        if self._ml_agent is None:
            print("Loading ML Expert Agent...")
            self._ml_agent = MLExpertAgent()
        return self._ml_agent

    # ── Routing ───────────────────────────────────────────────────────────────
    def _route(self, query: str) -> str:
        """Ask the LLM which agent should handle this query."""
        response = self.client.chat.completions.create(
            model=CHAT_DEPLOY,
            messages=[
                {"role": "user", "content": ROUTING_PROMPT.format(query=query)}
            ],
            temperature=0,
            max_tokens=10,
        )
        decision = response.choices[0].message.content.strip().lower()
        if "data" in decision:
            return "data_agent"
        elif "ml" in decision:
            return "ml_agent"
        else:
            return "document_agent"

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, query: str, patient_data: dict = None) -> dict:
        """
        Run the orchestrator.
        - query       : user's natural-language question
        - patient_data: (optional) dict with patient fields for ml_agent

        Returns: { agent, response }
        """
        agent_name = self._route(query)

        # If patient data is provided, always use ml_agent
        if patient_data:
            agent_name = "ml_agent"

        print(f"→ Routing to: {agent_name}")

        if agent_name == "document_agent":
            response = self.doc_agent.run(query)
        elif agent_name == "data_agent":
            response = self.data_agent.run(query)
        elif agent_name == "ml_agent":
            if patient_data:
                response = self.ml_agent.run(patient_data)
            else:
                response = (
                    "To predict diabetes risk, please provide patient details: "
                    "age, gender, BMI, HbA1c level, blood glucose level, "
                    "hypertension (0/1), heart disease (0/1), smoking history."
                )
        else:
            response = "I couldn't determine which agent to use. Please rephrase your question."

        return {"agent": agent_name, "response": response}


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    orch = Orchestrator()

    tests = [
        # (query, patient_data)
        ("What is the normal HbA1c level?", None),
        ("How many patients in the dataset have diabetes?", None),
        ("What are the complications of diabetes?", None),
        ("What percentage of patients have hypertension?", None),
        ("Predict risk for this patient", {
            "gender": "Female", "age": 62, "hypertension": 1,
            "heart_disease": 1, "smoking_history": "current",
            "bmi": 35.0, "HbA1c_level": 7.8, "blood_glucose_level": 200,
        }),
    ]

    for query, patient in tests:
        print(f"\n{'='*60}")
        print(f" Query: {query}")
        result = orch.run(query, patient)
        print(f" Agent : {result['agent']}")
        print(f" Reply :\n{result['response']}")
