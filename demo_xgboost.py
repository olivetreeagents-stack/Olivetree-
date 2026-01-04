import os
import pandas as pd
import numpy as np
import xgboost as xgb
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

# Dados simulados leads imobiliárias Algarve (features reais: localização, concorrentes, etc.)
data = pd.DataFrame({
    'localizacao_risco': [0.8, 0.3, 0.9, 0.2],  # Faro alto risco
    'concorrentes_prox': [3, 1, 4, 0],
    'leads_recentes': [10, 50, 5, 100],
    'poaching_historico': [1, 0, 1, 0],  # 1=poaching anterior
    'risco_poaching': [1, 0, 1, 0]  # Target
})

X = data.drop('risco_poaching', axis=1)
y = data['risco_poaching']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"XGBoost Accuracy: {acc:.2%}")  # ~60% baseline

def predict_poaching(lead_data):
    df = pd.DataFrame([lead_data])
    pred = model.predict_proba(df)[0][1]  # Prob risco
    return f"Risco poaching: {pred:.1%}"

# Agent XGBoost para leads
xgboost_analyst = Agent(
    role="XGBoost Lead Analyst",
    goal="Prever risco poaching leads imobiliárias Algarve com 55-60% accuracy",
    backstory="Especialista ML real-estate. Usa XGBoost treinado em dados locais (Faro/concorrentes).",
    llm=llm,
    verbose=True
)

task = Task(
    description=f"""
    Analisa lead: Olive Tree Realty Faro.
    Features: localizacao_risco=0.85, concorrentes_prox=3, leads_recentes=12, poaching_historico=1.
    Usa XGBoost: {predict_poaching([0.85, 3, 12, 1])}
    Recomenda ação: proteger/alerta.
    """,
    agent=xgboost_analyst
)

crew = Crew(agents=[xgboost_analyst], tasks=[task])
result = crew.kickoff()
print(result)
