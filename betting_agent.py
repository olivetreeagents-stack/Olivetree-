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

# Dados simulados football Portugal (features: form, H2H, odds)
np.random.seed(42)
data = pd.DataFrame({
    'home_form': np.random.uniform(0, 1, 1000),
    'away_form': np.random.uniform(0, 1, 1000),
    'h2h_home': np.random.uniform(0, 1, 1000),
    'odds_home': np.random.uniform(1.5, 4.0, 1000),
    'value_bet': np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # 30% value bets
})

X = data.drop('value_bet', axis=1)
y = data['value_bet']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"XGBoost Value Betting Accuracy: {acc:.1%}")

def predict_value_bet(match_data):
    df = pd.DataFrame([match_data])
    prob = model.predict_proba(df)[0][1]
    ev = prob * (df['odds_home'].iloc[0] - 1) - (1 - prob)
    return f"Prob value: {prob:.1%} | EV: {ev:.2f} | Bet: {'SIM' if ev > 0.05 else 'NO'}"

# Agent Betting
betting_pro = Agent(
    role="Value Betting XGBoost Pro",
    goal="Detectar value bets football Portugal (Liga/Primeira) com EV+",
    backstory="Analista ML sports betting. XGBoost em form/H2H/odds para 55-60% picks.",
    llm=llm,
    verbose=True
)

task = Task(
    description=f"""
    Analisa Benfica vs Porto.
    Features: home_form=0.75, away_form=0.65, h2h_home=0.55, odds_home=2.10.
    XGBoost: {predict_value_bet([0.75, 0.65, 0.55, 2.10])}
    Recomenda stake/bankroll 2% se EV>0.05.
    """,
    agent=betting_pro
)

crew = Crew(agents=[betting_pro], tasks=[task])
result = crew.kickoff()
print(result)
