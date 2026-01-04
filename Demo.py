import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Agente Lead Protector para Imobiliárias Algarve
lead_protector = Agent(
    role="Lead Protector IA",
    goal="Proteger leads imobiliárias de poaching concorrentes + otimizar AL Algarve (+25% Airbnb)",
    backstory="""Especialista IA em real-estate Algarve. Detecta poaching em leads, otimiza Airbnb listings,
    gera +30% leads autónomo. Stack: CrewAI + XGBoost 55-60% accuracy.""",
    llm=llm,
    verbose=True
)

task = Task(
    description="Analisa este lead: Imobiliária Olive Tree Realty, Faro. Contacto novo via Google Business. Risco poaching alto (concorrentes Remax/era). Otimiza AL listing + responde auto.",
    agent=lead_protector
)

crew = Crew(agents=[lead_protector], tasks=[task])
result = crew.kickoff()
print(result)
