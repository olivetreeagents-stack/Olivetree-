import os
import logging
import argparse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. LLM client may fail if an API key is required.")

try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
except Exception:
    logger.exception("Failed to import required packages. Install crewai and langchain-openai.")
    raise

EXAMPLE_LEAD_DESCRIPTION = (
    "Analisa este lead: Imobiliária Olive Tree Realty, Faro. Contacto novo via Google Business. "
    "Risco poaching alto (concorrentes Remax/era). Otimiza AL listing + responde auto."
)

def build_agent(llm):
    return Agent(
        role="Lead Protector IA",
        goal="Proteger leads imobiliárias de poaching concorrentes + otimizar AL Algarve (+25% Airbnb)",
        backstory=(
            "Especialista IA em real-estate Algarve. Detecta poaching em leads, otimiza Airbnb listings, "
            "gera +30% leads autónomo. Stack: CrewAI + XGBoost 55-60% accuracy."
        ),
        llm=llm,
        verbose=True,
    )


def main(dry_run: bool = False):
    if dry_run:
        logger.info("Dry run enabled — simulating agent/crew behavior without calling the LLM.")
        simulated_result = {
            "task": "Analyze lead",
            "lead": "Olive Tree Realty, Faro",
            "risk": "high (poaching by Remax/era)",
            "recommended_actions": [
                "Prioritize follow-up call within 30 minutes",
                "Send exclusive listing access link with expiring token",
                "Mark lead as 'at-risk' and assign senior agent",
                "Create optimized Airbnb listing title and 3 bullet points"
            ],
            "airbnb_optimization": {
                "title": "Charming Faro Apartment — Steps from Old Town",
                "bullets": [
                    "Great location: 200m to historic centre",
                    "Fast Wi-Fi & dedicated workspace",
                    "Flexible check-in and professional cleaning"
                ],
                "estimated_uplift_pct": 25
            }
        }
        print(simulated_result)
        return

    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = build_agent(llm)

    task = Task(description=EXAMPLE_LEAD_DESCRIPTION, agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    try:
        result = crew.kickoff()
        logger.info("Crew finished successfully.")
        print(result)
    except Exception:
        logger.exception("Error running crew.kickoff()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Olivetree Demo Agent")
    parser.add_argument("--dry-run", action="store_true", help="Simulate agent behavior without calling the LLM")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
