````markdown name=README.md
```markdown
# Olivetree Demo Agent

A small demo showing how to run a "Lead Protector" agent locally using CrewAI and a LangChain OpenAI LLM wrapper.

Prerequisites
- Python 3.8+
- An OpenAI API key (or compatible provider) set in the environment:
  - `export OPENAI_API_KEY="sk-..."`

Install

```bash
pip install -r requirements.txt
```

Run

Dry run (no LLM calls):

```bash
python Demo.py --dry-run
```

Live run (will call the LLM):

```bash
python Demo.py
```

Notes
- Do NOT commit your API keys. Use environment variables or a secrets manager.
- If you want CI-friendly tests, use `--dry-run` in your test suite to avoid external calls.
```
````
