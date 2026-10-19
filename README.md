# fin-agent

A financial-filings analysis agent built on a hand-written agent harness, with an evaluation suite that compares
**LLM-only**, **RAG**, and **full agent** setups on questions about US public companies (SEC EDGAR data).

> 🚧 Work in progress — built step by step.

## Setup
```bash
conda create -n fin-agent python=3.11 -y
conda activate fin-agent
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
python scripts/check_setup.py
```

## Project structure
```
fin-agent/
├── src/finagent/      # core package
│   └── config.py      # settings loaded from .env (OpenAI-compatible LLM)
├── scripts/           # runnable entry points
├── tests/             # unit tests
└── data/              # local data & SQLite DB (git-ignored)
```
