# Agentic AI & Apache Superset Multi-Container Infrastructure

An automated end-to-end benchmarking and analytics pipeline integrating Apache Superset, Ollama (Qwen 2.5), PostgreSQL, and Redis within a single Docker Compose environment.

## 🏗️ Architecture Overview

The platform orchestrates six interconnected containerized services:
* **`agentic-superset`**: Apache Superset Business Intelligence web engine.
* **`agentic-analyzer`**: Custom Python pipeline that feeds datasets (`Iris`, `Wine`, `Diabetes`) to Ollama and measures prompt execution performance.
* **`agentic-ollama`**: Local LLM inference engine running Qwen 2.5.
* **`agentic-ollama-pull`**: Automated model puller ensuring model availability prior to evaluation execution.
* **`agentic-db`**: PostgreSQL database storing Superset metadata and AI analysis output tables (`ai_benchmark_results`).
* **`agentic-redis`**: Caching layer for Superset query performance.

## 🚀 Quickstart

```bash
docker compose up -d --build
docker logs -f agentic-analyzer
