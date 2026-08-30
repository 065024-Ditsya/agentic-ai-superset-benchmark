import os
import json
import time
import requests
import pandas as pd
import psycopg2
from sklearn.datasets import load_iris, load_wine, load_diabetes

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
MODEL = os.getenv("MODEL", "qwen2.5:latest")
DB_URI = os.getenv("DB_URI", "postgresql://superset:superset_password@db:5432/superset")

def init_db():
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_benchmark_results (
                id SERIAL PRIMARY KEY,
                dataset_name VARCHAR(50),
                question TEXT,
                response TEXT,
                wall_clock_seconds FLOAT,
                eval_count INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Database table 'ai_benchmark_results' verified/created.")
    except Exception as e:
        print(f"Database initialization warning: {e}")

def save_to_db(dataset_name, question, response, wall_clock, eval_count):
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ai_benchmark_results (dataset_name, question, response, wall_clock_seconds, eval_count)
            VALUES (%s, %s, %s, %s, %s);
        """, (dataset_name, question, response, wall_clock, eval_count))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database insertion warning: {e}")

def create_summary(df, dataset_name):
    return json.dumps({
        "dataset_name": dataset_name,
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "sample_records": df.head(5).round(2).to_dict(orient="records"),
        "numeric_statistics": df.describe().round(2).to_dict()
    }, indent=2, default=str)

def ask_ollama(dataset_name, summary, question):
    prompt = f"Dataset: {dataset_name}\nSummary:\n{summary}\n\nQuestion: {question}\nProvide a concise analysis."
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    start_time = time.time()

    res = requests.post(OLLAMA_URL, json=payload, timeout=300)
    res.raise_for_status()
    wall_clock = round(time.time() - start_time, 2)
    data = res.json()

    response_text = data.get("response", "")
    eval_count = data.get("eval_count", 0)

    save_to_db(dataset_name, question, response_text, wall_clock, eval_count)

    return {
        "question": question,
        "response": response_text,
        "metrics": {
            "wall_clock_seconds": wall_clock,
            "eval_count": eval_count,
            "total_duration_ns": data.get("total_duration")
        }
    }

def test_dataset(dataset_name, df):
    summary = create_summary(df, dataset_name)
    questions = [
        "Give three key insights from this dataset.",
        "Identify the primary pattern or trend.",
        "Provide a business recommendation based on this data."
    ]
    return [ask_ollama(dataset_name, summary, q) for q in questions]

def main():
    print("=" * 80)
    print(f"STARTING MULTI-DATASET AGENTIC EVALUATION | MODEL: {MODEL}")
    print("=" * 80)

    init_db()

    iris = load_iris(as_frame=True)
    wine = load_wine(as_frame=True)
    diabetes = load_diabetes(as_frame=True)

    results = {
        "model": MODEL,
        "datasets_tested": 3,
        "Iris Dataset": test_dataset("Iris Dataset", iris.frame),
        "Wine Dataset": test_dataset("Wine Dataset", wine.frame),
        "Diabetes Dataset": test_dataset("Diabetes Dataset", diabetes.frame)
    }

    os.makedirs("/app/results", exist_ok=True)
    with open("/app/results/model_test_results.json", "w") as f:
        json.dump(results, f, indent=4, default=str)

    print("=" * 80)
    print("ALL DATASETS TESTED SUCCESSFULLY | TOTAL AI RESULTS: 9")
    print("=" * 80)

if __name__ == "__main__":
    main()
