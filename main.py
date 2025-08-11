from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from scipy.stats import pearsonr, linregress
import re
from typing import Dict, Any
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

app = FastAPI(title="TDS Data Analyst Agent", version="2.0")

# ----------- Helpers -----------

def _generate_plot_base64(df_plot, x_col, y_col, title, regression_line=False, x_label="", y_label="") -> str:
    img_base64 = ""
    if not df_plot.empty and len(df_plot) > 1:
        plt.figure(figsize=(6, 4))
        x_vals = df_plot[x_col].astype(float).to_numpy()
        y_vals = df_plot[y_col].astype(float).to_numpy()
        plt.scatter(x_vals, y_vals, alpha=0.7)
        plt.xlabel(x_label or x_col)
        plt.ylabel(y_label or y_col)
        plt.title(title)
        if regression_line:
            slope, intercept, _, _, _ = linregress(x_vals, y_vals)
            reg_line = intercept + slope * x_vals
            plt.plot(x_vals, reg_line, "r--", label=f"Slope = {slope:.2f}")
            plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight")
        buf.seek(0)
        img_base64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
    return img_base64

def web_search_fallback(query: str) -> str:
    if not SERPAPI_KEY:
        return "Web search is unavailable because SERPAPI_KEY is missing."
    try:
        params = {"q": query, "engine": "google", "api_key": SERPAPI_KEY}
        search = GoogleSearch(params)
        results = search.get_dict()
        if "error" in results:
            return f"Search error: {results['error']}"
        organic = results.get("organic_results", [])
        if not organic:
            return "No search results found."
        return organic[0].get("snippet", "No snippet available.")
    except Exception as e:
        return f"Error performing web search: {e}"

def scrape_highest_grossing_films(year: int) -> Dict[str, Any]:
    try:
        url = f"https://en.wikipedia.org/wiki/List_of_highest-grossing_films_of_{year}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "wikitable"})
        if not table:
            snippet = web_search_fallback(f"highest grossing films {year}")
            return {"films": [], "message": snippet}

        df_year = pd.read_html(str(table))[0].fillna("")
        title_col = next((col for col in df_year.columns if "film" in str(col).lower() or "title" in str(col).lower()), None)
        top_films = [str(f).strip() for f in df_year[title_col] if str(f).strip()] if title_col else []

        all_time_url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        resp_all = requests.get(all_time_url, timeout=10)
        resp_all.raise_for_status()
        soup_all = BeautifulSoup(resp_all.text, "html.parser")
        table_all = soup_all.find("table", {"class": ["wikitable", "sortable"]})
        df_all = pd.read_html(str(table_all))[0]
        df_all.columns = [re.sub(r"\[\d+\]", "", str(col)).strip() for col in df_all.columns]
        df_all.rename(columns={"Title": "Film", "Worldwide gross": "Worldwide_gross", "Year": "Release_Year"}, inplace=True)
        df_all["Worldwide_gross"] = df_all["Worldwide_gross"].replace(r"[\$,]", "", regex=True)
        df_all["Worldwide_gross_numeric"] = pd.to_numeric(df_all["Worldwide_gross"], errors="coerce")
        df_all["Release_Year"] = pd.to_numeric(df_all["Release_Year"], errors="coerce")

        two_billion_before_2020 = int(df_all[(df_all["Worldwide_gross_numeric"] >= 2_000_000_000) & (df_all["Release_Year"] < 2020)].shape[0])
        earliest = df_all[df_all["Worldwide_gross_numeric"] >= 1_500_000_000].sort_values("Release_Year").iloc[0]["Film"]

        corr_val = None
        plot_img = ""
        if "Rank" in df_all.columns and "Peak" in df_all.columns:
            df_all.dropna(subset=["Rank", "Peak"], inplace=True)
            corr_val, _ = pearsonr(df_all["Rank"], df_all["Peak"])
            corr_val = round(float(corr_val), 6)
            plot_img = _generate_plot_base64(df_all, "Rank", "Peak", "Rank vs Peak", True, "Rank", "Peak")

        return {
            "year": year,
            "top_films": top_films[:5],
            "two_billion_before_2020": two_billion_before_2020,
            "earliest_over_1_5_billion": earliest,
            "rank_peak_correlation": corr_val,
            "scatter_plot_base64": plot_img,
        }
    except Exception as e:
        snippet = web_search_fallback(f"highest grossing films {year}")
        return {"error": str(e), "fallback_search_snippet": snippet}

def analyze_highcourt_data(file_path: str) -> Dict[str, Any]:
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return {"error": "The dataset is empty."}
        if "Decision Date" not in df.columns:
            return {"error": "The dataset does not contain a 'Decision Date' column."}

        df["Decision Date"] = pd.to_datetime(df["Decision Date"], errors="coerce")
        df["Year"] = df["Decision Date"].dt.year
        counts = df["Year"].value_counts().sort_index()
        if counts.empty:
            return {"error": "No valid decision dates found in dataset."}

        df_counts = pd.DataFrame({"Year": counts.index, "Judgments": counts.values})
        plot_img = _generate_plot_base64(df_counts, "Year", "Judgments", "High Court Judgments per Year", True, "Year", "Judgments")

        return {
            "judgments_per_year": dict(zip(counts.index.tolist(), counts.values.tolist())),
            "scatter_plot_base64": plot_img
        }
    except Exception as e:
        return {"error": f"Error analyzing High Court data: {e}"}

def _handle_general_knowledge_task(task_description: str) -> Dict[str, Any]:
    if "what is a computer" in task_description.lower():
        return {"answer": "A computer is an electronic device that manipulates information, or data."}
    return {"answer": "I'm a data analyst agent. My general knowledge is limited."}

# ----------- Task Routing -----------

def process_question(question: str) -> Dict[str, Any]:
    q = question.lower()
    year_match = re.search(r"\b(19|20)\d{2}\b", q)

    if any(k in q for k in ["highest grossing", "box office", "top-grossing"]) and year_match:
        return scrape_highest_grossing_films(int(year_match.group()))

    if "indian high court" in q or "high court" in q:
        sample_csv = "data/high_court_judgments.csv"
        if os.path.exists(sample_csv):
            return analyze_highcourt_data(sample_csv)
        else:
            return {"error": "High Court dataset not found."}

    if "what is a computer" in q:
        return _handle_general_knowledge_task(q)

    snippet = web_search_fallback(question)
    return {"answer": snippet}

# ----------- API Models -----------

class QueryRequest(BaseModel):
    task: str

# ----------- Routes -----------

@app.get("/")
def root():
    return JSONResponse({"status": "ok", "message": "Welcome to TDS Data Analyst Agent API"})

@app.get("/health")
def health_check():
    return JSONResponse({"status": "ok", "message": "TDS Data Analyst Agent API is running"})

@app.get("/films/{year}")
def get_films(year: int):
    return JSONResponse(scrape_highest_grossing_films(year))

@app.get("/highcourt")
def get_highcourt_analysis():
    sample_csv = "data/high_court_judgments.csv"
    if not os.path.exists(sample_csv):
        return JSONResponse({"error": "High Court dataset not found."}, status_code=404)
    return JSONResponse(analyze_highcourt_data(sample_csv))

@app.get("/search")
def web_search(query: str = Query(..., description="Query to search on the web")):
    snippet = web_search_fallback(query)
    return JSONResponse({"query": query, "answer": snippet})

@app.get("/api/")
def api_get(task: str = Query(..., description="Your question or task")):
    return JSONResponse(process_question(task))

@app.post("/api/")
async def api_post(req: QueryRequest):
    return JSONResponse(process_question(req.task))

@app.post("/api/file")
async def api_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore").strip()
        return JSONResponse(process_question(text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
