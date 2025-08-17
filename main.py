import io
import re
import os
import base64
import logging
import requests
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
import httpx
from bs4 import BeautifulSoup

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

app = FastAPI(title="TDS Data Analyst Agent", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# ----------------- Utilities -----------------

def encode_plot_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    dpi = 70
    while True:
        buf.seek(0)
        buf.truncate(0)
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        size = buf.tell()
        if size < 100_000 or dpi <= 20:
            break
        dpi -= 10
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def create_scatterplot_with_regression(x, y, xlabel, ylabel, title) -> str:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x, y, alpha=0.7)
    X = x.values.reshape(-1, 1)
    Y = y.values
    model = LinearRegression()
    model.fit(X, Y)
    y_pred = model.predict(X)
    ax.plot(x, y_pred, "r--", label="Regression line")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return encode_plot_base64(fig)

def contains_any(keywords: List[str], text: str) -> bool:
    text = text.lower()
    return any(k.lower() in text for k in keywords)

def split_questions(text: str) -> List[str]:
    return [q.strip() for q in text.splitlines() if q.strip()]

# ----------------- Wikipedia Scraper -----------------

async def scrape_wikipedia_table(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

    table = soup.find("table", {"class": "wikitable"})
    if table is None:
        raise ValueError("Could not find the table on Wikipedia page")

    df = pd.read_html(str(table))[0]

    col_map = {}
    for c in df.columns:
        c_new = c.lower().strip()
        if "rank" in c_new:
            col_map[c] = "Rank"
        elif "peak" in c_new:
            col_map[c] = "Peak"
        elif "title" in c_new:
            col_map[c] = "Title"
        elif "worldwide" in c_new or "gross" in c_new:
            col_map[c] = "WorldwideGross"
        elif "year" in c_new:
            col_map[c] = "Year"
        else:
            col_map[c] = c

    df.rename(columns=col_map, inplace=True)

    def parse_billion(x):
        if pd.isna(x):
            return np.nan
        m = re.search(r"([\d\.]+)", str(x))
        if m:
            return float(m.group(1))
        return np.nan

    df["WorldwideGross"] = df["WorldwideGross"].apply(parse_billion)

    for col in ["Rank", "Peak", "Year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Rank", "Peak", "WorldwideGross", "Year", "Title"])

def answers_from_wiki_df(df: pd.DataFrame) -> Dict[str, Any]:
    answers = {}

    two_bn_before_2000 = df[(df["WorldwideGross"] >= 2.0) & (df["Year"] < 2000)]
    answers["How many $2 billion movies were released before 2000?"] = int(len(two_bn_before_2000))

    over_1_5bn = df[df["WorldwideGross"] > 1.5]
    if not over_1_5bn.empty:
        earliest = over_1_5bn.sort_values("Year").iloc[0]["Title"]
    else:
        earliest = None
    answers["Which is the earliest film that grossed over $1.5 billion?"] = earliest

    rank_peak_corr = df["Rank"].corr(df["Peak"])
    answers["What's the correlation between the Rank and Peak?"] = round(rank_peak_corr, 6) if not pd.isna(rank_peak_corr) else None

    img_data_uri = create_scatterplot_with_regression(df["Rank"], df["Peak"], "Rank", "Peak", "Rank vs Peak with Regression Line")
    answers["Scatterplot of Rank and Peak with regression line"] = img_data_uri

    return answers

# ----------------- Indian High Court Analysis -----------------

def parse_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def highcourt_analysis(df: pd.DataFrame, question: str) -> Any:
    question_lower = question.lower()
    df = parse_dates(df, ["date_of_registration", "decision_date"])

    if "date_of_registration" in df.columns and "decision_date" in df.columns:
        df["days_delay"] = (df["decision_date"] - df["date_of_registration"]).dt.days

    # 1. Which high court disposed the most cases from 2019 to 2022?
    if "disposed the most cases" in question_lower:
        df_filtered = df[(df["date_of_registration"].dt.year >= 2019) & (df["date_of_registration"].dt.year <= 2022)]
        if "court" in df.columns:
            counts = df_filtered["court"].value_counts()
            if counts.empty:
                return "No cases found in given period."
            most_court = counts.idxmax()
            most_count = counts.max()
            return f"High court '{most_court}' disposed the most cases: {most_count} cases (2019-2022)."
        else:
            return "Court column missing in dataset."

    # 2. Regression slope of date_of_registration vs decision_date for court=33_10
    if "regression slope" in question_lower and "33_10" in question_lower:
        df_court = df[df["court"] == "33_10"]
        df_court = df_court.dropna(subset=["date_of_registration", "decision_date"])
        if df_court.empty:
            return "No data for court=33_10 with valid dates."
        X = df_court["date_of_registration"].map(datetime.toordinal).to_numpy().reshape(-1, 1)
        y = df_court["decision_date"].map(datetime.toordinal).to_numpy()
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        return f"Regression slope (date_of_registration vs decision_date) for court=33_10 is {slope:.4f}."

    # 3. Plot year vs number of days delay with regression line
    if "plot year vs number of days delay" in question_lower or "plot year vs days delay" in question_lower:
        if "date_of_registration" not in df.columns or "days_delay" not in df.columns:
            return "Required columns missing for plotting."
        df_plot = df.dropna(subset=["date_of_registration", "days_delay"]).copy()
        df_plot["year"] = df_plot["date_of_registration"].dt.year
        if df_plot.empty:
            return "No valid data for plotting."
        plt.figure(figsize=(6, 4))
        plt.scatter(df_plot["year"], df_plot["days_delay"], label="Data points")
        X = np.array(df_plot["year"]).reshape(-1, 1)
        y = df_plot["days_delay"].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        plt.plot(df_plot["year"], y_pred, "r--", label="Regression line")
        plt.xlabel("Year")
        plt.ylabel("Days Delay")
        plt.title("Year vs Days Delay with Regression Line")
        plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_bytes = buf.read()
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64_img}"

    return "Question not recognized for Indian High Court data."

# ----------------- Generic URL scraper -----------------

def scrape_any_url(url: str, save_local: bool = False, filename: Optional[str] = None) -> Dict[str, Any]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TDSDataAgent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        visible_text = "\n".join(lines[:50])

        # Save full visible text locally if requested
        if save_local:
            if not filename:
                import hashlib
                safe_name = re.sub(r'[^a-zA-Z0-9]', '_', url)
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"scraped_{safe_name}_{url_hash}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logging.info(f"Scraped data saved to {filename}")

        tables = []
        try:
            dfs = pd.read_html(resp.text)
            for idx, df in enumerate(dfs):
                clean_df = df.replace({np.nan: None})
                tables.append(
                    {
                        "table_index": idx,
                        "columns": clean_df.columns.tolist(),
                        "row_count": len(clean_df),
                        "head": clean_df.head(3).to_dict(orient="records"),
                    }
                )
        except Exception:
            pass

        return {
            "url": url,
            "visible_text_snippet": visible_text,
            "num_tables_found": len(tables),
            "tables_preview": tables,
            "saved_to_file": filename if save_local else None,
        }
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
        return {"error": f"Failed to scrape website: {e}"}

# ----------------- SerpAPI fallback -----------------

def internet_search_fallback(query: str) -> Dict[str, Any]:
    if not SERPAPI_API_KEY:
        return {"error": "SERPAPI_API_KEY not set for internet search fallback."}
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 3,
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        snippets = []
        organic_results = results.get("organic_results", [])
        for res in organic_results[:3]:
            snippet = res.get("snippet") or res.get("title")
            if snippet:
                snippets.append(snippet)

        if not snippets:
            return {"error": "No results found from internet search."}
        return {"snippets": snippets}
    except Exception as e:
        logging.error(f"SerpAPI error: {e}")
        return {"error": f"Error performing web search: {e}"}

# ----------------- Main question processor -----------------

async def process_question(question: str, files: Optional[Dict[str, UploadFile]] = None) -> Any:
    q_lower = question.lower()

    # Wikipedia questions
    if contains_any(["highest grossing", "box office", "gross"], q_lower):
        try:
            wiki_df = await scrape_wikipedia_table("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
            wiki_ans = answers_from_wiki_df(wiki_df)
            return wiki_ans
        except Exception as e:
            return {"error": f"Failed Wikipedia scrape: {str(e)}"}

    # High court questions with uploaded CSV
    if contains_any(["high court", "court", "judgement", "disposed"], q_lower):
        if not files:
            return {"error": "Indian High Court dataset CSV file required for analysis."}
        csv_files = [file for file in files.values() if file.filename and file.filename.endswith(".csv")]
        if not csv_files:
            return {"error": "No CSV file uploaded for High Court data."}
        try:
            content = await csv_files[0].read()
            df = pd.read_csv(io.BytesIO(content))
            hc_ans = highcourt_analysis(df, question)
            return hc_ans
        except Exception as e:
            return {"error": f"Failed High Court analysis: {str(e)}"}

    # Scrape generic URL commands
    if q_lower.startswith("scrape ") or q_lower.startswith("scrapeurl "):
        url_match = re.search(r"(https?://[^\s]+)", question)
        if url_match:
            url = url_match.group(1)
            return scrape_any_url(url)
        else:
            return {"error": "No valid URL found to scrape."}

    # Fallback to internet search
    fallback = await asyncio.to_thread(internet_search_fallback, question)
    return fallback

# ----------------- API endpoints -----------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Welcome to TDS Data Analyst Agent API"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

@app.post("/api/")
async def analyze(
    questions_txt: UploadFile = File(..., description="Text file with questions, one per line"),
    files: Optional[List[UploadFile]] = File(None, description="Optional uploaded files like High Court CSV")
):
    try:
        content = await questions_txt.read()
        text = content.decode("utf-8", errors="ignore").strip()
        questions = split_questions(text)
        files_dict = {f.filename: f for f in files} if files else {}

        results = {}
        for i, question in enumerate(questions, 1):
            answer = await process_question(question, files_dict)
            results[f"question_{i}"] = {"question": question, "answer": answer}
        return results
    except Exception as e:
        logging.error(f"Error in /api/: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/films")
async def films():
    try:
        wiki_df = await scrape_wikipedia_table("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
        wiki_ans = answers_from_wiki_df(wiki_df)
        return wiki_ans
    except Exception as e:
        return {"error": f"Failed Wikipedia scrape: {str(e)}"}

@app.get("/scrape")
def scrape(
    url: str = Query(..., description="URL to scrape"),
    save: bool = Query(False, description="Save scraped text to local file")
):
    if not url:
        return JSONResponse({"error": "No URL provided"}, status_code=400)
    return scrape_any_url(url, save_local=save)

if __name__ == "__main__":
    import uvicorn
    # Cloud Run deployment uses $PORT env var, local uses 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
