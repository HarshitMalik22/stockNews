import os
import re
import time
import asyncio
import aiohttp
import requests
import feedparser
from newspaper import Article
from tqdm import tqdm
from typing import List, Tuple
import dotenv
import streamlit as st
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    pipeline,
    logging
)

# Configuration
dotenv.load_dotenv()
logging.set_verbosity_error()

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}
TIMEOUT = aiohttp.ClientTimeout(total=10)
MAX_RETRIES = 2
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Get from https://newsapi.org

# Streamlit config
st.set_page_config(page_title="Stock Pulse", layout="wide", page_icon="📈")
st.title("📈 Stock Pulse: Real-Time News Analyzer")
st.markdown("""
AI-powered market intelligence with:
- **Smart Summarization** (Pegasus-X)
- **Sentiment Scoring**
- **Multi-source Verification**
""")

# --- Core Functions ---
@st.cache_resource
def load_models() -> Tuple:
    """Load ML models with hardware optimization"""
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Load summarization model
    tokenizer = PegasusTokenizer.from_pretrained(
        "human-centered-summarization/financial-summarization-pegasus"
    )
    model = PegasusForConditionalGeneration.from_pretrained(
        "human-centered-summarization/financial-summarization-pegasus"
    ).to(device)
    
    # Load sentiment analysis pipeline
    sentiment = pipeline(
        "text-classification",
        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        device=0 if device == "cuda" else -1
    )
    
    return tokenizer, model, sentiment

async def fetch_news(ticker: str, num_articles: int) -> List[str]:
    """Fetch news URLs from multiple reliable sources"""
    sources = [
        f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&language=en&pageSize={num_articles}",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    ]
    
    urls = []
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for source in sources:
            try:
                async with session.get(source, timeout=TIMEOUT) as response:
                    content = await response.text()
                    
                    if "newsapi" in source:
                        data = await response.json()
                        urls.extend([article['url'] for article in data.get('articles', [])])
                    else:
                        feed = feedparser.parse(content)
                        urls.extend([entry.link for entry in feed.entries])
                        
            except Exception as e:
                st.error(f"Error fetching {source}: {str(e)}")
                continue
                
    return list(dict.fromkeys(urls))[:num_articles]  # Remove duplicates

async def process_article(session: aiohttp.ClientSession, url: str) -> str:
    """Process article with retries and sanitization"""
    for retry in range(MAX_RETRIES):
        try:
            async with session.get(url, timeout=TIMEOUT) as response:
                html = await response.text()
                article = Article(url)
                article.download(input_html=html)
                article.parse()
                
                # Clean content
                text = re.sub(r'\s+', ' ', article.text).strip()
                if len(text.split()) > 100:  # Validate meaningful content
                    return ' '.join(text.split()[:500])  # Limit to 500 words
                
        except Exception as e:
            if retry == MAX_RETRIES - 1:
                st.error(f"Failed to process {url}: {str(e)}")
            await asyncio.sleep(1)
    
    return None

def analyze_content(tokenizer, model, sentiment, articles: List[str]) -> Tuple:
    """Batch process articles for efficiency"""
    # Summarization
    summaries = []
    inputs = tokenizer(articles, truncation=True, padding=True, return_tensors="pt", max_length=512)
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=65,
        num_beams=5,
        early_stopping=True
    )
    summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]
    
    # Sentiment analysis
    scores = sentiment(summaries)
    
    return summaries, scores

# --- Streamlit UI ---
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    num_articles = st.slider("Articles to Analyze", 1, 20, 7)
    analyze_btn = st.button("Analyze Market News")

if analyze_btn:
    tokenizer, model, sentiment = load_models()
    
    with st.status("🔍 Gathering Market Intelligence..."):
        try:
            # Phase 1: News Collection
            urls = asyncio.run(fetch_news(ticker, num_articles))
            if not urls:
                st.error("No articles found. Try popular tickers like AAPL, TSLA, or GOOG")
                st.stop()
            
            # Phase 2: Article Processing
            articles = []
            async with aiohttp.ClientSession(headers=HEADERS) as session:
                tasks = [process_article(session, url) for url in urls]
                for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                    article = await future
                    if article:
                        articles.append(article)
            
            # Phase 3: AI Analysis
            summaries, scores = analyze_content(tokenizer, model, sentiment, articles)
            
        except Exception as e:
            st.error(f"Critical Failure: {str(e)}")
            st.stop()
    
    # Display Results
    st.success(f"✅ Analyzed {len(summaries)} articles for {ticker}")
    
    for idx, (summary, score) in enumerate(zip(summaries, scores)):
        with st.expander(f"{idx+1}. {summary[:60]}... ({score['label']} {score['score']:.2f})"):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**Sentiment Confidence**\n{score['score']*100:.1f}%")
                st.markdown(f"**Source**\n{urls[idx].split('//')[1].split('/')[0]}")
            with col2:
                st.markdown(f"**Summary**\n{summary}")
                st.link_button("View Full Article", urls[idx])
    
    # Export
    csv_data = "Ticker,Summary,Sentiment,Score,URL\n"
    csv_data += "\n".join(
        [f"{ticker},{s},{sc['label']},{sc['score']},{u}" 
         for s, sc, u in zip(summaries, scores, urls)]
    )
    
    st.download_button(
        "📥 Download Analysis",
        csv_data,
        f"{ticker}_analysis.csv",
        "text/csv"
    )
