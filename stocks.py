import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
import requests
import re
import csv
import time
import torch
from newspaper import Article
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}
TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5

# Set up page configuration
st.set_page_config(
    page_title="Stock News Analyzer",
    layout="wide",
    page_icon="📈"
)

# Title and description
st.title("📈 Real-Time Stock News Analyzer")
st.markdown("""
This app analyzes financial news using:
- **Pegasus Summarization** (Financial-Specific)
- **Sentiment Analysis** (FinancialBERT)
- **Reliable Article Parsing** (Newspaper3k)
""")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    num_articles = st.slider("Number of Articles", 1, 20, 5)
    analyze_button = st.button("Analyze News")

# Configure requests session with retries
session = requests.Session()
retry = Retry(
    total=MAX_RETRIES,
    backoff_factor=BACKOFF_FACTOR,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('https://', HTTPAdapter(max_retries=retry))

# Load models with caching
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Financial summarization model
    tokenizer = PegasusTokenizer.from_pretrained(
        "human-centered-summarization/financial-summarization-pegasus"
    )
    model = PegasusForConditionalGeneration.from_pretrained(
        "human-centered-summarization/financial-summarization-pegasus"
    ).to(device)
    
    # Financial sentiment analysis
    sentiment = pipeline(
        "text-classification",
        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        device=0 if device == "cuda" else -1
    )
    
    return tokenizer, model, sentiment

tokenizer, model, sentiment = load_models()

def get_news_links(ticker: str) -> list:
    """Get news links from reliable RSS feed with error handling"""
    try:
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        response = session.get(rss_url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        return [item.link.text for item in items][:num_articles]
    
    except Exception as e:
        st.error(f"News search error: {str(e)}")
        return []

def fetch_article(url: str) -> str:
    """Fetch and parse article content with newspaper3k"""
    try:
        article = Article(url)
        article.download(
            input_html=session.get(url, headers=HEADERS, timeout=TIMEOUT).text
        )
        article.parse()
        text = re.sub(r'\s+', ' ', article.text).strip()
        return ' '.join(text.split()[:400]) if len(text.split()) > 50 else None
    
    except Exception as e:
        st.error(f"Article processing failed: {str(e)}")
        return None

def analyze_news(ticker: str):
    """Main analysis workflow with progress tracking"""
    try:
        with st.status(f"🔍 Analyzing {ticker} news..."):
            # Get validated URLs
            urls = get_news_links(ticker)
            if not urls:
                return [], [], [], []
            
            # Process articles
            articles = []
            for url in urls[:num_articles]:
                if article := fetch_article(url):
                    articles.append(article)
                if len(articles) >= num_articles:
                    break
            
            # Batch processing for efficiency
            if articles:
                inputs = tokenizer(
                    articles,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                summaries = tokenizer.batch_decode(
                    model.generate(
                        inputs.input_ids,
                        max_length=65,
                        num_beams=5,
                        early_stopping=True
                    ),
                    skip_special_tokens=True
                )
                scores = sentiment(summaries)
            else:
                summaries, scores = [], []
            
            return urls[:len(articles)], articles, summaries, scores
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return [], [], [], []

# Main execution
if analyze_button:
    urls, articles, summaries, scores = analyze_news(ticker)
    
    if not summaries:
        st.warning(f"""No articles found for {ticker}. Try:
- Major tickers (AAPL, MSFT, GOOG)
- Check network connection
- Retry in 1 minute""")
        st.stop()

    st.success(f"✅ Successfully analyzed {len(summaries)} articles")
    
    # Prepare CSV data
    csv_data = [['Ticker','Summary','Sentiment','Score','URL']]
    for i, (summary, score) in enumerate(zip(summaries, scores)):
        with st.expander(f"{i+1}. {summary[:70]}... ({score['label']} {score['score']:.2f})"):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Confidence", f"{score['score']*100:.1f}%")
                st.caption(f"Source: {urls[i].split('//')[-1].split('/')[0]}")
            with col2:
                st.markdown(f"**Summary**: {summary}")
                with st.popover("View Details"):
                    st.markdown(f"**URL**: [Link]({urls[i]})")
                    if article := articles[i]:
                        st.write(article[:1000] + "...")
        
        csv_data.append([
            ticker,
            f'"{summary}"',
            score['label'],
            score['score'],
            urls[i]
        ])
    
    # Download handler
    st.download_button(
        "📥 Download Report",
        "\n".join([",".join(map(str, row)) for row in csv_data]),
        f"{ticker}_news_analysis.csv",
        "text/csv"
    )
