import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
import requests
import re
import csv
import time
from requests.exceptions import RequestException
import torch
import asyncio
import httpx  

# Configuration
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/91.0.4472.124 Safari/537.36'}
TIMEOUT = 10
MAX_RETRIES = 2

# Set up page config
st.set_page_config(page_title="Stock News Analyzer", layout="wide")

# Title and description
st.title("📈 Real-Time Stock News Analyzer")
st.markdown("""
This app analyzes financial news for selected stocks using:
- **Pegasus** for summarization
- **Sentiment Analysis** pipeline
- Optimized web scraping
""")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Enter Stock Ticker", "ETH").upper()
    num_articles = st.slider("Number of Articles", 1, 20, 5)
    analyze_button = st.button("Analyze News")

# Load models with caching
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "human-centered-summarization/financial-summarization-pegasus"
    
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    return tokenizer, model, sentiment_pipeline

tokenizer, model, sentiment = load_models()

# Async functions using httpx
async def fetch_url(client, url):
    try:
        response = await client.get(url, timeout=TIMEOUT, headers=HEADERS)
        return response.text
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None

async def scrape_articles(urls):
    async with httpx.AsyncClient() as client:
        tasks = [fetch_url(client, url) for url in urls]
        return await asyncio.gather(*tasks)

def clean_article_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text(strip=True) for p in paragraphs])
    return ' '.join(text.split()[:350])  # Limit to 350 words

def analyze_news(ticker):
    with st.status(f"Searching news for {ticker}..."):
        try:
            # Get news URLs
            search_url = f'https://news.google.com/rss/search?q={ticker}+stock+site:yahoo.com&ceid=US:en&hl=en-US&gl=US'
            response = requests.get(search_url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            urls = [item.link.text for item in items][:num_articles]

            # Scrape articles with retries
            articles = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            html_contents = loop.run_until_complete(scrape_articles(urls))
            
            for html in html_contents:
                if html:
                    article = clean_article_text(html)
                    if article:
                        articles.append(article)

            # Summarization
            summaries = []
            for article in articles:
                inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=512)
                summary_ids = model.generate(
                    inputs.input_ids,
                    max_length=55,
                    num_beams=5,
                    early_stopping=True
                )
                summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

            # Sentiment analysis
            scores = sentiment(summaries) if summaries else []
            
            return urls, articles, summaries, scores

        except RequestException as e:
            st.error(f"Network error: {str(e)}")
            return [], [], [], []
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return [], [], [], []

# Display results
if analyze_button:
    urls, articles, summaries, scores = analyze_news(ticker)
    
    if not summaries:
        st.warning("No articles found or error occurred. Try a different ticker or check network.")
        st.stop()

    st.success(f"Analyzed {len(summaries)} articles for {ticker}")
    
    # Create download data
    csv_data = [['Ticker','Summary', 'Sentiment', 'Sentiment Score', 'URL']]
    
    for i in range(len(summaries)):
        with st.expander(f"Article {i+1}: {summaries[i][:50]}...", expanded=False):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.image("https://via.placeholder.com/150", caption="Article Image")
                st.caption(f"Source: {urls[i].split('//')[1].split('/')[0]}")
                
            with col2:
                st.markdown(f"**Summary:** {summaries[i]}")
                st.markdown(f"**Sentiment:** {scores[i]['label']} ({scores[i]['score']:.2f})")
                st.markdown(f"**URL:** [Link]({urls[i]})")
                
                if st.checkbox("Show full text", key=f"text_{i}"):
                    st.write(articles[i][:1000] + "...")
            
            csv_data.append([ticker, summaries[i], scores[i]['label'], scores[i]['score'], urls[i]])

    # Download button
    st.download_button(
        label="Download Results as CSV",
        data='\n'.join([','.join(map(str, row)) for row in csv_data]),
        file_name=f'{ticker}_news_analysis.csv',
        mime='text/csv'
    )
