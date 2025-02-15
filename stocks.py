import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
import requests
import re
import csv
import time

# Configuration
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
TIMEOUT = 8
MAX_ARTICLES = 20

# Set up page config
st.set_page_config(page_title="Stock News Analyzer", layout="wide")

# Title and description
st.title("📈 Real-Time Stock News Analyzer")
st.markdown("""
This app analyzes financial news for selected stocks using:
- **Pegasus** for summarization
- **Sentiment Analysis** pipeline
- Reliable news sources
""")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    num_articles = st.slider("Number of Articles", 1, 20, 5)
    analyze_button = st.button("Analyze News")

# Load models with caching
@st.cache_resource
def load_models():
    model_name = "human-centered-summarization/financial-summarization-pegasus"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis")
    return tokenizer, model, sentiment_pipeline

tokenizer, model, sentiment = load_models()

def get_news_links(ticker):
    """Get news links from Google News with better search parameters"""
    try:
        url = f"https://news.google.com/search?q={ticker}%20stock&hl=en-US&gl=US&ceid=US%3Aen"
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')[:MAX_ARTICLES]
        
        urls = []
        for article in articles:
            link = article.find('a', href=True)
            if link:
                url = 'https://news.google.com' + link['href'].replace('./', '')
                urls.append(url)
        return urls[:num_articles]
    
    except Exception as e:
        st.error(f"News search failed: {str(e)}")
        return []

def fetch_article(url):
    """Fetch article content with retries and error handling"""
    for _ in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find main article content
            article = soup.find('article') or soup.find('div', class_=re.compile('content|body'))
            if not article:
                return None
                
            text = ' '.join([p.get_text() for p in article.find_all('p')])
            return ' '.join(text.split()[:400])  # Limit to 400 words
            
        except Exception as e:
            time.sleep(1)
    
    return None

def analyze_news(ticker):
    try:
        with st.status(f"Searching news for {ticker}..."):
            urls = get_news_links(ticker)
            if not urls:
                return [], [], [], []
            
            articles = []
            for url in urls:
                article = fetch_article(url)
                if article:
                    articles.append(article)
                if len(articles) >= num_articles:
                    break
            
            # Summarization
            summaries = []
            for article in articles:
                inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=512)
                summary_ids = model.generate(
                    inputs.input_ids,
                    max_length=65,
                    num_beams=5,
                    early_stopping=True
                )
                summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
            
            # Sentiment analysis
            scores = sentiment(summaries) if summaries else []
            
            return urls[:len(articles)], articles, summaries, scores
    
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return [], [], [], []

# Display results
if analyze_button:
    urls, articles, summaries, scores = analyze_news(ticker)
    
    if not summaries:
        st.warning(f"No articles found for {ticker}. Try:")
        st.markdown("- More common ticker (AAPL, TSLA, etc)")
        st.markdown("- Check network connection")
        st.markdown("- Try again later")
        st.stop()

    # Display results
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
