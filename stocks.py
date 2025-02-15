import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
import requests
import re
import csv
import time

# Set up page config
st.set_page_config(page_title="Stock News Analyzer", layout="wide")

# Title and description
st.title("📈 Real-Time Stock News Analyzer")
st.markdown("""
This app analyzes financial news for selected stocks using:
- **Pegasus** for summarization
- **Sentiment Analysis** pipeline
- Web scraping from Google/Yahoo News
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
    model_name = "human-centered-summarization/financial-summarization-pegasus"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis")
    return tokenizer, model, sentiment_pipeline

tokenizer, model, sentiment = load_models()

def get_news_links(ticker):
    """Search Google for Yahoo Finance news links."""
    try:
        search_url = f'https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws'
        r = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        atags = soup.find_all('a')
        hrefs = [link['href'] for link in atags if link.get('href')]
        
        # Clean URLs: filter out common unwanted paths
        exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
        cleaned_urls = []
        for url in hrefs:
            if 'https://' in url and not any(exc in url for exc in exclude_list):
                # Extract URL up to the first '&'
                match = re.findall(r'(https?://\S+)', url)
                if match:
                    clean = match[0].split('&')[0]
                    cleaned_urls.append(clean)
        cleaned_urls = list(set(cleaned_urls))[:num_articles]
        return cleaned_urls
    except Exception as e:
        st.error(f"News search failed: {str(e)}")
        return []

def fetch_article(url):
    """Fetch article content using improved extraction."""
    for _ in range(2):
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # Attempt to find a main article container
            article_tag = soup.find('article')
            if not article_tag:
                article_tag = soup.find('div', class_=re.compile(r'(caas-body|article-body|main-content)'))
            if not article_tag:
                # Fallback: use all paragraphs
                paragraphs = soup.find_all('p')
            else:
                paragraphs = article_tag.find_all('p')
            
            text = ' '.join(p.get_text(strip=True) for p in paragraphs)
            # Remove common boilerplate phrases (adjust as needed)
            text = re.sub(r'All photographs subject to copyright.*', '', text, flags=re.I)
            return ' '.join(text.split()[:400])  # Limit to 400 words
        except Exception:
            time.sleep(1)
    return None

def analyze_news(ticker):
    try:
        with st.spinner(f"Searching news for {ticker}..."):
            urls = get_news_links(ticker)
            if not urls:
                return [], [], [], []
            
            articles = []
            for url in urls:
                article_text = fetch_article(url)
                if article_text:
                    articles.append(article_text)
                if len(articles) >= num_articles:
                    break
            
            # Summarize articles using Pegasus
            summaries = []
            for article in articles:
                input_ids = tokenizer.encode(article, return_tensors="pt")
                output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
                summary = tokenizer.decode(output[0], skip_special_tokens=True)
                summaries.append(summary)
            
            # Perform sentiment analysis
            scores = sentiment(summaries) if summaries else []
            
            return urls[:len(articles)], articles, summaries, scores
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return [], [], [], []

if analyze_button:
    urls, articles, summaries, scores = analyze_news(ticker)
    
    if not summaries:
        st.warning("No articles found for this ticker")
        st.stop()

    st.success(f"Found {len(summaries)} articles for {ticker}")
    
    # Prepare CSV data for download
    csv_data = [['Ticker', 'Summary', 'Sentiment', 'Sentiment Score', 'URL']]
    for i in range(len(summaries)):
        with st.expander(f"Article {i+1}: {summaries[i][:50]}...", expanded=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image("https://via.placeholder.com/150", caption="Article Image")
                try:
                    domain = urls[i].split('//')[1].split('/')[0]
                except Exception:
                    domain = "Unknown Source"
                st.caption(f"Source: {domain}")
            with col2:
                st.markdown(f"**Summary:** {summaries[i]}")
                st.markdown(f"**Sentiment:** {scores[i]['label']} ({scores[i]['score']:.2f})")
                st.markdown(f"**URL:** [Link]({urls[i]})")
                if st.checkbox("Show full text", key=f"text_{i}"):
                    st.write(articles[i][:1000] + "...")
            csv_data.append([ticker, summaries[i], scores[i]['label'], scores[i]['score'], urls[i]])
    
    st.download_button(
        label="Download Results as CSV",
        data='\n'.join([','.join(map(str, row)) for row in csv_data]),
        file_name=f'{ticker}_news_analysis.csv',
        mime='text/csv'
    )
