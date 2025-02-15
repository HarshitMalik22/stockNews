import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
import requests
import re
import csv

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

# Main processing function
def analyze_news(ticker):
    # Search for news links
    with st.status(f"Searching news for {ticker}..."):
        search_url = f'https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws'
        r = requests.get(search_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        atags = soup.find_all('a')
        hrefs = [link['href'] for link in atags]

        # Clean URLs
        exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
        cleaned_urls = []
        for url in hrefs:
            if 'https://' in url and not any(exc in url for exc in exclude_list):
                res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                cleaned_urls.append(res)
        cleaned_urls = list(set(cleaned_urls))[:num_articles]

        # Scrape articles
        articles = []
        for url in cleaned_urls:
            article_text = ""
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text = [p.text for p in paragraphs]
                words = ' '.join(text).split(' ')[:350]
                articles.append(' '.join(words))
            except:
                continue

        # Summarize articles
        summaries = []
        for article in articles:
            input_ids = tokenizer.encode(article, return_tensors="pt")
            output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            summaries.append(summary)

        # Sentiment analysis
        scores = sentiment(summaries) if summaries else []

    return cleaned_urls, articles, summaries, scores

# Display results
if analyze_button:
    urls, articles, summaries, scores = analyze_news(ticker)
    
    if not summaries:
        st.warning("No articles found for this ticker")
        st.stop()

    # Display results
    st.success(f"Found {len(summaries)} articles for {ticker}")
    
    # Create download data
    csv_data = [['Ticker','Summary', 'Sentiment', 'Sentiment Score', 'URL']]
    
    for i in range(len(summaries)):
        with st.expander(f"Article {i+1}: {summaries[i][:50]}...", expanded=True):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Add image placeholder (you can implement actual image scraping)
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
        data='\n'.join([','.join(map(str,row)) for row in csv_data]),
        file_name=f'{ticker}_news_analysis.csv',
        mime='text/csv'
    )
"
