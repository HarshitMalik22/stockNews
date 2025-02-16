import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
import requests
import re

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
    with st.spinner(f"Searching news for {ticker}..."):
        search_url = f'https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws'
        # Add a User-Agent to avoid potential blocks or captchas
        r = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, 'html.parser')
        atags = soup.find_all('a')
        hrefs = [link['href'] for link in atags]

        # Clean URLs
        exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
        cleaned_urls = []
        for url in hrefs:
            if 'https://' in url and not any(exc in url for exc in exclude_list):
                matches = re.findall(r'(https?://\S+)', url)
                if matches:
                    res = matches[0].split('&')[0]
                    cleaned_urls.append(res)

        cleaned_urls = list(set(cleaned_urls))[:num_articles]

        # Scrape articles
        articles = []
        for url in cleaned_urls:
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(r.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text = [p.text for p in paragraphs]
                # Limit to first ~350 words
                words = ' '.join(text).split(' ')[:350]
                articles.append(' '.join(words))
            except Exception as e:
                # If scraping fails for any reason, skip this URL
                print(f"Skipping {url} due to error: {e}")
                continue

        # Summarize articles
        summaries = []
        for article in articles:
            input_ids = tokenizer.encode(article, return_tensors="pt", truncation=True)
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

    st.success(f"Found {len(summaries)} articles for {ticker}")
    
    # Prepare CSV data
    csv_data = [['Ticker','Summary', 'Sentiment', 'Sentiment Score', 'URL']]
    
    for i in range(len(summaries)):
        # Short preview for the expander label
        preview_text = summaries[i][:50] + "..."
        with st.expander(f"Article {i+1}: {preview_text}", expanded=True):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Placeholder image
                st.image("https://via.placeholder.com/150", caption="Article Image")
                # Show domain as source
                domain = urls[i].split('//')[1].split('/')[0]
                st.caption(f"Source: {domain}")
                
            with col2:
                st.markdown(f"**Summary:** {summaries[i]}")
                st.markdown(f"**Sentiment:** {scores[i]['label']} ({scores[i]['score']:.2f})")
                st.markdown(f"**URL:** [Link]({urls[i]})")
                
                if st.checkbox("Show full text", key=f"text_{i}"):
                    # Show partial text to avoid flooding
                    st.write(articles[i][:1000] + "...")
            
            csv_data.append([ticker, summaries[i], scores[i]['label'], scores[i]['score'], urls[i]])

    # Create CSV string
    csv_string = '\n'.join([','.join(map(str, row)) for row in csv_data])
    
    # Download button
    st.download_button(
        label="Download Results as CSV",
        data=csv_string,
        file_name=f'{ticker}_news_analysis.csv',
        mime='text/csv'
    )
