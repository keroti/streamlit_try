import pandas as pd
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
import pickle
import os
import streamlit as st

# -----------------------
# 🛠️ Load everything
# -----------------------
st.set_page_config(page_title="Bible Semantic Search", page_icon="📖", layout="centered")
st.title("📖 Bible Semantic Search")
st.write("Search across multiple Bible versions using intelligent semantic + keyword matching.")

# Load Bible data
@st.cache_data
def load_data():
    df = pd.read_excel('merged_bibles.xlsx')
    def combine_versions(row):
        return f"KJV: {row['Content-kjv']}\nASV: {row['Content-asv']}\nERV: {row['Content-erv']}"
    df['Combined'] = df.apply(combine_versions, axis=1)
    return df

# Load or compute embeddings
@st.cache_resource
def load_embeddings_and_model(df):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    if os.path.exists('corpus_embeddings.pkl'):
        with open('corpus_embeddings.pkl', 'rb') as f:
            corpus_embeddings = pickle.load(f)
    else:
        corpus_embeddings = model.encode(df['Combined'].tolist(), convert_to_tensor=True)
        with open('corpus_embeddings.pkl', 'wb') as f:
            pickle.dump(corpus_embeddings, f)
    return model, corpus_embeddings

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = load_data()
model, corpus_embeddings = load_embeddings_and_model(df)

# -----------------------
# 🚀 Streamlit app
# -----------------------
query = st.text_input("🔍 Enter a Bible search query:")

if query:
    # Extract core keywords
    words = query.lower().split()
    core_keywords = [w for w in words if w not in stop_words]
    st.write(f"🧠 Core keywords extracted: {core_keywords}")

    # Semantic similarity
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]
    df['Score'] = similarities.cpu().numpy() * 100

    # Keyword boosting
    def apply_boost_dynamic(row):
        text = row['Combined'].lower()
        for keyword in core_keywords:
            if keyword in text:
                re
