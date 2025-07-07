import pandas as pd
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
import streamlit as st

# -----------------------
# ⚙️ Streamlit + Page
# -----------------------
st.set_page_config(page_title="Bible Semantic Search",
                   page_icon="📖", layout="centered")
st.title("📖 Bible Semantic Search")
st.write("Search across multiple Bible versions using intelligent semantic + keyword matching.")

# -----------------------
# 📚 Load Bible data
# -----------------------


@st.cache_data
def load_data():
    df = pd.read_excel('merged_bibles.xlsx')

    def combine_versions(row):
        return f"KJV: {row['Content-kjv']}\nASV: {row['Content-asv']}\nERV: {row['Content-erv']}"
    df['Combined'] = df.apply(combine_versions, axis=1)
    return df

# -----------------------
# 🔥 Load model & embeddings
# -----------------------


@st.cache_resource
def load_embeddings_and_model(df):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # Encode fresh on this platform (avoids pickle device errors)
    corpus_embeddings = model.encode(
        df['Combined'].tolist(), convert_to_tensor=True)
    return model, corpus_embeddings


# -----------------------
# 📝 Load stopwords
# -----------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------
# 🚀 Main app flow
# -----------------------
df = load_data()
model, corpus_embeddings = load_embeddings_and_model(df)

query = st.text_input("🔍 Enter a Bible search query:")

if query:
    # Extract core keywords
    words = query.lower().split()
    core_keywords = [w for w in words if w not in stop_words]
    st.write(f"🧠 Core keywords extracted: {core_keywords}")

    # Compute semantic similarity
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]
    df['Score'] = similarities.cpu().numpy() * 100

    # Keyword boosting
    def apply_boost_dynamic(row):
        text = row['Combined'].lower()
        for keyword in core_keywords:
            if keyword in text:
                return row['Score'] + 20
        return row['Score']

    df['AdjustedScore'] = df.apply(apply_boost_dynamic, axis=1)

    # Top results
    results = df.sort_values('AdjustedScore', ascending=False).head(5)

    if results['AdjustedScore'].max() < 20:
        st.warning(
            "⚠️ No strong matches found (all under 20%). Try rephrasing your query.")
    else:
        st.subheader(f"📖 Top matches for: \"{query}\" (showing top 5)")
        for _, row in results.iterrows():
            ref = f"{row['Book']} {row['Chapter']}:{row['VerseNum']}"
            st.markdown(
                f"**🔹 {ref} — {row['AdjustedScore']:.2f}% adjusted match**")
            st.code(row['Combined'], language='text')

        books_chapters = results.apply(
            lambda row: f"{row['Book']} {row['Chapter']}", axis=1).unique()
        joined_books = ', '.join(books_chapters)
        st.info(
            f"📝 Insight:\nThis topic appears prominently in {joined_books}.")
