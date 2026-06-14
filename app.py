import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD


# 1. KONFIGURASI HALAMAN STREAMLIT

st.set_page_config(
    page_title="Gov-Discovery Feed",
    page_icon="🏛️",
    layout="wide"
)


# 2. LOAD DATA & CACHING (Agar Web Cepat)

import os

print("Current working directory:", os.getcwd())
print("Files in dataset folder:", os.listdir("dataset"))

path_places = "dataset/clean_places_surabaya.csv"
path_ratings = "dataset/clean_ratings_surabaya.csv"

@st.cache_data
def load_data():
    df_p = pd.read_csv(path_places)
    df_r = pd.read_csv(path_ratings)
    return df_p, df_r

df_places, df_ratings = load_data()


# 3. KREASI MODEL (TF-IDF & SVD)

# Content-Based Matrix
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_places['Cleaned_Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Collaborative SVD
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_ratings[['User_Id', 'Place_Id', 'Place_Ratings']], reader)
trainset = data.build_full_trainset()
model_svd = SVD()
model_svd.fit(trainset)


# 4. FUNGSI LOGIKA HYBRID

def get_hybrid_recommendations(user_id, top_n=9): # Di-set 9 biar pas membentuk grid 3x3
    user_history = df_ratings[(df_ratings['User_Id'] == user_id) & (df_ratings['Place_Ratings'] >= 4)]
    
    if user_history.empty:
        return df_places.sort_values(by='Rating', ascending=False).head(top_n)
    
    last_liked_place_id = user_history.iloc[-1]['Place_Id']
    
    try:
        idx = df_places[df_places['Place_Id'] == last_liked_place_id].index[0]
    except IndexError:
        idx = 0

    sim_scores = list(enumerate(cosine_sim[idx]))
    
    hybrid_scores = []
    for i, content_score in sim_scores:
        place_id = df_places.iloc[i]['Place_Id']
        predicted_rating = model_svd.predict(user_id, place_id).est
        normalized_collab_score = (predicted_rating - 1) / 4
        
        final_score = (0.5 * content_score) + (0.5 * normalized_collab_score)
        
        # Boost Khusus Event Pemerintah (ID 901 ke atas)
        if place_id >= 901:
            final_score += 0.2
            
        hybrid_scores.append((i, final_score))
    
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    recommend_indices = [i[0] for i in hybrid_scores if df_places.iloc[i[0]]['Place_Id'] != last_liked_place_id][:top_n]
    
    return df_places.iloc[recommend_indices]


# 5. DESAIN ANTARMUKA / USER INTERFACE (UI)

import streamlit as st

# Sidebar dengan nuansa budaya
st.sidebar.image("https://github.com/shitiych09-del/my-Gov-discovery-feed-app/blob/main/dataset/images.jfif", width=180)
st.sidebar.title("🏙️ Portal Warga Surabaya")
user_id = st.sidebar.number_input("Masukkan ID Warga (1-300):", min_value=1, max_value=300, value=1)
st.sidebar.button("🔑 Log In & Refresh Feed")

# Header dengan nuansa budaya
st.markdown("<h1 style='color: crimson;'>🎉 Gov-Discovery Feed Surabaya 🎉</h1>", unsafe_allow_html=True)
st.markdown("**Sistem Rekomendasi Event Budaya & Wisata Publik Kota Surabaya**")
st.write("🌺 Selamat datang! Nikmati rekomendasi agenda kota dengan nuansa budaya khas Surabaya.")

# Filter interaktif
st.subheader("🔍 Filter Rekomendasi")
kategori = st.selectbox("Pilih kategori event:", ["Semua", "Budaya", "Kuliner", "Wisata"])
rating_min = st.slider("Minimal rating:", 0.0, 5.0, 4.0)

# Feed rekomendasi dengan gaya card budaya
st.subheader("🌸 Rekomendasi Agenda & Wisata Publik Untuk Anda")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://via.placeholder.com/200x150?text=Festival+Rujak+Uleg", caption="Festival Rujak Uleg")
    st.write("Kategori: Budaya | ⭐ 4.8")
    st.button("👍 Hadiri")

with col2:
    st.image("https://via.placeholder.com/200x150?text=Bazar+UMKM+Kenjeran", caption="Bazar UMKM Kenjeran")
    st.write("Kategori: Kuliner | ⭐ 4.5")
    st.button("📌 Simpan")

with col3:
    st.image("https://via.placeholder.com/200x150?text=Surabaya+Vaganza", caption="Surabaya Vaganza")
    st.write("Kategori: Budaya | ⭐ 4.9")
    st.button("🔔 Ingatkan")

# Statistik tambahan
st.subheader("📊 Statistik Event Budaya Surabaya")
st.bar_chart({"Budaya": [12], "Kuliner": [8], "Wisata": [6]})
