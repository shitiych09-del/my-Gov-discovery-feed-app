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

st.title("🏛️ Gov-Discovery Feed")
st.markdown("### *Sistem Rekomendasi Event Budaya dan Wisata Publik Kota Surabaya*")
st.write("Selamat datang di portal personalisasi layanan publik daerah. Sistem ini mempelajari preferensi kunjungan Anda untuk menyajikan agenda kota dan fasilitas publik terbaik.")

st.divider()

# Komponen Sidebar untuk Simulasi Login Warga
st.sidebar.header("🔐 Autentikasi Warga")
st.sidebar.write("Simulasikan akun warga untuk melihat personalisasi feed:")
user_select = st.sidebar.number_input("Masukkan ID Warga (1 - 300):", min_value=1, max_value=300, value=1)

# Tombol Eksekusi
if st.sidebar.button("Log In & Refresh Feed"):
    st.sidebar.success(f"Berhasil masuk sebagai Warga ID: {user_select}")

# TAMPILAN UTAMA: FEED "FOR YOU"
st.subheader("📌 Rekomendasi Agenda & Wisata Publik Untuk Anda (Feed For You)")

# Panggil fungsi rekomendasi berdasarkan user yang dipilih (muncul 9 rekomendasi)
rekomendasi_df = get_hybrid_recommendations(user_id=user_select, top_n=9)

# Tampilkan dalam bentuk Grid (3 Kolom x 3 Baris)
cols = st.columns(3)
for idx, (_, row) in enumerate(rekomendasi_df.iterrows()):
    col_target = cols[idx % 3] # Bergantian mengisi kolom 1, 2, 3
    with col_target:
        # Berikan warna border berbeda jika itu merupakan Event Pemda (ID >= 901)
        if row['Place_Id'] >= 901:
            st.info(f"✨ **EVENT KOTA: {row['Place_Name']}**")
        else:
            st.success(f"📍 **{row['Place_Name']}**")
            
        st.caption(f"**Kategori:** {row['Category']} | ⭐ **Rating:** {row['Rating']}")
        st.write(f"_{row['Description']}_")
        st.write(f"💵 **Tiket Masuk:** Rp {row['Price']:,}")
        st.markdown("---")
