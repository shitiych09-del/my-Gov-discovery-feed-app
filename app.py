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


# 5. DESAIN ANTARMUKA / USER INTERFACE (UI)


# Menyisipkan CSS Kustom untuk mempercantik kartu rekomendasi (Card UI)
st.markdown("""
    <style>
    /* Desain dasar kartu rekomendasi pariwisata */
    .wisata-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #10b981; /* Warna hijau mint */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 20px;
    }
    /* Desain khusus untuk Event Pemerintah Kota */
    .event-card {
        background-color: #fffbeb; /* Kuning soft */
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #f59e0b; /* Warna kuning emas benderang */
        box-shadow: 0 4px 10px rgba(245, 158, 11, 0.15);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 20px;
    }
    /* Efek interaktif hover (saat mouse mendekat) */
    .wisata-card:hover, .event-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
    }
    /* Efek badge teks */
    .badge {
        background-color: #f1f5f9;
        color: #334155;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

#  HEADER APPS 
# Membuat layout header dengan kolom untuk menempatkan judul besar daerah
st.title("🏛️ Gov-Discovery Feed")
st.markdown("### *Portal Personalisasi Event Kebudayaan & Fasilitas Publik Kota Surabaya*")

# Menggunakan st.info untuk ucapan selamat datang yang rapi dan berwarna biru cerah
st.info(
    "👋 **Selamat datang di platform pintar Smart City Surabaya!** "
    "Sistem cerdas kami secara otomatis mempelajari histori preferensi kunjungan Anda "
    "untuk menyajikan rekomendasi agenda kota dan destinasi publik terbaik yang paling relevan."
)

st.divider()

#  KOMPONEN SIDEBAR (AUTENTIKASI WARGA) 
with st.sidebar:
    st.markdown("### 🔐 Kendali Akun Warga")
    st.markdown("Silakan simulasikan profil warga untuk menguji dinamika perubahan personalisasi umpan (*feed*):")
    
    # Input ID Warga dengan komponen interaktif slider/number
    user_select = st.number_input(
        "🎯 Masukkan ID Warga (1 - 300):", 
        min_value=1, 
        max_value=300, 
        value=1,
        step=1
    )
    
    # Tombol masuk dengan desain yang mencolok
    btn_login = st.button("🔄 Log In & Segarkan Feed", use_container_width=True, type="primary")
    
    if btn_login:
        st.success(f"🔓 Berhasil memuat data Warga ID: {user_select}")
        st.balloons() # Efek balon perayaan interaktif di layar saat klik login!

#  TAMPILAN UTAMA (FEED "FOR YOU") 
st.markdown(f"#### 📌 Rekomendasi Khusus Untuk Anda — **Umpan Warga #{user_select}**")

# Memanggil fungsi mesin rekomendasi hybrid kita (top 9)
rekomendasi_df = get_hybrid_recommendations(user_id=user_select, top_n=9)

# Menampilkan data dalam struktur layout Grid 3 Kolom yang sangat estetik
cols = st.columns(3)

for idx, (_, row) in enumerate(rekomendasi_df.iterrows()):
    col_target = cols[idx % 3] # Alokasi bergantian ke kolom 1, 2, dan 3
    
    with col_target:
        # Cek kondisi: Apakah item termasuk Event Pemda (ID >= 901) atau Wisata Umum
        if row['Place_Id'] >= 901:
            # Menggunakan elemen HTML kustom dengan class 'event-card' (Warna Kuning Emas)
            st.markdown(f"""
                <div class="event-card">
                    <h4 style="margin:0 0 8px 0; color:#b45309;">✨ EVENT KOTA: {row['Place_Name']}</h4>
                    <p style="margin:0 0 10px 0;"><span class="badge">🎪 {row['Category']}</span> &nbsp; <span style="color:#f59e0b;">⭐ {row['Rating']}</span></p>
                    <p style="font-size:0.95rem; color:#4b5563; font-style:italic; line-height:1.4; min-height:80px;">"{row['Description']}"</p>
                    <hr style="margin:10px 0; border:0; border-top:1px solid #fde68a;">
                    <p style="margin:0; font-weight:bold; color:#1e293b; font-size:0.9rem;">💵 Tiket Masuk: <span style="color:#b45309;">Rp {row['Price']:,}</span></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Menggunakan elemen HTML kustom dengan class 'wisata-card' (Warna Hijau Mint)
            st.markdown(f"""
                <div class="wisata-card">
                    <h4 style="margin:0 0 8px 0; color:#065f46;">📍 {row['Place_Name']}</h4>
                    <p style="margin:0 0 10px 0;"><span class="badge">🌳 {row['Category']}</span> &nbsp; <span style="color:#10b981;">⭐ {row['Rating']}</span></p>
                    <p style="font-size:0.95rem; color:#4b5563; line-height:1.4; min-height:80px;">{row['Description']}</p>
                    <hr style="margin:10px 0; border:0; border-top:1px solid #e2e8f0;">
                    <p style="margin:0; font-weight:bold; color:#1e293b; font-size:0.9rem;">💵 Tiket Masuk: <span style="color:#047857;">Rp {row['Price']:,}</span></p>
                </div>
                """, unsafe_allow_html=True)
