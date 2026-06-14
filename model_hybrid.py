import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD

# 1. LOAD DATA YANG SUDAH BERSIH
path_places = r'C:\Users\Dell\OneDrive - Politeknik Elektronika Negeri Surabaya\chiluy - KULIAH EUY\SEMESTER 4\Rekomendasi System\Projek Kecil\Dataset\clean_places_surabaya.csv'
path_ratings = r'C:\Users\Dell\OneDrive - Politeknik Elektronika Negeri Surabaya\chiluy - KULIAH EUY\SEMESTER 4\Rekomendasi System\Projek Kecil\Dataset\clean_ratings_surabaya.csv'

df_places = pd.read_csv(path_places)
df_ratings = pd.read_csv(path_ratings)


# BAGIAN A: CONTENT-BASED FILTERING (TF-IDF)

# Menghitung kemiripan tempat berdasarkan deskripsi teks yang sudah dibersihkan
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_places['Cleaned_Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# BAGIAN B: COLLABORATIVE FILTERING (SVD)

# Melatih model SVD untuk memprediksi rating user menggunakan library Surprise
reader = Reader(rating_scale=(1, 5))
# Biasanya di dataset Kaggle tersebut nama kolomnya adalah 'Place_Ratings'
data = Dataset.load_from_df(df_ratings[['User_Id', 'Place_Id', 'Place_Ratings']], reader)
trainset = data.build_full_trainset()

model_svd = SVD()
model_svd.fit(trainset)


# BAGIAN C: LOGIKA HYBRID RECOMMENDATION

def get_hybrid_recommendations(user_id, top_n=5):
    # 1. Ambil histori tempat yang pernah dikunjungi user dengan rating tertinggi (> 3)
    user_history = df_ratings[(df_ratings['User_Id'] == user_id) & (df_ratings['Place_Ratings'] >= 4)]
    
    # Jika user baru/tidak punya histori bagus, ambil tempat random dengan rating tertinggi secara umum
    if user_history.empty:
        return df_places.sort_values(by='Rating', ascending=False).head(top_n)
    
    # Ambil salah satu Place_Id terakhir yang disukai user sebagai "anchor" (pemicu)
    last_liked_place_id = user_history.iloc[-1]['Place_Id']
    
    # Cari indeks tempat tersebut di dalam dataframe lokal Surabaya
    try:
        idx = df_places[df_places['Place_Id'] == last_liked_place_id].index[0]
    except IndexError:
        # Jika id tidak ditemukan di data Surabaya, ambil indeks pertama saja
        idx = 0

    # 2. Ambil skor kemiripan Content-Based dari tempat tersebut dengan seluruh tempat lain
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    hybrid_scores = []
    for i, content_score in sim_scores:
        place_id = df_places.iloc[i]['Place_Id']
        
        # 3. Hitung skor Collaborative-Based (Prediksi SVD untuk user ini pada tempat ke-i)
        predicted_rating = model_svd.predict(user_id, place_id).est
        # Normalisasi rating (skala 1-5) menjadi skala 0-1 agar setara dengan Cosine Similarity
        normalized_collab_score = (predicted_rating - 1) / 4
        
        # 4. GABUNGKAN KEDUA SKOR (Hybrid Bobot Rata-rata 50:50)
        final_score = (0.5 * content_score) + (0.5 * normalized_collab_score)
        
        # Jika tempat tersebut adalah Event Pemda (ID 900+), beri sedikit bonus bobot (Boost)
        if place_id >= 901:
            final_score += 0.2
            
        hybrid_scores.append((i, final_score))
    
    # Sortir berdasarkan skor akhir tertinggi
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    
    # Ambil top N tempat teratas (kecuali tempat yang menjadi anchor tadi)
    recommend_indices = [i[0] for i in hybrid_scores if df_places.iloc[i[0]]['Place_Id'] != last_liked_place_id][:top_n]
    
    return df_places.iloc[recommend_indices]


# UJI COBA LOGIKA (SIMULASI UNTUK USER ID 1)

print("\n Menguji Sistem Rekomendasi Hybrid ")
target_user = 1
rekomendasi = get_hybrid_recommendations(user_id=target_user, top_n=3)

print(f"\nRekomendasi 'Gov-Discovery Feed' Teratas untuk Warga ID {target_user}:")
for index, row in rekomendasi.iterrows():
    print(f"- {row['Place_Name']} [{row['Category']}] - Rating Wisata: {row['Rating']}")