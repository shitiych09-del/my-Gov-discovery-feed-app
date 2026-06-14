import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. LOAD DATASET MENTAH
df_wisata = pd.read_csv(r'C:\Users\Dell\OneDrive - Politeknik Elektronika Negeri Surabaya\chiluy - KULIAH EUY\SEMESTER 4\Rekomendasi System\Projek Kecil\Dataset\tourism_with_id.csv')
df_rating = pd.read_csv(r'C:\Users\Dell\OneDrive - Politeknik Elektronika Negeri Surabaya\chiluy - KULIAH EUY\SEMESTER 4\Rekomendasi System\Projek Kecil\Dataset\tourism_rating.csv')

print(f"Total data awal: {len(df_wisata)} tempat wisata.")

# 2. FILTER KOTA (Fokus ke Surabaya untuk simulasi Smart City)
df_surabaya = df_wisata[df_wisata['City'] == 'Surabaya'].copy()

# 3. SIMULASI TAMBAHAN EVENT PEMERINTAH 
# Kita tambahkan beberapa event budaya/agenda kota temporer secara manual
event_pemda = [
    {
        "Place_Id": 901,
        "Place_Name": "Festival Rujak Uleg 2026",
        "Description": "Festival kebudayaan tahunan kota Surabaya menyambut hari jadi kota, menampilkan pembuatan rujak uleg massal dan pawai budaya lokal.",
        "Category": "Budaya",
        "City": "Surabaya",
        "Price": 0,
        "Rating": 4.8,
        "Lat": -7.245,
        "Long": 112.738
    },
    {
        "Place_Id": 902,
        "Place_Name": "Bazar UMKM dan Kuliner Kenjeran",
        "Description": "Pameran produk lokal kreatif dan sentra kuliner hasil laut olahan UMKM binaan Dinas Koperasi dan Perdagangan kota.",
        "Category": "Pusat Perbelanjaan",
        "City": "Surabaya",
        "Price": 0,
        "Rating": 4.5,
        "Lat": -7.231,
        "Long": 112.799
    },
    {
        "Place_Id": 903,
        "Place_Name": "Surabaya Vaganza (Pawai Bunga)",
        "Description": "Pawai kendaraan hias bunga dan parade budaya yang diselenggarakan oleh Dinas Kebudayaan dan Pariwisata menyusuri jalan protokol.",
        "Category": "Budaya",
        "City": "Surabaya",
        "Price": 0,
        "Rating": 4.9,
        "Lat": -7.265,
        "Long": 112.740
    }
]

# Gabungkan data asli Surabaya dengan simulasi event pemda
df_event = pd.DataFrame(event_pemda)
df_surabaya = pd.concat([df_surabaya, df_event], ignore_index=True)

# 4. TEXT CLEANING (Pembersihan deskripsi untuk TF-IDF nanti)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Mengubah jadi huruf kecil
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca
    
    # Daftar kata umum (stopwords) Indonesia sederhana untuk dibuang
    stopwords = ['dan', 'yang', 'di', 'ke', 'dari', 'ini', 'itu', 'adalah', 'sebagai', 'dengan', 'untuk', 'sebuah', 'kota']
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords]
    
    return " ".join(cleaned_words)

df_surabaya['Cleaned_Description'] = df_surabaya['Description'].apply(clean_text)

# 5. FILTER DATA RATING
# Kita hanya ambil data rating dari user yang pernah menilai tempat-tempat di Surabaya saja
surabaya_place_ids = df_surabaya['Place_Id'].tolist()
df_rating_surabaya = df_rating[df_rating['Place_Id'].isin(surabaya_place_ids)].copy()

# 6. SIMPAN DATA YANG SUDAH BERSIH KE FOLDER BARU
df_surabaya.to_csv(r'C:\Users\Dell\OneDrive - Politeknik Elektronika Negeri Surabaya\chiluy - KULIAH EUY\SEMESTER 4\Rekomendasi System\Projek Kecil\Dataset\clean_places_surabaya.csv', index=False)
df_rating_surabaya.to_csv(r'C:\Users\Dell\OneDrive - Politeknik Elektronika Negeri Surabaya\chiluy - KULIAH EUY\SEMESTER 4\Rekomendasi System\Projek Kecil\Dataset\clean_ratings_surabaya.csv', index=False)

print("\n Preprocessing Selesai ")
print(f"Jumlah tempat & event di Surabaya: {len(df_surabaya)}")
print(f"Jumlah baris interaksi/rating warga: {len(df_rating_surabaya)}")
print("File 'clean_places_surabaya.csv' dan 'clean_ratings_surabaya.csv' berhasil disimpan di folder data!")