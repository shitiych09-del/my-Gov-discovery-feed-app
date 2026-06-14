import pandas as pd
import matplotlib
# Biar RAM laptop gak capek, simpan langsung tanpa buka jendela popup
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# Load data bersih
path_places = r'C:\Users\Dell\OneDrive - Politeknik Elektronika Negeri Surabaya\chiluy - KULIAH EUY\SEMESTER 4\Rekomendasi System\Projek Kecil\Dataset\clean_places_surabaya.csv'
path_ratings = r'C:\Users\Dell\OneDrive - Politeknik Elektronika Negeri Surabaya\chiluy - KULIAH EUY\SEMESTER 4\Rekomendasi System\Projek Kecil\Dataset\clean_ratings_surabaya.csv'

df_places = pd.read_csv(path_places)
df_ratings = pd.read_csv(path_ratings)

# Set tema dasar seaborn
sns.set_theme(style="whitegrid")

# GRAPH 1: Distribusi Kategori Wisata/Event (Sudah fix dari Future Warning)
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(
    data=df_places, 
    y='Category', 
    order=df_places['Category'].value_counts().index, 
    hue='Category', # Biar ga kena FutureWarning
    legend=False, 
    palette='viridis', 
    ax=ax
)
ax.set_title('Distribusi Kategori Wisata dan Event di Surabaya')
ax.set_xlabel('Jumlah Tempat')
ax.set_ylabel('Kategori')
plt.tight_layout()
fig.savefig('distribusi_kategori.png', dpi=300)
plt.close(fig)

# GRAPH 2: Distribusi Rating Warga (Sudah fix dari warna 'creams' dan Future Warning)
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.countplot(
    data=df_ratings, 
    x='Place_Ratings', 
    hue='Place_Ratings', # Biar ga kena FutureWarning
    legend=False, 
    palette='Blues', # Diubah ke palet warna resmi yang estetik dan valid
    ax=ax2
) 
ax2.set_title('Distribusi Pemberian Rating oleh Warga')
ax2.set_xlabel('Rating (Skala 1-5)')
ax2.set_ylabel('Total Interaksi')
plt.tight_layout()
fig2.savefig('distribusi_rating.png', dpi=300)
plt.close(fig2)

print("EDA Sukses Besar! Gambar 'distribusi_kategori.png' dan 'distribusi_rating.png' berhasil ")