# Studi Kasus Machine Learning: Clustering Dataset Marketing Campaign

### kode ini membandingkan algoritma K-Means dan DBSCAN dengan tiga metrik jarak (Euclidean, Cosine, dan Manhattan) pada data pelanggan. Kualitas model dinilai berdasarkan Silhouette Score, waktu eksekusi, dan penggunaan memori.


## Bagian 1 — Walkthrough Proyek

Berikut adalah tahapan pengerjaan proyek ini dari awal hingga akhir:


### Tahap 1: Pemuatan Data (Load Data)

Dataset marketing_campaign.csv berisi 2.240 data demografi dan riwayat belanja pelanggan. Karena formatnya dipisahkan dengan tab (\t), data dibaca menggunakan pandas dengan kode berikut:

```python
df = pd.read_csv('data/marketing_campaign.csv', sep='\t')
```

### Tahap 2: Seleksi Fitur

Dari seluruh kolom yang tersedia, dipilih 16 fitur untuk keperluan klasterisasi:

| # | Fitur | Deskripsi |
|---|---|---|
| 1 | `Year_Birth` | Tahun lahir pelanggan |
| 2 | `Income` | Pendapatan tahunan |
| 3 | `Kidhome` | Jumlah anak kecil di rumah |
| 4 | `Teenhome` | Jumlah remaja di rumah |
| 5 | `Recency` | Hari sejak pembelian terakhir |
| 6 | `MntWines` | Pengeluaran untuk anggur (2 tahun terakhir) |
| 7 | `MntFruits` | Pengeluaran untuk buah-buahan |
| 8 | `MntMeatProducts` | Pengeluaran untuk daging |
| 9 | `MntFishProducts` | Pengeluaran untuk ikan |
| 10 | `MntSweetProducts` | Pengeluaran untuk makanan manis |
| 11 | `MntGoldProds` | Pengeluaran untuk produk emas/premium |
| 12 | `NumDealsPurchases` | Jumlah pembelian dengan diskon |
| 13 | `NumWebPurchases` | Jumlah pembelian via website |
| 14 | `NumCatalogPurchases` | Jumlah pembelian via katalog |
| 15 | `NumStorePurchases` | Jumlah pembelian langsung di toko |
| 16 | `NumWebVisitsMonth` | Jumlah kunjungan website per bulan |

```python
kolom_fitur = [
    'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth'
]
df_fitur = df[kolom_ada].dropna()
```

Setelah penghapusan baris yang mengandung nilai kosong (dropna()), tersisa 2.216 baris data.

---

### Tahap 3: Prapemrosesan Data (`Preprocessing`)

#### a. Encoding Kolom Kategorikal
Jika ada data kategori atau teks, data tersebut diubah menjadi angka menggunakan pd.get_dummies() :

```python
kolom_teks = df_fitur.select_dtypes(include='object').columns.tolist()
if kolom_teks:
    df_fitur = pd.get_dummies(df_fitur, columns=kolom_teks, drop_first=True)
```

#### b. Standardisasi (Z-Score Scaling)
Seluruh data disesuaikan skalanya dengan StandardScaler. Ini dilakukan agar kolom dengan angka besar (seperti pendapatan) tidak mendominasi perhitungan :

```python
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_fitur)
# Hasil: Shape (2216, 16)
```

#### c. Normalisasi L2 (Khusus Metrik Cosine)
Untuk cosine, data dinormalisasi menggunakan Normalizer(norm='l2') agar hasil perhitungan jarak antar datanya lebih akurat :

```python
from sklearn.preprocessing import Normalizer
X_l2 = Normalizer(norm='l2').fit_transform(X_scaled)
```

---

### Tahap 4: Reduksi Dimensi dengan PCA

Tujuan PCA :

1. Mempermudah pembuatan grafik saat mencari jumlah klaster terbaik (seperti pada metode Elbow).
2. Menampilkan hasil klaster ke dalam grafik agar pola dan sebaran data pelanggan lebih mudah dilihat dan dipahami.

```python
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
v1, v2 = pca.explained_variance_ratio_ * 100
# v1, v2 = persentase variansi yang dijelaskan oleh PC1 dan PC2
```

### Tahap 5: Penentuan Jumlah Klaster (Elbow Method — K-Means)

Metode Elbow mencari jumlah kelompok terbaik di K-Means dengan melihat grafik, pilih titik saat garisnya menekuk tajam, karena di titik itu pembagian kelompok paling optimal.

```python
K_range = range(1, 11)
sse = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sse.append(km.inertia_)
```

Berdasarkan hasil elbow pada grafik, nilai K = 3 dipilih sebagai jumlah klaster optimal untuk ketiga variasi metrik K-Means.

### Tahap 6: Tuning Parameter DBSCAN (K-Distance Graph)

Untuk algoritma DBSCAN, parameter kritis adalah eps (radius lingkungan) dan min_samples (jumlah minimum tetangga). Nilai eps optimal ditentukan menggunakan K-Distance Graph :

```python
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=MIN_SAMPLES, metric='cosine')
nn.fit(X_l2)
distances, _ = nn.kneighbors(X_l2)
k_dist = np.sort(distances[:, -1])[::-1]
```

Titik patahan (*knee*) pada grafik jarak ini menunjukkan nilai eps yang ideal. Grafik ini disimpan di outputs/a4_kdistance_cosine.png.

Parameter DBSCAN yang digunakan : 

| Metrik | `eps` | `min_samples` |
|---|---|---|
| Euclidean | 2.0 | 8 |
| Cosine | 0.15 | 8 |
| Manhattan | 2.0 | 8 |

### Tahap 7: Pelatihan Algoritma Klasterisasi

Setiap kombinasi algoritma + metrik dijalankan dengan pengukuran waktu eksekusi (time) dan penggunaan memori puncak (tracemalloc).

#### A. K-Means

```python
tracemalloc.start()
t_mulai = time.time()

km = KMeans(n_clusters=K_TERBAIK, random_state=42, n_init=10)
lbl = km.fit_predict(X_scaled)   # Untuk Euclid & Manhattan
# atau
lbl = km.fit_predict(X_l2)       # Untuk Cosine

t_selesai = time.time()
_, mem_puncak = tracemalloc.get_traced_memory()
tracemalloc.stop()
```

#### B. DBSCAN

```python
tracemalloc.start()
t_mulai = time.time()

db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='euclidean')  # atau 'cosine'/'manhattan'
lbl = db.fit_predict(X_scaled)

t_selesai = time.time()
_, mem_puncak = tracemalloc.get_traced_memory()
tracemalloc.stop()
```

DBSCAN secara otomatis menentukan jumlah klaster dan mengidentifikasi noise (label = -1), yaitu data yang tidak masuk dalam klaster manapun.

### Tahap 8: Evaluasi dengan Silhouette Score

Silhouette Score digunakan untuk mengukur kualitas klasterisasi. Nilai dari -1 (sangat buruk) hingga +1 (sempurna), nilai mendekati 1 menandakan klaster kompak dan terpisah dengan baik.

```python
sil = silhouette_score(X_scaled, labels, metric='euclidean')  # atau 'cosine'/'manhattan'
```

Untuk DBSCAN, Silhouette Score hanya dihitung dari titik non-noise (menghilangkan label `-1`):

```python
mask_valid = labels != -1
sil = silhouette_score(X_scaled[mask_valid], labels[mask_valid], metric='euclidean')
```

### Tahap 9: Visualisasi dan Penyimpanan Hasil

Hasil klasterisasi divisualisasikan dalam scatter plot menggunakan koordinat PCA. Setiap klaster ditampilkan dengan warna berbeda, dan centroid K-Means ditandai dengan simbol bintang merah.

Untuk DBSCAN, titik noise ditampilkan dengan simbol x berwarna hitam, memudahkan identifikasi outlier.

Semua visualisasi disimpan ke folder outputs/.

## Bagian 2 — Tabel Komparasi Hasil

Berikut adalah ringkasan hasil evaluasi dari keenam eksperimen klasterisasi. Data dikumpulkan langsung dari output notebook masing-masing.

### Tabel Utama: Perbandingan Semua Metode

| Algoritma & Metrik Jarak | Jumlah Klaster | Waktu Eksekusi | Penggunaan Memori | Silhouette Score |
|:---|:---:|:---:|:---:|:---:|
| K-Means + Euclidean | 4 | 0.1410 detik | 766.62 KB (0.7486 MB) | 0.1638 |
| K-Means + Cosine | 4 | 0.1382 detik | 766.92 KB (0.7489 MB) | 0.3363 |
| K-Means + Manhattan | 4 | 0.1285 detik | 766.83 KB (0.7489 MB) | 0.1928 |
| DBSCAN + Euclidean | 2 | 2.5240 detik | 1,353.69 KB (1.3220 MB) | 0.2378 |
| DBSCAN + Cosine | 5 | 0.3687 detik | 96,628.66 KB (94.3639 MB) | 0.2801 |
| DBSCAN + Manhattan | 2 | 1.7148 detik | 1,213.09 KB (1.1847 MB) | 0.3888 |

### Tabel Detail: DBSCAN — Informasi Tambahan Titik Noise

| Algoritma & Metrik Jarak | Jumlah Klaster | Titik Noise | Persentase Noise | Silhouette Score |
|:---|:---:|:---:|:---:|:---:|
| DBSCAN + Euclidean | 2 | 893 titik | 40.3% | 0.2378 |
| DBSCAN + Cosine | 5 | 617 titik | 27.8% | 0.2801 |
| DBSCAN + Manhattan | 2 | 1,819 titik | 82.1% | 0.3888 |

## Bagian 3 — Kesimpulan & Analisis

### Temuan Utama

Berdasarkan tabel di atas, K-Means + Cosine — Kandidat Terbaik Secara Keseluruhan. 

K-Means + Cosine mencatat Silhouette Score = 0.3363, tertinggi di antara semua varian K-Means :
- Waktu eksekusi tercepat di antara K-Means : 0.1382 detik
- Penggunaan memori efisien: hanya ~767 KB
- Membentuk 4 klaster yang relatif seimbang dan terpisah

### Rekomendasi Metode Optimal

| Kriteria | Metode Terbaik |
|---|---|
| Kualitas klaster tertinggi (K-Means) | K-Means + Cosine (Silhouette: 0.3363) |
| Eksekusi tercepat | K-Means + Manhattan (0.1285 detik) |
| Efisiensi memori | Semua K-Means (~767 KB) |
| Granularitas segmen | DBSCAN + Cosine (5 klaster, 27.8% noise) |
| Rekomendasi keseluruhan | K-Means + Cosine |

### Alasan Rekomendasi K-Means + Cosine

1. Skor Silhouette tertinggi (0,3363): Membuktikan kelompok data terbentuk rapi dan terpisah jelas.
2. Semua data terpakai: Seluruh 2.216 pelanggan berhasil dikelompokkan tanpa ada data yang terbuang (noise).
3. Cepat dan hemat: Proses komputasi sangat singkat (~0,14 detik) dan minim memori (~767 KB).
4. Fokus pada pola kebiasaan: Metrik Cosine melihat kemiripan perilaku belanja, bukan nominal uangnya—pelanggan dengan kebiasaan serupa tetap masuk kelompok yang sama meski beda pendapatan.
5. Mudah dieksekusi: Pembagian 3 kelompok sangat pas untuk aksi pemasaran (misal: pelanggan premium, pemburu promo, kasual).