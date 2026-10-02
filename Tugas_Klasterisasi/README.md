# Studi Kasus Machine Learning: Clustering Dataset Marketing Campaign

> Proyek ini membandingkan algoritma K-Means dan DBSCAN dengan tiga metrik jarak (Euclidean, Cosine, dan Manhattan) pada data pelanggan. Kualitas model dinilai berdasarkan Silhouette Score, waktu eksekusi, dan penggunaan memori.


## Bagian 1 — Walkthrough Proyek

Berikut adalah tahapan pengerjaan proyek ini dari awal hingga akhir:

---

### Tahap 1: Pemuatan Data (`Load Data`)

Dataset marketing_campaign.csv berisi 2.240 data demografi dan riwayat belanja pelanggan. Karena formatnya dipisahkan dengan tab (\t), data dibaca menggunakan pandas dengan kode berikut:

```python
df = pd.read_csv('data/marketing_campaign.csv', sep='\t')
```

Dataset ini memuat informasi demografis dan perilaku pembelian pelanggan sebuah perusahaan. Terdapat **2.240 baris** data awal yang mewakili pelanggan individu.

---

### Tahap 2: Seleksi Fitur

Dari seluruh kolom yang tersedia, dipilih **16 fitur numerik yang relevan** untuk keperluan klasterisasi:

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

Setelah penghapusan baris yang mengandung nilai kosong (`dropna()`), tersisa **2.216 baris** data yang siap diproses.

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
Seluruh data disesuaikan skalanya dengan StandardScaler. Langkah ini sangat penting agar kolom dengan angka besar (seperti Pendapatan) tidak mendominasi perhitungan :

```python
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_fitur)
# Hasil: Shape (2216, 16)
```

#### c. Normalisasi L2 (Khusus Metrik Cosine)
Khusus untuk pengujian metrik Cosine, data dinormalisasi kembali menggunakan Normalizer(norm='l2') agar hasil perhitungan jarak antar datanya lebih akurat :

```python
from sklearn.preprocessing import Normalizer
X_l2 = Normalizer(norm='l2').fit_transform(X_scaled)
```

---

### Tahap 4: Reduksi Dimensi dengan PCA

**Principal Component Analysis (PCA)** digunakan dengan dua tujuan:

1. Mempermudah pembuatan grafik saat mencari jumlah klaster terbaik (seperti pada metode Elbow).
2. Menampilkan hasil pengelompokan (klaster) ke dalam grafik 2D agar pola dan sebaran data pelanggan lebih mudah dilihat dan dipahami.

```python
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
v1, v2 = pca.explained_variance_ratio_ * 100
# v1, v2 = persentase variansi yang dijelaskan oleh PC1 dan PC2
```

PCA menghasilkan dua komponen utama (PC1 dan PC2) yang menangkap sebagian besar variansi data, memungkinkan visualisasi klaster dalam ruang 2D.

---

### Tahap 5: Penentuan Jumlah Klaster (Elbow Method — K-Means)

Untuk algoritma K-Means, jumlah klaster terbaik dicari menggunakan Metode Elbow. Caranya adalah dengan menguji pembagian data dari 1 hingga 10 klaster, lalu mencatat tingkat kesalahannya. Jumlah klaster yang paling ideal ditandai oleh titik yang membelok tajam seperti siku pada grafik hasil pengujian.

```python
K_range = range(1, 11)
sse = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sse.append(km.inertia_)
```

Berdasarkan hasil elbow pada grafik, nilai **K = 4** dipilih sebagai jumlah klaster optimal untuk ketiga variasi metrik K-Means.

**Grafik Elbow yang dihasilkan:**
- `outputs/a1_elbow.png` — Elbow Method untuk K-Means Euclidean & Manhattan
- `outputs/a2_elbow.png` — Elbow Method untuk K-Means Cosine (data L2-normalized)

---

### Tahap 6: Tuning Parameter DBSCAN (K-Distance Graph)

Untuk algoritma **DBSCAN**, parameter kritis adalah `eps` (radius lingkungan) dan `min_samples` (jumlah minimum tetangga). Nilai `eps` optimal ditentukan menggunakan **K-Distance Graph**:

```python
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=MIN_SAMPLES, metric='cosine')
nn.fit(X_l2)
distances, _ = nn.kneighbors(X_l2)
k_dist = np.sort(distances[:, -1])[::-1]
```

Titik patahan (*knee*) pada grafik jarak ini menunjukkan nilai `eps` yang ideal. Grafik ini disimpan di `outputs/a4_kdistance_cosine.png`.

**Parameter DBSCAN yang digunakan:**

| Metrik | `eps` | `min_samples` |
|---|---|---|
| Euclidean | 2.0 | 8 |
| Cosine | 0.15 | 8 |
| Manhattan | 2.0 | 8 |

> **Catatan:** Nilai `eps` untuk Cosine lebih kecil (0.15) karena rentang nilai jarak Cosine (0–2) berbeda secara signifikan dengan jarak Euclidean dan Manhattan pada data berdimensi tinggi.

---

### Tahap 7: Pelatihan Algoritma Klasterisasi

Setiap kombinasi algoritma + metrik dijalankan dengan pengukuran waktu eksekusi (`time`) dan penggunaan memori puncak (`tracemalloc`).

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

> **Catatan Implementasi:** `sklearn.KMeans` secara native menggunakan jarak Euclidean. Untuk mensimulasikan metrik **Cosine**, data dinormalisasi L2 sebelum dilatih (setara matematis). Untuk metrik **Manhattan**, pelatihan tetap menggunakan Euclidean, namun **Silhouette Score** dihitung secara terpisah dengan `metric='manhattan'`.

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

DBSCAN secara otomatis menentukan jumlah klaster dan mengidentifikasi **titik noise** (label = `-1`), yaitu data yang tidak masuk dalam klaster manapun.

---

### Tahap 8: Evaluasi dengan Silhouette Score

**Silhouette Score** digunakan sebagai metrik evaluasi utama untuk mengukur kualitas klasterisasi. Nilai berkisar dari **-1** (sangat buruk) hingga **+1** (sempurna), di mana nilai mendekati 1 menandakan klaster yang kompak dan terpisah dengan baik.

```python
sil = silhouette_score(X_scaled, labels, metric='euclidean')  # atau 'cosine'/'manhattan'
```

Untuk DBSCAN, Silhouette Score hanya dihitung dari **titik non-noise** (menghilangkan label `-1`):

```python
mask_valid = labels != -1
sil = silhouette_score(X_scaled[mask_valid], labels[mask_valid], metric='euclidean')
```

---

### Tahap 9: Visualisasi dan Penyimpanan Hasil

Hasil klasterisasi divisualisasikan dalam **scatter plot 2D** menggunakan koordinat PCA. Setiap klaster ditampilkan dengan warna berbeda, dan centroid K-Means ditandai dengan simbol bintang merah.

Untuk DBSCAN, **titik noise** ditampilkan dengan simbol `x` berwarna hitam, memudahkan identifikasi outlier.

Semua visualisasi disimpan ke folder `outputs/`:

| File Output | Deskripsi |
|---|---|
| `a1_elbow.png` | Elbow Method — K-Means Euclidean & Manhattan |
| `a1_kmeans_plot.png` | Scatter Plot PCA — K-Means Euclidean/Manhattan |
| `a2_elbow.png` | Elbow Method — K-Means Cosine |
| `a2_kmeans_cosine_plot.png` | Scatter Plot PCA — K-Means Cosine |
| `a3_dbscan_plot.png` | Scatter Plot PCA — DBSCAN Euclidean |
| `a3_dbscan_manhattan_plot.png` | Scatter Plot PCA — DBSCAN Manhattan |
| `a4_dbscan_cosine_plot.png` | Scatter Plot PCA — DBSCAN Cosine |
| `a4_kdistance_cosine.png` | K-Distance Graph — Tuning `eps` DBSCAN Cosine |

---

## Bagian 2 — Tabel Komparasi Hasil

Berikut adalah ringkasan hasil evaluasi dari keenam eksperimen klasterisasi. Data dikumpulkan langsung dari output notebook masing-masing.

### Tabel Utama: Perbandingan Semua Metode

| Algoritma & Metrik Jarak | Jumlah Klaster | Waktu Eksekusi | Penggunaan Memori | Silhouette Score |
|:---|:---:|:---:|:---:|:---:|
| **K-Means + Euclidean** | 4 | 0.1410 detik | 766.62 KB (0.7486 MB) | 0.1638 |
| **K-Means + Cosine** | 4 | 0.1382 detik | 766.92 KB (0.7489 MB) | **0.3363** |
| **K-Means + Manhattan** | 4 | 0.1285 detik | 766.83 KB (0.7489 MB) | 0.1928 |
| **DBSCAN + Euclidean** | 2 | 2.5240 detik | 1,353.69 KB (1.3220 MB) | 0.2378 |
| **DBSCAN + Cosine** | 5 | 0.3687 detik | 96,628.66 KB (94.3639 MB) | 0.2801 |
| **DBSCAN + Manhattan** | 2 | 1.7148 detik | 1,213.09 KB (1.1847 MB) | 0.3888 |

### Tabel Detail: DBSCAN — Informasi Tambahan Titik Noise

| Algoritma & Metrik Jarak | Jumlah Klaster | Titik Noise | Persentase Noise | Silhouette Score |
|:---|:---:|:---:|:---:|:---:|
| **DBSCAN + Euclidean** | 2 | 893 titik | 40.3% | 0.2378 |
| **DBSCAN + Cosine** | 5 | 617 titik | 27.8% | 0.2801 |
| **DBSCAN + Manhattan** | 2 | 1,819 titik | 82.1% | 0.3888 |

> **Catatan:** Silhouette Score DBSCAN dihitung hanya pada titik non-noise. Angka ini bukan representasi kualitas seluruh dataset, melainkan kualitas klaster yang terbentuk saja.

---

## Bagian 3 — Kesimpulan & Analisis

### Temuan Utama

Berdasarkan tabel komparasi di atas, beberapa kesimpulan penting dapat ditarik:

#### 1. K-Means + Cosine — Kandidat Terbaik Secara Keseluruhan

K-Means dengan metrik Cosine mencatat **Silhouette Score = 0.3363**, tertinggi di antara semua varian K-Means. Pencapaian ini diraih dengan:
- Waktu eksekusi tercepat di antara K-Means: **0.1382 detik**
- Penggunaan memori efisien: hanya **~767 KB**
- Membentuk **4 klaster** yang relatif seimbang dan terpisah

Implementasinya menggunakan teknik **L2 Normalization** sebelum pelatihan KMeans, yang secara matematis setara dengan menggunakan jarak Cosine. Metrik Cosine cocok untuk data pemasaran karena lebih mengukur **kemiripan pola arah perilaku** pelanggan, bukan besaran absolut pengeluaran mereka.

#### 2. DBSCAN + Manhattan — Silhouette Score Tertinggi Secara Keseluruhan

DBSCAN dengan metrik Manhattan memiliki **Silhouette Score = 0.3888**, nilai tertinggi dari semua eksperimen. Namun, hasil ini perlu diinterpretasikan dengan hati-hati karena:
- Hanya **397 dari 2.216 titik** (17.9%) yang masuk dalam klaster — sisanya diklasifikasikan sebagai noise
- **82.1% data dianggap outlier/noise** — ini menandakan parameter `eps=2.0` terlalu ketat untuk metrik Manhattan
- Klaster yang terbentuk hanya **2 klaster**, yang tidak informatif secara bisnis

> Silhouette Score tinggi pada DBSCAN Manhattan bersifat *menyesatkan* — skor tersebut hanya mencerminkan kohesivitas sebagian kecil data, bukan kualitas segmentasi keseluruhan dataset.

#### 3. DBSCAN + Cosine — Paling Seimbang di Antara Varian DBSCAN

DBSCAN Cosine menghasilkan **5 klaster** dengan hanya **27.8% noise** — paling sedikit dibanding varian DBSCAN lain. Meskipun penggunaan memorinya sangat tinggi (**94.36 MB**), konfigurasi ini memberikan segmentasi yang lebih granular dan representatif.

#### 4. K-Means + Euclidean — Performa Terendah di Kelasnya

K-Means Euclidean memiliki Silhouette Score terendah (**0.1638**). Hal ini terjadi karena data marketing memiliki skala dan distribusi yang sangat bervariasi antar fitur; meskipun telah distandarisasi, jarak Euclidean di ruang berdimensi tinggi (16 dimensi) rentan terhadap *curse of dimensionality*.

---

### Rekomendasi Metode Optimal

Berdasarkan keseimbangan antara kualitas klasterisasi, kecepatan, efisiensi memori, dan interpretabilitas hasil bisnis:

| Kriteria | Metode Terbaik |
|---|---|
| **Kualitas klaster tertinggi (K-Means)** | K-Means + Cosine (Silhouette: 0.3363) |
| **Eksekusi tercepat** | K-Means + Manhattan (0.1285 detik) |
| **Efisiensi memori** | Semua K-Means (~767 KB) |
| **Granularitas segmen** | DBSCAN + Cosine (5 klaster, 27.8% noise) |
| **Rekomendasi keseluruhan** | **K-Means + Cosine** |

### Alasan Rekomendasi K-Means + Cosine

1. **Silhouette Score terbaik** di antara semua varian K-Means (0.3363), menandakan klaster yang lebih kompak dan terpisah.
2. **Semua 2.216 pelanggan** berhasil disegmentasikan tanpa ada yang dianggap noise — ideal untuk keperluan strategi pemasaran.
3. **Waktu dan memori efisien** — sangat cepat (~0.14 detik) dan hemat memori (~767 KB).
4. **Metrik Cosine relevan secara domain** — pada data pemasaran, pola relatif perilaku pelanggan (arah vektor) lebih bermakna dibanding besaran absolut. Dua pelanggan dengan pola pembelian serupa namun berbeda pendapatan akan ditempatkan dalam klaster yang sama.
5. **4 klaster** memberikan segmentasi yang actionable untuk strategi pemasaran (misalnya: pelanggan premium, pelanggan sensitif harga, pelanggan kasual, dan pelanggan tidak aktif).

---