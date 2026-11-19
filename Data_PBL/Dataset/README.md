# Dokumentasi Dataset PBL Cafind

Dokumen ini mendokumentasikan dataset asli yang digunakan dalam proyek PBL **Cafind**:
*"Sistem Rekomendasi Cafe Berdasarkan Suasana dan Kebutuhan Menggunakan Analisis Sentimen dan Klasifikasi Teks Ulasan Google Maps"*.

---

## 1. Sumber Data Asli

- **Nama File Sumber:** `dump-cafind-202611172252.sql`
- **Format:** PostgreSQL Plain Text SQL Dump
- **Ukuran File:** 4.06 MB (4,257,294 bytes)
- **Status Data:** Sumber data asli dan murni tanpa modifikasi manual atau data sintetis.

---

## 2. Proses Ekstraksi SQL Dump ke CSV

Dataset diekstraksi secara langsung dari blok perintah `COPY ... FROM stdin;` yang terdapat di dalam file dump PostgreSQL menggunakan skrip Python (`extract_sql_to_csv.py`).

### Karakteristik Format COPY PostgreSQL:
- Pemisah kolom menggunakan karakter tab (`\t`).
- Nilai `NULL` direpresentasikan dengan simbol `\N`.
- Karakter khusus (newline, backslash) di-escape secara standar oleh PostgreSQL.
- Penanda akhir data setiap tabel ditandai dengan baris `\.`.

### Langkah Ekstraksi:
1. Membaca isi file SQL dump dengan encoding UTF-8.
2. Menggunakan ekspresi reguler (*regex*) untuk mengekstrak header kolom dan seluruh baris data di antara `COPY public.<tabel> (...) FROM stdin;` dan `\.`.
3. Mengganti representasi `\N` menjadi `None` (missing value / NaN standar pada Pandas).
4. Menyimpan data ke dalam file CSV standar dengan pemisah koma (`,`) menggunakan konfigurasi minimal quoting.

---

## 3. Struktur dan Ringkasan Tabel Hasil Ekstraksi

Hasil ekstraksi menghasilkan 3 file CSV utama:

### A. `cafes.csv`
Menyimpan informasi profil, lokasi, dan metadata operasional dari masing-masing cafe.

- **Jumlah Baris:** 275 entri
- **Jumlah Kolom:** 18 kolom
- **Primary Key:** `id` (UUID)

| Nama Kolom | Tipe Data | Non-Null Count | Deskripsi |
|---|---|---|---|
| `id` | string (UUID) | 275 | Identifier unik untuk setiap cafe |
| `place_id` | string | 0 | Google Maps Place ID (semua bernilai NULL pada dump) |
| `name` | string | 275 | Nama cafe / warkop |
| `address` | string | 275 | Alamat lengkap cafe |
| `latitude` | float | 275 | Titik koordinat garis lintang (latitude) |
| `longitude` | float | 275 | Titik koordinat garis bujur (longitude) |
| `rating` | float | 275 | Nilai rata-rata rating cafe di Google Maps |
| `review_count` | integer | 275 | Total jumlah ulasan yang tercatat di Google Maps |
| `image_url` | string | 0 | Tautan gambar profil cafe (semua bernilai NULL) |
| `categories` | string | 275 | Kategori cafe (misal: "Kafe", "Kedai Kopi") |
| `phone` | string | 275 | Nomor telepon kontak cafe |
| `website` | string | 275 | Tautan website / media sosial cafe |
| `created_at` | timestamp | 275 | Waktu pencatatan cafe ke database |
| `updated_at` | timestamp | 275 | Waktu pembaruan data cafe |
| `price_range` | string | 239 | Rentang harga deskriptif (misal: "Rp 10.000 - 25.000") |
| `price_min` | float | 239 | Estimasi harga minimum menu |
| `price_max` | float | 239 | Estimasi harga maksimum menu |
| `opening_hours` | string (JSON/text) | 269 | Jadwal jam operasional cafe |

---

### B. `reviews.csv`
Menyimpan ulasan teks Google Maps yang diberikan oleh pengunjung cafe.

- **Jumlah Baris:** 7,736 entri
- **Jumlah Kolom:** 10 kolom
- **Primary Key:** `id` (UUID)
- **Foreign Key:** `cafe_id` mereferensikan `cafes.id`

| Nama Kolom | Tipe Data | Non-Null Count | Deskripsi |
|---|---|---|---|
| `id` | string (UUID) | 7,736 | Identifier unik ulasan |
| `cafe_id` | string (UUID) | 7,736 | ID cafe yang diulas |
| `author_name` | string | 7,736 | Nama reviewer / pengguna Google Maps |
| `text` | string | 7,736 | Teks ulasan yang telah diproses/dibersihkan |
| `raw_text` | string | 7,736 | Teks ulasan mentah asli dari Google Maps |
| `rating` | integer | 7,736 | Nilai rating bintang (skala 1 s.d. 5) |
| `published_at` | timestamp | 7,736 | Waktu ulasan dipublikasikan |
| `likes` | integer | 7,736 | Jumlah like/dukungan ulasan dari pengguna lain |
| `is_processed` | boolean | 7,736 | Status pemrosesan ulasan (True/False) |
| `created_at` | timestamp | 7,736 | Waktu ulasan dimasukkan ke sistem |

#### Distribusi Nilai Rating Ulasan:
- **Rating 5:** 6,117 ulasan (79.07%)
- **Rating 4:** 667 ulasan (8.62%)
- **Rating 1:** 557 ulasan (7.20%)
- **Rating 3:** 229 ulasan (2.96%)
- **Rating 2:** 166 ulasan (2.15%)

---

### C. `aspect_sentiments.csv`
Menyimpan hasil ekstraksi sentimen berbasis aspek (*Aspect-Based Sentiment Analysis / ABSA*) dari ulasan cafe.

- **Jumlah Baris:** 9,257 entri
- **Jumlah Kolom:** 6 kolom
- **Primary Key:** `id` (UUID)
- **Foreign Keys:**
  - `review_id` mereferensikan `reviews.id`
  - `cafe_id` mereferensikan `cafes.id`

| Nama Kolom | Tipe Data | Non-Null Count | Deskripsi |
|---|---|---|---|
| `id` | string (UUID) | 9,257 | Identifier unik entri sentimen aspek |
| `review_id` | string (UUID) | 9,257 | ID ulasan terkait |
| `cafe_id` | string (UUID) | 9,257 | ID cafe terkait |
| `aspect` | string | 9,257 | Kategori aspek ulasan |
| `sentiment` | string | 9,257 | Label sentimen (`positive`, `neutral`, `negative`) |
| `confidence` | float | 9,257 | Nilai keyakinan model anotasi (skala 0.0 – 1.0) |

#### Distribusi Kategori Aspek:
- **`food_beverage`:** 3,880 entri (41.91%)
- **`ambience`:** 3,055 entri (33.00%)
- **`facility`:** 1,294 entri (13.98%)
- **`price`:** 1,028 entri (11.11%)

#### Distribusi Label Sentimen:
- **`positive`:** 6,156 entri (66.50%)
- **`neutral`:** 2,716 entri (29.34%)
- **`negative`:** 385 entri (4.16%)

#### Ringkasan Nilai Confidence:
- **Minimum:** 0.7000
- **Kuartil 1 (25%):** 0.7000
- **Median (50%):** 0.8650
- **Rata-rata (Mean):** 0.8292
- **Kuartil 3 (75%):** 0.8842
- **Maksimum:** 0.9570

---

## 4. Integritas Relasi Data

Uji integritas referensial antar-tabel menunjukkan konsistensi penuh:
- Seluruh 5,591 `review_id` unik pada `aspect_sentiments.csv` terdaftar pada `reviews.csv` (100% valid).
- Seluruh 271 `cafe_id` unik pada `reviews.csv` terdaftar pada `cafes.csv` (100% valid).
- Seluruh 270 `cafe_id` unik pada `aspect_sentiments.csv` terdaftar pada `cafes.csv` (100% valid).
- Tidak ada duplikasi baris primary key (`id`) pada ketiga tabel.

### Catatan Penting untuk Modeling Machine Learning:
1. **Relasi Satu Ulasan ke Banyak Aspek (*One-to-Many*):** Satu ulasan dapat memiliki lebih dari satu baris aspek (misalnya membahas aspek suasana dan rasa makanan sekaligus). Saat menggabungkan tabel `reviews` dan `aspect_sentiments`, penggabungan harus dilakukan secara bijak sesuai tujuan pemodelan (klasifikasi ulasan global vs. klasifikasi aspek spesifik).
2. **Ketidakseimbangan Kelas (*Imbalanced Class*):** Kelas sentimen `negative` merupakan kelas minoritas (hanya 4.16%). Oleh karena itu, teknik pemodelan harus mempertimbangkan stratified splitting, pembobotan kelas (*class weight*), serta evaluasi komprehensif menggunakan Precision, Recall, dan Macro/Weighted F1-score.
