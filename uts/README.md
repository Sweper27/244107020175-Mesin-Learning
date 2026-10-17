# Pembagian Tugas Kelompok 2

## Rafi Adrian Prasetya : Dbscan Euclidean, Hdbscan Manhattan, Kmeans Cosine

## Rifqi Aries Saputra : Dbscan Manhattan, Kmeans Euclidean

## Nabillah Umi Purnama : Hdbscan Euclidean, Kmeans Manhattan

## Nayla Akas Oktavia : Dbscan Cosine, Hdbscan Cosine


# Summary eksperimen clustering

Struktur folder :

```text
data/
    cafes.csv
    reviews.csv
summarize_notebooks.py
requirements-clustering.txt
kmeans_*.ipynb      (3 notebook)
dbscan_*.ipynb      (3 notebook)
hdbscan_*.ipynb     (3 notebook)
```

Dari folder paket, pada environment Python yang memiliki dependensi:

```powershell
python -m pip install -r requirements-clustering.txt
python summarize_notebooks.py
```

Runner memakai data/cafes.csv, dengan fallback cafes.csv di sebelah skrip.

Hasil: reports/notebook_summary.csv dan reports/figures/ di dalam folder paket.
