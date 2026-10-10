# Prototipe WebGIS Prediksi White Spot Disease (XGBoost + SHAP)

Prototipe skripsi: penerapan XGBoost untuk memprediksi White Spot Disease (WSD) pada tambak udang,
dilengkapi penjelasan SHAP dan pemetaan GIS berbasis web (Streamlit + Folium).

Data latih: Hasan & Haque (2020), 233 tambak udang windu, Bangladesh. Input lokasi pada aplikasi: Indonesia (demonstrasi).

## Langkah menjalankan (VS Code, Windows)

Buka folder `prototipe_wsd_xgboost` di VS Code, lalu buka Terminal (`Ctrl+` `` ` ``):

```powershell
# 1. Buat & aktifkan virtual environment (sekali saja)
python -m venv .venv
.venv\Scripts\activate
#    Bila muncul error "running scripts is disabled", jalankan dulu:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Pasang library (sekali saja)
pip install -r requirements.txt

# 3. Latih ulang model di laptop sendiri (WAJIB, ±1–3 menit)
python jalankan_pipeline.py
#    uji cepat: python jalankan_pipeline.py --cepat

# 4. Jalankan aplikasi web → otomatis terbuka di http://localhost:8501
streamlit run aplikasi/Beranda.py
```

Pilih interpreter `.venv` di VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter*.

> Langkah 3 wajib karena file model (`.joblib`) terikat pada versi library. Melatih ulang di laptop
> sendiri menjamin model cocok dengan versi yang terpasang. Hasil identik karena `RANDOM_STATE` tetap.

## Struktur folder

```
prototipe_wsd_xgboost/
├── data/mentah/            Data asli (jangan diedit)
├── data/olahan/            Hasil pembersihan & split (dibuat otomatis)
├── notebook/               EDA (01_eksplorasi_data.ipynb)
├── kode_sumber/            LAPISAN I – pengolah data
│   ├── konfigurasi.py      Semua pengaturan: path, seed, k-fold, hyperparameter, kamus fitur
│   ├── muat_data.py        T1 baca & validasi skema
│   ├── praproses.py        T2 pembersihan statis, T4 split stratified 80:20
│   ├── pelatihan.py        T5–T7 Pipeline, RandomizedSearchCV + Stratified k-Fold, ablasi
│   ├── evaluasi.py         T8 evaluasi holdout + baseline, T9 validasi spasial
│   ├── interpretasi.py     T10 SHAP (beeswarm, bar, waterfall)
│   └── ekspor_artefak.py   T11 simpan model & metadata
├── model/                  LAPISAN II – artefak model (dibaca aplikasi)
├── hasil/gambar, tabel/    Bahan BAB IV (dibuat otomatis)
├── aplikasi/               LAPISAN III – WebGIS
│   ├── Beranda.py          Halaman utama (entry point)
│   ├── pages/              Prediksi Risiko, Peta Sebaran, Riwayat, Tentang Model, Panduan
│   └── layanan/            Modul: artefak, validasi, prediksi, penjelas (SHAP), peta, basis_data
├── jalankan_pipeline.py    Menjalankan T1–T11 berurutan
└── requirements.txt
```

## Fitur input lokasi (halaman Prediksi Risiko)

| Cara | Keterangan | Butuh internet |
|---|---|---|
| 🗺️ Peta | Klik peta atau ketik koordinat | Ya (tampilan peta) |
| 🏛️ Wilayah | Pilih berjenjang: Negara → Provinsi → Kabupaten/Kota → Kecamatan → Desa/Kelurahan | Ya |
| 🔎 Alamat | Ketik nama tempat bebas (bisa di luar Indonesia) | Ya |
| 📱 Lokasi saya | GPS/lokasi perangkat (izinkan akses lokasi di browser) | Tidak* |

Setelah lokasi ditetapkan, nama wilayah terisi otomatis (reverse geocoding) dan **dapat dikoreksi** sebelum disimpan.
Sumber data: API Wilayah Indonesia (emsifa) dan Nominatim OpenStreetMap (maks. 1 permintaan/detik).
\* Lokasi perangkat hanya berfungsi di `localhost` atau HTTPS; pengisian nama wilayahnya tetap memerlukan internet.

## Mengubah eksperimen

Semua parameter ada di `kode_sumber/konfigurasi.py` (jumlah fold, `N_ITER_SEARCH`, ruang hyperparameter,
ambang kategori risiko, metrik utama). Setelah mengubah, jalankan ulang `python jalankan_pipeline.py`.
