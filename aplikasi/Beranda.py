"""
Halaman utama aplikasi WebGIS Prediksi WSD.
Jalankan dari folder prototipe_wsd_xgboost:
    streamlit run aplikasi/Beranda.py
"""
import streamlit as st

from layanan.artefak import muat_metadata
from layanan import basis_data

st.set_page_config(page_title="WebGIS Prediksi WSD", page_icon="🦐", layout="wide")

meta = muat_metadata()
skenario = meta["skenario_terpilih"]
cv = meta["ringkasan_cv"][skenario]
uji = meta["metrik_uji"][skenario]

st.title("🦐 WebGIS Prediksi Risiko White Spot Disease")
st.markdown(
    "Prototipe sistem prediksi risiko **White Spot Disease (WSD)** pada tambak udang menggunakan algoritma "
    "**XGBoost**, dilengkapi penjelasan **SHAP** (*Explainable AI*) dan pemetaan **GIS**. "
    "Pengguna memilih lokasi tambak pada peta, mengisi kondisi tambak, lalu sistem menampilkan "
    "probabilitas risiko beserta faktor-faktor penyebabnya."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Data penelitian", f"{meta['jumlah_data']['total']} tambak")
k2.metric("Fitur dipakai model", f"{len(meta['fitur_terpilih'])} dari {len(meta['fitur_input'])}")
k3.metric("ROC-AUC (cross-validation)", f"{cv['cv_roc_auc_mean']:.3f}")
k4.metric("Prediksi tersimpan", len(basis_data.semua()))

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Cara penggunaan")
    st.markdown(
        "1. Buka **Prediksi Risiko** → klik lokasi tambak pada peta.\n"
        "2. Isi kondisi tambak pada formulir.\n"
        "3. Klik **Prediksi** → lihat probabilitas, kategori risiko, dan faktor penyebab.\n"
        "4. Simpan hasil → titik tambak muncul di **Peta Sebaran**."
    )
    st.page_link("pages/1_Prediksi_Risiko.py", label="Mulai prediksi", icon="▶️")
with c2:
    st.subheader("Penting diketahui")
    st.warning(
        "Model dilatih dari data **233 tambak udang windu di Bangladesh** (Hasan & Haque, 2020). "
        "Untuk lokasi di Indonesia, hasil bersifat **demonstrasi/indikatif** dan bukan diagnosis. "
        "Konfirmasi penyakit tetap memerlukan uji laboratorium (PCR).",
        icon="⚠️",
    )
    st.page_link("pages/5_Panduan_dan_Batasan.py", label="Baca panduan & batasan", icon="📘")
