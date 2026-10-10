"""Halaman Panduan & Batasan."""
import streamlit as st

from layanan.artefak import muat_metadata

st.set_page_config(page_title="Panduan & Batasan", page_icon="📘", layout="wide")
meta = muat_metadata()
a = meta["ambang"]

st.title("Panduan & Batasan Sistem")

st.header("Arti kategori risiko")
st.markdown(
    f"| Kategori | Probabilitas | Makna |\n|---|---|---|\n"
    f"| 🟢 Rendah | < {a['sedang']:.0%} | Kondisi tambak mirip tambak yang tidak terdeteksi WSSV |\n"
    f"| 🟠 Sedang | {a['sedang']:.0%} – < {a['tinggi']:.0%} | Waspada; perbaiki faktor yang menaikkan risiko |\n"
    f"| 🔴 Tinggi | ≥ {a['tinggi']:.0%} | Kondisi mirip tambak terserang; perketat biosekuriti & lakukan uji PCR |"
)

st.header("Cara membaca penjelasan SHAP")
st.markdown(
    "- Batang **merah** = faktor yang **menaikkan** risiko pada tambak tersebut.\n"
    "- Batang **hijau** = faktor yang **menurunkan** risiko.\n"
    "- Panjang batang menunjukkan besar pengaruhnya. Penjelasan bersifat lokal (khusus tambak yang diinput)."
)

st.header("Batasan")
st.markdown(
    "1. **Sumber data.** Model dilatih dari 233 tambak udang windu (*Penaeus monodon*) di Khulna, Bagerhat, "
    "dan Satkhira, Bangladesh (Hasan & Haque, 2020). Penerapan pada tambak udang vaname di Indonesia bersifat "
    "**demonstrasi** dan memerlukan validasi dengan data lokal.\n"
    "2. **Lokasi tidak memengaruhi prediksi.** Koordinat dipakai untuk pemetaan saja; prediksi dihitung dari "
    "kondisi dan praktik pengelolaan tambak.\n"
    "3. **Validasi antarwilayah lemah.** Pengujian latih-di-dua-wilayah/uji-di-wilayah-ketiga menunjukkan kinerja "
    "turun mendekati tebakan acak; model belum dapat digeneralisasi ke wilayah baru tanpa data tambahan.\n"
    "4. **Bukan alat diagnosis.** Konfirmasi WSD wajib melalui uji laboratorium (PCR).\n"
    "5. **Input di luar jangkauan data latih** akan diberi peringatan karena prediksi menjadi kurang andal."
)
