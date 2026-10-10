"""Halaman Riwayat Prediksi: tabel, detail, unduh CSV, hapus."""
import json

import pandas as pd
import streamlit as st

from layanan import basis_data
from layanan.artefak import muat_kamus
from layanan.geolokasi import KOLOM_ALAMAT, teks_alamat
from layanan.penjelas import format_nilai

st.set_page_config(page_title="Riwayat Prediksi", page_icon="📋", layout="wide")
st.title("Riwayat Prediksi")

df = basis_data.semua()
if df.empty:
    st.info("Belum ada prediksi tersimpan. Lakukan prediksi terlebih dahulu.")
    st.page_link("pages/1_Prediksi_Risiko.py", label="Ke halaman Prediksi Risiko", icon="▶️")
    st.stop()

kamus = muat_kamus()
tampil = df[["id", "waktu", "nama_tambak", "desa", "kecamatan", "kabupaten", "provinsi", "negara",
             "latitude", "longitude", "probabilitas", "kategori"]].copy()
st.dataframe(tampil, hide_index=True, width="stretch",
             column_config={"probabilitas": st.column_config.ProgressColumn(
                 "Probabilitas WSD", format="percent", min_value=0, max_value=1)})
st.download_button("⬇️ Unduh CSV", df.to_csv(index=False).encode("utf-8"),
                   "riwayat_prediksi_wsd.csv", "text/csv")

st.divider()
pilih = st.selectbox("Lihat detail prediksi", df["id"],
                     format_func=lambda i: f"#{i} – {df.set_index('id').loc[i, 'nama_tambak'] or 'Tanpa nama'}")
baris = df.set_index("id").loc[pilih]
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Probabilitas:** {baris['probabilitas']:.1%} — **{baris['kategori']}**")
    alamat = teks_alamat({k: baris[k] or "" for k in KOLOM_ALAMAT})
    st.markdown(f"**Lokasi:** {alamat or '-'} ({baris['latitude']:.5f}, {baris['longitude']:.5f})")
    st.markdown("**Faktor dominan (SHAP):**")
    st.dataframe(pd.DataFrame(json.loads(baris["faktor_json"])), hide_index=True, width="stretch")
with c2:
    st.markdown("**Data input:**")
    masukan = json.loads(baris["input_json"])
    st.dataframe(pd.DataFrame([(kamus[f]["label"], format_nilai(f, v if v is not None else float("nan"), kamus))
                               for f, v in masukan.items()], columns=["Variabel", "Nilai"]),
                 hide_index=True, width="stretch")
if st.button(f"🗑️ Hapus prediksi #{pilih}"):
    basis_data.hapus(int(pilih))
    st.rerun()
