"""Halaman Peta Sebaran: layer data penelitian + layer hasil prediksi + heatmap."""
import folium
import streamlit as st
from streamlit_folium import st_folium

from layanan import basis_data, peta
from layanan.artefak import muat_geojson_penelitian

st.set_page_config(page_title="Peta Sebaran Risiko", page_icon="🗺️", layout="wide")
st.title("Peta Sebaran Risiko WSD")

df = basis_data.semua()
c1, c2 = st.columns([1, 3])
with c1:
    fokus = st.radio("Fokus wilayah", ["Indonesia (hasil prediksi)", "Bangladesh (data penelitian)"])
    kategori = st.multiselect("Kategori risiko", ["Rendah", "Sedang", "Tinggi"],
                              default=["Rendah", "Sedang", "Tinggi"])
    st.markdown(peta.legenda_html(), unsafe_allow_html=True)
    st.caption("Gunakan ikon lapisan di kanan atas peta untuk menampilkan/menyembunyikan layer.")
    provinsi = sorted(p for p in df["provinsi"].dropna().unique() if p)
    pilih_prov = st.multiselect("Provinsi", provinsi, placeholder="Semua provinsi")
    tersaring = df[df["kategori"].isin(kategori)]
    if pilih_prov:
        tersaring = tersaring[tersaring["provinsi"].isin(pilih_prov)]
    kabupaten = sorted(k for k in tersaring["kabupaten"].dropna().unique() if k)
    pilih_kab = st.multiselect("Kabupaten/Kota", kabupaten, placeholder="Semua kabupaten/kota")
    if pilih_kab:
        tersaring = tersaring[tersaring["kabupaten"].isin(pilih_kab)]
    st.metric("Prediksi ditampilkan", len(tersaring))
    if not tersaring.empty:
        st.dataframe(tersaring["kategori"].value_counts().rename("jumlah"), width="stretch")

with c2:
    if fokus.startswith("Indonesia"):
        if not tersaring.empty:
            m = peta.peta_dasar((tersaring["latitude"].mean(), tersaring["longitude"].mean()), 6)
        else:
            m = peta.peta_dasar(peta.PUSAT_INDONESIA, 5)
    else:
        m = peta.peta_dasar(peta.PUSAT_BANGLADESH, 9)
    peta.layer_penelitian(muat_geojson_penelitian()).add_to(m)
    titik, panas = peta.layer_prediksi(tersaring)
    titik.add_to(m)
    panas.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=600, use_container_width=True, returned_objects=[], key="peta_sebaran")
