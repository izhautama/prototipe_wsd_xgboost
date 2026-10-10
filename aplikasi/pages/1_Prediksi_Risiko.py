"""Halaman Prediksi Risiko: lokasi (peta/wilayah/alamat/GPS) → formulir → hasil + SHAP → simpan."""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation

from layanan import basis_data, geolokasi, penjelas, peta, prediksi, validasi
from layanan.artefak import muat_kamus, muat_metadata, muat_pipeline

st.set_page_config(page_title="Prediksi Risiko WSD", page_icon="🦐", layout="wide")

pipeline, meta, kamus = muat_pipeline(), muat_metadata(), muat_kamus()
fitur_form = meta["fitur_terpilih"]          # hanya fitur yang dipakai model yang ditanyakan
rentang = meta["rentang_data_latih"]
st.session_state.setdefault("lat", None)
st.session_state.setdefault("lon", None)
st.session_state.setdefault("sumber_lokasi", "")
st.session_state.setdefault("minta_geo", 0)   # >0 = sedang meminta lokasi perangkat (angka = kunci unik)
for _k in geolokasi.KOLOM_ALAMAT:
    st.session_state.setdefault(f"al_{_k}", "")


def set_lokasi(lat, lon, sumber: str, alamat: dict | None = None):
    """Menetapkan koordinat tambak + mengisi alamat (reverse geocoding bila alamat belum diketahui)."""
    st.session_state.lat, st.session_state.lon = round(float(lat), 6), round(float(lon), 6)
    st.session_state.sumber_lokasi = sumber
    if alamat is None:
        try:
            alamat = geolokasi.balik_geokode(st.session_state.lat, st.session_state.lon)
        except geolokasi.GagalLokasi:
            alamat = {k: "" for k in geolokasi.KOLOM_ALAMAT}
    for k in geolokasi.KOLOM_ALAMAT:   # mengisi kolom alamat yang dapat dikoreksi pengguna
        st.session_state[f"al_{k}"] = alamat.get(k, "")


st.title("Prediksi Risiko White Spot Disease")

# ---------------------------------------------------------------- 1. LOKASI
st.subheader("1. Lokasi tambak")
kiri, kanan = st.columns([3, 2])
with kiri:
    keluaran = st_folium(peta.peta_pilih_lokasi(st.session_state.lat, st.session_state.lon),
                         height=460, use_container_width=True, returned_objects=["last_clicked"],
                         key="peta_lokasi")
    klik = (keluaran or {}).get("last_clicked")
    if klik and (round(klik["lat"], 6), round(klik["lng"], 6)) != (st.session_state.lat, st.session_state.lon):
        with st.spinner("Mengambil nama wilayah..."):
            set_lokasi(klik["lat"], klik["lng"], "Klik peta")
        st.rerun()

with kanan:
    tab_peta, tab_wilayah, tab_cari, tab_saya = st.tabs(
        ["🗺️ Peta", "🏛️ Wilayah", "🔎 Alamat", "📱 Lokasi saya"])

    # --- Cara 1: klik peta atau koordinat manual
    with tab_peta:
        st.caption("Klik lokasi tambak pada peta, atau ketik koordinatnya.")
        lat = st.number_input("Latitude", -90.0, 90.0, value=st.session_state.lat, format="%.6f",
                              placeholder="contoh: -7.450000")
        lon = st.number_input("Longitude", -180.0, 180.0, value=st.session_state.lon, format="%.6f",
                              placeholder="contoh: 112.720000")
        if lat is not None and lon is not None and (lat, lon) != (st.session_state.lat, st.session_state.lon):
            with st.spinner("Mengambil nama wilayah..."):
                set_lokasi(lat, lon, "Koordinat manual")
            st.rerun()

    # --- Cara 2: wilayah administratif berjenjang
    with tab_wilayah:
        st.selectbox("Negara", ["Indonesia"], help="Data wilayah berjenjang tersedia untuk Indonesia. "
                                                   "Untuk negara lain gunakan tab Cari alamat.")
        try:
            prov = st.selectbox("Provinsi", geolokasi.daftar_provinsi(), index=None,
                                format_func=lambda w: geolokasi.rapikan_nama(w["name"]), placeholder="Pilih provinsi")
            kab = st.selectbox("Kabupaten/Kota", geolokasi.daftar_kabupaten(prov["id"]) if prov else [],
                               index=None, format_func=lambda w: geolokasi.rapikan_nama(w["name"]),
                               placeholder="Pilih kabupaten/kota", disabled=prov is None)
            kec = st.selectbox("Kecamatan", geolokasi.daftar_kecamatan(kab["id"]) if kab else [],
                               index=None, format_func=lambda w: geolokasi.rapikan_nama(w["name"]),
                               placeholder="Pilih kecamatan", disabled=kab is None)
            desa = st.selectbox("Desa/Kelurahan", geolokasi.daftar_desa(kec["id"]) if kec else [],
                                index=None, format_func=lambda w: geolokasi.rapikan_nama(w["name"]),
                                placeholder="Pilih desa/kelurahan", disabled=kec is None)
            if st.button("📍 Tampilkan wilayah di peta", disabled=prov is None, width="stretch"):
                nama = {k: geolokasi.rapikan_nama(w["name"]) if w else ""
                        for k, w in [("provinsi", prov), ("kabupaten", kab), ("kecamatan", kec), ("desa", desa)]}
                with st.spinner("Mencari koordinat wilayah..."):
                    hasil_geo = geolokasi.geokode_wilayah(nama["provinsi"], nama["kabupaten"],
                                                          nama["kecamatan"], nama["desa"])
                if hasil_geo is None:
                    st.error("Koordinat wilayah tidak ditemukan. Silakan klik lokasi pada peta.")
                else:
                    lat_w, lon_w, ketelitian = hasil_geo
                    set_lokasi(lat_w, lon_w, f"Wilayah administratif (titik pusat {ketelitian})",
                               dict(negara="Indonesia", alamat_detail="", **nama))
                    st.rerun()
        except geolokasi.GagalLokasi as e:
            st.warning(str(e))

    # --- Cara 3: pencarian alamat bebas
    with tab_cari:
        teks = st.text_input("Nama tempat / alamat", placeholder="contoh: Kalanganyar, Sedati, Sidoarjo")
        semua_negara = st.checkbox("Cari di luar Indonesia")
        if st.button("Cari", disabled=not teks.strip(), width="stretch"):
            try:
                with st.spinner("Mencari..."):
                    st.session_state.hasil_cari = geolokasi.cari_alamat(teks.strip(), not semua_negara)
                if not st.session_state.hasil_cari:
                    st.info("Alamat tidak ditemukan. Coba kata kunci yang lebih umum.")
            except geolokasi.GagalLokasi as e:
                st.warning(str(e))
        if st.session_state.get("hasil_cari"):
            pilihan = st.radio("Hasil pencarian", range(len(st.session_state.hasil_cari)),
                               format_func=lambda i: st.session_state.hasil_cari[i]["nama"])
            if st.button("Gunakan lokasi ini", type="primary", width="stretch"):
                h = st.session_state.hasil_cari[pilihan]
                set_lokasi(h["lat"], h["lon"], "Pencarian alamat", h["alamat"])
                st.session_state.hasil_cari = []
                st.rerun()

    # --- Cara 4: lokasi perangkat (GPS/browser)
    with tab_saya:
        st.caption("Memakai GPS/lokasi perangkat. Browser akan meminta izin akses lokasi; "
                   "paling akurat bila dibuka dari ponsel di area tambak.")
        if st.button("📱 Ambil lokasi saya", width="stretch"):
            st.session_state.minta_geo += 1
        if st.session_state.minta_geo:
            posisi = get_geolocation(component_key=f"geo_{st.session_state.minta_geo}")
            if posisi is None:
                st.info("Menunggu izin & sinyal lokasi dari browser...")
            elif "coords" in posisi:
                st.session_state.minta_geo = 0
                with st.spinner("Mengambil nama wilayah..."):
                    set_lokasi(posisi["coords"]["latitude"], posisi["coords"]["longitude"],
                               f"Lokasi perangkat (±{posisi['coords'].get('accuracy', 0):.0f} m)")
                st.rerun()
            else:
                st.session_state.minta_geo = 0
                st.error("Lokasi tidak dapat diambil (izin ditolak atau GPS nonaktif).")

# --- Ringkasan lokasi + alamat yang dapat dikoreksi
if st.session_state.lat is not None:
    st.success(f"📍 Lat {st.session_state.lat:.5f} | Lon {st.session_state.lon:.5f} — "
               f"sumber: {st.session_state.sumber_lokasi or '-'}")
    for p in validasi.peringatan_lokasi(st.session_state.lat, st.session_state.lon):
        st.info(p)
    with st.expander("🏠 Detail alamat tambak (terisi otomatis, dapat dikoreksi)", expanded=True):
        kol_alamat = st.columns(3)
        for i, k in enumerate(geolokasi.KOLOM_ALAMAT):
            with kol_alamat[i % 3]:
                st.text_input(geolokasi.LABEL_ALAMAT[k], key=f"al_{k}", max_chars=120)

# ---------------------------------------------------------------- 2. FORMULIR
st.subheader("2. Kondisi tambak")
kelompok = {}
for f in fitur_form:
    kelompok.setdefault(kamus[f]["kelompok"], []).append(f)


def widget(f: str):
    m = kamus[f]
    if m["tipe"] in ("kategori", "biner"):
        kode = list(m["opsi"])
        if m["tipe"] == "biner":
            return st.radio(m["label"], kode, format_func=m["opsi"].get, horizontal=True, key=f)
        return st.selectbox(m["label"], kode, format_func=m["opsi"].get, key=f)
    r = rentang.get(f, {})
    label = f"{m['label']} ({m['satuan']})" if m.get("satuan") else m["label"]
    if m.get("opsional"):
        if st.checkbox(f"{m['label']} tidak diukur", key=f"kosong_{f}"):
            return np.nan
    return st.number_input(label, value=float(r.get("median", 0.0)), step=0.1, format="%.2f", key=f)


with st.form("form_prediksi"):
    nama_tambak = st.text_input("Nama tambak / pemilik (opsional)", max_chars=80)
    nilai = {}
    tab = st.tabs(list(kelompok))
    for t, (nama_kel, daftar) in zip(tab, kelompok.items()):
        with t:
            kol = st.columns(2)
            for i, f in enumerate(daftar):
                with kol[i % 2]:
                    nilai[f] = widget(f)
    if "PreviousPrevalence" in fitur_form:
        st.caption("Tambak baru? Centang 'belum pernah beroperasi' – riwayat penyakit akan dikosongkan.")
        if st.checkbox("Tambak belum pernah beroperasi", key="tambak_baru"):
            nilai["PreviousPrevalence"] = np.nan
    kirim = st.form_submit_button("🔍 Prediksi", type="primary", width="stretch")

if kirim:
    kesalahan = validasi.validasi_lokasi(st.session_state.lat, st.session_state.lon)
    salah, peringatan = validasi.validasi_input(nilai, kamus, rentang)
    kesalahan += salah
    if kesalahan:
        for e in kesalahan:
            st.error(e)
        st.session_state.pop("hasil", None)
    else:
        X = prediksi.susun_baris(nilai, meta["fitur_input"])
        prob = prediksi.prediksi(pipeline, X)
        kategori, warna = prediksi.kategori_risiko(prob, meta["ambang"])
        st.session_state.hasil = dict(
            prob=prob, kategori=kategori, warna=warna, peringatan=peringatan, nama=nama_tambak.strip(),
            alamat={k: st.session_state[f"al_{k}"].strip() for k in geolokasi.KOLOM_ALAMAT},
           
            lat=st.session_state.lat, lon=st.session_state.lon,
            nilai={f: (None if isinstance(v, float) and math.isnan(v) else v) for f, v in nilai.items()},
            faktor=penjelas.jelaskan(pipeline, X, kamus), tersimpan=False,
        )

# ---------------------------------------------------------------- 3. HASIL
h = st.session_state.get("hasil")
if h:
    st.subheader("3. Hasil prediksi")
    if any(h["alamat"].values()):
        st.caption(f"📍 {geolokasi.teks_alamat(h['alamat'])}")
    for p in h["peringatan"]:
        st.warning(p)
    g, d = st.columns([1, 2])
    with g:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=h["prob"] * 100, number={"suffix": "%"},
            title={"text": f"Risiko <b>{h['kategori']}</b>"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": h["warna"]},
                   "steps": [{"range": [0, meta["ambang"]["sedang"] * 100], "color": "#e8f5e9"},
                             {"range": [meta["ambang"]["sedang"] * 100, meta["ambang"]["tinggi"] * 100],
                              "color": "#fff3e0"},
                             {"range": [meta["ambang"]["tinggi"] * 100, 100], "color": "#ffebee"}]}))
        gauge.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge, width="stretch")
        st.caption("Probabilitas tambak terdeteksi WSSV menurut model. Bukan diagnosis laboratorium.")
    with d:
        f = h["faktor"].iloc[::-1]
        bar = go.Figure(go.Bar(
            x=f["shap"], y=[f"{a} = {b}" for a, b in zip(f["faktor"], f["nilai"])], orientation="h",
            marker_color=["#d62728" if s > 0 else "#2ca02c" for s in f["shap"]]))
        bar.update_layout(title="5 faktor paling berpengaruh (SHAP)", height=300,
                          xaxis_title="Kontribusi terhadap risiko (log-odds)  ← menurunkan | menaikkan →",
                          margin=dict(t=50, b=40, l=10, r=10))
        st.plotly_chart(bar, width="stretch")

    saran = h["faktor"][h["faktor"]["saran"] != ""]
    if not saran.empty:
        st.markdown("**Saran tindakan berdasarkan faktor yang menaikkan risiko:**")
        for _, r in saran.iterrows():
            st.markdown(f"- **{r['faktor']}** ({r['nilai']}): {r['saran']}")

    if h["tersimpan"]:
        st.success("Hasil telah disimpan. Lihat di halaman Peta Sebaran.")
    elif st.button("💾 Simpan ke peta & riwayat", type="primary"):
        basis_data.simpan(h["nama"] or None, h["lat"], h["lon"], h["prob"], h["kategori"], h["nilai"],
                          h["faktor"], meta["dilatih_pada"], h["alamat"])
        h["tersimpan"] = True
        st.rerun()
