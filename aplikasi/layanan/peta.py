"""M5 – Komponen peta (Folium/Leaflet)."""
from html import escape

import folium
import pandas as pd
from folium.plugins import HeatMap

PUSAT_INDONESIA = (-2.5, 118.0)
PUSAT_BANGLADESH = (22.6, 89.4)
WARNA_KATEGORI = {"Rendah": "green", "Sedang": "orange", "Tinggi": "red"}


def peta_dasar(pusat, zoom) -> folium.Map:
    peta = folium.Map(location=pusat, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Citra satelit").add_to(peta)
    return peta


def peta_pilih_lokasi(lat=None, lon=None) -> folium.Map:
    """Peta untuk memilih lokasi tambak dengan klik."""
    if lat is not None:
        peta = peta_dasar((lat, lon), 13)
        folium.Marker((lat, lon), tooltip="Lokasi tambak", icon=folium.Icon(color="blue", icon="home")).add_to(peta)
    else:
        peta = peta_dasar(PUSAT_INDONESIA, 5)
    folium.LayerControl().add_to(peta)
    return peta


def layer_penelitian(geojson: dict) -> folium.FeatureGroup:
    grup = folium.FeatureGroup(name="Data penelitian (Bangladesh, n=233)")
    for f in geojson["features"]:
        lon, lat = f["geometry"]["coordinates"]
        p = f["properties"]
        positif = p["status_wsd"] == 1
        folium.CircleMarker(
            (lat, lon), radius=4, weight=1, fill=True, fill_opacity=0.8,
            color="#b22222" if positif else "#1f77b4",
            popup=folium.Popup(f"<b>Tambak #{p['id']}</b><br>Wilayah: {p['wilayah']}<br>"
                               f"Status: {'Terdeteksi WSD' if positif else 'Tidak terdeteksi'}<br>"
                               f"Prevalensi sebelumnya: {p['prevalensi_sebelumnya']:.0f}%", max_width=220),
        ).add_to(grup)
    return grup


def layer_prediksi(df: pd.DataFrame) -> tuple:
    titik = folium.FeatureGroup(name="Hasil prediksi pengguna")
    for _, r in df.iterrows():
        folium.Marker(
            (r["latitude"], r["longitude"]),
            icon=folium.Icon(color=WARNA_KATEGORI.get(r["kategori"], "gray"), icon="info-sign"),
            tooltip=f"{escape(r['nama_tambak'] or 'Tambak')} – risiko {r['kategori']}",
            popup=folium.Popup(f"<b>{escape(r['nama_tambak'] or 'Tambak tanpa nama')}</b><br>"
                               f"{_alamat(r)}"
                               f"Probabilitas WSD: {r['probabilitas']:.1%}<br>Kategori: {r['kategori']}<br>"
                               f"Waktu: {r['waktu']}", max_width=240),
        ).add_to(titik)
    panas = folium.FeatureGroup(name="Heatmap risiko", show=False)
    if not df.empty:
        HeatMap(df[["latitude", "longitude", "probabilitas"]].values.tolist(), radius=25).add_to(panas)
    return titik, panas


def _alamat(r) -> str:
    # escape(): teks input pengguna tidak boleh dieksekusi sebagai HTML (mencegah XSS)
    bagian = [escape(r.get(k)) for k in ("desa", "kecamatan", "kabupaten", "provinsi") if isinstance(r.get(k), str) and r.get(k)]
    return f"{', '.join(bagian)}<br>" if bagian else ""


def legenda_html() -> str:
    return ("<div style='font-size:0.85rem'>"
            "<b>Hasil prediksi:</b> 🟢 Rendah &nbsp; 🟠 Sedang &nbsp; 🔴 Tinggi<br>"
            "<b>Data penelitian:</b> <span style='color:#b22222'>●</span> Terdeteksi WSD &nbsp; "
            "<span style='color:#1f77b4'>●</span> Tidak terdeteksi</div>")
