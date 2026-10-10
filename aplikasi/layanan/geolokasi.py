"""
M1b – Layanan lokasi: wilayah administratif Indonesia, geocoding, dan reverse geocoding.

Sumber data (memerlukan koneksi internet):
- Daftar wilayah  : API Wilayah Indonesia (emsifa) – provinsi → kabupaten/kota → kecamatan → desa/kelurahan
- Koordinat/alamat: Nominatim OpenStreetMap (kebijakan: maks. 1 permintaan/detik + User-Agent jelas)
Semua hasil di-cache agar tidak membebani server dan aplikasi tetap cepat.
"""
import time

import requests
import streamlit as st

URL_WILAYAH = "https://www.emsifa.com/api-wilayah-indonesia/api"
URL_NOMINATIM = "https://nominatim.openstreetmap.org"
HEADER = {"User-Agent": "PrototipeWSD-Skripsi/1.0 (aplikasi prediksi WSD tambak udang)"}
BATAS_WAKTU = 10  # detik

KOLOM_ALAMAT = ["negara", "provinsi", "kabupaten", "kecamatan", "desa", "alamat_detail"]
LABEL_ALAMAT = {"negara": "Negara", "provinsi": "Provinsi", "kabupaten": "Kabupaten/Kota",
                "kecamatan": "Kecamatan", "desa": "Desa/Kelurahan",
                "alamat_detail": "Alamat detail (dusun, RT/RW, jalan)"}

# Kunci alamat Nominatim bervariasi antardaerah → urutan prioritas
_KUNCI_NOMINATIM = {
    "desa": ["village", "suburb", "hamlet", "neighbourhood", "quarter"],
    "kecamatan": ["city_district", "district", "municipality", "subdistrict"],
    "kabupaten": ["county", "city", "regency", "town"],
    "provinsi": ["state", "region"],
    "negara": ["country"],
}


class GagalLokasi(Exception):
    """Dilempar bila layanan lokasi tidak dapat dihubungi."""


_terakhir = [0.0]


def _get(url: str, params=None):
    # Throttle: jeda minimal 1 detik antar-permintaan ke Nominatim
    if url.startswith(URL_NOMINATIM):
        jeda = 1.0 - (time.time() - _terakhir[0])
        if jeda > 0:
            time.sleep(jeda)
        _terakhir[0] = time.time()
    try:
        r = requests.get(url, params=params, headers=HEADER, timeout=BATAS_WAKTU)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        raise GagalLokasi(f"Layanan lokasi tidak dapat dihubungi ({type(e).__name__}). "
                          "Periksa koneksi internet, atau gunakan klik peta/koordinat manual.") from e


def rapikan_nama(nama: str) -> str:
    """'KABUPATEN SIDOARJO' → 'Kabupaten Sidoarjo'."""
    return " ".join(k.capitalize() for k in nama.split()) if nama else ""


# ------------------------------------------------------------- Wilayah administratif
@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def daftar_provinsi() -> list:
    return _get(f"{URL_WILAYAH}/provinces.json")


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def daftar_kabupaten(id_provinsi: str) -> list:
    return _get(f"{URL_WILAYAH}/regencies/{id_provinsi}.json")


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def daftar_kecamatan(id_kabupaten: str) -> list:
    return _get(f"{URL_WILAYAH}/districts/{id_kabupaten}.json")


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def daftar_desa(id_kecamatan: str) -> list:
    return _get(f"{URL_WILAYAH}/villages/{id_kecamatan}.json")


# ------------------------------------------------------------- Geocoding
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cari_alamat(teks: str, hanya_indonesia: bool = True, batas: int = 5) -> list:
    """Pencarian bebas, mis. 'Desa Kalanganyar, Sedati, Sidoarjo'."""
    params = {"q": teks, "format": "jsonv2", "addressdetails": 1, "limit": batas, "accept-language": "id"}
    if hanya_indonesia:
        params["countrycodes"] = "id"
    hasil = _get(f"{URL_NOMINATIM}/search", params)
    return [dict(lat=float(h["lat"]), lon=float(h["lon"]), nama=h["display_name"],
                 alamat=_dari_nominatim(h.get("address", {}))) for h in hasil]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geokode_wilayah(provinsi: str, kabupaten: str = "", kecamatan: str = "", desa: str = ""):
    """Mencari koordinat wilayah, mulai dari tingkat paling rinci lalu naik bila tidak ditemukan.
    Mengembalikan (lat, lon, tingkat_ketelitian) atau None."""
    tingkat = [("desa", [desa, kecamatan, kabupaten, provinsi]),
               ("kecamatan", [kecamatan, kabupaten, provinsi]),
               ("kabupaten", [kabupaten, provinsi]),
               ("provinsi", [provinsi])]
    for nama_tingkat, bagian in tingkat:
        if not bagian[0]:
            continue
        kueri = ", ".join(rapikan_nama(b) for b in bagian if b) + ", Indonesia"
        hasil = _get(f"{URL_NOMINATIM}/search",
                     {"q": kueri, "format": "jsonv2", "limit": 1, "countrycodes": "id"})
        if hasil:
            return float(hasil[0]["lat"]), float(hasil[0]["lon"]), nama_tingkat
    return None


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def balik_geokode(lat: float, lon: float) -> dict:
    """Koordinat → nama wilayah (negara s.d. desa)."""
    h = _get(f"{URL_NOMINATIM}/reverse",
             {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18, "addressdetails": 1, "accept-language": "id"})
    if "error" in h:
        return {k: "" for k in KOLOM_ALAMAT}
    alamat = _dari_nominatim(h.get("address", {}))
    alamat["alamat_detail"] = h.get("address", {}).get("road", "")
    return alamat


def _dari_nominatim(a: dict) -> dict:
    hasil = {kolom: next((a[k] for k in kunci if a.get(k)), "") for kolom, kunci in _KUNCI_NOMINATIM.items()}
    hasil.setdefault("alamat_detail", "")
    return hasil


def teks_alamat(alamat: dict) -> str:
    urutan = ["alamat_detail", "desa", "kecamatan", "kabupaten", "provinsi", "negara"]
    return ", ".join(alamat[k] for k in urutan if alamat.get(k))
