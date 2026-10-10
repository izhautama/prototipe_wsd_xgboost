"""Memuat artefak model (Lapisan II). Aplikasi hanya MEMBACA, tidak pernah melatih ulang."""
import json
from pathlib import Path

import joblib
import streamlit as st

AKAR = Path(__file__).resolve().parents[2]
DIR_MODEL = AKAR / "model"
DIR_GAMBAR = AKAR / "hasil" / "gambar"
DIR_TABEL = AKAR / "hasil" / "tabel"
PATH_DB = AKAR / "aplikasi" / "riwayat_prediksi.db"


def _baca_json(nama: str):
    with open(DIR_MODEL / nama, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Memuat model...")
def muat_pipeline():
    path = DIR_MODEL / "pipeline_wsd.joblib"
    if not path.exists():
        st.error("Model belum tersedia. Jalankan terlebih dahulu: `python jalankan_pipeline.py`")
        st.stop()
    return joblib.load(path)


@st.cache_data
def muat_metadata() -> dict:
    return _baca_json("metadata_model.json")


@st.cache_data
def muat_kamus() -> dict:
    """Kamus fitur; kunci opsi dikembalikan dari string JSON menjadi integer."""
    kamus = _baca_json("kamus_data.json")
    for meta in kamus.values():
        if "opsi" in meta:
            meta["opsi"] = {int(kode): label for kode, label in meta["opsi"].items()}
    return kamus


@st.cache_data
def muat_geojson_penelitian() -> dict:
    return _baca_json("data_penelitian.geojson")
