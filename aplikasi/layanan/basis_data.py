"""M6 – Penyimpanan riwayat prediksi (SQLite)."""
import json
import sqlite3
from datetime import datetime

import pandas as pd

from layanan.artefak import PATH_DB
from layanan.geolokasi import KOLOM_ALAMAT

SKEMA = """
CREATE TABLE IF NOT EXISTS prediksi (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    waktu         TEXT NOT NULL,
    nama_tambak   TEXT,
    latitude      REAL NOT NULL,
    longitude     REAL NOT NULL,
    probabilitas  REAL NOT NULL,
    kategori      TEXT NOT NULL,
    input_json    TEXT NOT NULL,
    faktor_json   TEXT NOT NULL,
    versi_model   TEXT
)"""


def _koneksi():
    kon = sqlite3.connect(PATH_DB)
    kon.execute(SKEMA)
    # Migrasi: menambah kolom alamat pada basis data versi lama tanpa menghapus riwayat
    ada = {baris[1] for baris in kon.execute("PRAGMA table_info(prediksi)")}
    for kolom in KOLOM_ALAMAT:
        if kolom not in ada:
            kon.execute(f"ALTER TABLE prediksi ADD COLUMN {kolom} TEXT")  # nama kolom dari konstanta, aman
    return kon


def simpan(nama_tambak, lat, lon, prob, kategori, nilai_input: dict, faktor: pd.DataFrame, versi_model: str,
           alamat: dict | None = None) -> int:
    alamat = alamat or {}
    kolom_alamat = ", ".join(KOLOM_ALAMAT)
    with _koneksi() as kon:  # parameterized query: mencegah SQL injection
        cur = kon.execute(
            "INSERT INTO prediksi (waktu, nama_tambak, latitude, longitude, probabilitas, kategori, "
            f"input_json, faktor_json, versi_model, {kolom_alamat}) "
            f"VALUES ({', '.join(['?'] * (9 + len(KOLOM_ALAMAT)))})",
            (datetime.now().isoformat(timespec="seconds"), nama_tambak, lat, lon, prob, kategori,
             json.dumps(nilai_input, ensure_ascii=False),
             faktor[["faktor", "nilai", "pengaruh"]].to_json(orient="records", force_ascii=False), versi_model,
             *[alamat.get(k) or None for k in KOLOM_ALAMAT]),
        )
        return cur.lastrowid


def semua() -> pd.DataFrame:
    with _koneksi() as kon:
        return pd.read_sql_query("SELECT * FROM prediksi ORDER BY id DESC", kon)


def hapus(id_prediksi: int):
    with _koneksi() as kon:
        kon.execute("DELETE FROM prediksi WHERE id = ?", (id_prediksi,))
