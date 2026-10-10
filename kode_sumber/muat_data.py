"""T1 – Akuisisi data mentah dan validasi skema."""
import pandas as pd

from kode_sumber import konfigurasi as k

KOLOM_WAJIB = 48


def muat_data_mentah(path=k.PATH_DATA_MENTAH) -> pd.DataFrame:
    """Membaca sheet data utama dan memastikan struktur sesuai sumber (Hasan & Haque, 2020)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_excel(path, sheet_name=k.SHEET_DATA)
    if df.shape[1] != KOLOM_WAJIB:
        raise ValueError(f"Jumlah kolom {df.shape[1]}, seharusnya {KOLOM_WAJIB}")
    if k.TARGET not in df.columns:
        raise ValueError(f"Kolom target '{k.TARGET}' tidak ditemukan")
    return df
