"""
T2 – Pembersihan statis (Zona A) dan T4 – pembagian data.
Zona A hanya berisi transformasi deterministik (tidak menghitung statistik dari data),
sehingga aman dilakukan sebelum split tanpa menimbulkan data leakage.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from kode_sumber import konfigurasi as k


def bersihkan(df_mentah: pd.DataFrame) -> pd.DataFrame:
    df = df_mentah.rename(columns=k.GANTI_NAMA).copy()
    df = df.drop(columns=k.KOLOM_BOCOR)

    # Pendidikan bersifat bertingkat -> kode ordinal 0..7
    df["FarmerEducation"] = df["FarmerEducation"].map(k.ORDINAL_PENDIDIKAN)
    # Variabel dua kategori yang dikodekan 1/3 atau 1/2 diseragamkan menjadi 0/1
    for kolom, peta in k.BINER_ULANG.items():
        df[kolom] = df[kolom].map(peta)

    # Validasi hasil pembersihan
    fitur = k.FITUR_SKENARIO_A
    hilang = set(fitur + k.KOLOM_METADATA + [k.TARGET]) - set(df.columns)
    if hilang:
        raise ValueError(f"Kolom tidak ditemukan setelah pembersihan: {hilang}")
    kolom_boleh_kosong = [f for f, m in k.KAMUS_FITUR.items() if m.get("opsional")]
    kosong_tak_wajar = df[[f for f in fitur if f not in kolom_boleh_kosong]].isna().sum()
    if kosong_tak_wajar.any():
        raise ValueError(f"Nilai kosong/gagal dipetakan:\n{kosong_tak_wajar[kosong_tak_wajar > 0]}")
    for f, m in k.KAMUS_FITUR.items():  # setiap kode kategori harus tercantum di kamus
        if m["tipe"] in ("kategori", "biner"):
            asing = set(df[f].dropna().unique()) - set(m["opsi"])
            if asing:
                raise ValueError(f"Kode tidak dikenal pada {f}: {asing}")
    if df.duplicated(subset=fitur + [k.TARGET]).any():
        raise ValueError("Terdapat baris duplikat")

    return df[k.KOLOM_METADATA + fitur + [k.TARGET]]


def bagi_data(df: pd.DataFrame):
    """Split stratified 80:20; metadata ikut terbagi dengan indeks yang sama."""
    latih, uji = train_test_split(df, test_size=k.UKURAN_UJI, stratify=df[k.TARGET],
                                  random_state=k.RANDOM_STATE)
    return latih, uji


def pisahkan(df: pd.DataFrame, fitur: list):
    return df[fitur], df[k.TARGET], df[k.KOLOM_METADATA]


def simpan_data_olahan(df, latih, uji):
    k.DIR_OLAHAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(k.DIR_OLAHAN / "data_bersih.csv", index=False)
    latih.to_csv(k.DIR_OLAHAN / "data_latih.csv", index=False)
    uji.to_csv(k.DIR_OLAHAN / "data_uji.csv", index=False)
