"""M3 – Prediksi probabilitas WSD."""
import numpy as np
import pandas as pd


def susun_baris(nilai: dict, fitur_input: list) -> pd.DataFrame:
    """Menyusun 1 baris sesuai urutan kolom model.
    Fitur yang tidak lolos seleksi tidak ditanyakan di form; diisi NaN karena dibuang oleh tahap seleksi."""
    baris = {f: nilai.get(f, np.nan) for f in fitur_input}
    return pd.DataFrame([baris], columns=fitur_input).astype(float)


def kategori_risiko(prob: float, ambang: dict) -> tuple:
    if prob >= ambang["tinggi"]:
        return "Tinggi", "#d62728"
    if prob >= ambang["sedang"]:
        return "Sedang", "#ff9f1c"
    return "Rendah", "#2ca02c"


def prediksi(pipeline, X: pd.DataFrame) -> float:
    return float(pipeline.predict_proba(X)[0, 1])
