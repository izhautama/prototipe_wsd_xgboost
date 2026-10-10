"""T11 – Ekspor artefak model (Lapisan II): jembatan antara riset dan aplikasi."""
import json
import platform
from datetime import datetime

import joblib
import pandas as pd
import shap
import sklearn
import xgboost

from kode_sumber import konfigurasi as k


def _ke_json(obj):
    """Konversi tipe numpy/pandas agar dapat ditulis ke JSON."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Tipe tidak didukung: {type(obj)}")


def simpan_pipeline(pipeline):
    k.DIR_MODEL.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, k.DIR_MODEL / "pipeline_wsd.joblib")


def simpan_metadata(skenario, fitur, pipeline, latih, ringkasan_cv, metrik_uji, spasial, kepentingan):
    rentang = {}
    for f in fitur:
        if k.KAMUS_FITUR[f]["tipe"] == "numerik":
            kolom = latih[f].dropna()
            rentang[f] = dict(min=float(kolom.min()), max=float(kolom.max()), median=float(kolom.median()))
    metadata = dict(
        skenario_terpilih=skenario,
        fitur_input=fitur,
        fitur_terpilih=list(pipeline.named_steps["seleksi"].get_feature_names_out()),
        rentang_data_latih=rentang,
        ambang=dict(sedang=k.AMBANG_SEDANG, tinggi=k.AMBANG_TINGGI),
        ringkasan_cv=ringkasan_cv,
        metrik_uji=metrik_uji,
        validasi_spasial=spasial.to_dict(orient="records"),
        kepentingan_shap=kepentingan.head(15).to_dict(orient="records"),
        jumlah_data=dict(total=len(latih) + metrik_uji[skenario]["TP"] + metrik_uji[skenario]["TN"]
                         + metrik_uji[skenario]["FP"] + metrik_uji[skenario]["FN"], latih=len(latih)),
        versi=dict(python=platform.python_version(), xgboost=xgboost.__version__,
                   scikit_learn=sklearn.__version__, shap=shap.__version__, pandas=pd.__version__),
        dilatih_pada=datetime.now().isoformat(timespec="seconds"),
        sumber_data="Hasan & Haque (2020), Data in Brief – tambak Penaeus monodon, Khulna/Bagerhat/Satkhira, Bangladesh",
    )
    with open(k.DIR_MODEL / "metadata_model.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=_ke_json)


def simpan_kamus():
    kamus = {}
    for f, m in k.KAMUS_FITUR.items():
        isi = {kunci: nilai for kunci, nilai in m.items() if kunci != "opsi"}
        if "opsi" in m:
            isi["opsi"] = {str(kode): label for kode, label in m["opsi"].items()}
        kamus[f] = isi
    with open(k.DIR_MODEL / "kamus_data.json", "w", encoding="utf-8") as f:
        json.dump(kamus, f, ensure_ascii=False, indent=2)


def simpan_geojson(df_bersih: pd.DataFrame):
    """233 titik tambak penelitian sebagai layer referensi peta."""
    fitur_geo = []
    for _, r in df_bersih.iterrows():
        fitur_geo.append(dict(
            type="Feature",
            geometry=dict(type="Point", coordinates=[float(r["Longitude"]), float(r["Latitude"])]),
            properties=dict(id=int(r["SlNo"]), wilayah=r["Zone"], status_wsd=int(r[k.TARGET]),
                            prevalensi_sebelumnya=float(r["PreviousPrevalence"])),
        ))
    with open(k.DIR_MODEL / "data_penelitian.geojson", "w", encoding="utf-8") as f:
        json.dump(dict(type="FeatureCollection", features=fitur_geo), f, ensure_ascii=False)
