"""M4 – Penjelasan lokal prediksi dengan SHAP + saran tindakan."""
import math

import pandas as pd
import shap
import streamlit as st

# Saran ditampilkan bila faktor tersebut MENAIKKAN risiko pada tambak pengguna
SARAN = {
    "WaterExchangeFrequency": "Atur ulang jadwal pergantian air; gunakan air yang telah diendapkan/disterilkan di tandon.",
    "Temperature": "Pantau suhu harian; fluktuasi dan suhu ekstrem menurunkan imunitas udang terhadap WSSV.",
    "pH": "Jaga pH stabil (±7,5–8,5) dengan pengapuran terukur.",
    "Salinity": "Pantau salinitas dan hindari perubahan mendadak saat pergantian air.",
    "InletOutlet": "Pisahkan saluran masuk dan keluar agar air buangan tidak masuk kembali.",
    "WaterSource_DirectNatural": "Endapkan dan desinfeksi air dari sumber alami sebelum digunakan.",
    "WaterSource_IndirectNatural": "Hindari air yang melewati tambak lain; gunakan tandon untuk pengendapan.",
    "WaterComingViaOtherFarms": "Putus aliran air dari tambak lain; buat saluran masuk tersendiri.",
    "Reservoir": "Sediakan kolam tandon untuk pengendapan dan sterilisasi air.",
    "SourceOfFingerling": "Gunakan benur dari hatchery bersertifikat dengan hasil uji PCR bebas WSSV.",
    "PLQuality": "Lakukan seleksi benur (uji stres & PCR) sebelum penebaran.",
    "StockingAge_Days": "Tebar benur pada umur yang dianjurkan (PL ≥ 12) agar lebih tahan penyakit.",
    "StockingDensity_PL40m2": "Sesuaikan padat tebar dengan daya dukung tambak.",
    "PeriodOfFallow": "Perpanjang masa istirahat kolam untuk memutus siklus patogen.",
    "GherDryAfterHarvest": "Keringkan dasar tambak setelah panen hingga retak.",
    "SludgeRemovalMethods": "Buang lumpur ke luar area tambak, jangan ditimbun di pematang.",
    "SludgeRemovalInterval": "Lakukan pembuangan lumpur setiap siklus/tahun.",
    "MaintainAndRepairDikes": "Perbaiki pematang dengan tanah bersih, bukan tanah dari tambak lain.",
    "LimitedAccess": "Batasi akses orang dan kendaraan ke area tambak.",
    "BirdScare": "Pasang jaring/pengusir burung untuk mencegah pembawa virus.",
    "CrabFence": "Pasang pagar penghalang kepiting (carrier WSSV).",
    "FootBath": "Sediakan bak celup kaki berdesinfektan di pintu masuk.",
    "SameEquipmentForWholeCultureArea": "Gunakan peralatan terpisah per kolam atau desinfeksi setiap pemakaian.",
    "FeedType": "Gunakan pakan komersial berkualitas; hindari pakan hidup yang berpotensi carrier.",
    "FeedAdditives": "Pertimbangkan imunostimulan/probiotik sebagai aditif pakan.",
    "ChemicalUsePondPreparation": "Gunakan bahan persiapan kolam sesuai dosis anjuran.",
    "ChemicalUseWaterTreatment": "Gunakan bahan perlakuan air sesuai dosis anjuran.",
    "Aerator": "Gunakan aerator untuk menjaga oksigen terlarut.",
    "CultureMethod": "Evaluasi sistem budidaya yang diterapkan bersama penyuluh.",
    "CropRotation": "Evaluasi pola rotasi komoditas untuk memutus siklus patogen.",
    "PreviousPrevalence": "Riwayat serangan tinggi: perketat seluruh protokol biosekuriti.",
}
FAKTOR_TETAP = {"FarmerAge", "FarmerEducation", "InvolveWithShrimpFarming", "Area_ha", "SoilType",
                "CanalDepth_ft", "GherDepth_ft", "PriorLandUse", "Operator", "CulturePractice"}


@st.cache_resource
def _penjelas(_model):
    return shap.TreeExplainer(_model)


def format_nilai(fitur: str, x, kamus: dict) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "tidak diisi"
    meta = kamus[fitur]
    if "opsi" in meta:
        return meta["opsi"].get(int(x), str(x))
    return f"{x:g} {meta.get('satuan', '')}".strip()


def jelaskan(pipeline, X: pd.DataFrame, kamus: dict, n_teratas: int = 5) -> pd.DataFrame:
    seleksi, model = pipeline.named_steps["seleksi"], pipeline.named_steps["model"]
    fitur = list(seleksi.get_feature_names_out())
    X_sel = pd.DataFrame(seleksi.transform(X), columns=fitur)
    nilai_shap = _penjelas(model)(X_sel).values[0]
    tabel = pd.DataFrame({
        "fitur": fitur,
        "faktor": [kamus[f]["label"] for f in fitur],
        "nilai": [format_nilai(f, X_sel.iloc[0][f], kamus) for f in fitur],
        "shap": nilai_shap,
    })
    tabel["pengaruh"] = tabel["shap"].apply(lambda s: "Menaikkan risiko" if s > 0 else "Menurunkan risiko")
    tabel["saran"] = [
        (SARAN.get(f, "Konsultasikan dengan penyuluh perikanan.") if s > 0 and f not in FAKTOR_TETAP else "")
        for f, s in zip(tabel["fitur"], tabel["shap"])
    ]
    return tabel.reindex(tabel["shap"].abs().sort_values(ascending=False).index).head(n_teratas)
