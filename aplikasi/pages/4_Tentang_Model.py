"""Halaman Tentang Model: transparansi metrik, ablasi, validasi spasial, dan SHAP global."""
import pandas as pd
import streamlit as st

from layanan.artefak import DIR_GAMBAR, muat_kamus, muat_metadata

st.set_page_config(page_title="Tentang Model", page_icon="📊", layout="wide")
meta, kamus = muat_metadata(), muat_kamus()
s = meta["skenario_terpilih"]

st.title("Tentang Model")
st.markdown(
    f"Model: **XGBoost** dengan seleksi fitur *SelectFromModel*, dioptimasi menggunakan **RandomizedSearchCV** "
    f"dan **Stratified 5-Fold Cross-Validation** pada {meta['jumlah_data']['latih']} data latih. "
    f"Skenario terpilih: **{s}** "
    f"({'tanpa' if s == 'B' else 'dengan'} variabel riwayat penyakit). Sumber data: {meta['sumber_data']}."
)

st.subheader("Eksperimen ablasi (cross-validation, data latih)")
ablasi = pd.DataFrame(meta["ringkasan_cv"]).T
kolom = {"cv_f1_macro_mean": "F1-macro", "cv_f1_macro_std": "± SD", "cv_accuracy_mean": "Accuracy",
         "cv_precision_mean": "Precision", "cv_recall_mean": "Recall", "cv_roc_auc_mean": "ROC-AUC",
         "gap_overfitting": "Selisih latih–validasi"}
st.dataframe(ablasi[list(kolom)].rename(columns=kolom).round(3), width="stretch")

st.subheader("Evaluasi pada data uji (holdout)")
st.dataframe(pd.DataFrame(meta["metrik_uji"]).T.round(3), width="stretch")
st.caption("Baseline = model yang selalu menebak kelas mayoritas. Model bermakna bila melampaui baseline "
           "pada F1-macro dan ROC-AUC.")

st.subheader("Validasi spasial (latih 2 wilayah, uji wilayah ke-3)")
st.dataframe(pd.DataFrame(meta["validasi_spasial"]).round(3), hide_index=True, width="stretch")

st.subheader("Interpretasi global (SHAP)")
c1, c2 = st.columns(2)
for kol, nama, judul in [(c1, f"shap_beeswarm_{s}.png", "Beeswarm"), (c2, f"shap_bar_{s}.png", "Mean |SHAP|")]:
    path = DIR_GAMBAR / nama
    if path.exists():
        kol.image(str(path), caption=judul, width="stretch")
c3, c4 = st.columns(2)
for kol, nama in [(c3, f"confusion_matrix_{s}.png"), (c4, f"kurva_roc_{s}.png")]:
    path = DIR_GAMBAR / nama
    if path.exists():
        kol.image(str(path), width="stretch")

with st.expander("Fitur yang dipakai model setelah seleksi"):
    st.write(pd.DataFrame([(kamus[f]["kelompok"], kamus[f]["label"]) for f in meta["fitur_terpilih"]],
                          columns=["Kelompok", "Fitur"]))
with st.expander("Versi perangkat lunak"):
    st.json(meta["versi"])
