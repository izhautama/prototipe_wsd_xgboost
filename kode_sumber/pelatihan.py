"""
T5–T7 – Pipeline (Zona B), RandomizedSearchCV + Stratified k-Fold, dan eksperimen ablasi.
Seleksi fitur diletakkan DI DALAM Pipeline sehingga dipilih ulang pada setiap fold
hanya dari data latih fold tersebut (mencegah data leakage).
"""
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from kode_sumber import konfigurasi as k

METRIK_CV = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
             "f1": "f1", "f1_macro": "f1_macro", "roc_auc": "roc_auc"}


def buat_pipeline() -> Pipeline:
    # Nilai kosong (Temperature, pH, Salinity) dibiarkan NaN: XGBoost mempelajari
    # arah cabang default untuk data kosong (sparsity-aware split finding).
    penyeleksi = SelectFromModel(
        XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                      eval_metric="logloss", random_state=k.RANDOM_STATE, n_jobs=1),
        threshold="median",
    )
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          random_state=k.RANDOM_STATE, n_jobs=1)
    return Pipeline([("seleksi", penyeleksi), ("model", model)])


def tuning(X: pd.DataFrame, y: pd.Series, n_iter: int = k.N_ITER_SEARCH) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=k.K_FOLD, shuffle=True, random_state=k.RANDOM_STATE)
    ruang = dict(k.RUANG_HYPERPARAMETER)
    # Bobot kelas: 1 (tanpa pembobotan) atau rasio negatif/positif
    ruang["model__scale_pos_weight"] = [1.0, round((y == 0).sum() / (y == 1).sum(), 4)]
    pencari = RandomizedSearchCV(
        buat_pipeline(), ruang, n_iter=n_iter, scoring=METRIK_CV, refit=k.METRIK_UTAMA,
        cv=cv, random_state=k.RANDOM_STATE, n_jobs=-1, return_train_score=True, error_score="raise",
    )
    return pencari.fit(X, y)


def ringkas_cv(pencari: RandomizedSearchCV) -> dict:
    """Skor CV kombinasi terbaik: rata-rata ± simpangan baku, plus selisih latih–validasi."""
    hasil = pd.DataFrame(pencari.cv_results_)
    baris = hasil.loc[pencari.best_index_]
    ringkasan = {}
    for m in METRIK_CV:
        ringkasan[f"cv_{m}_mean"] = float(baris[f"mean_test_{m}"])
        ringkasan[f"cv_{m}_std"] = float(baris[f"std_test_{m}"])
    m = k.METRIK_UTAMA
    ringkasan[f"latih_{m}_mean"] = float(baris[f"mean_train_{m}"])
    ringkasan["gap_overfitting"] = ringkasan[f"latih_{m}_mean"] - ringkasan[f"cv_{m}_mean"]
    return ringkasan


def eksperimen_ablasi(latih: pd.DataFrame, n_iter: int = k.N_ITER_SEARCH) -> dict:
    """Melatih Skenario A (dengan riwayat penyakit) dan B (tanpa riwayat penyakit)."""
    y = latih[k.TARGET]
    hasil = {}
    for nama, fitur in [("A", k.FITUR_SKENARIO_A), ("B", k.FITUR_SKENARIO_B)]:
        print(f"  Tuning Skenario {nama} ({len(fitur)} fitur, {n_iter} kombinasi × {k.K_FOLD} fold)...")
        pencari = tuning(latih[fitur], y, n_iter)
        hasil[nama] = dict(pencari=pencari, fitur=fitur, ringkasan=ringkas_cv(pencari))
    return hasil


def pilih_skenario(hasil: dict) -> str:
    """Skenario B (berlaku untuk petani baru) diutamakan selama skor CV tidak turun melebihi toleransi."""
    kunci = f"cv_{k.METRIK_UTAMA}_mean"
    skor_a, skor_b = hasil["A"]["ringkasan"][kunci], hasil["B"]["ringkasan"][kunci]
    return "B" if skor_b >= skor_a - k.TOLERANSI_ABLASI else "A"


def simpan_hasil_cv(hasil: dict):
    k.DIR_TABEL.mkdir(parents=True, exist_ok=True)
    for nama, h in hasil.items():
        cv = pd.DataFrame(h["pencari"].cv_results_).sort_values(f"rank_test_{k.METRIK_UTAMA}")
        cv.to_csv(k.DIR_TABEL / f"hasil_cv_skenario_{nama}.csv", index=False)
        param = {p.replace("model__", "").replace("seleksi__", "seleksi_"): v
                 for p, v in h["pencari"].best_params_.items()}
        pd.Series(param, name="nilai").to_csv(k.DIR_TABEL / f"parameter_terbaik_{nama}.csv")
    tabel = pd.DataFrame({n: h["ringkasan"] for n, h in hasil.items()}).T
    tabel.insert(0, "jumlah_fitur", [len(h["fitur"]) for h in hasil.values()])
    tabel.to_csv(k.DIR_TABEL / "ablasi_skenario.csv")
    return tabel
