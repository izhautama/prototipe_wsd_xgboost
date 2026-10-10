"""T8 – Evaluasi pada data uji (holdout) dan T9 – validasi spasial antarwilayah."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut

from kode_sumber import konfigurasi as k

LABEL_KELAS = ["Tidak terdeteksi (0)", "Terdeteksi WSD (1)"]


def hitung_metrik(y_benar, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_benar, y_pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y_benar, y_pred),
                precision=precision_score(y_benar, y_pred, zero_division=0),
                recall=recall_score(y_benar, y_pred, zero_division=0),
                f1=f1_score(y_benar, y_pred, zero_division=0),
                f1_macro=f1_score(y_benar, y_pred, average="macro", zero_division=0),
                roc_auc=roc_auc_score(y_benar, y_prob) if len(set(y_benar)) > 1 else np.nan,
                TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


def evaluasi_holdout(model, X_uji, y_uji, nama: str) -> dict:
    prob = model.predict_proba(X_uji)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrik = hitung_metrik(y_uji, pred, prob)

    k.DIR_GAMBAR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_uji, pred, labels=[0, 1]),
                           display_labels=LABEL_KELAS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix – Skenario {nama} (data uji, n={len(y_uji)})")
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    plt.setp(ax.get_xticklabels(), fontsize=8); plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout(); fig.savefig(k.DIR_GAMBAR / f"confusion_matrix_{nama}.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_uji, prob, ax=ax, name=f"XGBoost Skenario {nama}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title(f"Kurva ROC – Skenario {nama} (data uji)")
    fig.tight_layout(); fig.savefig(k.DIR_GAMBAR / f"kurva_roc_{nama}.png", dpi=200); plt.close(fig)
    return metrik


def evaluasi_baseline(y_latih, y_uji) -> dict:
    """Pembanding minimum: model yang selalu menebak kelas mayoritas data latih.
    Model XGBoost baru bermakna bila skornya melampaui baseline ini."""
    mayoritas = int(y_latih.mode()[0])
    pred = np.full(len(y_uji), mayoritas)
    return hitung_metrik(y_uji, pred, pred.astype(float))


def validasi_spasial(pipeline_terbaik, X, y, wilayah) -> pd.DataFrame:
    """Leave-One-Zone-Out: latih di dua kabupaten, uji di kabupaten ketiga.
    Menguji apakah model sekadar menghafal wilayah (spatial overfitting)."""
    baris = []
    for idx_latih, idx_uji in LeaveOneGroupOut().split(X, y, wilayah):
        m = clone(pipeline_terbaik).fit(X.iloc[idx_latih], y.iloc[idx_latih])
        prob = m.predict_proba(X.iloc[idx_uji])[:, 1]
        metrik = hitung_metrik(y.iloc[idx_uji], (prob >= 0.5).astype(int), prob)
        baris.append(dict(wilayah_uji=wilayah.iloc[idx_uji].iloc[0], n_uji=len(idx_uji), **metrik))
    return pd.DataFrame(baris)
