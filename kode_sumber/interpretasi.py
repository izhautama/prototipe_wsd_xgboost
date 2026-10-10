"""T10 – Interpretasi model dengan SHAP (TreeExplainer)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from kode_sumber import konfigurasi as k


def label_fitur(nama: str) -> str:
    return k.KAMUS_FITUR.get(nama, {}).get("label", nama)


def hitung_shap(pipeline, X: pd.DataFrame) -> shap.Explanation:
    """Menghitung nilai SHAP pada ruang fitur hasil seleksi."""
    seleksi, model = pipeline.named_steps["seleksi"], pipeline.named_steps["model"]
    fitur_terpilih = list(seleksi.get_feature_names_out())
    X_sel = pd.DataFrame(seleksi.transform(X), columns=fitur_terpilih, index=X.index)
    penjelas = shap.TreeExplainer(model)
    nilai = penjelas(X_sel)
    nilai.feature_names = [label_fitur(f) for f in fitur_terpilih]
    return nilai


def simpan_plot_shap(nilai: shap.Explanation, nama: str):
    k.DIR_GAMBAR.mkdir(parents=True, exist_ok=True)
    # Beeswarm: sebaran pengaruh tiap fitur untuk setiap observasi
    plt.figure()
    shap.plots.beeswarm(nilai, max_display=15, show=False)
    plt.title(f"SHAP Summary (Beeswarm) – Skenario {nama}")
    plt.tight_layout(); plt.savefig(k.DIR_GAMBAR / f"shap_beeswarm_{nama}.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    # Bar: kepentingan global = rata-rata |SHAP|
    plt.figure()
    shap.plots.bar(nilai, max_display=15, show=False)
    plt.title(f"Kepentingan Fitur Global (mean |SHAP|) – Skenario {nama}")
    plt.tight_layout(); plt.savefig(k.DIR_GAMBAR / f"shap_bar_{nama}.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    # Waterfall: contoh penjelasan lokal satu tambak (observasi pertama data uji)
    plt.figure()
    shap.plots.waterfall(nilai[0], max_display=10, show=False)
    plt.title(f"Contoh Penjelasan Lokal (SHAP Waterfall) – Skenario {nama}")
    plt.tight_layout(); plt.savefig(k.DIR_GAMBAR / f"shap_waterfall_contoh_{nama}.png", dpi=200,
                                    bbox_inches="tight")
    plt.close("all")


def tabel_kepentingan(nilai: shap.Explanation) -> pd.DataFrame:
    return (pd.DataFrame({"fitur": nilai.feature_names, "mean_abs_shap": abs(nilai.values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
