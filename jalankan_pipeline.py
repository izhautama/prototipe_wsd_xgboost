"""
Menjalankan seluruh Lapisan I (T1–T11) secara berurutan.

Pemakaian (dari folder prototipe_wsd_xgboost):
    python jalankan_pipeline.py              # eksperimen penuh (100 kombinasi)
    python jalankan_pipeline.py --cepat      # uji cepat (10 kombinasi)
"""
import argparse
import time
import warnings

import pandas as pd

from kode_sumber import konfigurasi as k
from kode_sumber import ekspor_artefak, evaluasi, interpretasi, pelatihan, praproses
from kode_sumber.muat_data import muat_data_mentah

warnings.filterwarnings("ignore", category=UserWarning)


def main(n_iter: int):
    mulai = time.time()
    print("[T1] Memuat data mentah...")
    mentah = muat_data_mentah()
    print(f"     {mentah.shape[0]} baris × {mentah.shape[1]} kolom")

    print("[T2] Pembersihan statis (Zona A)...")
    bersih = praproses.bersihkan(mentah)
    print(f"     Fitur Skenario A: {len(k.FITUR_SKENARIO_A)} | Skenario B: {len(k.FITUR_SKENARIO_B)}")

    print("[T4] Split stratified 80:20...")
    latih, uji = praproses.bagi_data(bersih)
    praproses.simpan_data_olahan(bersih, latih, uji)
    print(f"     Latih {len(latih)} | Uji {len(uji)} | proporsi positif latih "
          f"{latih[k.TARGET].mean():.3f}, uji {uji[k.TARGET].mean():.3f}")

    print("[T5–T7] Tuning & ablasi...")
    hasil = pelatihan.eksperimen_ablasi(latih, n_iter)
    tabel_ablasi = pelatihan.simpan_hasil_cv(hasil)
    skenario = pelatihan.pilih_skenario(hasil)
    m = k.METRIK_UTAMA
    print(tabel_ablasi[["jumlah_fitur", f"cv_{m}_mean", f"cv_{m}_std", "cv_f1_mean", "cv_recall_mean",
                        "cv_roc_auc_mean", "gap_overfitting"]].round(3).to_string())
    print(f"     → Skenario terpilih: {skenario}")

    print("[T8] Evaluasi holdout...")
    metrik_uji = {}
    for nama, h in hasil.items():
        X_uji = uji[h["fitur"]]
        metrik_uji[nama] = evaluasi.evaluasi_holdout(h["pencari"].best_estimator_, X_uji, uji[k.TARGET], nama)
    metrik_uji["Baseline"] = evaluasi.evaluasi_baseline(latih[k.TARGET], uji[k.TARGET])
    tabel_uji = pd.DataFrame(metrik_uji).T
    tabel_uji.to_csv(k.DIR_TABEL / "metrik_uji.csv")
    print(tabel_uji.round(3).to_string())

    terbaik = hasil[skenario]
    pipeline = terbaik["pencari"].best_estimator_

    print("[T9] Validasi spasial (Leave-One-Zone-Out)...")
    spasial = evaluasi.validasi_spasial(pipeline, bersih[terbaik["fitur"]], bersih[k.TARGET], bersih["Zone"])
    spasial.to_csv(k.DIR_TABEL / "validasi_spasial.csv", index=False)
    print(spasial[["wilayah_uji", "n_uji", "accuracy", "f1", "f1_macro", "roc_auc"]].round(3).to_string(index=False))

    print("[T10] SHAP...")
    for nama, h in hasil.items():
        nilai = interpretasi.hitung_shap(h["pencari"].best_estimator_, uji[h["fitur"]])
        interpretasi.simpan_plot_shap(nilai, nama)
        tabel = interpretasi.tabel_kepentingan(nilai)
        tabel.to_csv(k.DIR_TABEL / f"kepentingan_shap_{nama}.csv", index=False)
        if nama == skenario:
            kepentingan = tabel
    print(kepentingan.head(10).round(4).to_string(index=False))

    print("[T11] Ekspor artefak...")
    ekspor_artefak.simpan_pipeline(pipeline)
    ekspor_artefak.simpan_metadata(skenario, terbaik["fitur"], pipeline, latih,
                                   {n: h["ringkasan"] for n, h in hasil.items()},
                                   metrik_uji, spasial, kepentingan)
    ekspor_artefak.simpan_kamus()
    ekspor_artefak.simpan_geojson(bersih)
    print(f"Selesai dalam {time.time() - mulai:.1f} detik. Artefak tersimpan di: {k.DIR_MODEL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cepat", action="store_true", help="10 kombinasi saja (uji coba)")
    args = parser.parse_args()
    main(10 if args.cepat else k.N_ITER_SEARCH)
