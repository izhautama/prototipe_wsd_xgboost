"""
Konfigurasi terpusat seluruh eksperimen.
Ubah nilai di sini (bukan di modul lain) agar eksperimen tetap reprodusibel.
"""
from pathlib import Path

from scipy.stats import loguniform, randint, uniform

# ----------------------------------------------------------------------------
# 1. PATH
# ----------------------------------------------------------------------------
AKAR = Path(__file__).resolve().parents[1]
PATH_DATA_MENTAH = AKAR / "data" / "mentah" / "WSSVRiskFactor_rev.xlsx"
SHEET_DATA = "Dateset"  # nama sheet asli (typo dari sumber)
DIR_OLAHAN = AKAR / "data" / "olahan"
DIR_MODEL = AKAR / "model"
DIR_GAMBAR = AKAR / "hasil" / "gambar"
DIR_TABEL = AKAR / "hasil" / "tabel"

# ----------------------------------------------------------------------------
# 2. PARAMETER EKSPERIMEN
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
UKURAN_UJI = 0.20          # holdout 20% (47 data), dikunci sampai evaluasi akhir
K_FOLD = 5                 # Stratified k-Fold pada data latih
N_ITER_SEARCH = 100        # jumlah kombinasi RandomizedSearchCV
# F1-macro (rata-rata F1 kedua kelas) dipakai sebagai dasar refit, BUKAN F1 kelas positif:
# karena kelas positif mayoritas (61,8%), model yang menebak "positif" untuk semua tambak
# sudah memperoleh F1-positif 0,764 tanpa belajar apa pun.
METRIK_UTAMA = "f1_macro"
TOLERANSI_ABLASI = 0.02    # Skenario B dipilih bila skor CV-nya >= skor CV A - 0,02

# Ambang kategori risiko pada aplikasi
AMBANG_SEDANG = 0.40
AMBANG_TINGGI = 0.60

# ----------------------------------------------------------------------------
# 3. ATURAN PEMBERSIHAN STATIS (ZONA A)
# ----------------------------------------------------------------------------
TARGET = "VirusDetected"

GANTI_NAMA = {
    "FarmerEduaction": "FarmerEducation",
    "CurrentPrevalance(%)": "CurrentPrevalance",
    "PreviousPrevalence(%)": "PreviousPrevalence",
    "StockingDensity_PL/40MeterSquare": "StockingDensity_PL40m2",
    "CrabFench": "CrabFence",
}
# Disimpan sebagai metadata spasial (untuk GIS & validasi spasial), bukan fitur
KOLOM_METADATA = ["SlNo", "Latitude", "Longitude", "Zone"]
# Dibuang total: kebocoran target (prevalensi saat ini hanya diketahui setelah terserang)
KOLOM_BOCOR = ["CurrentPrevalance"]

ORDINAL_PENDIDIKAN = {"No": 0, "Primary": 1, "Junior Secondary": 2, "Secondary": 3,
                      "Higher Secondary": 4, "Diploma": 5, "Bachelor's": 6, "Master's": 7}
BINER_ULANG = {
    "PriorLandUse": {1: 0, 3: 1},
    "ChemicalUsePondPreparation": {1: 0, 3: 1},
    "ChemicalUseWaterTreatment": {1: 0, 3: 1},
    "CulturePractice": {1: 0, 2: 1},
}

FITUR_RIWAYAT = "PreviousPrevalence"   # diuji lewat eksperimen ablasi

# ----------------------------------------------------------------------------
# 4. KAMUS FITUR (urutan = urutan kolom model; dipakai juga oleh form aplikasi)
#    tipe: "numerik" | "biner" | "kategori"; opsi = {kode: label}
# ----------------------------------------------------------------------------
YA_TIDAK = {0: "Tidak", 1: "Ya"}

KAMUS_FITUR = {
    # --- Profil petani ---
    "FarmerAge": dict(kelompok="Profil Petani", label="Usia petani", tipe="numerik", satuan="tahun"),
    "FarmerEducation": dict(kelompok="Profil Petani", label="Pendidikan terakhir", tipe="kategori",
                            opsi={0: "Tidak sekolah", 1: "SD", 2: "SMP", 3: "SMA (kelas 10)",
                                  4: "SMA/SMK lulus", 5: "Diploma", 6: "Sarjana (S1)", 7: "Magister (S2)"}),
    "InvolveWithShrimpFarming": dict(kelompok="Profil Petani", label="Lama pengalaman budidaya udang",
                                     tipe="numerik", satuan="tahun"),
    # --- Karakteristik tapak ---
    "PriorLandUse": dict(kelompok="Karakteristik Tapak", label="Penggunaan lahan sebelumnya", tipe="kategori",
                         opsi={0: "Lahan basah / lainnya", 1: "Sawah / lahan pertanian"}),
    "Area_ha": dict(kelompok="Karakteristik Tapak", label="Luas kolam produksi", tipe="numerik", satuan="ha"),
    "SoilType": dict(kelompok="Karakteristik Tapak", label="Jenis tanah dominan", tipe="kategori",
                     opsi={1: "Liat (clay)", 2: "Lempung (loam)", 3: "Berpasir"}),
    "CanalDepth_ft": dict(kelompok="Karakteristik Tapak", label="Rata-rata kedalaman kanal", tipe="numerik",
                          satuan="kaki (ft)"),
    "GherDepth_ft": dict(kelompok="Karakteristik Tapak", label="Rata-rata kedalaman tambak", tipe="numerik",
                         satuan="kaki (ft)"),
    # --- Manajemen tambak ---
    "Operator": dict(kelompok="Manajemen Tambak", label="Tambak dikelola langsung oleh pemilik", tipe="biner",
                     opsi=YA_TIDAK),
    "FertilizerUse": dict(kelompok="Manajemen Tambak", label="Penggunaan pupuk", tipe="kategori",
                          opsi={1: "Campuran organik & anorganik", 2: "Organik", 3: "Anorganik", 4: "Tidak memakai"}),
    "ChemicalUsePondPreparation": dict(kelompok="Manajemen Tambak", label="Bahan pada persiapan kolam",
                                       tipe="kategori", opsi={0: "Perlakuan terapeutik", 1: "Perlakuan kimia"}),
    "ChemicalUseWaterTreatment": dict(kelompok="Manajemen Tambak", label="Bahan pada perlakuan air",
                                      tipe="kategori", opsi={0: "Perlakuan terapeutik", 1: "Perlakuan kimia"}),
    "Aerator": dict(kelompok="Manajemen Tambak", label="Menggunakan aerator", tipe="biner", opsi=YA_TIDAK),
    "GherDryAfterHarvest": dict(kelompok="Manajemen Tambak", label="Tambak dikeringkan setelah panen",
                                tipe="biner", opsi=YA_TIDAK),
    "SludgeRemovalMethods": dict(kelompok="Manajemen Tambak", label="Metode pembuangan lumpur", tipe="kategori",
                                 opsi={1: "Dibilas, lumpur dibuang ke luar tambak",
                                       2: "Dibilas, lumpur di dalam & luar tambak",
                                       3: "Dibilas, lumpur ditimbun di area tambak", 5: "Tidak dibuang"}),
    "SludgeRemovalInterval": dict(kelompok="Manajemen Tambak", label="Interval pembuangan lumpur", tipe="kategori",
                                  opsi={1: "Tidak pernah", 2: "Setiap 1 tahun", 3: "Setiap ≥ 2 tahun"}),
    "MaintainAndRepairDikes": dict(kelompok="Manajemen Tambak", label="Perbaikan pematang", tipe="kategori",
                                   opsi={1: "Memakai tanah dari lahan bera",
                                         2: "Memakai tanah dasar kolam sendiri",
                                         4: "Tidak diperbaiki / memakai tanah tambak lain"}),
    "PeriodOfFallow": dict(kelompok="Manajemen Tambak", label="Masa istirahat (bera) kolam", tipe="numerik",
                           satuan="hari"),
    # --- Manajemen air ---
    "WaterSource_DirectNatural": dict(kelompok="Manajemen Air", label="Sumber air alami langsung", tipe="kategori",
                                      opsi={0: "Bukan sumber alami langsung", 1: "Langsung dari laut/sungai (pasang)",
                                            2: "Sumur bor", 3: "Air hujan"}),
    "WaterSource_IndirectNatural": dict(kelompok="Manajemen Air", label="Sumber air alami tidak langsung",
                                        tipe="kategori",
                                        opsi={0: "Bukan sumber tidak langsung", 1: "Air yang sudah diolah",
                                              2: "Kanal dari laut/sungai", 4: "Melewati tambak lain"}),
    "WaterComingViaOtherFarms": dict(kelompok="Manajemen Air", label="Air masuk melewati tambak lain",
                                     tipe="biner", opsi=YA_TIDAK),
    "Reservoir": dict(kelompok="Manajemen Air", label="Memiliki kolam tandon (reservoir)", tipe="biner",
                      opsi=YA_TIDAK),
    "WaterExchangeFrequency": dict(kelompok="Manajemen Air", label="Frekuensi pergantian air", tipe="kategori",
                                   opsi={1: "Tidak pernah", 2: "> 42 hari sekali", 3: "29–42 hari sekali",
                                         4: "7–28 hari sekali"}),
    "InletOutlet": dict(kelompok="Manajemen Air", label="Saluran masuk & keluar air sama", tipe="biner",
                        opsi=YA_TIDAK),
    # --- Manajemen budidaya ---
    "CultureMethod": dict(kelompok="Manajemen Budidaya", label="Metode budidaya", tipe="kategori",
                          opsi={1: "Polikultur (udang + ikan)", 3: "Polikultur (udang + udang galah)",
                                4: "Monokultur"}),
    "SourceOfFingerling": dict(kelompok="Manajemen Budidaya", label="Sumber benur (PL)", tipe="kategori",
                               opsi={1: "Alam (liar)", 2: "Hatchery swasta terdaftar",
                                     3: "Campuran / hatchery tidak terdaftar"}),
    "StockingDensity_PL40m2": dict(kelompok="Manajemen Budidaya", label="Padat tebar", tipe="numerik",
                                   satuan="ekor PL per 40 m²"),
    "StockingAge_Days": dict(kelompok="Manajemen Budidaya", label="Umur benur saat tebar", tipe="numerik",
                             satuan="hari"),
    "PLQuality": dict(kelompok="Manajemen Budidaya", label="Kualitas benur", tipe="kategori",
                      opsi={1: "Tinggi", 2: "Sedang", 3: "Rendah"}),
    "CropRotation": dict(kelompok="Manajemen Budidaya", label="Melakukan rotasi tanaman/komoditas",
                         tipe="biner", opsi=YA_TIDAK),
    "CulturePractice": dict(kelompok="Manajemen Budidaya", label="Sistem budidaya", tipe="kategori",
                            opsi={0: "Semi-intensif", 1: "Ekstensif (tradisional)"}),
    # --- Pakan ---
    "FeedType": dict(kelompok="Manajemen Pakan", label="Jenis pakan", tipe="kategori",
                     opsi={1: "Tanpa pakan tambahan", 2: "Pelet komersial", 3: "Campuran pelet buatan & komersial",
                           4: "Pelet buatan sendiri", 5: "Pakan hidup"}),
    "FeedAdditives": dict(kelompok="Manajemen Pakan", label="Menggunakan aditif pakan", tipe="biner",
                          opsi=YA_TIDAK),
    # --- Biosekuriti ---
    "BirdScare": dict(kelompok="Biosekuriti", label="Jaring/pengusir burung", tipe="biner", opsi=YA_TIDAK),
    "CrabFence": dict(kelompok="Biosekuriti", label="Pagar penghalang kepiting", tipe="biner", opsi=YA_TIDAK),
    "FootBath": dict(kelompok="Biosekuriti", label="Bak celup kaki (foot bath)", tipe="biner", opsi=YA_TIDAK),
    "LimitedAccess": dict(kelompok="Biosekuriti", label="Akses orang ke tambak dibatasi", tipe="biner",
                          opsi=YA_TIDAK),
    "SameEquipmentForWholeCultureArea": dict(kelompok="Biosekuriti", label="Peralatan sama untuk semua kolam",
                                             tipe="biner", opsi=YA_TIDAK),
    # --- Lingkungan (boleh kosong) ---
    "Temperature": dict(kelompok="Kualitas Air", label="Suhu air", tipe="numerik", satuan="°C", opsional=True),
    "pH": dict(kelompok="Kualitas Air", label="pH air", tipe="numerik", satuan="", opsional=True),
    "Salinity": dict(kelompok="Kualitas Air", label="Salinitas", tipe="numerik", satuan="ppt", opsional=True),
    # --- Riwayat penyakit ---
    "PreviousPrevalence": dict(kelompok="Riwayat Penyakit", label="Prevalensi WSD siklus sebelumnya",
                               tipe="numerik", satuan="%"),
}

FITUR_SKENARIO_A = list(KAMUS_FITUR)                                    # 42 fitur
FITUR_SKENARIO_B = [f for f in FITUR_SKENARIO_A if f != FITUR_RIWAYAT]  # 41 fitur

# ----------------------------------------------------------------------------
# 5. RUANG HYPERPARAMETER (RandomizedSearchCV)
#    Rentang dibuat konservatif (pohon dangkal + regularisasi) karena data kecil.
# ----------------------------------------------------------------------------
RUANG_HYPERPARAMETER = {
    "model__n_estimators": randint(50, 501),
    "model__max_depth": randint(2, 7),
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.5, 0.5),
    "model__min_child_weight": randint(1, 11),
    "model__gamma": uniform(0, 5),
    "model__reg_alpha": uniform(0, 10),
    "model__reg_lambda": uniform(0, 10),
    "seleksi__threshold": ["median", "mean", "0.5*mean"],
}
