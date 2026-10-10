"""M2 – Validasi input terhadap rentang data latih."""
import math

# Batas wajar fisik (input di luar batas ini ditolak)
BATAS_FISIK = {
    "Temperature": (0, 45), "pH": (0, 14), "Salinity": (0, 60), "PreviousPrevalence": (0, 100),
    "FarmerAge": (10, 100), "InvolveWithShrimpFarming": (0, 80), "Area_ha": (0.001, 1000),
}
BATAS_INDONESIA = dict(lat=(-11.5, 6.5), lon=(94.5, 141.5))


def validasi_lokasi(lat, lon) -> list:
    if lat is None or lon is None:
        return ["Lokasi tambak belum dipilih pada peta."]
    return []


def peringatan_lokasi(lat, lon) -> list:
    (lat_min, lat_maks), (lon_min, lon_maks) = BATAS_INDONESIA["lat"], BATAS_INDONESIA["lon"]
    if not (lat_min <= lat <= lat_maks and lon_min <= lon <= lon_maks):
        return ["Lokasi berada di luar wilayah Indonesia."]
    return []


def validasi_input(nilai: dict, kamus: dict, rentang: dict):
    """Mengembalikan (daftar_kesalahan, daftar_peringatan)."""
    kesalahan, peringatan = [], []
    for fitur, x in nilai.items():
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue
        label = kamus[fitur]["label"]
        if fitur in BATAS_FISIK:
            bawah, atas = BATAS_FISIK[fitur]
            if not bawah <= x <= atas:
                kesalahan.append(f"{label} = {x} tidak wajar (batas {bawah}–{atas}).")
                continue
        if fitur in rentang:
            r = rentang[fitur]
            if not r["min"] <= x <= r["max"]:
                peringatan.append(f"{label} = {x} berada di luar jangkauan data latih "
                                  f"({r['min']:g}–{r['max']:g}); prediksi kurang dapat diandalkan.")
    return kesalahan, peringatan
