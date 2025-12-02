import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SAW & TOPSIS Cloud Storage", layout="wide")
st.title("Analisis Metode SAW & TOPSIS untuk Pemilihan Cloud Storage")

# ============================================================
# 1. DEFINISI KRITERIA (Fixed)
# ============================================================
kriteria = ["C1", "C2", "C3", "C4", "C5"]
nama_kriteria = {
    "C1": "Biaya Penyimpanan",
    "C2": "Biaya Egress",
    "C3": "Latency / Kecepatan Akses",
    "C4": "Skalabilitas & Integrasi",
    "C5": "Keamanan & Compliance"
}
atribut = {"C1": "cost", "C2": "cost", "C3": "benefit", "C4": "benefit", "C5": "benefit"}
bobot = {"C1": 0.25, "C2": 0.20, "C3": 0.20, "C4": 0.15, "C5": 0.20}

# ============================================================
# 2. TABEL CRIPS
# ============================================================
def konversi_crips(kode, nilai):
    # C1 Biaya penyimpanan
    if kode == "C1":
        if nilai > 0.025: return 40
        if nilai <= 0.025 and nilai > 0.015: return 60
        if nilai <= 0.015 and nilai > 0.005: return 80
        if nilai <= 0.005: return 100

    # C2 Biaya Egress
    if kode == "C2":
        if nilai > 0.10: return 40
        if nilai <= 0.10 and nilai > 0.05: return 60
        if nilai <= 0.05 and nilai > 0.01: return 80
        if nilai <= 0.01: return 100

    # C3 Latency
    if kode == "C3":
        if nilai > 50: return 40
        if 31 <= nilai <= 50: return 60
        if 21 <= nilai <= 30: return 80
        if nilai <= 20: return 100

    return None

map_kategori_C4 = {
    "Rendah": 40,
    "Sedang": 60,
    "Baik": 80,
    "Sangat Baik": 100
}

map_kategori_C5 = {
    "Kurang": 40,
    "Cukup": 60,
    "Baik": 80,
    "Sangat Baik": 100
}

}

# ============================================================
# INPUT JUMLAH ALTERNATIF
# ============================================================
st.subheader("Input Alternatif yang Akan Dianalisis")

jumlah_alt = st.selectbox("Jumlah Alternatif:", [1, 2, 3, 4, 5], index=4)

# DataFrame penampung
data_input = []

for i in range(jumlah_alt):
    st.markdown(f"### Alternatif A{i+1}")
    nama = st.text_input(f"Nama Alternatif A{i+1}", key=f"nama_{i}")

    c1 = st.number_input(f"C1 Biaya Penyimpanan ($/GB)", min_value=0.0, key=f"c1_{i}")
    c2 = st.number_input(f"C2 Biaya Egress ($/GB)", min_value=0.0, key=f"c2_{i}")
    c3 = st.number_input(f"C3 Latency (ms)", min_value=0.0, key=f"c3_{i}")

    c4 = st.selectbox("C4 Skalabilitas & Kemudahan Integrasi", ["Rendah", "Sedang", "Baik", "Sangat Baik"], key=f"c4_{i}")
    c5 = st.selectbox("C5 Keamanan & Compliance", ["Kurang", "Cukup", "Baik", "Sangat Baik"], key=f"c5_{i}")

data_input.append([
    nama,
    konversi_crips("C1", c1),
    konversi_crips("C2", c2),
    konversi_crips("C3", c3),
    map_kategori_C4[c4],
    map_kategori_C5[c5]
])


# ============================================================
# HITUNG SAW & TOPSIS
# ============================================================
if st.button("Hitung SAW dan TOPSIS"):
    df = pd.DataFrame(data_input, columns=["Alternatif", "C1", "C2", "C3", "C4", "C5"])
    st.subheader("📌 Nilai Crips")
    st.dataframe(df)

    # --------------------------------------------------------
    # SAW NORMALISASI
    # --------------------------------------------------------
    df_saw = df.copy()
    for c in kriteria:
        if atribut[c] == "benefit":
            df_saw[c] = df[c] / df[c].max()
        else:  # cost
            df_saw[c] = df[c].min() / df[c]

    # hitung skor akhir
    df_saw["Skor_SAW"] = sum(df_saw[c] * bobot[c] for c in kriteria)
    df_saw = df_saw.sort_values("Skor_SAW", ascending=False)

    st.subheader("📊 Hasil SAW")
    st.dataframe(df_saw[["Alternatif", "Skor_SAW"]])

    # --------------------------------------------------------
    # TOPSIS
    # --------------------------------------------------------
    df_t = df.copy()
    R = df_t[kriteria] / np.sqrt((df_t[kriteria]**2).sum())

    # bobot
    V = R * np.array(list(bobot.values()))

    ideal_plus = []
    ideal_minus = []

    for i, c in enumerate(kriteria):
        if atribut[c] == "benefit":
            ideal_plus.append(V[c].max())
            ideal_minus.append(V[c].min())
        else:
            ideal_plus.append(V[c].min())
            ideal_minus.append(V[c].max())

    D_plus = np.sqrt(((V - ideal_plus)**2).sum(axis=1))
    D_minus = np.sqrt(((V - ideal_minus)**2).sum(axis=1))

    preferensi = D_minus / (D_plus + D_minus)
    df_t["Skor_TOPSIS"] = preferensi
    df_t = df_t.sort_values("Skor_TOPSIS", ascending=False)

    st.subheader("📊 Hasil TOPSIS")
    st.dataframe(df_t[["Alternatif", "Skor_TOPSIS"]])
