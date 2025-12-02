import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SAW & TOPSIS Cloud Storage", layout="wide")

st.title("Analisis Perbandingan SAW & TOPSIS untuk Pemilihan Cloud Storage")

# -----------------------------
# Fixed Kriteria, Bobot, Atribut
# -----------------------------
kriteria = ["C1 - Biaya Penyimpanan",
            "C2 - Biaya Egress",
            "C3 - Latency / Kecepatan Akses",
            "C4 - Skalabilitas & Kemudahan Integrasi",
            "C5 - Keamanan & Compliance"]

atribut = ["cost", "cost", "benefit", "benefit", "benefit"]
bobot = np.array([0.25, 0.20, 0.20, 0.15, 0.20])

# Mapping kategori ke angka
map_C4 = {
    "Rendah": 1,
    "Sedang": 2,
    "Baik": 3,
    "Sangat Baik": 4
}

map_C5 = {
    "Kurang": 1,
    "Cukup": 2,
    "Baik": 3,
    "Sangat Baik": 4
}

# -----------------------------
# Input jumlah alternatif
# -----------------------------
st.header("Input Alternatif")
jumlah_alternatif = st.number_input("Jumlah Alternatif (maksimal 5)", min_value=1, max_value=5, value=3)

nama_alternatif = []
for i in range(jumlah_alternatif):
    nama = st.text_input(f"Nama Alternatif A{i+1}", value=f"Alternatif {i+1}")
    nama_alternatif.append(nama)

st.write("---")

# -----------------------------
# Input Nilai Kriteria
# -----------------------------
st.header("Input Nilai Setiap Kriteria")
data = []

for i in range(jumlah_alternatif):
    st.subheader(nama_alternatif[i])
    nilai = []

    # C1
    c1 = st.number_input(f"Masukkan nilai untuk C1 - Biaya Penyimpanan", min_value=0.0, value=1.0, key=f"{i}-c1")
    nilai.append(c1)

    # C2
    c2 = st.number_input(f"Masukkan nilai untuk C2 - Biaya Egress", min_value=0.0, value=1.0, key=f"{i}-c2")
    nilai.append(c2)

    # C3
    c3 = st.number_input(f"Masukkan nilai untuk C3 - Latency/Kecepatan Akses", min_value=0.0, value=1.0, key=f"{i}-c3")
    nilai.append(c3)

    # C4 → dropdown
    c4 = st.selectbox(
        "Pilih kategori untuk C4 - Skalabilitas & Kemudahan Integrasi",
        list(map_C4.keys()),
        key=f"{i}-c4"
    )
    nilai.append(map_C4[c4])

    # C5 → dropdown
    c5 = st.selectbox(
        "Pilih kategori untuk C5 - Keamanan & Compliance",
        list(map_C5.keys()),
        key=f"{i}-c5"
    )
    nilai.append(map_C5[c5])

    data.append(nilai)

data = np.array(data)
df_data = pd.DataFrame(data, columns=kriteria, index=nama_alternatif)

st.write("### Tabel Input")
st.dataframe(df_data)

st.write("---")


# ============================
# ====== METODE SAW ==========
# ============================
def normalisasi_saw(data, atribut):
    norm = np.zeros_like(data)
    for j in range(data.shape[1]):
        if atribut[j] == "benefit":
            norm[:, j] = data[:, j] / data[:, j].max()
        else:  # cost
            norm[:, j] = data[:, j].min() / data[:, j]
    return norm

norm_saw = normalisasi_saw(data, atribut)
hasil_saw = norm_saw.dot(bobot)
df_saw = pd.DataFrame({
    "Alternatif": nama_alternatif,
    "Nilai SAW": hasil_saw
}).sort_values("Nilai SAW", ascending=False)


# ============================
# ===== METODE TOPSIS ========
# ============================
def topsis(data, atribut, bobot):
    # Normalisasi
    norm = data / np.sqrt((data**2).sum(axis=0))

    # Pembobotan
    y = norm * bobot

    # Ideal positif & negatif
    ideal_plus = np.zeros(data.shape[1])
    ideal_minus = np.zeros(data.shape[1])

    for j in range(data.shape[1]):
        if atribut[j] == "benefit":
            ideal_plus[j] = y[:, j].max()
            ideal_minus[j] = y[:, j].min()
        else:
            ideal_plus[j] = y[:, j].min()
            ideal_minus[j] = y[:, j].max()

    # Jarak
    D_plus = np.sqrt(((y - ideal_plus)**2).sum(axis=1))
    D_minus = np.sqrt(((y - ideal_minus)**2).sum(axis=1))

    # Nilai preferensi
    V = D_minus / (D_plus + D_minus)
    return V

hasil_topsis = topsis(data, atribut, bobot)

df_topsis = pd.DataFrame({
    "Alternatif": nama_alternatif,
    "Nilai TOPSIS": hasil_topsis
}).sort_values("Nilai TOPSIS", ascending=False)


# -----------------------------
# TAMPILKAN HASIL
# -----------------------------
st.header("Hasil Perhitungan SAW")
st.dataframe(df_saw)

st.header("Hasil Perhitungan TOPSIS")
st.dataframe(df_topsis)

# -----------------------------
# Rangking Gabungan (opsional)
# -----------------------------
ranking_final = pd.DataFrame({
    "Alternatif": nama_alternatif,
    "SAW": hasil_saw,
    "TOPSIS": hasil_topsis,
    "Rata-rata Ranking": (hasil_saw + hasil_topsis) / 2
}).sort_values("Rata-rata Ranking", ascending=False)

st.header("Rangking Akhir (Gabungan SAW & TOPSIS)")
st.dataframe(ranking_final)
