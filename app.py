import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SAW & TOPSIS Cloud Storage", layout="wide")

st.title("Analisis Perbandingan SAW & TOPSIS untuk Pemilihan Cloud Storage")

st.write("Aplikasi ini digunakan untuk menghitung perbandingan lima kriteria Cloud Storage menggunakan metode SAW dan TOPSIS.")


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
    for k in kriteria:
        v = st.number_input(f"Masukkan nilai untuk {k}", min_value=0.0, value=1.0, step=0.1, key=f"{i}-{k}")
        nilai.append(v)
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
    # Step 1: Normalisasi
    norm = data / np.sqrt((data**2).sum(axis=0))

    # Step 2: Pembobotan
    y = norm * bobot

    # Step 3: A+ dan A-
    ideal_plus = np.zeros(data.shape[1])
    ideal_minus = np.zeros(data.shape[1])

    for j in range(data.shape[1]):
        if atribut[j] == "benefit":
            ideal_plus[j] = y[:, j].max()
            ideal_minus[j] = y[:, j].min()
        else:
            ideal_plus[j] = y[:, j].min()
            ideal_minus[j] = y[:, j].max()

    # Step 4: Jarak ke A+ dan A-
    D_plus = np.sqrt(((y - ideal_plus)**2).sum(axis=1))
    D_minus = np.sqrt(((y - ideal_minus)**2).sum(axis=1))

    # Step 5: Nilai Preferensi
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
