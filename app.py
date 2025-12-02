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
# HITUNG SAW & TOPSIS (VERSI DETAIL)
# ============================================================
if st.button("Hitung SAW dan TOPSIS"):
    df = pd.DataFrame(data_input, columns=["Alternatif", "C1", "C2", "C3", "C4", "C5"])
    st.subheader("📌 Tabel Crips")
    st.dataframe(df)

    # ============================================================
    # -------------------------- SAW -----------------------------
    # ============================================================
    st.header("📘 Perhitungan Metode SAW")

    # --- Normalisasi SAW ---
    st.subheader("3️⃣ Tahap Normalisasi SAW")

    df_saw_norm = df.copy()

    for c in kriteria:
        if atribut[c] == "benefit":
            df_saw_norm[c] = df[c] / df[c].max()
        else:
            df_saw_norm[c] = df[c].min() / df[c]

    df_saw_norm_display = df_saw_norm.copy()
    df_saw_norm_display.index = [f"A{i+1}" for i in range(len(df))]

    st.dataframe(df_saw_norm_display[kriteria])

    # --- Nilai Akhir SAW ---
    st.subheader("4️⃣ Nilai Akhir SAW dan Ranking")

    df_saw = df_saw_norm.copy()
    df_saw["Skor_SAW"] = sum(df_saw[c] * bobot[c] for c in kriteria)
    df_saw = df_saw.sort_values("Skor_SAW", ascending=False)
    df_saw["Ranking"] = range(1, len(df_saw) + 1)

    st.dataframe(df_saw[["Alternatif", "Skor_SAW", "Ranking"]])

    # ============================================================
    # ------------------------- TOPSIS ----------------------------
    # ============================================================
    st.header("📗 Perhitungan Metode TOPSIS")

    df_t = df.copy()
    
    # --- 1. Matriks Normalisasi R (Bukan Vektor Normalisasi!) ---
    st.subheader("3️⃣ Matriks Ternormalisasi (R)")
    
    # R = x_ij / max(x_j)
    R = df_t[kriteria] / df_t[kriteria].max()
    
    R.index = df['Alternatif']
    st.dataframe(R)
    
    # --- 2. Matriks Ternormalisasi Terbobot (Y) ---
    st.subheader("4️⃣ Matriks Ternormalisasi Terbobot (Y)")
    
    bobot_array = np.array(list(bobot.values()))
    Y = R * bobot_array
    st.dataframe(Y)
    
    # --- 3. Solusi Ideal Positif & Negatif ---
    st.subheader("5️⃣ Solusi Ideal Positif (A+) dan Negatif (A-)")
    
    A_plus = []
    A_minus = []
    
    for c in kriteria:
        # Semua kriteria dianggap COST seperti manual
        A_plus.append(Y[c].min())   # ideal positif cost → min
        A_minus.append(Y[c].max())  # ideal negatif cost → max
    
    A_plus_df = pd.DataFrame([A_plus], columns=kriteria)
    A_minus_df = pd.DataFrame([A_minus], columns=kriteria)
    
    st.write("✔ **A+ (Ideal Positif)**")
    st.dataframe(A_plus_df)
    
    st.write("✔ **A– (Ideal Negatif)**")
    st.dataframe(A_minus_df)
    
    # --- 4. Jarak S+ dan S- ---
    st.subheader("6️⃣ Jarak ke Solusi Ideal (S+ dan S-)")
    
    S_plus = np.sqrt(((Y - A_plus) ** 2).sum(axis=1))
    S_minus = np.sqrt(((Y - A_minus) ** 2).sum(axis=1))
    
    df_distance = pd.DataFrame({
        "Alternatif": df["Alternatif"],
        "S+": S_plus,
        "S-": S_minus
    })
    
    st.dataframe(df_distance)
    
    # --- 5. Nilai Preferensi (C+) ---
    st.subheader("7️⃣ Nilai Preferensi (C+) dan Ranking")
    
    C_plus = S_minus / (S_plus + S_minus)
    
    df_topsis = pd.DataFrame({
        "Alternatif": df["Alternatif"],
        "S+": S_plus,
        "S-": S_minus,
        "C+": C_plus
    })
    
    df_topsis = df_topsis.sort_values("C+", ascending=False)
    df_topsis["Ranking"] = range(1, len(df_topsis) + 1)
    
    st.dataframe(df_topsis)
