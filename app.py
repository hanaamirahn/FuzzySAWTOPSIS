import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisis SAW & TOPSIS - Cloud Storage", layout="wide")

st.title("📊 Analisis Perbandingan Metode SAW dan TOPSIS FUZZY")
st.subheader("Penentuan Pemilihan Layanan Cloud Storage pada Lima Alternatif Provider")


# ============================
# DATA DASAR
# ============================

# Bobot kriteria
weights = {
    "C1": 0.25,
    "C2": 0.20,
    "C3": 0.20,
    "C4": 0.15,
    "C5": 0.20,
}

# Atribut (cost = minimum lebih baik, benefit = maksimum lebih baik)
attributes = {
    "C1": "cost",
    "C2": "cost",
    "C3": "benefit",
    "C4": "benefit",
    "C5": "benefit",
}

# Data crips mapping
crips = {
    "C1": [(0.025, 40), (0.025, 60), (0.015, 80), (0.005, 100)],
    "C2": [(0.10, 40), (0.10, 60), (0.05, 80), (0.01, 100)],
}

# Alternatif input mentah
raw_data = pd.DataFrame({
    "Alternatif": ["Amazon S3", "Google Cloud Storage", "Azure Blob Storage", "Backblaze B2", "Wasabi"],
    "C1": [0.023, 0.020, 0.018, 0.006, 0.00699],
    "C2": [0.09, 0.11, 0.08, 0.00, 0.00],
    "C3": [20, 20, 20, 250, 40],
    "C4": ["Sangat Baik", "Sangat Baik", "Sangat Baik", "Sedang", "Baik"],
    "C5": ["Sangat Baik", "Sangat Baik", "Sangat Baik", "Baik", "Baik"],
})

st.header("📥 Data Alternatif")
st.dataframe(raw_data, use_container_width=True)



# ============================
# KONVERSI CRISP → NILAI
# ============================

def convert_c1(x):
    if x > 0.025: return 40
    elif x <= 0.025 and x > 0.015: return 60
    elif x <= 0.015 and x > 0.005: return 80
    else: return 100

def convert_c2(x):
    if x > 0.10: return 40
    elif x <= 0.10 and x > 0.05: return 60
    elif x <= 0.05 and x > 0.01: return 80
    else: return 100

def convert_c3(x):
    if x > 50: return 40
    elif 31 <= x <= 50: return 60
    elif 21 <= x <= 30: return 80
    else: return 100

def convert_text(value):
    mapping = {"Kurang":40, "Cukup":60, "Baik":80, "Sangat Baik":100}
    return mapping[value]


processed = raw_data.copy()
processed["C1"] = processed["C1"].apply(convert_c1)
processed["C2"] = processed["C2"].apply(convert_c2)
processed["C3"] = processed["C3"].apply(convert_c3)
processed["C4"] = processed["C4"].apply(convert_text)
processed["C5"] = processed["C5"].apply(convert_text)

st.header("📗 Data Setelah Konversi Crips → Nilai")
st.dataframe(processed, use_container_width=True)


# ============================
# SAW METHOD
# ============================

st.header("🔎 Perhitungan Metode SAW")

df = processed.copy()
criteria = ["C1","C2","C3","C4","C5"]

# Normalisasi
norm = df.copy()
for c in criteria:
    if attributes[c] == "benefit":
        norm[c] = df[c] / df[c].max()
    else:  # cost
        norm[c] = df[c].min() / df[c]

# Hitung nilai SAW
norm["SAW"] = (
    norm["C1"]*weights["C1"] +
    norm["C2"]*weights["C2"] +
    norm["C3"]*weights["C3"] +
    norm["C4"]*weights["C4"] +
    norm["C5"]*weights["C5"]
)

saw_result = norm[["Alternatif","SAW"]].copy()
saw_result["Ranking"] = saw_result["SAW"].rank(ascending=False).astype(int)
saw_result = saw_result.sort_values("Ranking")

st.subheader("📌 Hasil SAW")
st.dataframe(saw_result, use_container_width=True)



# ============================
# TOPSIS METHOD
# ============================

st.header("🔎 Perhitungan Metode TOPSIS")

X = df[criteria].values.astype(float)

# Normalisasi
R = X / np.sqrt((X**2).sum(axis=0))

# Normalisasi terbobot
W = np.array([weights[c] for c in criteria])
Y = R * W

# Solusi ideal
ideal_positive = np.zeros(len(criteria))
ideal_negative = np.zeros(len(criteria))

for i, c in enumerate(criteria):
    if attributes[c] == "benefit":
        ideal_positive[i] = Y[:, i].max()
        ideal_negative[i] = Y[:, i].min()
    else:
        ideal_positive[i] = Y[:, i].min()
        ideal_negative[i] = Y[:, i].max()

# Jarak
D_pos = np.sqrt(((Y - ideal_positive) ** 2).sum(axis=1))
D_neg = np.sqrt(((Y - ideal_negative) ** 2).sum(axis=1))

# Nilai preferensi TOPSIS
C = D_neg / (D_pos + D_neg)

topsis_result = pd.DataFrame({
    "Alternatif": raw_data["Alternatif"],
    "TOPSIS": C,
})
topsis_result["Ranking"] = topsis_result["TOPSIS"].rank(ascending=False).astype(int)
topsis_result = topsis_result.sort_values("Ranking")

st.subheader("📌 Hasil TOPSIS")
st.dataframe(topsis_result, use_container_width=True)



# ============================
# GRAFIK PERBANDINGAN
# ============================

st.header("📈 Grafik Perbandingan SAW vs TOPSIS")

graph_data = pd.DataFrame({
    "Alternatif": raw_data["Alternatif"],
    "SAW": saw_result.sort_values("Ranking")["SAW"].values,
    "TOPSIS": topsis_result.sort_values("Ranking")["TOPSIS"].values,
})

st.bar_chart(graph_data.set_index("Alternatif"))


st.success("Analisis SAW dan TOPSIS berhasil dihitung dan divisualisasikan.")
