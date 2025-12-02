import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="SAW vs TOPSIS - Fuzzy Multicriteria", layout='wide')

st.title("Analisis Perbandingan Metode Fuzzy SAW dan TOPSIS\nPemilihan Layanan Cloud Storage")

st.markdown("Kriteria: C1=Biaya penyimpanan (Cost), C2=Biaya Egress (Cost), "
            "C3=Latency (Benefit), C4=Skalabilitas & Integrasi (Benefit), "
            "C5=Keamanan & Compliance (Benefit). Bobot: [0.25,0.20,0.20,0.15,0.20].")

# Fixed metadata
criteria = [
    {"code":"C1","name":"Biaya penyimpanan","type":"cost","weight":0.25},
    {"code":"C2","name":"Biaya Egress","type":"cost","weight":0.20},
    {"code":"C3","name":"Latency / Kecepatan Akses","type":"benefit","weight":0.20},
    {"code":"C4","name":"Skalabilitas & Kemudahan Integrasi","type":"benefit","weight":0.15},
    {"code":"C5","name":"Keamanan & Compliance","type":"benefit","weight":0.20},
]

linguistic_map = {
    "Sangat Buruk": (0.0, 0.0, 0.25),
    "Buruk": (0.0, 0.25, 0.5),
    "Cukup": (0.25, 0.5, 0.75),
    "Baik": (0.5, 0.75, 1.0),
    "Sangat Baik": (0.75, 1.0, 1.0),
}

# === Dropdown Alternatif Maksimal 5 ===
st.sidebar.header("Pengaturan Alternatif")
available_alternatives = [
    "AWS S3",
    "Google Cloud Storage",
    "Microsoft Azure Blob",
    "Backblaze B2",
    "DigitalOcean Spaces"
]

selected_names = st.sidebar.multiselect(
    "Pilih Alternatif (maks 5)",
    options=available_alternatives,
    default=available_alternatives[:3],
    max_selections=5
)

num = len(selected_names)

if num == 0:
    st.warning("Pilih minimal satu alternatif untuk dianalisis.")
    st.stop()

# === Input nilai untuk setiap alternatif ===
st.header("Input Data Alternatif")

values = {c['code']: [] for c in criteria}

for i in range(num):
    st.subheader(f"Alternatif: {selected_names[i]}")

    c1 = st.number_input(f"{selected_names[i]} - C1: Biaya penyimpanan ($/GB)", min_value=0.0, value=0.03, key=f"c1_{i}")
    c2 = st.number_input(f"{selected_names[i]} - C2: Biaya Egress ($/GB)", min_value=0.0, value=0.10, key=f"c2_{i}")
    c3 = st.number_input(f"{selected_names[i]} - C3: Latency (ms)", min_value=0.0, value=20.0, key=f"c3_{i}")
    c4 = st.selectbox(f"{selected_names[i]} - C4: Skalabilitas", options=list(linguistic_map.keys()), index=3, key=f"c4_{i}")
    c5 = st.selectbox(f"{selected_names[i]} - C5: Keamanan", options=list(linguistic_map.keys()), index=3, key=f"c5_{i}")

    values['C1'].append(c1)
    values['C2'].append(c2)
    values['C3'].append(c3)
    values['C4'].append(c4)
    values['C5'].append(c5)

# === Fungsi utilitas Fuzzy ===
def to_tri_from_numeric(x, rel=0.05):
    if x == 0:
        return (0.0, 0.0, 0.0)
    return (x*(1-rel), x, x*(1+rel))

def tri_div(A, B):
    denom_l = B[0] if B[0] != 0 else 1e-9
    denom_m = B[1] if B[1] != 0 else 1e-9
    denom_u = B[2] if B[2] != 0 else 1e-9
    return (A[0]/denom_u, A[1]/denom_m, A[2]/denom_l)

def tri_mul_scalar(A, s):
    return (A[0]*s, A[1]*s, A[2]*s)

def tri_add(A, B):
    return (A[0]+B[0], A[1]+B[1], A[2]+B[2])

def tri_distance(A, B):
    return math.sqrt(((A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2)/3)

def defuzzify(A):
    return (A[0]+A[1]+A[2]) / 3.0

# === Bangun matriks fuzzy ===
fuzzy_matrix = []
for i in range(num):
    row = []
    row.append(to_tri_from_numeric(values['C1'][i], 0.05))
    row.append(to_tri_from_numeric(values['C2'][i], 0.05))

    lat = values['C3'][i] if values['C3'][i] > 0 else 1e-6
    row.append(to_tri_from_numeric(1/lat, 0.05))

    row.append(linguistic_map[values['C4'][i]])
    row.append(linguistic_map[values['C5'][i]])
    fuzzy_matrix.append(row)

# === Tampilkan matriks fuzzy ===
st.subheader("Matriks Fuzzy")
disp = []
for i in range(num):
    r = {"Alternatif": selected_names[i]}
    for j, c in enumerate(criteria):
        r[c["code"]] = str(tuple(round(v,4) for v in fuzzy_matrix[i][j]))
    disp.append(r)
st.dataframe(pd.DataFrame(disp))

# === FUZZY SAW ===
max_upper = [max(fuzzy_matrix[i][j][2] for i in range(num)) for j in range(5)]
min_lower = [min(fuzzy_matrix[i][j][0] for i in range(num)) for j in range(5)]

norm = []
for i in range(num):
    row = []
    for j, c in enumerate(criteria):
        A = fuzzy_matrix[i][j]
        if c['type'] == 'benefit':
            denom = (min_lower[j], (min_lower[j]+max_upper[j])/2, max_upper[j])
            n = tri_div(A, denom)
        else:
            nume = (min_lower[j], min_lower[j], min_lower[j])
            n = tri_div(nume, A)
        row.append(n)
    norm.append(row)

w_norm = [[tri_mul_scalar(norm[i][j], criteria[j]['weight']) for j in range(5)] for i in range(num)]

saw_fuzzy = []
for i in range(num):
    s = (0,0,0)
    for j in range(5):
        s = tri_add(s, w_norm[i][j])
    saw_fuzzy.append(s)

saw_score = [defuzzify(s) for s in saw_fuzzy]

# === FUZZY TOPSIS ===
pis, nis = [], []
for j, c in enumerate(criteria):
    if c['type'] == 'benefit':
        pis.append((max(fuzzy_matrix[i][j][0] for i in range(num)),
                    max(fuzzy_matrix[i][j][1] for i in range(num)),
                    max(fuzzy_matrix[i][j][2] for i in range(num))))
        nis.append((min(fuzzy_matrix[i][j][0] for i in range(num)),
                    min(fuzzy_matrix[i][j][1] for i in range(num)),
                    min(fuzzy_matrix[i][j][2] for i in range(num))))
    else:
        pis.append((min(fuzzy_matrix[i][j][0] for i in range(num)),
                    min(fuzzy_matrix[i][j][1] for i in range(num)),
                    min(fuzzy_matrix[i][j][2] for i in range(num))))
        nis.append((max(fuzzy_matrix[i][j][0] for i in range(num)),
                    max(fuzzy_matrix[i][j][1] for i in range(num)),
                    max(fuzzy_matrix[i][j][2] for i in range(num))))

dist_p, dist_n = [], []
for i in range(num):
    dp = 0; dn = 0
    for j in range(5):
        wp = tri_mul_scalar(pis[j], criteria[j]['weight'])
        wn = tri_mul_scalar(nis[j], criteria[j]['weight'])
        dp += tri_distance(w_norm[i][j], wp)
        dn += tri_distance(w_norm[i][j], wn)
    dist_p.append(dp)
    dist_n.append(dn)

cc = [dist_n[i] / (dist_p[i] + dist_n[i]) for i in range(num)]

# === TABEL HASIL ===
df = pd.DataFrame({
    "Alternatif": selected_names,
    "SAW_score_defuzz": [round(x,6) for x in saw_score],
    "TOPSIS_closeness": [round(x,6) for x in cc],
})

df["Rank_SAW"] = df["SAW_score_defuzz"].rank(ascending=False).astype(int)
df["Rank_TOPSIS"] = df["TOPSIS_closeness"].rank(ascending=False).astype(int)

st.subheader("Hasil Perhitungan & Perangkingan")
st.dataframe(df.sort_values("Rank_TOPSIS"))
