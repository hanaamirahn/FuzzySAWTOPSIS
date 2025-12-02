import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

st.set_page_config(page_title="Fuzzy SAW & TOPSIS - Cloud Storage", layout="wide")

st.title("Analisis Perbandingan: SAW & TOPSIS (Fuzzy) \nPemilihan Layanan Cloud Storage")
st.write("Aplikasi Streamlit untuk menghitung dan membandingkan alternatif menggunakan metode Fuzzy SAW dan Fuzzy TOPSIS.")

# Fixed criteria metadata
CRITERIA = [
    {"kode": "C1", "nama": "Biaya penyimpanan", "atribut": "Cost", "bobot": 0.25},
    {"kode": "C2", "nama": "Biaya Egress", "atribut": "Cost", "bobot": 0.20},
    {"kode": "C3", "nama": "Latency / Kecepatan Akses", "atribut": "Benefit", "bobot": 0.20},
    {"kode": "C4", "nama": "Skalabilitas & Kemudahan Integrasi", "atribut": "Benefit", "bobot": 0.15},
    {"kode": "C5", "nama": "Keamanan & Compliance", "atribut": "Benefit", "bobot": 0.20},
]

WEIGHTS = np.array([c['bobot'] for c in CRITERIA])
ATTRIBUTES = [c['atribut'] for c in CRITERIA]
CR_NAMES = [c['nama'] for c in CRITERIA]
CR_CODES = [c['kode'] for c in CRITERIA]

st.sidebar.header("Pengaturan Input")
num_alt = st.sidebar.number_input("Jumlah alternatif (provider)", min_value=2, max_value=12, value=5, step=1)

input_mode = st.sidebar.selectbox("Mode input nilai", ["Angka (kritis: nilai numerik)", "Linguistic (Very Low..Very High)"])

# Default providers (common cloud storage)
default_providers(n):
    default = ["Amazon S3", "Google Cloud Storage", "Azure Blob Storage", "Backblaze B2", "Wasabi Hot Cloud Storage"]
    out = []
    for i in range(n):
        if i < len(default):
            out.append(default[i])
        else:
            out.append(f"Alternatif {i+1}")
    return out

alts = st.sidebar.text_area("Nama alternatif, pisahkan tiap baris (atau biarkan default)", "\n".join(default_providers(num_alt)))
alt_list = [a.strip() for a in alts.splitlines() if a.strip()][:num_alt]
if len(alt_list) < num_alt:
    # pad
    alt_list += [f"Alternatif {i+1}" for i in range(len(alt_list), num_alt)]

st.subheader("Data Input Alternatif")
st.write("Masukkan nilai untuk setiap alternatif pada masing-masing kriteria. Jika memilih mode linguistik, pilih label untuk tiap sel.")

# Linguistic scale -> triangular fuzzy numbers (l,m,u)
LINGUISTIC_SCALE = {
    "VL": (0.0, 0.0, 0.25),  # Very Low
    "L": (0.0, 0.25, 0.5),
    "M": (0.25, 0.5, 0.75),
    "H": (0.5, 0.75, 1.0),
    "VH": (0.75, 1.0, 1.0),
}
LINGUISTIC_LABELS = {"Very Low": "VL", "Low": "L", "Medium": "M", "High": "H", "Very High": "VH"}

# Build input DataFrame structure
if input_mode.startswith("Angka"):
    # numeric input table with default placeholders
    default_vals = {
        'C1 (Cost - lower is better)': [0.02, 0.03, 0.025, 0.01, 0.015][:num_alt],
        'C2 (Cost - lower is better)': [0.01, 0.02, 0.02, 0.015, 0.01][:num_alt],
        'C3 (Benefit - higher is better)': [80, 70, 75, 60, 65][:num_alt],
        'C4 (Benefit - higher is better)': [8, 7, 7.5, 6, 6.5][:num_alt],
        'C5 (Benefit - higher is better)': [9, 8.5, 8.8, 7.5, 8][:num_alt],
    }
    df_input = pd.DataFrame(default_vals, index=alt_list)
    edited = st.experimental_data_editor(df_input, num_rows="dynamic")
    # validation: convert to numeric
    try:
        matrix = edited.astype(float).values
    except Exception:
        st.error("Semua nilai harus numerik pada mode Angka. Perbaiki input Anda.")
        st.stop()
else:
    # Linguistic mode: provide selectboxes per cell
    st.write("Gunakan label linguistik: Very Low, Low, Medium, High, Very High")
    cols = st.columns(len(CRITERIA))
    data = {cr['kode']: [] for cr in CRITERIA}
    for i, alt in enumerate(alt_list):
        row_vals = []
        for j, cr in enumerate(CRITERIA):
            key = f"{alt}_C{j}"
            val = cols[j].selectbox(f"{alt} - {cr['kode']}", options=list(LINGUISTIC_LABELS.keys()), key=key, index=2)
            data[cr['kode']].append(LINGUISTIC_LABELS[val])
    # Build matrix of triangular numbers
    # We'll keep both labels and triangular fuzzy numbers later
    df_labels = pd.DataFrame({f"{c['kode']}": data[c['kode']] for c in CRITERIA}, index=alt_list)
    st.table(df_labels)
    matrix = None  # will build fuzzy matrix below

# Helper: triangular fuzzy number operations
class TFN:
    def __init__(self, l, m, u):
        self.l = float(l)
        self.m = float(m)
        self.u = float(u)
    def __repr__(self):
        return f"({self.l:.3f},{self.m:.3f},{self.u:.3f})"

def tfn_add(a: TFN, b: TFN):
    return TFN(a.l + b.l, a.m + b.m, a.u + b.u)

def tfn_mul_scalar(a: TFN, k: float):
    return TFN(a.l * k, a.m * k, a.u * k)

def tfn_div(a: TFN, b: TFN):
    # divide a by b (approx): use pointwise division of vertices (assuming positive)
    return TFN(a.l / b.u, a.m / b.m, a.u / b.l)

def defuzzify_centroid(a: TFN):
    return (a.l + a.m + a.u) / 3.0

def tfn_distance(a: TFN, b: TFN):
    # Euclidean distance between two triangular fuzzy numbers (vertex method)
    return sqrt((1.0/3.0) * ((a.l - b.l)**2 + (a.m - b.m)**2 + (a.u - b.u)**2))

# Construct fuzzy decision matrix
if input_mode.startswith("Angka"):
    # Convert crisp numeric matrix to fuzzy triangular numbers by attaching small spread
    # For costs and benefits we normalize differently later; here we convert each crisp x to (x*0.95, x, x*1.05)
    fuzz_matrix = []
    for i in range(len(alt_list)):
        row = []
        for j in range(len(CRITERIA)):
            x = float(matrix[i, j])
            if x < 0:
                st.warning("Nilai negatif ditemukan; konversi fuzzy menggunakan spread absolut. Pastikan data benar.")
            row.append(TFN(0.95*x, x, 1.05*x))
        fuzz_matrix.append(row)
else:
    # linguistic -> map to normalized triangular numbers from LINGUISTIC_SCALE
    fuzz_matrix = []
    for i in range(len(alt_list)):
        row = []
        for j in range(len(CRITERIA)):
            lab = df_labels.iloc[i, j]  # code like 'M','H'
            tri = LINGUISTIC_SCALE[lab]
            row.append(TFN(*tri))
        fuzz_matrix.append(row)

# Normalization for fuzzy numbers
# For benefit criteria: r_ij = a_ij / a_j_max
# For cost criteria: r_ij = a_j_min / a_ij

def fuzzy_max_per_col(matrix):
    # max by upper value
    n = len(matrix)
    m = len(matrix[0])
    cols = []
    for j in range(m):
        max_u = max(matrix[i][j].u for i in range(n))
        cols.append(max_u)
    return cols

def fuzzy_min_per_col(matrix):
    n = len(matrix)
    m = len(matrix[0])
    cols = []
    for j in range(m):
        min_l = min(matrix[i][j].l for i in range(n))
        cols.append(min_l)
    return cols

n_alt = len(alt_list)

# compute normalized fuzzy decision matrix
fuzzy_norm = [[None]*len(CRITERIA) for _ in range(n_alt)]
max_u = fuzzy_max_per_col(fuzz_matrix)
min_l = fuzzy_min_per_col(fuzz_matrix)

for j, attr in enumerate(ATTRIBUTES):
    for i in range(n_alt):
        aij = fuzz_matrix[i][j]
        if attr == 'Benefit':
            # r = aij / max_u_j  => TFN division by crisp max
            denom = TFN(max_u[j], max_u[j], max_u[j])
            fuzzy_norm[i][j] = tfn_div(aij, denom)
        else:
            # Cost: r = min_l_j / aij
            numer = TFN(min_l[j], min_l[j], min_l[j])
            fuzzy_norm[i][j] = tfn_div(numer, aij)

# Apply weights
weighted_fuzzy = [[None]*len(CRITERIA) for _ in range(n_alt)]
for i in range(n_alt):
    for j in range(len(CRITERIA)):
        w = WEIGHTS[j]
        weighted_fuzzy[i][j] = tfn_mul_scalar(fuzzy_norm[i][j], w)

# --- FUZZY SAW ---
# Aggregate by summing weighted criteria
fuzzy_saw_agg = []
score_saw_defuzz = []
for i in range(n_alt):
    s = TFN(0,0,0)
    for j in range(len(CRITERIA)):
        s = tfn_add(s, weighted_fuzzy[i][j])
    fuzzy_saw_agg.append(s)
    score_saw_defuzz.append(defuzzify_centroid(s))

# Ranking for SAW
saw_ranking_idx = np.argsort(score_saw_defuzz)[::-1]

# --- FUZZY TOPSIS ---
# Positive ideal (PIS): for benefit - max of u, for cost - min of l  BUT we need TFN PIS and NIS per criterion
pis = []
nis = []
for j, attr in enumerate(ATTRIBUTES):
    if attr == 'Benefit':
        # PIS = max of upper among weighted normalized
        max_u = max(weighted_fuzzy[i][j].u for i in range(n_alt))
        max_m = max(weighted_fuzzy[i][j].m for i in range(n_alt))
        max_l = max(weighted_fuzzy[i][j].l for i in range(n_alt))
        pis.append(TFN(max_l, max_m, max_u))
        min_u = min(weighted_fuzzy[i][j].u for i in range(n_alt))
        min_m = min(weighted_fuzzy[i][j].m for i in range(n_alt))
        min_l = min(weighted_fuzzy[i][j].l for i in range(n_alt))
        nis.append(TFN(min_l, min_m, min_u))
    else:
        # Cost: ideal is minimum
        min_l = min(weighted_fuzzy[i][j].l for i in range(n_alt))
        min_m = min(weighted_fuzzy[i][j].m for i in range(n_alt))
        min_u = min(weighted_fuzzy[i][j].u for i in range(n_alt))
        pis.append(TFN(min_l, min_m, min_u))
        max_l = max(weighted_fuzzy[i][j].l for i in range(n_alt))
        max_m = max(weighted_fuzzy[i][j].m for i in range(n_alt))
        max_u = max(weighted_fuzzy[i][j].u for i in range(n_alt))
        nis.append(TFN(max_l, max_m, max_u))

# Distances to PIS and NIS
d_pos = []
d_neg = []
for i in range(n_alt):
    sum_pos = 0.0
    sum_neg = 0.0
    for j in range(len(CRITERIA)):
        sum_pos += tfn_distance(weighted_fuzzy[i][j], pis[j])**2
        sum_neg += tfn_distance(weighted_fuzzy[i][j], nis[j])**2
    dpos = sqrt(sum_pos)
    dneg = sqrt(sum_neg)
    d_pos.append(dpos)
    d_neg.append(dneg)

cc = []
for i in range(n_alt):
    denom = d_pos[i] + d_neg[i]
    if denom == 0:
        cc.append(0)
    else:
        cc.append(d_neg[i] / denom)

topsis_ranking_idx = np.argsort(cc)[::-1]

# Show results
st.subheader("Hasil Perhitungan")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Fuzzy SAW - Nilai Agregat (Triangular TFN) & Defuzzified Score**")
    saw_df = pd.DataFrame({
        'Alternatif': alt_list,
        'TFN (l,m,u)': [str(t) for t in fuzzy_saw_agg],
        'Defuzzified Score': score_saw_defuzz
    })
    saw_df = saw_df.sort_values('Defuzzified Score', ascending=False).reset_index(drop=True)
    st.dataframe(saw_df)
    st.markdown("**Ranking SAW**")
    for i, r in enumerate(saw_df['Alternatif']):
        st.write(f"{i+1}. {r} (score: {saw_df.loc[i,'Defuzzified Score']:.4f})")

with col2:
    st.markdown("**Fuzzy TOPSIS - Closeness Coefficient (CC)**")
    topsis_df = pd.DataFrame({
        'Alternatif': alt_list,
        'D+ (to PIS)': d_pos,
        'D- (to NIS)': d_neg,
        'Closeness Coef (CC)': cc
    })
    topsis_df = topsis_df.sort_values('Closeness Coef (CC)', ascending=False).reset_index(drop=True)
    st.dataframe(topsis_df)
    st.markdown("**Ranking TOPSIS**")
    for i, r in enumerate(topsis_df['Alternatif']):
        st.write(f"{i+1}. {r} (CC: {topsis_df.loc[i,'Closeness Coef (CC)']:.4f})")

# Combined comparison
st.subheader("Perbandingan Ranking SAW vs TOPSIS")
combined = pd.DataFrame({
    'Alternatif': alt_list,
    'SAW Score': score_saw_defuzz,
    'SAW Rank': np.argsort(score_saw_defuzz)[::-1] + 1,
    'TOPSIS CC': cc,
    'TOPSIS Rank': np.argsort(cc)[::-1] + 1
})
combined = combined.sort_values('SAW Rank')
st.dataframe(combined)

# Simple bar charts
st.subheader("Visualisasi Skor & CC")
fig, ax = plt.subplots()
ax.bar(alt_list, score_saw_defuzz)
ax.set_title('Defuzzified SAW Score per Alternatif')
ax.set_ylabel('Score')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

fig2, ax2 = plt.subplots()
ax2.bar(alt_list, cc)
ax2.set_title('TOPSIS Closeness Coef per Alternatif')
ax2.set_ylabel('CC')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig2)

st.markdown("---")
st.info("Catatan: Implementasi ini menggunakan pendekatan fuzzy sederhana: jika input numerik diberikan, setiap nilai diubah menjadi TFN dengan spread ±5%. Untuk input linguistik, digunakan mapping TFN yang dinormalisasi (0..1). Bobot kriteria yang dipakai adalah bobot crisp yang Anda berikan. Metode normalisasi dan operasi TFN menggunakan pendekatan titik/vertex untuk pembagian dan jarak.")
st.markdown("Jika Anda mau, saya bisa: (1) menambahkan opsi penyimpanan hasil ke CSV, (2) menyesuaikan skala linguistik, atau (3) menggunakan metode defuzzifikasi lain. Beritahu saya fitur mana yang Anda inginkan selanjutnya.")
