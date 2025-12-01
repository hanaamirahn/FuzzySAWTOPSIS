import streamlit as st
import pandas as pd
import numpy as np
import math
import io

st.set_page_config(page_title="SAW vs TOPSIS - Fuzzy Multicriteria", layout='wide')

st.title("Analisis Perbandingan Metode Fuzzy SAW dan TOPSIS\nPemilihan Layanan Cloud Storage (5 Kriteria)")
st.markdown("Kriteria: C1=Biaya penyimpanan (Cost), C2=Biaya Egress (Cost), C3=Latency (Benefit), C4=Skalabilitas & Integrasi (Benefit), C5=Keamanan & Compliance (Benefit). Bobot: [0.25,0.20,0.20,0.15,0.20].")

# Fixed weights and meta
criteria = [
    {"code":"C1","name":"Biaya penyimpanan","type":"cost","weight":0.25},
    {"code":"C2","name":"Biaya Egress","type":"cost","weight":0.20},
    {"code":"C3","name":"Latency / Kecepatan Akses","type":"benefit","weight":0.20},
    {"code":"C4","name":"Skalabilitas & Kemudahan Integrasi","type":"benefit","weight":0.15},
    {"code":"C5","name":"Keamanan & Compliance","type":"benefit","weight":0.20},
]
weights = [c['weight'] for c in criteria]

# Linguistic to triangular fuzzy number mapping for qualitative criteria
linguistic_map = {
    "Sangat Buruk": (0.0, 0.0, 0.25),
    "Buruk": (0.0, 0.25, 0.5),
    "Cukup": (0.25, 0.5, 0.75),
    "Baik": (0.5, 0.75, 1.0),
    "Sangat Baik": (0.75, 1.0, 1.0),
}

st.sidebar.header("Pengaturan Input")
num = st.sidebar.number_input("Jumlah alternatif", min_value=2, max_value=10, value=5, step=1)

with st.sidebar.expander("Petunjuk"):
    st.write("Masukkan nilai numerik untuk C1 ($/GB), C2 ($/GB), C3 (ms). Untuk C4 dan C5 pilih bahasa kualitatif (Sangat Buruk..Sangat Baik). Aplikasi akan mengubah semua menjadi bilangan fuzzy segitiga dan menjalankan perhitungan Fuzzy SAW dan Fuzzy TOPSIS.")

# Collect alternatives
alts = []
st.header("Input Data Alternatif")
cols = st.columns([1,1,1,1,1,1])
with cols[0]:
    names = []
    for i in range(num):
        names.append(st.text_input(f"Nama Alternatif #{i+1}", value=f"A{i+1}"))

# Create a form-like grid for numeric/qualitative inputs
values = {c['code']: [] for c in criteria}
for i in range(num):
    st.subheader(f"Alternatif: {names[i]}")
    c1 = st.number_input(f"{names[i]} - C1: Biaya penyimpanan ($/GB)", min_value=0.0, value=0.03, key=f"c1_{i}")
    c2 = st.number_input(f"{names[i]} - C2: Biaya Egress ($/GB)", min_value=0.0, value=0.10, key=f"c2_{i}")
    c3 = st.number_input(f"{names[i]} - C3: Latency (ms) - lebih kecil lebih baik? (angka) ", min_value=0.0, value=20.0, key=f"c3_{i}")
    c4 = st.selectbox(f"{names[i]} - C4: Skalabilitas & Kemudahan Integrasi", options=list(linguistic_map.keys()), index=3, key=f"c4_{i}")
    c5 = st.selectbox(f"{names[i]} - C5: Keamanan & Compliance", options=list(linguistic_map.keys()), index=3, key=f"c5_{i}")
    values['C1'].append(c1)
    values['C2'].append(c2)
    values['C3'].append(c3)
    values['C4'].append(c4)
    values['C5'].append(c5)

# Utility: triangular fuzzy operations

def to_tri_from_numeric(x, rel=0.05):
    # make small fuzziness around crisp number
    if x == 0:
        return (0.0, 0.0, 0.0)
    return (x * (1 - rel), x, x * (1 + rel))

def tri_div(A, B):
    # A and B are triangular tuples (l,m,u)
    # A / B approximated by (lA/uB, mA/mB, uA/lB) guard div by zero
    l = A[0] / (B[2] if B[2] != 0 else 1e-9)
    m = A[1] / (B[1] if B[1] != 0 else 1e-9)
    u = A[2] / (B[0] if B[0] != 0 else 1e-9)
    return (l, m, u)

def tri_mul_scalar(A, scalar):
    return (A[0]*scalar, A[1]*scalar, A[2]*scalar)

def tri_add(A, B):
    return (A[0]+B[0], A[1]+B[1], A[2]+B[2])

def tri_distance(A, B):
    # Vertex method
    return math.sqrt(((A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2)/3.0)

def defuzzify_centroid(A):
    return (A[0] + A[1] + A[2]) / 3.0

# Build fuzzy decision matrix
fuzzy_matrix = []  # list of list of triangular numbers
for i in range(num):
    row = []
    # C1 (cost)
    t1 = to_tri_from_numeric(values['C1'][i], rel=0.05)
    row.append(t1)
    # C2 (cost)
    t2 = to_tri_from_numeric(values['C2'][i], rel=0.05)
    row.append(t2)
    # C3 (benefit: latency lower is better -> we invert numeric to be benefit-like by using 1/latency)
    latency = values['C3'][i]
    if latency <= 0:
        latency = 1e-6
    latency_inv = 1.0 / latency
    t3 = to_tri_from_numeric(latency_inv, rel=0.05)
    row.append(t3)
    # C4 qualitative
    row.append(linguistic_map[values['C4'][i]])
    # C5 qualitative
    row.append(linguistic_map[values['C5'][i]])
    fuzzy_matrix.append(row)

# Show fuzzy matrix
st.subheader("Matriks Fuzzy (Triangular) — representasi internal")
fm_display = []
for i in range(num):
    row = {"Alternatif": names[i]}
    for j, c in enumerate(criteria):
        row[c['code']] = str(tuple(round(v,4) for v in fuzzy_matrix[i][j]))
    fm_display.append(row)
st.dataframe(pd.DataFrame(fm_display))

# Fuzzy SAW
st.header("Hasil Analisis: Fuzzy SAW dan Fuzzy TOPSIS")

# Normalization for SAW: for benefit criteria, divide by max; for cost criteria, min/val
# For triangular we need fuzzy normalization: r_ij = a_ij / a_jmax (for benefit), or a_jmin / a_ij (for cost)
# We'll compute per-criterion fuzzy max (by upper bound) and min (by lower bound)

# compute per-criterion max upper and min lower
max_upper = []
min_lower = []
for j in range(len(criteria)):
    uppers = [fuzzy_matrix[i][j][2] for i in range(num)]
    lowers = [fuzzy_matrix[i][j][0] for i in range(num)]
    max_upper.append(max(uppers))
    min_lower.append(min(lowers))

# SAW normalized fuzzy matrix
norm_fuzzy = [[(0,0,0) for _ in range(len(criteria))] for _ in range(num)]
for i in range(num):
    for j, c in enumerate(criteria):
        A = fuzzy_matrix[i][j]
        if c['type'] == 'benefit':
            B = (min_lower[j], (min_lower[j]+max_upper[j])/2.0, max_upper[j])
            # To normalize benefit: A / max_upper_j approximately
            denom = (min_lower[j], (min_lower[j]+max_upper[j])/2.0, max_upper[j])
            norm = tri_div(A, denom)
        else:
            # cost: normalize by min / A
            numerator = (min_lower[j], (min_lower[j]+min_lower[j])/2.0, min_lower[j])
            norm = tri_div(numerator, A)
        norm_fuzzy[i][j] = norm

# Weighted normalized
weighted_fuzzy = [[(0,0,0) for _ in range(len(criteria))] for _ in range(num)]
for i in range(num):
    for j, c in enumerate(criteria):
        weighted_fuzzy[i][j] = tri_mul_scalar(norm_fuzzy[i][j], c['weight'])

# SAW score (sum of weighted fuzzy numbers) then defuzzify
saw_scores_fuzzy = [ (0.0,0.0,0.0) for _ in range(num)]
for i in range(num):
    s = (0.0,0.0,0.0)
    for j in range(len(criteria)):
        s = tri_add(s, weighted_fuzzy[i][j])
    saw_scores_fuzzy[i] = s
saw_scores = [defuzzify_centroid(s) for s in saw_scores_fuzzy]

# TOPSIS fuzzy
# Determine fuzzy PIS and NIS per criterion
pis = [None]*len(criteria)
nis = [None]*len(criteria)
for j, c in enumerate(criteria):
    # For benefit: PIS = max upper among alternatives, NIS = min lower
    if c['type'] == 'benefit':
        pis[j] = (max([fuzzy_matrix[i][j][0] for i in range(num)]),
                  max([fuzzy_matrix[i][j][1] for i in range(num)]),
                  max([fuzzy_matrix[i][j][2] for i in range(num)]))
        nis[j] = (min([fuzzy_matrix[i][j][0] for i in range(num)]),
                  min([fuzzy_matrix[i][j][1] for i in range(num)]),
                  min([fuzzy_matrix[i][j][2] for i in range(num)]))
    else:
        # cost: PIS should be minimum (best small), NIS maximum
        pis[j] = (min([fuzzy_matrix[i][j][0] for i in range(num)]),
                  min([fuzzy_matrix[i][j][1] for i in range(num)]),
                  min([fuzzy_matrix[i][j][2] for i in range(num)]))
        nis[j] = (max([fuzzy_matrix[i][j][0] for i in range(num)]),
                  max([fuzzy_matrix[i][j][1] for i in range(num)]),
                  max([fuzzy_matrix[i][j][2] for i in range(num)]))

# For TOPSIS we should use weighted normalized fuzzy matrix
# We'll reuse weighted_fuzzy from SAW (approx) — acceptable for demonstration

# Compute distance to PIS and NIS for each alternative
dist_to_pis = [0.0]*num
dist_to_nis = [0.0]*num
for i in range(num):
    dp = 0.0
    dn = 0.0
    for j in range(len(criteria)):
        # multiply pis/nis by weight too
        w = criteria[j]['weight']
        wp = tri_mul_scalar(pis[j], w)
        wn = tri_mul_scalar(nis[j], w)
        dij_p = tri_distance(weighted_fuzzy[i][j], wp)
        dij_n = tri_distance(weighted_fuzzy[i][j], wn)
        dp += dij_p
        dn += dij_n
    dist_to_pis[i] = dp
    dist_to_nis[i] = dn

# Closeness coefficient
cc = []
for i in range(num):
    denom = dist_to_pis[i] + dist_to_nis[i]
    if denom == 0:
        cc.append(0)
    else:
        cc.append(dist_to_nis[i] / denom)

# Prepare result tables
result_df = pd.DataFrame({
    'Alternatif': names,
    'SAW_fuzzy_l': [round(x[0],4) for x in saw_scores_fuzzy],
    'SAW_fuzzy_m': [round(x[1],4) for x in saw_scores_fuzzy],
    'SAW_fuzzy_u': [round(x[2],4) for x in saw_scores_fuzzy],
    'SAW_score_defuzz': [round(x,6) for x in saw_scores],
    'TOPSIS_closeness': [round(x,6) for x in cc]
})

result_df['Rank_SAW'] = result_df['SAW_score_defuzz'].rank(ascending=False, method='min').astype(int)
result_df['Rank_TOPSIS'] = result_df['TOPSIS_closeness'].rank(ascending=False, method='min').astype(int)

st.subheader("Tabel Hasil dan Perangkingan")
st.dataframe(result_df.sort_values('Rank_TOPSIS'))

# Present top alternatives
st.subheader("Ringkasan Rekomendasi")
best_saw = result_df.sort_values('Rank_SAW').iloc[0]
best_topsis = result_df.sort_values('Rank_TOPSIS').iloc[0]
col1, col2 = st.columns(2)
with col1:
    st.metric("Peringkat Terbaik (SAW)", f"{best_saw['Alternatif']} (Rank {best_saw['Rank_SAW']})")
with col2:
    st.metric("Peringkat Terbaik (TOPSIS)", f"{best_topsis['Alternatif']} (Rank {best_topsis['Rank_TOPSIS']})")

# Download results
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(result_df)
st.download_button("Download hasil (CSV)", data=csv, file_name='hasil_saw_topsis.csv', mime='text/csv')


st.graphviz_chart(flow)

st.markdown("---")
st.caption("Aplikasi demonstrasi: metode fuzzy SAW dan TOPSIS yang disederhanakan untuk konteks pembelajaran/penelitian. Untuk penelitian formal, Anda mungkin ingin menyesuaikan pemetaan fuzzy, metode normalisasi, dan cara defuzzifikasi sesuai literatur.")
