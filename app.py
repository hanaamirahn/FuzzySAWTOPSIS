import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisis SAW vs TOPSIS (Fuzzy→Defuzz)", layout="wide")

st.title("Analisis Perbandingan Metode SAW dan TOPSIS\nuntuk Pemilihan Layanan Cloud Storage")
st.markdown(
    """
Aplikasi ini menerima alternatif (nama) dan nilai kriteria (angka atau nilai linguistik),
melakukan *fuzzification* sederhana + *defuzzification* (centroid), lalu menghitung
peringkat menggunakan SAW dan TOPSIS.
"""
)

# -------------------------
# Kriteria (fixed sesuai permintaan user)
# -------------------------
criteria = [
    {"code": "C1", "name": "Biaya penyimpanan", "type": "cost", "weight": 0.25},
    {"code": "C2", "name": "Biaya Egress", "type": "cost", "weight": 0.20},
    {"code": "C3", "name": "Latency / Kecepatan Akses", "type": "benefit", "weight": 0.20},
    {"code": "C4", "name": "Skalabilitas & Kemudahan Integrasi", "type": "benefit", "weight": 0.15},
    {"code": "C5", "name": "Keamanan & Compliance", "type": "benefit", "weight": 0.20},
]
crit_codes = [c["code"] for c in criteria]
crit_names = [c["name"] for c in criteria]
weights = np.array([c["weight"] for c in criteria])
types = [c["type"] for c in criteria]

st.sidebar.header("Pengaturan")
n_alt = st.sidebar.number_input("Jumlah alternatif", min_value=2, max_value=20, value=5, step=1)

# Linguistic scale contoh
linguistic_options = [
    "Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"
]

st.sidebar.markdown("**Input style**: Anda bisa memasukkan nilai numerik (mis. 10, 200) atau\npilih nilai linguistik (mis. 'Sedang').\nJika teks linguistik digunakan, akan di-*fuzzify* dan di-defuzzify otomatis.")

# -------------------------
# Mapping linguistik ke TFN (0..1 scale)
# -------------------------
# TFN triangular: (l, m, u) dalam skala 0..1
LINGUISTIC_TO_TFN = {
    "Sangat Rendah": (0.0, 0.0, 0.25),
    "Rendah":        (0.0, 0.25, 0.5),
    "Sedang":        (0.25, 0.5, 0.75),
    "Tinggi":        (0.5, 0.75, 1.0),
    "Sangat Tinggi": (0.75, 1.0, 1.0),
}

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def defuzzify_tfn(tfn):
    # simple centroid defuzzification for triangular fuzzy numbers
    l, m, u = tfn
    return (l + m + u) / 3.0

# -------------------------
# Create editable input table
# -------------------------
st.header("1) Input Alternatif dan Nilai Kriteria")
st.caption("Masukkan nama alternatif lalu untuk setiap kriteria isi nilai (angka atau linguistik).")

# Prepare default dataframe
default_rows = []
for i in range(n_alt):
    name = f"Alt-{i+1}"
    row = {"Alternatif": name}
    # fill with default linguistic 'Sedang'
    for code in crit_codes:
        row[code] = "Sedang"
    default_rows.append(row)

if "df_inputs" not in st.session_state or st.session_state.get("last_n_alt", None) != n_alt:
    st.session_state["df_inputs"] = pd.DataFrame(default_rows)
    st.session_state["last_n_alt"] = n_alt

# Allow editing
df_inputs = st.experimental_data_editor(st.session_state["df_inputs"], num_rows="dynamic")
st.session_state["df_inputs"] = df_inputs

# -------------------------
# Preprocess inputs: convert to numeric by defuzzifying when needed
# -------------------------
def parse_cell(val):
    # if numeric string or number -> return numeric
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    if is_number(s):
        return float(s)
    # try to map linguistic
    if s in LINGUISTIC_TO_TFN:
        return defuzzify_tfn(LINGUISTIC_TO_TFN[s])
    # try case-insensitive match
    for k in LINGUISTIC_TO_TFN:
        if s.lower() == k.lower():
            return defuzzify_tfn(LINGUISTIC_TO_TFN[k])
    # fallback: NaN
    return np.nan

# Build decision matrix (defuzzified numeric between 0..1 ideally)
D = []
names = []
for idx, row in df_inputs.fillna("").iterrows():
    names.append(row.get("Alternatif", f"Alt-{idx+1}"))
    values = []
    for code in crit_codes:
        raw = row.get(code, "")
        val = parse_cell(raw)
        values.append(val)
    D.append(values)
D = np.array(D, dtype=float)  # shape (n_alt, n_crit)

# If any NaN present, show a warning and stop calc until fixed
if np.isnan(D).any():
    st.warning("Terdapat nilai kosong atau nilai yang tidak dikenali. Pastikan semua sel kriteria diisi dengan angka atau nilai linguistik yang valid (contoh: 'Sedang').")
    st.stop()

# Note: since many cost inputs (harga) might be in different scales (e.g., IDR), we assume user either:
# - enters normalized values (0..1) OR
# - enters raw numbers (like IDR) but mixed numeric with others -> we'll normalize per criterion below.

# -------------------------
# Normalize input numeric matrix to comparable scale
# Approach: For each criterion:
#   - If users entered values mostly within [0,1], keep.
#   - Otherwise do min-max normalization to 0..1 (per criterion).
# Then apply cost/benefit transformations for SAW/TOPSIS as needed.
# -------------------------
st.header("2) Proses Perhitungan")
st.markdown("Menormalkan data per kriteria (min-max => 0..1). Jika Anda sudah memasukkan nilai 0..1, hasilnya akan tetap sama.")

# Min-max normalize per column
D_norm = np.zeros_like(D)
for j in range(D.shape[1]):
    col = D[:, j]
    cmin, cmax = np.min(col), np.max(col)
    if np.isclose(cmax, cmin):
        # flat column -> set to 1.0 (no discrimination)
        D_norm[:, j] = 1.0
    else:
        D_norm[:, j] = (col - cmin) / (cmax - cmin)

# For cost criteria: lower is better; for normalization we will invert for benefit sense when required.
# But for SAW we will use standard benefit/cost formulas; for TOPSIS we do vector normalization.
st.write("Contoh nilai (setelah defuzzifikasi awal dan normalisasi min-max):")
df_norm_example = pd.DataFrame(D_norm, columns=crit_codes)
df_norm_example.insert(0, "Alternatif", names)
st.dataframe(df_norm_example.style.format({c: "{:.3f}" for c in crit_codes}), height=240)

# -------------------------
# SAW (Simple Additive Weighting)
# Implementation after defuzzification and min-max normalization:
# For SAW we need normalization that respects cost/benefit:
#  - benefit: r_ij = x_ij / max_j
#  - cost:    r_ij = min_j / x_ij  (but if x_j normalized to 0..1, better to use 1 - normalized)
# To keep consistency, we will operate on original defuzzified D (not D_norm),
# using a robust approach: if criterion is benefit -> r = x / max; if cost -> r = min / x (but guard zeros).
# However since we already min-maxed to D_norm 0..1, use:
#  - benefit: r = D_norm
#  - cost: r = 1 - D_norm
# -------------------------
def compute_saw_scores(D_norm, weights, types):
    # D_norm range 0..1
    R = np.copy(D_norm)
    for j, t in enumerate(types):
        if t == "cost":
            R[:, j] = 1.0 - R[:, j]
    # weighted sum
    scores = R.dot(weights)
    return scores, R

saw_scores, saw_R = compute_saw_scores(D_norm, weights, types)

# -------------------------
# TOPSIS
# Steps (classical) using defuzzified numeric values (we already normalized to 0..1 so we can:
# Option A: apply vector normalization (divide by sqrt(sum squares) on original defuzzified D)
# We'll use vector normalization on original defuzzified values (D) for TOPSIS.
# -------------------------
def compute_topsis(D_raw, weights, types):
    # vector normalization
    denom = np.sqrt(np.sum(D_raw ** 2, axis=0))
    # avoid division by zero
    denom[denom == 0] = 1e-9
    Rv = D_raw / denom
    # Apply criterion type: for cost, smaller is better — but we handle ideal/bad properly below
    # Weighted normalized matrix
    W = weights
    V = Rv * W
    # ideal best and ideal worst
    ideal_best = np.zeros(V.shape[1])
    ideal_worst = np.zeros(V.shape[1])
    for j, t in enumerate(types):
        if t == "benefit":
            ideal_best[j] = np.max(V[:, j])
            ideal_worst[j] = np.min(V[:, j])
        else:  # cost
            ideal_best[j] = np.min(V[:, j])   # for cost, best is minimum (lower cost)
            ideal_worst[j] = np.max(V[:, j])
    # distances
    dist_best = np.sqrt(np.sum((V - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((V - ideal_worst) ** 2, axis=1))
    # closeness coefficient
    cc = dist_worst / (dist_best + dist_worst + 1e-12)
    return cc, V, ideal_best, ideal_worst, dist_best, dist_worst

topsis_scores, V_mat, ideal_best, ideal_worst, db, dw = compute_topsis(D, weights, types)

# -------------------------
# Combine results and show
# -------------------------
results_df = pd.DataFrame({
    "Alternatif": names,
    "SAW_Score": saw_scores,
    "TOPSIS_Score": topsis_scores
})
results_df["SAW_Rank"] = results_df["SAW_Score"].rank(ascending=False, method="min").astype(int)
results_df["TOPSIS_Rank"] = results_df["TOPSIS_Score"].rank(ascending=False, method="min").astype(int)
results_df = results_df.sort_values(by=["TOPSIS_Score"], ascending=False).reset_index(drop=True)

st.header("3) Hasil dan Peringkat")
st.markdown("Tabel di bawah menunjukkan skor dan peringkat dari setiap metode.")
st.dataframe(results_df.style.format({
    "SAW_Score": "{:.4f}", "TOPSIS_Score": "{:.4f}"
}), height=300)

# Show bar charts side-by-side
st.subheader("Perbandingan Skor (bar chart)")
fig, ax = plt.subplots(figsize=(8,4))
x = np.arange(len(names))
width = 0.35
ax.bar(x - width/2, results_df.set_index("Alternatif")["SAW_Score"].loc[names], width=width, label="SAW")
ax.bar(x + width/2, results_df.set_index("Alternatif")["TOPSIS_Score"].loc[names], width=width, label="TOPSIS")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.set_ylabel("Nilai")
ax.legend()
st.pyplot(fig)

# Offer CSV download
csv_buf = io.StringIO()
results_df.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode('utf-8')
st.download_button("Unduh Hasil (CSV)", csv_bytes, file_name="hasil_ranking.csv", mime="text/csv")

# -------------------------
# Additional info / transparency
# -------------------------
st.header("Catatan metodologi & asumsi penting (baca sebelum menggunakan hasil)")
st.markdown(
    """
- **Fuzziness**: aplikasi ini memetakan nilai linguistik ke *triangular fuzzy numbers* (TFN) pada skala 0..1, lalu **defuzzifikasi** menggunakan centroid (rata-rata titik TFN) menjadi nilai numerik tunggal.  
  *Implikasi:* ini menyederhanakan perhitungan (lebih stabil/cepat) tapi **mengurangi** keunggulan penuh metode fuzzy (ketidakpastian dihapus saat defuzzifikasi).
- **Normalisasi**: untuk SAW saya gunakan `benefit = D_norm` dan `cost = 1 - D_norm` (setelah min-max). Untuk TOPSIS saya gunakan vektor normalisasi (pembagi = sqrt(sum squares)) pada nilai defuzzified asli.  
- **Bobot**: bobot kriteria tetap sesuai yang Anda berikan. Hati-hati — bobot yang subjektif dapat mengubah peringkat signifikan.
- **Validasi**: jalankan *sensitivity analysis* (mis. ubah bobot ±10% atau coba seluruh input numerik) untuk mengamati stabilitas peringkat.
"""
)

# -------------------------
# Practical suggestions / next steps (intellectual sparring)
# -------------------------
st.header("Saran peningkatan & kritik terhadap asumsi (pendekatan 'sparring intelektual')")
st.markdown(
    """
1. **Asumsi defuzzifikasi centroid**: saya pakai cara sederhana untuk practical use — namun hal ini *membuang* informasi fuzzy.  
   *Alternatif lebih ketat:* terapkan operasi fuzzy lengkap (penjumlahan, perkalian TFN) untuk SAW/TOPSIS sehingga hasilnya tetap fuzzy, baru lakukan ranking berdasarkan ranking fuzzy (mis. menggunakan peluang/indeks centroid).
2. **Skala input campuran**: aplikasi sekarang otomatis min-max per kriteria. Jika Anda memasukkan angka nyata (mis. IDR/TB), pertimbangkan standarisasi satuan (semua biaya di IDR/GB-blabla) agar transformasi masuk akal.
3. **Ketergantungan bobot**: bobot tetap bisa mem-bias hasil. Lakukan uji sensitivitas bobot (panel kontrol untuk mengubah bobot secara interaktif).
4. **Validasi eksternal**: cocokkan peringkat dengan studi kasus/ground-truth (jika ada) atau bandingkan hasil dua metode untuk identifikasi konsensus.
5. **Ketidakpastian data**: jika data asli memang sangat tidak pasti (perkiraan kasar), pertimbangkan metode fuzzy penuh atau probabilistic MCDM.
"""
)

st.success("Selesai — Anda bisa mengunduh kode (file ini) ke GitHub dan jalankan `streamlit run app.py`.")
