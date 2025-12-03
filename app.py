import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SAW & TOPSIS Cloud Storage", layout="wide")
st.title("Analisis Metode SAW & TOPSIS untuk Pemilihan Cloud Storage")

# ============================================================
# INFORMASI & PANDUAN PENGGUNAAN APLIKASI
# ============================================================
st.markdown("---")
st.header("📘 Panduan Penggunaan Aplikasi")

with st.expander("Klik untuk melihat tata cara penggunaan aplikasi"):
    st.markdown("""
    **Langkah-langkah menjalankan analisis SAW & TOPSIS:**

    **1. Tentukan jumlah alternatif**  
    Pilih berapa provider/cloud storage yang ingin dibandingkan (1–5).

    **2. Isi data masing-masing alternatif**  
    Masukkan:
    - Nama alternatif (misalnya Amazon S3, Google Cloud Storage)
    - Biaya penyimpanan per GB  
    - Biaya egress per GB  
    - Latency dalam milidetik  
    - Penilaian kualitas integrasi  
    - Penilaian keamanan & compliance  

    **3. Tekan tombol “Hitung SAW dan TOPSIS”**  
    Sistem akan otomatis:
    - Mengonversi nilai menjadi Crips  
    - Melakukan Normalisasi SAW  
    - Menghitung Skor SAW  
    - Menghitung normalisasi TOPSIS  
    - Menentukan solusi ideal, jarak S+, S−  
    - Menghasilkan nilai preferensi (C+)  
    - Mengurutkan ranking dari terbaik ke terendah  

    **4. Bandingkan hasil SAW dan TOPSIS**  
    Di bagian akhir terdapat tabel perbandingan ranking + analisis otomatis.
    """)

st.markdown("---")

# ============================================================
# TABEL INFORMASI KRITERIA
# ============================================================
st.subheader("📌 Informasi Kriteria yang Digunakan")

df_info_kriteria = pd.DataFrame({
    "Kode Kriteria": ["C1", "C2", "C3", "C4", "C5"],
    "Nama Kriteria": [
        "Biaya penyimpanan",
        "Biaya Egress",
        "Latency / Kecepatan Akses",
        "Skalabilitas & Kemudahan Integrasi",
        "Keamanan & Compliance"
    ],
    "Atribut": ["Cost", "Cost", "Benefit", "Benefit", "Benefit"],
    "Bobot": [0.25, 0.20, 0.20, 0.15, 0.20]
})

st.dataframe(df_info_kriteria, use_container_width=True)
st.markdown("---")


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

    # Perbaikan: mendukung 3–7 angka di belakang koma
    c1 = st.number_input(
        f"C1 Biaya Penyimpanan ($/GB)",
        min_value=0.0,
        step=0.0000001,
        format="%.7f",
        key=f"c1_{i}"
    )
    c2 = st.number_input(
        f"C2 Biaya Egress ($/GB)",
        min_value=0.0,
        step=0.0000001,
        format="%.7f",
        key=f"c2_{i}"
    )
    c3 = st.number_input(
        f"C3 Latency (ms)",
        min_value=0.0,
        step=0.0000001,
        format="%.7f",
        key=f"c3_{i}"
    )

    c4 = st.selectbox(
        "C4 Skalabilitas & Kemudahan Integrasi",
        ["Rendah", "Sedang", "Baik", "Sangat Baik"],
        key=f"c4_{i}"
    )
    c5 = st.selectbox(
        "C5 Keamanan & Compliance",
        ["Kurang", "Cukup", "Baik", "Sangat Baik"],
        key=f"c5_{i}"
    )

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
    st.subheader("Tahap Normalisasi SAW")

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
    st.subheader("Nilai Akhir SAW dan Ranking")

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
    
    # Pastikan semua alternatif punya nama unik
    for i in range(len(df_t)):
        if df_t.loc[i, "Alternatif"] == "" or pd.isna(df_t.loc[i, "Alternatif"]):
            df_t.loc[i, "Alternatif"] = f"A{i+1}"
    
    df_t.set_index("Alternatif", inplace=True)   # <-- konsisten mulai sini!
    
    # --- 1. Normalisasi Matriks R ---
    st.subheader("Matriks Ternormalisasi (R)")
    
    R = df_t[kriteria].astype(float) / np.sqrt((df_t[kriteria].astype(float)**2).sum())
    st.dataframe(R)
    
    # --- 2. Matriks Ternormalisasi Terbobot (Y) ---
    st.subheader("Matriks Ternormalisasi Terbobot (Y)")
    
    bobot_array = np.array(list(bobot.values()))
    Y = R * bobot_array
    st.dataframe(Y)
    
    # --- 3. Solusi Ideal Positif dan Negatif ---
    st.subheader("Solusi Ideal Positif (A+) dan Negatif (A-)")
    
    A_plus = []
    A_minus = []
    
    for c in kriteria:
        if atribut[c] == "benefit":
            A_plus.append(Y[c].max())
            A_minus.append(Y[c].min())
        else:
            A_plus.append(Y[c].min())
            A_minus.append(Y[c].max())
    
    A_plus_df = pd.DataFrame([A_plus], columns=kriteria, index=["A+"])
    A_minus_df = pd.DataFrame([A_minus], columns=kriteria, index=["A-"])
    
    st.dataframe(A_plus_df)
    st.dataframe(A_minus_df)
    
    # --- 4. Jarak S+ dan S- ---
    st.subheader("Jarak ke Solusi Ideal")
    
    S_plus = np.sqrt(((Y - A_plus)**2).sum(axis=1))
    S_minus = np.sqrt(((Y - A_minus)**2).sum(axis=1))
    
    df_distance = pd.DataFrame({
        "S+": S_plus,
        "S-": S_minus
    }, index=Y.index)
    
    st.dataframe(df_distance)
    
    # --- 5. Preferensi C+ dan Ranking ---
    st.subheader("Nilai Preferensi (C+) dan Ranking")
    
    C_plus = S_minus / (S_plus + S_minus)
    
    df_topsis = pd.DataFrame({
        "S+": S_plus,
        "S-": S_minus,
        "C+": C_plus
    })
    
    df_topsis = df_topsis.sort_values("C+", ascending=False)
    df_topsis["Ranking"] = range(1, len(df_topsis) + 1)
    
    st.dataframe(df_topsis)

    # ============================================================
    # -------------- PERBANDINGAN RANKING SAW VS TOPSIS ----------
    # ============================================================
    st.header("📊 Perbandingan Hasil Ranking SAW dan TOPSIS")
    
    # Ambil ranking SAW (index default sudah acak karena sorted)
    df_saw_rank = df_saw[["Alternatif", "Ranking"]].copy()
    df_saw_rank.rename(columns={"Ranking": "Ranking_SAW"}, inplace=True)
    
    # Ambil ranking TOPSIS
    df_topsis_rank = df_topsis.copy()
    df_topsis_rank.reset_index(inplace=True)
    df_topsis_rank.rename(columns={"index": "Alternatif", "Ranking": "Ranking_TOPSIS"}, inplace=True)
    
    # Gabungkan
    df_compare = pd.merge(df_saw_rank, df_topsis_rank[["Alternatif", "Ranking_TOPSIS"]], on="Alternatif")
    
    # Hitung selisih peringkat
    df_compare["Selisih"] = df_compare["Ranking_TOPSIS"] - df_compare["Ranking_SAW"]
    
    # Urutkan berdasarkan ranking TOPSIS atau SAW (bebas, tapi lebih logis urut SAW)
    df_compare = df_compare.sort_values("Ranking_SAW").reset_index(drop=True)
    
    st.dataframe(df_compare)
    
    # Kesimpulan otomatis
    st.subheader("📌 Analisis Singkat Konsistensi Ranking")
    
    jumlah_sama = sum(df_compare["Ranking_SAW"] == df_compare["Ranking_TOPSIS"])
    total_alt = len(df_compare)
    
    if jumlah_sama == total_alt:
        st.success("🎉 **Kedua metode menghasilkan urutan ranking yang sama persis.** Ini menunjukkan konsistensi penuh antara SAW dan TOPSIS.")
    elif jumlah_sama > 0:
        st.warning(f"⚠ **Sebagian alternatif memiliki ranking yang sama**, tetapi tidak semuanya.\n\n"
                   f"- Jumlah peringkat yang sama: **{jumlah_sama}/{total_alt}**\n"
                   f"- Ada perbedaan yang berarti antara metode SAW dan TOPSIS.")
    else:
        st.error("❗ **Tidak ada ranking yang sama antara SAW dan TOPSIS.** "
                 "Ini menunjukkan kedua metode memberikan perspektif berbeda dalam evaluasi alternatif.")
