import streamlit as st
import pandas as pd
import numpy as np

from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import plotly.express as px
import plotly.graph_objects as go

# ====================================================
# Konfigurasi Halaman
# ====================================================
st.set_page_config(
    page_title="PMA Surabaya - K-Means Clustering",
    page_icon="💹",
    layout="wide"
)

# Sedikit styling CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #0f172a, #1d4ed8);
        padding: 20px 25px;
        border-radius: 18px;
        color: white;
        margin-bottom: 15px;
    }
    .main-header h1 {
        font-size: 28px;
        margin-bottom: 5px;
    }
    .main-header p {
        font-size: 14px;
        margin-top: 0;
        opacity: 0.9;
    }
    .small-note {
        font-size: 12px;
        color: #6b7280;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================================
# Helper Functions
# ====================================================

def winsorize_like_r(df, lower_q=0.05, upper_q=0.95):
    """
    Meniru logika winsorizing di skrip R:
    - hitung Q1, Q3, IQR
    - nilai di bawah (Q1 - 1.5*IQR) diganti quantile lower_q
    - nilai di atas (Q3 + 1.5*IQR) diganti quantile upper_q
    """
    df = df.copy()
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        x = df[col].to_numpy(dtype=float)
        q1, q3 = np.quantile(x, [0.25, 0.75])
        caps = np.quantile(x, [lower_q, upper_q])
        H = 1.5 * (q3 - q1)

        lower_bound = q1 - H
        upper_bound = q3 + H

        x = np.where(x < lower_bound, caps[0], x)
        x = np.where(x > upper_bound, caps[1], x)

        df[col] = x

    return df


def compute_elbow_silhouette(X, k_min=2, k_max=10, random_state=42):
    """
    Menghitung SSE (untuk Elbow) dan Silhouette Score untuk rentang K.
    """
    sse = []
    Ks = list(range(1, k_max + 1))
    sil_scores = []

    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sse.append(km.inertia_)

        if k >= k_min:
            sil = silhouette_score(X, labels)
            sil_scores.append((k, sil))

    return Ks, sse, sil_scores


def add_cluster_to_df(df_raw, labels, pca_coords, negara_col="Negara"):
    """
    Menggabungkan label cluster + koordinat PCA ke data asli.
    """
    df_result = df_raw.copy()
    df_result["Cluster"] = labels

    pca_df = pd.DataFrame(
        pca_coords,
        columns=["PC1", "PC2"]
    )

    df_result = pd.concat([df_result, pca_df], axis=1)

    if negara_col in df_result.columns:
        cols = ["Cluster", "PC1", "PC2"] + [c for c in df_result.columns if c not in ["Cluster", "PC1", "PC2"]]
        df_result = df_result[cols]

    return df_result


def download_csv_button(df, filename="hasil_cluster.csv", label="📥 Download Hasil Clustering (.csv)"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


# ====================================================
# HEADER
# ====================================================
st.markdown("""
<div class="main-header">
  <h1>💹 Clustering Penanaman Modal Asing (PMA) – Kota Surabaya</h1>
  <p>Pipeline Machine Learning: Winsorizing → Z-Score → PCA → Penentuan K → K-Means → Davies-Bouldin Index</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p class='small-note'>Gunakan sidebar untuk mengatur parameter dan upload dataset. "
    "Grafik bersifat interaktif: bisa di-zoom, di-pan, dan di-hover.</p>",
    unsafe_allow_html=True
)

# ====================================================
# SIDEBAR
# ====================================================
st.sidebar.title("⚙️ Pengaturan Utama")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset PMA (Excel)",
    type=["xlsx"],
    help="Gunakan file yang berisi kolom: Negara, Nilai Investasi, Jumlah Proyek, TKI, TKA"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Pengaturan Preprocessing")

winsor_lower = st.sidebar.slider("Quantile Winsor Bawah", 0.0, 0.2, 0.05, 0.01)
winsor_upper = st.sidebar.slider("Quantile Winsor Atas", 0.8, 1.0, 0.95, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Pengaturan K-Means")

k_max_elbow = st.sidebar.slider("Maksimum K untuk Elbow & Silhouette", 4, 12, 8, 1)
k_selected = st.sidebar.slider("Jumlah Cluster K-Means (K)", 2, 8, 2, 1)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.info("Tips: Untuk meniru hasil jurnal & skrip R, gunakan K = 2.")

# ====================================================
# CEK FILE
# ====================================================
if uploaded_file is None:
    st.warning("Silakan upload file Excel terlebih dahulu melalui sidebar.")
    st.stop()

# ====================================================
# LOAD DATA
# ====================================================
with st.spinner("📥 Membaca dataset..."):
    df_raw = pd.read_excel(uploaded_file)

st.success(f"Dataset berhasil dimuat. Baris: {df_raw.shape[0]}, Kolom: {df_raw.shape[1]}")

st.subheader("📦 Data Asli")
st.dataframe(df_raw.head(), use_container_width=True)

# Deteksi kolom negara
negara_col = None
for candidate in ["Negara", "Country", "NEGARA"]:
    if candidate in df_raw.columns:
        negara_col = candidate
        break

numeric_cols = [c for c in df_raw.columns if np.issubdtype(df_raw[c].dtype, np.number)]
default_features = [c for c in numeric_cols]

st.markdown("### 🔧 Pemilihan Fitur Numerik untuk Clustering")
selected_features = st.multiselect(
    "Pilih fitur numerik (minimal 2):",
    options=numeric_cols,
    default=default_features,
    help="Umumnya: Nilai Investasi, Jumlah Proyek, TKI, TKA"
)

if len(selected_features) < 2:
    st.error("Minimal pilih 2 fitur numerik untuk PCA & K-Means.")
    st.stop()

data_selected = df_raw[selected_features].copy()

# ====================================================
# TABS UTAMA
# ====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Preprocessing",
    "🧬 PCA & Cari K Terbaik",
    "🧩 Clustering & Insight",
    "📥 Download & Ringkasan"
])

# ====================================================
# TAB 1 – EDA & PREPROCESSING
# ====================================================
with tab1:
    st.header("1️⃣ EDA & Preprocessing")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Ringkasan Statistik (Sebelum Winsorizing)")
        st.dataframe(data_selected.describe().T, use_container_width=True)

    with col2:
        st.subheader("📉 Boxplot Sebelum Winsorizing")
        df_melt_before = data_selected.melt(var_name="Fitur", value_name="Nilai")
        fig_box_before = px.box(
            df_melt_before,
            x="Fitur",
            y="Nilai",
            title="Distribusi Fitur Sebelum Winsorizing"
        )
        fig_box_before.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_box_before, use_container_width=True)

    # Winsorizing
    with st.spinner("✂️ Melakukan Winsorizing..."):
        data_winsor = winsorize_like_r(data_selected, lower_q=winsor_lower, upper_q=winsor_upper)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📌 Ringkasan Statistik (Setelah Winsorizing)")
        st.dataframe(data_winsor.describe().T, use_container_width=True)

    with col4:
        st.subheader("📉 Boxplot Setelah Winsorizing")
        df_melt_after = data_winsor.melt(var_name="Fitur", value_name="Nilai")
        fig_box_after = px.box(
            df_melt_after,
            x="Fitur",
            y="Nilai",
            title="Distribusi Fitur Setelah Winsorizing"
        )
        fig_box_after.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_box_after, use_container_width=True)

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_winsor)
    data_scaled = pd.DataFrame(X_scaled, columns=selected_features)

    st.subheader("📉 Boxplot Setelah Normalisasi (Z-Score)")
    df_melt_scaled = data_scaled.melt(var_name="Fitur", value_name="Nilai Z")
    fig_box_scaled = px.box(
        df_melt_scaled,
        x="Fitur",
        y="Nilai Z",
        title="Distribusi Fitur Setelah Normalisasi"
    )
    fig_box_scaled.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_box_scaled, use_container_width=True)

    st.subheader("🧱 Heatmap Korelasi (Setelah Normalisasi)")
    corr = data_scaled.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Korelasi Antar Fitur"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info(
        "Terlihat bahwa beberapa fitur saling berkorelasi cukup kuat. "
        "Ini mendukung penggunaan **PCA** untuk mereduksi dimensi sebelum K-Means."
    )

# ====================================================
# TAB 2 – PCA & K
# ====================================================
with tab2:
    st.header("2️⃣ PCA & Penentuan Jumlah Cluster (K)")

    with st.spinner("🧬 Menghitung PCA..."):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_

    st.subheader("📈 Rasio Varian yang Dijelaskan PCA")
    st.write(
        f"PC1: **{explained_var[0]:.3f}**, "
        f"PC2: **{explained_var[1]:.3f}**, "
        f"Total: **{explained_var.sum():.3f}**"
    )

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    if negara_col is not None:
        pca_df[negara_col] = df_raw[negara_col]

    st.subheader("🌐 Peta PCA (Tanpa Cluster)")
    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        hover_data=[negara_col] if negara_col is not None else None,
        title="Proyeksi Data ke Ruang PCA (2 Komponen Utama)",
    )
    fig_pca.update_layout(transition_duration=500)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("📉 Elbow Method (SSE vs K)")
    with st.spinner("📊 Menghitung SSE & Silhouette..."):
        Ks, sse, sil_scores = compute_elbow_silhouette(
            X_pca,
            k_min=2,
            k_max=k_max_elbow,
            random_state=random_state
        )

    # Elbow plot (Plotly)
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=Ks,
        y=sse,
        mode="lines+markers",
        name="SSE"
    ))
    fig_elbow.update_layout(
        xaxis_title="Jumlah Cluster (K)",
        yaxis_title="SSE (Within-Cluster Sum of Squares)",
        title="Elbow Method",
        hovermode="x unified",
        transition_duration=500
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.subheader("📈 Silhouette Score vs K")
    if sil_scores:
        ks_sil, vals_sil = zip(*sil_scores)
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=ks_sil,
            y=vals_sil,
            mode="lines+markers",
            name="Silhouette"
        ))
        fig_sil.update_layout(
            xaxis_title="Jumlah Cluster (K)",
            yaxis_title="Silhouette Score",
            title="Silhouette Score untuk Berbagai K",
            hovermode="x unified",
            transition_duration=500
        )
        st.plotly_chart(fig_sil, use_container_width=True)

        best_k, best_sil = max(sil_scores, key=lambda x: x[1])
        st.info(
            f"Silhouette tertinggi diperoleh pada K = **{best_k}** "
            f"dengan nilai **{best_sil:.3f}**. "
            f"Kamu bisa menyesuaikan slider K di sidebar mendekati nilai ini."
        )
    else:
        st.warning("Silhouette hanya dihitung mulai K=2. Coba naikkan parameter K maksimum di sidebar.")

# ====================================================
# TAB 3 – CLUSTERING & INSIGHT
# ====================================================
with tab3:
    st.header("3️⃣ K-Means Clustering & Insight")

    with st.spinner("🚀 Menjalankan K-Means..."):
        km = KMeans(n_clusters=k_selected, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        dbi = davies_bouldin_score(X_pca, labels)

        df_clustered = add_cluster_to_df(df_raw, labels, X_pca, negara_col=negara_col)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Jumlah Cluster (K)", k_selected)
    col_m2.metric("Silhouette Score", f"{sil:.3f}")
    col_m3.metric("Davies-Bouldin Index", f"{dbi:.3f}")

    st.subheader("🌐 Peta PCA dengan Label Cluster")
    plot_df = df_clustered.copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)

    hover_cols = ["Cluster"]
    if negara_col is not None:
        hover_cols.append(negara_col)

    fig_cluster = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=hover_cols,
        title=f"Pemetaan PCA dengan K-Means (K = {k_selected})",
        symbol="Cluster"
    )
    fig_cluster.update_traces(marker=dict(size=10, line=dict(width=1)))
    fig_cluster.update_layout(transition_duration=500)
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("📃 Tabel Hasil Clustering per Negara")
    st.dataframe(df_clustered, use_container_width=True)

    st.subheader("📊 Profil Rata-Rata Tiap Cluster (di Ruang Asli Fitur)")
    df_profile = df_clustered.groupby("Cluster")[selected_features].mean().reset_index()
    df_profile["Cluster"] = df_profile["Cluster"].astype(str)

    fig_prof = px.bar(
        df_profile.melt(id_vars="Cluster", var_name="Fitur", value_name="Rata-Rata"),
        x="Fitur",
        y="Rata-Rata",
        color="Cluster",
        barmode="group",
        title="Perbandingan Rata-Rata Fitur per Cluster",
    )
    fig_prof.update_layout(xaxis_tickangle=-45, transition_duration=500)
    st.plotly_chart(fig_prof, use_container_width=True)

    st.info(
        "Interpretasi umum:\n"
        "- Cluster dengan rata-rata **investasi, proyek, TKI, dan TKA tinggi** → negara penyumbang investasi utama.\n"
        "- Cluster dengan nilai lebih rendah → negara yang kontribusinya relatif kecil, namun bisa menjadi target peningkatan investasi."
    )

# ====================================================
# TAB 4 – DOWNLOAD & RINGKASAN
# ====================================================
with tab4:
    st.header("4️⃣ Download & Ringkasan")

    st.subheader("📥 Unduh Hasil Clustering")
    download_csv_button(df_clustered)

    st.subheader("🔍 Ringkasan Singkat untuk Laporan")
    n_cluster = df_clustered["Cluster"].nunique()
    st.markdown(f"""
    - Jumlah negara investor: **{df_clustered.shape[0]}**
    - Jumlah fitur yang digunakan: **{len(selected_features)}**
    - Jumlah cluster (K-Means): **{n_cluster}**
    - Silhouette Score: **{sil:.3f}**
    - Davies-Bouldin Index: **{dbi:.3f}**
    """)

    st.markdown("""
    Contoh kalimat narasi yang bisa dipakai di laporan:

    > Data Penanaman Modal Asing (PMA) Kota Surabaya dikelompokkan menggunakan algoritma K-Means 
    > setelah melalui tahap winsorizing dan normalisasi Z-Score. Reduksi dimensi dilakukan dengan 
    > Principal Component Analysis (PCA) menjadi dua komponen utama yang mampu menjelaskan sebagian besar 
    > keragaman data. Berdasarkan analisis Elbow dan Silhouette, pemilihan jumlah cluster K dilakukan pada 
    > nilai yang menghasilkan keseimbangan antara kompaknya cluster dan pemisahan antarcluster. 
    > Hasil clustering menunjukkan adanya perbedaan yang jelas antara negara dengan kontribusi investasi tinggi 
    > dan negara dengan kontribusi relatif rendah, yang tercermin dari rata-rata nilai investasi, jumlah proyek, 
    > serta serapan tenaga kerja (TKI dan TKA) pada masing-masing cluster.
    """)

    st.caption("Ringkasan ini bisa kamu sesuaikan lagi dengan gaya bahasa dan format tugas dosen.")


# Selesai
