import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# konfig
st.set_page_config(
    page_title="Bike Sharing Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Dashboard/all_data.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])
    
    # Mapping musim
    df["season_name"] = df["season"].map({
        1: "Spring", 
        2: "Summer", 
        3: "Fall", 
        4: "Winter"
    })
    
    # Kategorisasi suhu
    df['temp_category'] = pd.cut(
        df['temp'], 
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Dingin', 'Sejuk', 'Hangat', 'Panas']
    )
    
    return df

df = load_data()


# Sidebar untuk filter data
st.sidebar.title("🎛️ Filter Data")
st.sidebar.markdown("---")

# Filter Musim
season_options = ["Semua Musim"] + sorted(df["season_name"].unique().tolist())
selected_season = st.sidebar.selectbox("📅 Pilih Musim", season_options)

# Filter Tanggal
min_date = df["dteday"].min().date()
max_date = df["dteday"].max().date()

date_range = st.sidebar.date_input(
    "📆 Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

# Filter Kategori Suhu
temp_options = ["Semua Suhu"] + df["temp_category"].cat.categories.tolist()
selected_temp = st.sidebar.selectbox("🌡️ Kategori Suhu", temp_options)



# Apply filters
filtered_df = df.copy()

# Filter musim
if selected_season != "Semua Musim":
    filtered_df = filtered_df[filtered_df["season_name"] == selected_season]

# Filter tanggal
filtered_df = filtered_df[
    (filtered_df["dteday"].dt.date >= start_date) & 
    (filtered_df["dteday"].dt.date <= end_date)
]

# Filter suhu
if selected_temp != "Semua Suhu":
    filtered_df = filtered_df[filtered_df["temp_category"] == selected_temp]

# Header
st.title("🚴 Bike Sharing Analytics Dashboard")
st.markdown(f"""
**Periode Data:** {df['dteday'].min().strftime('%d %b %Y')} - {df['dteday'].max().strftime('%d %b %Y')}  
**Data Ditampilkan:** {len(filtered_df):,} hari dari total {len(df):,} hari
""")
st.markdown("---")

# Columns ringkasan
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_rentals = filtered_df['cnt'].sum()
    st.metric(
        label="📊 Total Penyewaan",
        value=f"{total_rentals:,}",
        delta=f"{(total_rentals/df['cnt'].sum()*100):.1f}% dari total" if len(filtered_df) < len(df) else None
    )

with col2:
    avg_rentals = filtered_df['cnt'].mean()
    st.metric(
        label="📈 Rata-rata Harian",
        value=f"{int(avg_rentals):,}",
        delta=f"{((avg_rentals - df['cnt'].mean())/df['cnt'].mean()*100):+.1f}%" if len(filtered_df) < len(df) else None
    )

with col3:
    total_casual = filtered_df['casual'].sum()
    st.metric(
        label="👥 Casual Users",
        value=f"{total_casual:,}",
        delta=f"{(total_casual/filtered_df['cnt'].sum()*100):.1f}%"
    )

with col4:
    total_registered = filtered_df['registered'].sum()
    st.metric(
        label="🎫 Registered Users",
        value=f"{total_registered:,}",
        delta=f"{(total_registered/filtered_df['cnt'].sum()*100):.1f}%"
    )

st.markdown("---")

# Pertanyaan 1: Pola Penyewaan Berdasarkan Musim
st.header("❄️ Pertanyaan 1: Pola Penyewaan Berdasarkan Musim")
st.markdown("**Bagaimana pola penyewaan sepeda berbeda di setiap musim?**")

# Hitung statistik per musim (dari filtered data)
season_stats = filtered_df.groupby('season_name')['cnt'].agg([
    ('rata_rata', 'mean'),
    ('total', 'sum'),
    ('minimum', 'min'),
    ('maksimum', 'max')
]).round(0)

season_order = ['Spring', 'Summer', 'Fall', 'Winter']
season_stats = season_stats.reindex([s for s in season_order if s in season_stats.index])

# Identifikasi musim terbaik
if len(season_stats) > 0:
    best_season = season_stats['rata_rata'].idxmax()
    worst_season = season_stats['rata_rata'].idxmin()
    best_avg = season_stats.loc[best_season, 'rata_rata']
    worst_avg = season_stats.loc[worst_season, 'rata_rata']
    perbedaan_persen = ((best_avg - worst_avg) / worst_avg * 100)

    st.info(f"""
    🏆 **Musim Terbaik:** {best_season} ({best_avg:,.0f} penyewaan/hari)  
    📉 **Musim Terendah:** {worst_season} ({worst_avg:,.0f} penyewaan/hari)  
    📊 **Selisih:** {perbedaan_persen:.1f}%
    """)

# Visualisasi
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Rata-rata Penyewaan per Musim")
    
    if len(season_stats) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        season_avg = season_stats['rata_rata'].sort_values(ascending=False)
        
        # Warna berbeda
        colors = []
        for musim in season_avg.index:
            if musim == best_season:
                colors.append('#2ecc71')
            elif musim == worst_season:
                colors.append('#e74c3c')
            else:
                colors.append('#3498db')
        
        bars = ax.bar(season_avg.index, season_avg.values, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        # Tambahkan nilai
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Rata-rata Penyewaan', fontsize=11)
        ax.set_xlabel('Musim', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Tidak ada data untuk filter yang dipilih")

with col2:
    st.subheader("👥 Casual vs Registered per Musim")
    
    if len(filtered_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        season_detail = filtered_df.groupby('season_name')[['casual', 'registered']].sum()
        season_detail = season_detail.reindex([s for s in season_order if s in season_detail.index])
        
        x = range(len(season_detail))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], season_detail['casual'], 
               width, label='Casual', color='#f39c12', alpha=0.8)
        ax.bar([i + width/2 for i in x], season_detail['registered'],
               width, label='Registered', color='#3498db', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(season_detail.index)
        ax.set_ylabel('Total Penyewaan', fontsize=11)
        ax.set_xlabel('Musim', fontsize=11)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Tidak ada data untuk filter yang dipilih")

# Visualisasi 
st.subheader("📦 Distribusi Penyewaan per Musim")

if len(filtered_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    
    available_seasons = [s for s in season_order if s in filtered_df['season_name'].values]
    
    sns.boxplot(data=filtered_df, x='season_name', y='cnt', 
                order=available_seasons, palette='Set2', ax=ax)
    
    ax.set_ylabel('Jumlah Penyewaan', fontsize=11)
    ax.set_xlabel('Musim', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.warning("Tidak ada data untuk filter yang dipilih")

# Tabel Statistik
st.subheader("📋 Tabel Statistik per Musim")
st.dataframe(season_stats, use_container_width=True)

st.markdown("---")

# PERTANYAAN 2: ANALISIS SUHU
st.header("🌡️ Pertanyaan 2: Pengaruh Suhu terhadap Penyewaan")
st.markdown("**Bagaimana pengaruh suhu udara terhadap pola penyewaan casual dan registered users?**")

# Hitung korelasi Pearson (gunakan filtered data)
if len(filtered_df) > 1:
    corr_casual, p_casual = pearsonr(filtered_df['temp'], filtered_df['casual'])
    corr_registered, p_registered = pearsonr(filtered_df['temp'], filtered_df['registered'])
    corr_total, p_total = pearsonr(filtered_df['temp'], filtered_df['cnt'])
    
    # Tampilkan metrik korelasi
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Korelasi Casual",
            value=f"{corr_casual:.3f}",
            delta="Signifikan ✓" if p_casual < 0.05 else "Tidak Signifikan"
        )
    
    with col2:
        st.metric(
            label="📊 Korelasi Registered",
            value=f"{corr_registered:.3f}",
            delta="Signifikan ✓" if p_registered < 0.05 else "Tidak Signifikan"
        )
    
    with col3:
        st.metric(
            label="📊 Korelasi Total",
            value=f"{corr_total:.3f}",
            delta="Signifikan ✓" if p_total < 0.05 else "Tidak Signifikan"
        )
    
    st.info(f"""
    💡 **Interpretasi:** Casual users (r={corr_casual:.3f}) lebih sensitif terhadap suhu dibanding 
    Registered users (r={corr_registered:.3f}). Semakin tinggi suhu, semakin banyak penyewaan, 
    terutama untuk casual users yang cenderung bersepeda untuk rekreasi.
    """)
    
    # Visualisasi
    st.subheader("📈 Hubungan Suhu dengan Jumlah Penyewaan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(filtered_df['temp'], filtered_df['casual'], 
                   alpha=0.5, color='#f39c12', s=30, label='Casual')
        ax.scatter(filtered_df['temp'], filtered_df['registered'], 
                   alpha=0.5, color='#3498db', s=30, label='Registered')
        
        ax.set_xlabel('Suhu (Normalized)', fontsize=11)
        ax.set_ylabel('Jumlah Penyewaan', fontsize=11)
        ax.set_title(f'Casual (r={corr_casual:.2f}) vs Registered (r={corr_registered:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Bar chart per kategori suhu
        fig, ax = plt.subplots(figsize=(7, 5))
        
        temp_analysis = filtered_df.groupby('temp_category')[['casual', 'registered']].mean()
        
        x = range(len(temp_analysis))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], temp_analysis['casual'], 
               width, label='Casual', color='#f39c12', alpha=0.8)
        ax.bar([i + width/2 for i in x], temp_analysis['registered'],
               width, label='Registered', color='#3498db', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(temp_analysis.index)
        ax.set_ylabel('Rata-rata Penyewaan', fontsize=11)
        ax.set_xlabel('Kategori Suhu', fontsize=11)
        ax.set_title('Rata-rata Penyewaan per Kategori Suhu')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Visualisasi
    st.subheader("📦 Distribusi Penyewaan per Kategori Suhu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=filtered_df, x='temp_category', y='casual', 
                    palette='Oranges', ax=ax)
        ax.set_ylabel('Casual Users', fontsize=11)
        ax.set_xlabel('Kategori Suhu', fontsize=11)
        ax.set_title('Distribusi Casual Users')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=filtered_df, x='temp_category', y='registered',
                    palette='Blues', ax=ax)
        ax.set_ylabel('Registered Users', fontsize=11)
        ax.set_xlabel('Kategori Suhu', fontsize=11)
        ax.set_title('Distribusi Registered Users')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

else:
    st.warning("Data tidak cukup untuk menghitung korelasi (minimal 2 data point)")

st.markdown("---")

# KESIMPULAN
st.header("📝 Kesimpulan")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🍂 Kesimpulan Pertanyaan 1")
    if len(season_stats) > 0:
        st.success(f"""
        **Temuan Utama:**
        - Musim **{best_season}** mencatat penyewaan tertinggi ({best_avg:,.0f}/hari)
        - Musim **{worst_season}** memiliki penyewaan terendah ({worst_avg:,.0f}/hari)
        - Terdapat gap **{perbedaan_persen:.1f}%** antar musim
        
        **Rekomendasi:**
        - Tingkatkan inventory di musim {best_season}
        - Implementasi promo di musim {worst_season}
        - Fokus retention untuk registered users
        """)
    else:
        st.info("Pilih filter untuk melihat kesimpulan")

with col2:
    st.subheader("🌡️ Kesimpulan Pertanyaan 2")
    if len(filtered_df) > 1:
        kekuatan = "kuat" if abs(corr_total) >= 0.7 else "sedang" if abs(corr_total) >= 0.4 else "lemah"
        st.success(f"""
        **Temuan Utama:**
        - Korelasi suhu-penyewaan: **{kekuatan}** (r={corr_total:.3f})
        - Casual users **lebih sensitif** terhadap suhu (r={corr_casual:.3f})
        - Registered users **lebih stabil** (r={corr_registered:.3f})
        
        **Rekomendasi:**
        - Weather-based marketing untuk casual users
        - Dynamic pricing berdasarkan forecast cuaca
        - Maintain service quality untuk registered users
        """)
    else:
        st.info("Pilih filter untuk melihat kesimpulan")

st.markdown("---")
