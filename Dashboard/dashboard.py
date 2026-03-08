import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Dashboard/all_data.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["season_label"] = df["season"].map({
        1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"
    })
    df["temp_c"] = (df["temp"] * 41).round(1)
    return df

df = load_data()

# Membuat Sidebar Filter
st.sidebar.title("Filter Data")

season_options = ["Semua"] + list(df["season_label"].unique())
season_sel  = st.sidebar.selectbox("Musim", season_options)

min_date = df["dteday"].min().date()
max_date = df["dteday"].max().date()

date_start = st.sidebar.date_input("Tanggal Mulai", value=min_date, min_value=min_date, max_value=max_date)
date_end   = st.sidebar.date_input("Tanggal Akhir", value=max_date, min_value=min_date, max_value=max_date)

# Apply filter sidebar
dff = df.copy()
if season_sel != "Semua":
    dff = dff[dff["season_label"] == season_sel]
dff = dff[(dff["dteday"].dt.date >= date_start) & (dff["dteday"].dt.date <= date_end)]

# Header
st.title("Bike Sharing Dashboard")
st.caption(f"Periode: {df['dteday'].min().strftime('%d %b %Y')} – {df['dteday'].max().strftime('%d %b %Y')} | Total data: {len(df):,} hari")
st.divider()

# Ringkasan metrik utama
sewa, rata, casual, registered = st.columns(4)
sewa.metric("Total Penyewaan",  f"{dff['cnt'].sum():,}")
rata.metric("Rata-rata Harian", f"{int(dff['cnt'].mean()):,}")
casual.metric("Casual Users",     f"{dff['casual'].sum():,}")
registered.metric("Registered Users", f"{dff['registered'].sum():,}")

st.divider()

# Diagram pertanyaan 1
st.subheader("Pertanyaan 1: Pengaruh Musim terhadap Penyewaan Sepeda")

season_agg = (
    df.groupby("season_label")[["cnt", "casual", "registered"]]
    .agg(total_cnt=("cnt","sum"), mean_cnt=("cnt","mean"),
         total_casual=("casual","sum"), total_registered=("registered","sum"))
    .reindex(["Spring","Summer","Fall","Winter"])
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3"]
    bars = ax.bar(season_agg["season_label"], season_agg["total_cnt"],
                  color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, season_agg["total_cnt"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Total Penyewaan per Musim")
    ax.set_xlabel("Musim")
    ax.set_ylabel("Total Penyewaan")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(season_agg))
    w = 0.35
    ax.bar(x - w/2, season_agg["total_casual"],     width=w, label="Casual",     color="#FF9800")
    ax.bar(x + w/2, season_agg["total_registered"], width=w, label="Registered", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels(season_agg["season_label"])
    ax.set_title("Casual vs Registered per Musim")
    ax.set_xlabel("Musim")
    ax.set_ylabel("Jumlah Pengguna")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.legend()
    st.pyplot(fig)
    plt.close()

# Boxplot
st.markdown("**Distribusi Penyewaan Harian per Musim**")
fig, ax = plt.subplots(figsize=(10, 3.5))
order = ["Spring", "Summer", "Fall", "Winter"]
sns.boxplot(data=df, x="season_label", y="cnt", order=order,
            palette=["#4CAF50","#FF9800","#F44336","#2196F3"], ax=ax)
ax.set_title("Distribusi Penyewaan Harian per Musim")
ax.set_xlabel("Musim")
ax.set_ylabel("Jumlah Penyewaan")
st.pyplot(fig)
plt.close()

# Tabel ringkasan
st.markdown("**Tabel Statistik per Musim**")
tabel = df.groupby("season_label")["cnt"].agg(
    Total="sum", Rata_rata="mean", Minimum="min", Maksimum="max", Std="std"
).reindex(order).round(1)
st.dataframe(tabel, use_container_width=True)

st.divider()

# Diagram pertanyaan 2 
st.subheader("Pertanyaan 2: Korelasi Suhu Udara dengan Jumlah Pengguna")

r_cas, p_cas = stats.pearsonr(df["temp_c"], df["casual"])
r_reg, p_reg = stats.pearsonr(df["temp_c"], df["registered"])
r_tot, p_tot = stats.pearsonr(df["temp_c"], df["cnt"])

# Koefisien korelasi
korcas, korreg, kortot = st.columns(3)
korcas.metric("Korelasi Casual",     f"{r_cas:.3f}", "Signifikan ✓" if p_cas < 0.05 else "Tidak Signifikan")
korreg.metric("Korelasi Registered", f"{r_reg:.3f}", "Signifikan ✓" if p_reg < 0.05 else "Tidak Signifikan")
kortot.metric("Korelasi Total",      f"{r_tot:.3f}", "Signifikan ✓" if p_tot < 0.05 else "Tidak Signifikan")

# Scatter plots
col3, col4, col5 = st.columns(3)

def scatter_plot(ax, x, y, ylabel, color, r):
    ax.scatter(x, y, alpha=0.4, s=12, color=color)
    m, b = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 200)
    ax.plot(xl, m*xl + b, color="red", lw=1.5, linestyle="--", label=f"r = {r:.3f}")
    ax.set_xlabel("Suhu (°C)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)

with col3:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    scatter_plot(ax, df["temp_c"], df["casual"], "Casual", "#FF9800", r_cas)
    ax.set_title("Suhu vs Casual Users")
    st.pyplot(fig); plt.close()

with col4:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    scatter_plot(ax, df["temp_c"], df["registered"], "Registered", "#2196F3", r_reg)
    ax.set_title("Suhu vs Registered Users")
    st.pyplot(fig); plt.close()

with col5:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    scatter_plot(ax, df["temp_c"], df["cnt"], "Total", "#9C27B0", r_tot)
    ax.set_title("Suhu vs Total Penyewaan")
    st.pyplot(fig); plt.close()

# Heatmap korelasi
st.markdown("**Heatmap Korelasi Variabel Numerik**")
fig, ax = plt.subplots(figsize=(8, 4))
corr_cols = df[["temp_c","hum","windspeed","casual","registered","cnt"]].rename(columns={
    "temp_c":"Suhu(°C)", "hum":"Kelembapan",
    "windspeed":"Angin", "casual":"Casual",
    "registered":"Registered", "cnt":"Total"
})
sns.heatmap(corr_cols.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax, linewidths=0.5)
ax.set_title("Heatmap Korelasi")
st.pyplot(fig); plt.close()

# Tren bulanan
st.markdown("**Tren Suhu & Penyewaan per Bulan**")
month_agg = df.groupby("mnth").agg(
    avg_temp=("temp_c","mean"), avg_cnt=("cnt","mean"),
    avg_casual=("casual","mean"), avg_registered=("registered","mean")
).reset_index()
MONTHS = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]

fig, ax1 = plt.subplots(figsize=(11, 4))
ax2 = ax1.twinx()
ax1.fill_between(month_agg["mnth"], month_agg["avg_casual"],     alpha=0.4, color="#FF9800", label="Casual")
ax1.fill_between(month_agg["mnth"], month_agg["avg_registered"], alpha=0.3, color="#2196F3", label="Registered")
ax1.plot(month_agg["mnth"], month_agg["avg_cnt"], color="black", lw=2, marker="o", markersize=4, label="Total")
ax2.plot(month_agg["mnth"], month_agg["avg_temp"], color="red", lw=2, linestyle="--", marker="s", markersize=4, label="Suhu °C")
ax1.set_xticks(range(1,13)); ax1.set_xticklabels(MONTHS)
ax1.set_ylabel("Rata-rata Penyewaan"); ax2.set_ylabel("Suhu (°C)", color="red")
ax2.tick_params(colors="red")
ax1.set_title("Tren Suhu & Penyewaan per Bulan")
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc="upper left", fontsize=8)
st.pyplot(fig); plt.close()

st.divider()

# KESIMPULAN

st.subheader("Kesimpulan")

best  = season_agg.loc[season_agg["total_cnt"].idxmax(), "season_label"]
worst = season_agg.loc[season_agg["total_cnt"].idxmin(), "season_label"]
pct   = (season_agg["total_cnt"].max() - season_agg["total_cnt"].min()) / season_agg["total_cnt"].min() * 100
level = "kuat" if abs(r_tot) >= 0.7 else "sedang"

c1, c2 = st.columns(2)
with c1:
    st.info(f"""**🍂 Musim & Penyewaan**

Musim **{best}** mencatat penyewaan tertinggi, sedangkan **{worst}** terendah 
dengan selisih hingga **{pct:.1f}%**. Musim panas dan gugur mendominasi karena 
cuaca lebih mendukung aktivitas luar ruangan.""")

with c2:
    st.info(f"""**🌡️ Suhu & Pengguna**

Terdapat korelasi positif **{level}** antara suhu dan total penyewaan (r = {r_tot:.3f}). 
Casual users (r = {r_cas:.3f}) lebih sensitif terhadap suhu dibanding 
registered users (r = {r_reg:.3f}), karena casual users cenderung bersepeda 
untuk rekreasi saat cuaca hangat.""")