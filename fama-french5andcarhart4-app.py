import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import plotly.express as px
import numpy as np
from datetime import datetime

# --- 頁面設定 ---
st.set_page_config(
    page_title="Fama-French 真實數據分析",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 美化 ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif !important; }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] label { color: #31333F !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# --- 數據讀取核心函數 ---
def process_zip_data(zip_file, file_type):
    """解析 Zip 檔案並清洗數據"""
    try:
        if isinstance(zip_file, bytes):
            z = zipfile.ZipFile(io.BytesIO(zip_file))
        else:
            z = zipfile.ZipFile(zip_file)
            
        csv_name = z.namelist()[0]
        try:
            df = pd.read_csv(z.open(csv_name), skiprows=3, index_col=0)
        except:
            df = pd.read_csv(z.open(csv_name), index_col=0)

        # 數據清洗標準流程
        # 1. 篩選有效日期列 (長度為6的字串, e.g., '202301')
        df = df[df.index.astype(str).str.len() == 6]
        # 2. 轉換索引為日期格式
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m")
        # 3. 數值正規化 (原始數據通常是百分比整數，需除以100)
        df = df.astype(float) / 100
        
        return df
    except Exception as e:
        st.error(f"解析 {file_type} 失敗: {e}")
        return None

# 自動下載函數 (作為備用)
@st.cache_data(ttl=3600)
def download_from_web(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            return r.content
    except:
        pass
    return None

# --- 側邊欄：數據控制中心 ---
with st.sidebar:
    st.title("📂 數據來源設定")
    st.info("💡 學校伺服器若擋 IP，請手動下載並上傳，保證 100% 真實數據。")
    
    st.markdown("### 1. 25 Portfolios (Size-Value)")
    st.markdown("[📥 點此下載 (Dartmouth)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_Formed_on_Size_and_Book_to_Market_CSV.zip)")
    file_25 = st.file_uploader("上傳 25_Portfolios.zip", type=["zip", "csv"], key="f25")

    st.markdown("### 2. Momentum (動能)")
    st.markdown("[📥 點此下載 (Dartmouth)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_CSV.zip)")
    file_mom = st.file_uploader("上傳 10_Portfolios.zip", type=["zip", "csv"], key="fmom")

    st.markdown("### 3. Fama-French 5 Factors")
    st.markdown("[📥 點此下載 (Dartmouth)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip)")
    file_ff = st.file_uploader("上傳 5_Factors.zip", type=["zip", "csv"], key="fff")

    st.divider()
    
    # 參數設定
    st.header("⚙️ 回測參數")
    start_year = st.slider("起始年份", 1927, 2024, 1990)
    initial_capital = st.number_input("初始本金", value=10000)

# --- 主程式邏輯 ---
st.title("🎓 Fama-French 因子分析 (真實數據版)")

# 變數初始化
df_25, df_mom, df_ff5 = None, None, None

# 1. 處理 25 Portfolios
if file_25:
    df_25 = process_zip_data(file_25, "25 Portfolios")
else:
    # 嘗試自動下載
    raw = download_from_web("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_Formed_on_Size_and_Book_to_Market_CSV.zip")
    if raw: df_25 = process_zip_data(raw, "25 Portfolios")

# 2. 處理 Momentum
if file_mom:
    df_mom = process_zip_data(file_mom, "Momentum")
else:
    raw = download_from_web("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_CSV.zip")
    if raw: df_mom = process_zip_data(raw, "Momentum")

# 3. 處理 Factors
if file_ff:
    df_ff5 = process_zip_data(file_ff, "5 Factors")
else:
    raw = download_from_web("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip")
    if raw: df_ff5 = process_zip_data(raw, "5 Factors")

# --- 檢查數據是否齊全 ---
if df_25 is None or df_mom is None or df_ff5 is None:
    st.error("❌ 無法獲取完整數據。")
    st.warning("""
    **請協助完成以下步驟以獲取真實數據：**
    1. 點擊側邊欄的連結下載 3 個 ZIP 檔案。
    2. 將檔案分別拖曳到側邊欄對應的上傳區。
    3. 系統將會自動開始分析。
    """)
    st.stop() # 停止執行，直到有數據為止

# --- 數據處理與分析 (有數據才會執行到這裡) ---
try:
    st.success("✅ 真實數據載入成功！開始運算...")
    
    # 時間篩選
    mask = df_25.index.year >= start_year
    df_25 = df_25[mask]
    df_mom = df_mom[mask]
    df_ff5 = df_ff5[mask]

    # 清洗欄位名稱
    df_25.columns = [c.strip() for c in df_25.columns]
    df_mom.columns = [c.strip() for c in df_mom.columns]
    df_ff5.columns = [c.strip() for c in df_ff5.columns]

    df_final = pd.DataFrame(index=df_25.index)
    
    # 嚴格映射 (不再隨機填充)
    style_map = {
        "Large Growth": ["BIG LoBM", "BIG Lo"], 
        "Large Blend": ["BIG BM2", "BIG 2"],
        "Large Value": ["BIG HiBM", "BIG Hi"],
        "Mid Growth": ["ME3 LoBM", "ME3 Lo"], 
        "Mid Blend": ["ME3 BM3", "ME3 3"], 
        "Mid Value": ["ME3 HiBM", "ME3 Hi"],
        "Small Growth": ["SMALL LoBM", "SMALL Lo"], 
        "Small Blend": ["SMALL BM3", "SMALL 3"], 
        "Small Value": ["SMALL HiBM", "SMALL Hi"]
    }

    for ui_name, possible_names in style_map.items():
        for pname in possible_names:
            if pname in df_25.columns:
                df_final[ui_name] = df_25[pname]
                break

    # 處理動能與市場
    if "Hi PRIOR" in df_mom.columns: df_final["Momentum"] = df_mom["Hi PRIOR"]
    elif "10" in df_mom.columns: df_final["Momentum"] = df_mom["10"] # 舊格式容錯

    mkt_col = "Mkt-RF" if "Mkt-RF" in df_ff5.columns else df_ff5.columns[0]
    rf_col = "RF" if "RF" in df_ff5.columns else df_ff5.columns[-1]
    df_final["Market"] = df_ff5[mkt_col] + df_ff5[rf_col]

    # 計算指標
    metrics = []
    for col in df_final.columns:
        s = df_final[col]
        tot_ret = (1 + s).prod()
        ann_ret = (tot_ret ** (12/len(s))) - 1 if len(s) > 0 else 0
        ann_vol = s.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd = (s + 1).cumprod().div((s + 1).cumprod().cummax()).sub(1).min()
        metrics.append({"Asset": col, "CAGR": ann_ret, "Vol": ann_vol, "Sharpe": sharpe, "MaxDD": max_dd})
        
    df_metrics = pd.DataFrame(metrics).set_index("Asset")
    mkt_cagr = df_metrics.loc["Market", "CAGR"]

    # --- UI 呈現 ---
    tab1, tab2, tab3 = st.tabs(["🧩 風格九宮格", "🚀 淨值走勢", "📋 統計報表"])

    with tab1:
        st.markdown(f"#### 美股風格績效 ({start_year}-Present)")
        rows = ["Large", "Mid", "Small"]
        for r in rows:
            cols = st.columns(3)
            types = ["Value", "Blend", "Growth"]
            for i, t in enumerate(types):
                name = f"{r} {t}"
                if name in df_metrics.index:
                    d = df_metrics.loc[name]
                    is_outperform = d["CAGR"] > mkt_cagr
                    icon = "🔥" if is_outperform else "❄️"
                    cols[i].metric(name, f"{d['CAGR']:.2%}", f"SR: {d['Sharpe']:.2f} {icon}")

    with tab2:
        st.markdown("#### 財富累積 (Log Scale)")
        subset = ["Small Value", "Momentum", "Market", "Large Growth"]
        valid = [x for x in subset if x in df_final.columns]
        df_cum = (1 + df_final[valid]).cumprod() * initial_capital
        fig = px.line(df_cum, log_y=True, template="plotly_dark")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### 詳細數據")
        try:
            st.dataframe(
                df_metrics.style.format("{:.2%}").background_gradient(cmap="RdYlGn"),
                use_container_width=True, height=500
            )
        except:
            st.dataframe(df_metrics, use_container_width=True)

except Exception as e:
    st.error(f"數據處理發生錯誤: {e}")
    st.write("請確認上傳的檔案是否為 Kenneth French 官網的原始 ZIP 檔。")
