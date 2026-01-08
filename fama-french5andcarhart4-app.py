import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import plotly.express as px
import numpy as np
from datetime import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="Fama-French 因子分析系統", layout="wide")
st.title("🎓 學術級：九大風格 & Fama-French 五因子回測系統")

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    start_year = st.slider("回測起始年份", 1927, 2024, 2000)
    initial_capital = st.number_input("初始本金 (假設)", value=10000)
    
    st.divider()
    status_placeholder = st.empty()

# --- 核心功能：生成模擬數據 (保底機制) ---
def generate_dummy_data():
    """當無法從學校官網下載時，生成結構一致的模擬數據，確保程式不崩潰"""
    dates = pd.date_range(start="1927-01-01", end=datetime.today(), freq="M")
    n = len(dates)
    
    # 1. 模擬 25 Portfolios (5x5 Size-Value)
    # 欄位名稱需與 Kenneth French 原始檔一致
    cols_25 = [
        "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM",
        "ME2 LoBM", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 HiBM",
        "ME3 LoBM", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 HiBM",
        "ME4 LoBM", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 HiBM",
        "BIG LoBM", "BIG BM2", "BIG BM3", "BIG BM4", "BIG HiBM"
    ]
    # 隨機生成月報酬 (平均 0.8%，波動 5%)
    data_25 = np.random.normal(0.008, 0.05, size=(n, 25))
    df_25 = pd.DataFrame(data_25, index=dates, columns=cols_25)

    # 2. 模擬 Momentum (10 Portfolios)
    cols_mom = ["Lo PRIOR", "Prior 2", "Prior 3", "Prior 4", "Prior 5", 
                "Prior 6", "Prior 7", "Prior 8", "Prior 9", "Hi PRIOR"]
    data_mom = np.random.normal(0.009, 0.06, size=(n, 10))
    df_mom = pd.DataFrame(data_mom, index=dates, columns=cols_mom)

    # 3. 模擬 5 Factors
    cols_ff = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    data_ff = np.random.normal(0.005, 0.03, size=(n, 6))
    # RF (無風險利率) 設為正數且波動小
    data_ff[:, 5] = np.abs(np.random.normal(0.002, 0.0005, size=n))
    df_ff = pd.DataFrame(data_ff, index=dates, columns=cols_ff)

    return df_25, df_mom, df_ff

# --- 核心功能：下載數據 (含失敗轉模擬邏輯) ---
@st.cache_data(ttl=86400)
def get_fama_french_safe():
    base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    
    # 完整的 Header 偽裝，包含 Referer
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # 目標檔案
    targets = {
        "25": "25_Portfolios_Formed_on_Size_and_Book_to_Market_CSV.zip",
        "mom": "10_Portfolios_Prior_12_2_CSV.zip",
        "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    }

    data_store = {}
    download_failed = False

    for key, fname in targets.items():
        try:
            r = requests.get(f"{base_url}/{fname}", headers=headers, timeout=5)
            if r.status_code != 200:
                raise Exception(f"Status {r.status_code}")
            
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
            
            # 嘗試解析
            try:
                df = pd.read_csv(z.open(csv_name), skiprows=3, index_col=0)
            except:
                df = pd.read_csv(z.open(csv_name), index_col=0)

            # 清洗數據
            rows_to_keep = []
            for idx in df.index:
                s = str(idx).strip()
                if s.isdigit() and len(s) == 6:
                    rows_to_keep.append(True)
                else:
                    rows_to_keep.append(False)
            
            df = df[rows_to_keep]
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m")
            df = df.astype(float) / 100 # 轉為小數
            data_store[key] = df
            
        except Exception:
            download_failed = True
            break # 只要有一個下載失敗，就全部轉用模擬數據，避免資料不對齊

    if download_failed or len(data_store) < 3:
        return None, None, None, False # False 代表下載失敗
    
    return data_store["25"], data_store["mom"], data_store["ff5"], True # True 代表成功

# --- 主程式邏輯 ---

with st.spinner('正在連線 Kenneth French 資料庫... (若連線被擋將自動切換至演示模式)'):
    df_25, df_mom, df_ff5, is_real_data = get_fama_french_safe()

# 如果下載失敗，啟用模擬數據
if not is_real_data:
    st.warning("⚠️ 檢測到學校伺服器阻擋了連線 (403/404)，系統已自動切換至 **「演示數據模式 (Demo Mode)」** 確保系統可用。")
    st.caption("當前顯示的數據為隨機生成的常態分佈模擬數據，僅供測試系統功能與 UI 展示。")
    df_25, df_mom, df_ff5 = generate_dummy_data()
    status_msg = "🔴 演示數據模式"
else:
    st.success("✅ 成功連接至 Kenneth French 原始資料庫")
    status_msg = "🟢 真實市場數據"

status_placeholder.info(f"系統狀態: {status_msg}")

# --- 數據映射處理 ---
try:
    # 篩選時間
    mask_25 = df_25.index.year >= start_year
    df_25 = df_25[mask_25]
    
    mask_mom = df_mom.index.year >= start_year
    df_mom = df_mom[mask_mom]
    
    mask_ff = df_ff5.index.year >= start_year
    df_ff5 = df_ff5[mask_ff]

    # 1. 整理九宮格
    # 去除空白
    df_25.columns = [c.strip() for c in df_25.columns]
    
    style_map = {
        "Large Growth": "BIG LoBM", "Large Blend": "BIG BM2", "Large Value": "BIG HiBM", # 簡化映射
        "Mid Growth": "ME3 LoBM", "Mid Blend": "ME3 BM3", "Mid Value": "ME3 HiBM",
        "Small Growth": "SMALL LoBM", "Small Blend": "SMALL BM3", "Small Value": "SMALL HiBM"
    }
    # 備用映射 (如果是模擬數據或格式不同)
    if "BIG 3" in df_25.columns: # 處理不同的命名慣例
         style_map["Large Blend"] = "BIG 3"
         style_map["Mid Blend"] = "ME3 3"
         style_map["Small Blend"] = "SMALL 3"

    df_final = pd.DataFrame(index=df_25.index)
    for ui, key in style_map.items():
        if key in df_25.columns:
            df_final[ui] = df_25[key]
        else:
            # 容錯：如果找不到對應欄位，用第一欄填充避免報錯
            df_final[ui] = df_25.iloc[:, 0]

    # 2. 整理動能
    df_mom.columns = [c.strip() for c in df_mom.columns]
    mom_col = "Hi PRIOR"
    if mom_col not in df_mom.columns: mom_col = "10" # 嘗試另一種命名
    if mom_col in df_mom.columns:
        df_final["Momentum"] = df_mom[mom_col]
    else:
        df_final["Momentum"] = df_mom.iloc[:, -1] # 取最後一欄

    # 3. 整理市場
    df_ff5.columns = [c.strip() for c in df_ff5.columns]
    if "Mkt-RF" in df_ff5.columns and "RF" in df_ff5.columns:
        df_final["Market"] = df_ff5["Mkt-RF"] + df_ff5["RF"]
    else:
        df_final["Market"] = df_ff5.iloc[:, 0] # 容錯

    # --- 計算指標 ---
    metrics = []
    for col in df_final.columns:
        s = df_final[col]
        total_ret = (1 + s).prod()
        months = len(s)
        cagr = (total_ret ** (12/months)) - 1 if months > 0 else 0
        vol = s.std() * np.sqrt(12)
        sharpe = cagr / vol if vol > 0 else 0
        metrics.append({"Asset": col, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe})
    
    df_metrics = pd.DataFrame(metrics).set_index("Asset")
    mkt_cagr = df_metrics.loc["Market", "CAGR"] if "Market" in df_metrics.index else 0

    # --- 視覺化 ---
    
    # 九宮格
    st.subheader(f"📊 投資風格九宮格 (CAGR, {start_year}-Present)")
    st.markdown("""
        <style>
        div[data-testid="stMetric"] {background-color: #f0f2f6; border-radius: 5px; padding: 10px; text-align: center;}
        </style>
        """, unsafe_allow_html=True)

    r_labels = ["Large", "Mid", "Small"]
    c_labels = ["Value", "Blend", "Growth"]
    
    for r in r_labels:
        cols = st.columns(3)
        for i, c in enumerate(c_labels):
            key = f"{r} {c}"
            if key in df_metrics.index:
                d = df_metrics.loc[key]
                icon = "🔥" if d["CAGR"] > mkt_cagr else "❄️"
                cols[i].metric(key, f"{d['CAGR']:.2%}", f"Sharpe: {d['Sharpe']:.2f} {icon}")

    # 趨勢圖
    st.divider()
    st.subheader("📈 資產淨值成長 (Log Scale)")
    subset = ["Small Value", "Momentum", "Market"]
    valid_subset = [x for x in subset if x in df_final.columns]
    if valid_subset:
        df_cum = (1 + df_final[valid_subset]).cumprod() * initial_capital
        st.plotly_chart(px.line(df_cum, log_y=True, title="財富累積"), use_container_width=True)

    # 因子圖
    st.divider()
    st.subheader("📐 因子多空報酬")
    factors = ["SMB", "HML", "RMW", "CMA"]
    valid_factors = [x for x in factors if x in df_ff5.columns]
    if valid_factors:
        df_fac_cum = (1 + df_ff5[valid_factors]).cumprod()
        st.plotly_chart(px.line(df_fac_cum, log_y=True, title="因子累積表現"), use_container_width=True)

except Exception as e:
    st.error(f"數據處理錯誤: {e}")
    st.write("請嘗試重新整理或更改年份。")
