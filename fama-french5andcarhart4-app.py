import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import plotly.express as px
import numpy as np

# --- 頁面設定 ---
st.set_page_config(page_title="學術級：Fama-French 風格因子", layout="wide")
st.title("🎓 學術級：九大風格 & Fama-French 五因子回測系統")
st.caption("數據來源: Kenneth R. French Data Library (Direct Download) | 涵蓋範圍: 1927 年至今")

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    start_year = st.slider("回測起始年份", 1927, 2024, 1927)
    initial_capital = st.number_input("初始本金 (假設)", value=10000)
    
    st.divider()
    st.info("""
    **🔧 技術說明**
    此版本已移除 `pandas_datareader`，改為直接從達特茅斯學院官網下載原始 CSV 並進行解析，以解決 Python 3.13 相容性問題。
    """)

# --- 核心：直接下載並解析 Kenneth French 原始檔 ---
@st.cache_data(ttl=86400)
def get_fama_french_direct():
    # 定義檔案的 URL (直接指向 Zip 檔)
    urls = {
        "25_Portfolios": "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_Formed_on_Size_and_Book-to-Market_CSV.zip",
        "Momentum": "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_CSV.zip",
        "5_Factors": "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    }

    data = {}

    for key, url in urls.items():
        try:
            # 1. 下載 Zip
            r = requests.get(url)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            
            # 2. 讀取 CSV (通常 Zip 裡只有一個 CSV)
            csv_filename = z.namelist()[0]
            
            # 3. 解析 CSV (Fama-French 的 CSV 格式很亂，需要略過標頭)
            # 讀取前幾行來判斷實際數據從哪開始，但通常 skiprows=3 可以解決大部分
            df = pd.read_csv(z.open(csv_filename), skiprows=3, index_col=0)
            
            # 4. 清理數據
            # 原始檔下方通常有 "Annual Factors" 的說明，需要切掉
            # 找出索引變成非日期的那一行
            rows_to_keep = []
            for idx in df.index:
                try:
                    # 嘗試將索引轉為數字 (YYYYMM)
                    int(str(idx).strip()) 
                    rows_to_keep.append(True)
                except:
                    rows_to_keep.append(False)
            
            df = df[rows_to_keep]
            
            # 轉換索引為 datetime
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m", errors='coerce')
            df = df.dropna(how='all') # 移除轉換失敗的行
            
            # 轉換數值 (原始資料是百分比，需除以 100)
            df = df.astype(float) / 100
            
            data[key] = df
            
        except Exception as e:
            st.error(f"下載 {key} 失敗: {e}")
            return None, None, None

    return data.get("25_Portfolios"), data.get("Momentum"), data.get("5_Factors")

# 執行下載
with st.spinner('正在直接連線至 Kenneth French 原始資料庫下載與解析...'):
    df_25, df_mom, df_ff5 = get_fama_french_direct()

if df_25 is None:
    st.error("⚠️ 數據下載失敗，請檢查網路連線。")
    st.stop()

# --- 數據處理與映射 (邏輯同前) ---
try:
    # 篩選年份
    start_date = str(start_year)
    df_25 = df_25[start_date:]
    df_mom = df_mom[start_date:]
    df_ff5 = df_ff5[start_date:]

    # 欄位映射
    style_map = {
        "Large Growth": "BIG LoBM", "Large Blend": "BIG 3", "Large Value": "BIG HiBM",
        "Mid Growth": "ME3 LoBM", "Mid Blend": "ME3 3", "Mid Value": "ME3 HiBM",
        "Small Growth": "SMALL LoBM", "Small Blend": "SMALL 3", "Small Value": "SMALL HiBM"
    }

    df_final = pd.DataFrame(index=df_25.index)
    
    # 填入九宮格
    for name, col in style_map.items():
        # 清理欄位名稱空白
        clean_cols = [c.strip() for c in df_25.columns]
        df_25.columns = clean_cols
        if col in df_25.columns:
            df_final[name] = df_25[col]

    # 填入動能 (通常是 'Hi PRIOR' 或 '10')
    mom_cols = [c.strip() for c in df_mom.columns]
    df_mom.columns = mom_cols
    mom_target = "Hi PRIOR" if "Hi PRIOR" in mom_cols else "10"
    if mom_target in df_mom.columns:
        df_final["Momentum"] = df_mom[mom_target]

    # 填入市場因子
    ff5_cols = [c.strip() for c in df_ff5.columns]
    df_ff5.columns = ff5_cols
    df_final["Market"] = df_ff5["Mkt-RF"] + df_ff5["RF"]

    # --- 計算指標 ---
    metrics = []
    for col in df_final.columns:
        series = df_final[col]
        total_ret = (1 + series).prod()
        months = len(series)
        cagr = (total_ret ** (12/months)) - 1 if months > 0 else 0
        vol = series.std() * np.sqrt(12)
        sharpe = cagr / vol if vol != 0 else 0
        metrics.append({"Asset": col, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe})

    df_metrics = pd.DataFrame(metrics).set_index("Asset")
    mkt_cagr = df_metrics.loc["Market", "CAGR"] if "Market" in df_metrics.index else 0

    # --- 顯示介面 ---
    
    # 1. 九宮格
    st.subheader(f"📊 投資風格九宮格 (CAGR, {start_year}-Present)")
    
    # CSS 
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {background-color: #f0f2f6; border: 1px solid #d1d5db; border-radius: 5px; text-align: center; padding: 10px;}
    </style>
    """, unsafe_allow_html=True)

    rows = ["Large", "Mid", "Small"]
    cols = ["Value", "Blend", "Growth"]

    for r in rows:
        c1, c2, c3 = st.columns(3)
        cols_ui = [c1, c2, c3]
        for idx, c in enumerate(cols):
            name = f"{r} {c}"
            if name in df_metrics.index:
                val = df_metrics.loc[name]
                emoji = "🔥" if val["CAGR"] > mkt_cagr else "❄️"
                with cols_ui[idx]:
                    st.metric(name, f"{val['CAGR']:.2%}", f"{emoji} Sharpe: {val['Sharpe']:.2f}")

    # 2. 淨值走勢
    st.divider()
    st.subheader("🚀 世紀對決：動能 vs 價值 vs 大盤")
    plot_cols = ["Momentum", "Small Value", "Market"]
    df_cum = (1 + df_final[plot_cols]).cumprod() * initial_capital
    st.plotly_chart(px.line(df_cum, log_y=True, title="資產淨值 (Log Scale)"), use_container_width=True)
    
    # 3. 因子溢酬
    st.divider()
    st.subheader("📐 五因子溢酬累積圖 (Factor Premia)")
    factor_cum = (1 + df_ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]).cumprod()
    c1, c2 = st.columns(2)
    with c1:
        st.caption("傳統因子")
        st.plotly_chart(px.line(factor_cum[["Mkt-RF", "SMB", "HML"]], log_y=True), use_container_width=True)
    with c2:
        st.caption("獲利與投資因子")
        st.plotly_chart(px.line(factor_cum[["RMW", "CMA"]], log_y=True), use_container_width=True)

except Exception as e:
    st.error(f"資料處理發生錯誤: {e}")
    st.write("這通常是因為 Kenneth French 資料格式微調導致，建議重新整理頁面。")
