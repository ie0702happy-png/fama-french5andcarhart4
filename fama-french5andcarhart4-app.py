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
    此版本直接從達特茅斯學院官網下載原始 CSV 並進行解析。
    已修復 URL 格式與 User-Agent 阻擋問題。
    """)

# --- 核心：直接下載並解析 Kenneth French 原始檔 ---
@st.cache_data(ttl=86400)
def get_fama_french_direct():
    # 修正後的正確 URL 列表
    base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    urls = {
        # 注意：這裡必須是用底線 _ 而非連字號 -
        "25_Portfolios": f"{base_url}/25_Portfolios_Formed_on_Size_and_Book_to_Market_CSV.zip",
        "Momentum": f"{base_url}/10_Portfolios_Prior_12_2_CSV.zip",
        "5_Factors": f"{base_url}/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    }

    # 偽裝成瀏覽器，避免 403/404 錯誤
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    data = {}

    for key, url in urls.items():
        try:
            # 發送請求
            r = requests.get(url, headers=headers)
            r.raise_for_status() # 檢查是否成功 (200 OK)
            
            # 解壓縮
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_filename = z.namelist()[0]
            
            # 讀取 CSV (跳過前 3 行說明文字)
            df = pd.read_csv(z.open(csv_filename), skiprows=3, index_col=0)
            
            # 清理數據 (去除底部的年度統計 Annual Factors)
            rows_to_keep = []
            for idx in df.index:
                try:
                    # 只有當索引是 6 位數日期 (YYYYMM) 時才保留
                    if len(str(idx).strip()) == 6 and str(idx).strip().isdigit():
                        rows_to_keep.append(True)
                    else:
                        rows_to_keep.append(False)
                except:
                    rows_to_keep.append(False)
            
            df = df[rows_to_keep]
            
            # 轉換索引格式
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m", errors='coerce')
            df = df.dropna(how='all') 
            
            # 轉換數值 (原始資料是百分比，需除以 100)
            df = df.astype(float) / 100
            
            data[key] = df
            
        except Exception as e:
            st.error(f"下載 {key} 失敗: {e} | URL: {url}")
            return None, None, None

    return data.get("25_Portfolios"), data.get("Momentum"), data.get("5_Factors")

# 執行下載
with st.spinner('正在連線至 Kenneth French 原始資料庫下載與解析...'):
    df_25, df_mom, df_ff5 = get_fama_french_direct()

if df_25 is None:
    st.error("⚠️ 數據下載失敗，請檢查網路連線或稍後再試。")
    st.stop()

# --- 數據處理與映射 ---
try:
    # 篩選年份
    start_date = str(start_year)
    df_25 = df_25[start_date:]
    df_mom = df_mom[start_date:]
    df_ff5 = df_ff5[start_date:]

    # 九宮格映射表
    style_map = {
        "Large Growth": "BIG LoBM", "Large Blend": "BIG 3", "Large Value": "BIG HiBM",
        "Mid Growth": "ME3 LoBM", "Mid Blend": "ME3 3", "Mid Value": "ME3 HiBM",
        "Small Growth": "SMALL LoBM", "Small Blend": "SMALL 3", "Small Value": "SMALL HiBM"
    }

    df_final = pd.DataFrame(index=df_25.index)
    
    # 填入九宮格數據
    # 先清理欄位名稱 (移除可能存在的空白)
    df_25.columns = [c.strip() for c in df_25.columns]
    
    for name, col in style_map.items():
        if col in df_25.columns:
            df_final[name] = df_25[col]

    # 填入動能數據
    df_mom.columns = [c.strip() for c in df_mom.columns]
    # 動能通常是 "Hi PRIOR" 或 "10"
    mom_target = "Hi PRIOR" if "Hi PRIOR" in df_mom.columns else "10"
    if mom_target in df_mom.columns:
        df_final["Momentum"] = df_mom[mom_target]

    # 填入市場因子與其他因子
    df_ff5.columns = [c.strip() for c in df_ff5.columns]
    # 市場報酬 = Mkt-RF (超額報酬) + RF (無風險利率)
    df_final["Market"] = df_ff5["Mkt-RF"] + df_ff5["RF"]

    # --- 計算指標 ---
    metrics = []
    for col in df_final.columns:
        series = df_final[col]
        # 總報酬
        total_ret = (1 + series).prod()
        months = len(series)
        # 年化報酬 CAGR
        cagr = (total_ret ** (12/months)) - 1 if months > 0 else 0
        # 年化波動率
        vol = series.std() * np.sqrt(12)
        # 夏普值 (簡化版)
        sharpe = cagr / vol if vol != 0 else 0
        metrics.append({"Asset": col, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe})

    df_metrics = pd.DataFrame(metrics).set_index("Asset")
    mkt_cagr = df_metrics.loc["Market", "CAGR"] if "Market" in df_metrics.index else 0

    # --- 顯示介面 ---
    
    # 1. 九宮格
    st.subheader(f"📊 投資風格九宮格 (年化報酬 CAGR, {start_year}-Present)")
    
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
    st.caption("Log Scale (對數座標) 顯示長期複利效果")
    
    plot_cols = ["Momentum", "Small Value", "Market"]
    # 確保欄位存在
    existing_cols = [c for c in plot_cols if c in df_final.columns]
    
    if existing_cols:
        df_cum = (1 + df_final[existing_cols]).cumprod() * initial_capital
        st.plotly_chart(px.line(df_cum, log_y=True, title="資產淨值成長"), use_container_width=True)
    
    # 3. 因子溢酬
    st.divider()
    st.subheader("📐 五因子溢酬累積圖 (Factor Premia)")
    st.caption("顯示因子多空對沖後的累積報酬 (Long-Short Return)")
    
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    existing_factors = [c for c in factor_cols if c in df_ff5.columns]
    
    if existing_factors:
        factor_cum = (1 + df_ff5[existing_factors]).cumprod()
        
        c1, c2 = st.columns(2)
        with c1:
            st.caption("傳統三因子")
            st.plotly_chart(px.line(factor_cum[["Mkt-RF", "SMB", "HML"]], log_y=True), use_container_width=True)
        with c2:
            st.caption("獲利與投資因子")
            st.plotly_chart(px.line(factor_cum[["RMW", "CMA"]], log_y=True), use_container_width=True)

except Exception as e:
    st.error(f"資料處理發生錯誤: {e}")
    st.write("這通常是數據源格式微調導致，請嘗試重新整理頁面。")
