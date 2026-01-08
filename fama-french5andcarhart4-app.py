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
    **🔧 系統狀態**
    - 模式：直接連線 (不依賴 pandas_datareader)
    - Python版本相容：已修復 (支援 3.10+)
    - 連線修正：已加入瀏覽器偽裝與多重路徑備援
    """)

# --- 核心：智慧下載與解析函數 ---
@st.cache_data(ttl=86400)
def get_fama_french_data():
    base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    
    # 定義檔案的可能網址 (因為教授有時候會微調檔名，這裡設定了備援機制)
    # 格式: (Key, [List of possible filenames])
    files_config = {
        "25_Portfolios": [
            "25_Portfolios_Formed_on_Size_and_Book_to_Market_CSV.zip",       # 標準格式
            "25_Portfolios_Formed_on_Size_and_Book_to_Market_5_x_5_CSV.zip" # 變體格式
        ],
        "Momentum": [
            "10_Portfolios_Prior_12_2_CSV.zip"
        ],
        "5_Factors": [
            "F-F_Research_Data_5_Factors_2x3_CSV.zip"
        ]
    }

    # 偽裝成 Chrome 瀏覽器，避免被伺服器誤判為機器人而阻擋 (404/403)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    results = {}

    for key, filenames in files_config.items():
        success = False
        for fname in filenames:
            url = f"{base_url}/{fname}"
            try:
                # 嘗試下載
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    # 解壓縮
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    csv_name = z.namelist()[0]
                    
                    # 智慧讀取：嘗試跳過不同行數來尋找正確的表頭
                    # Kenneth French 的 CSV 通常前 3 行是說明文字
                    df = None
                    try:
                        df = pd.read_csv(z.open(csv_name), skiprows=3, index_col=0)
                    except:
                        # 如果失敗，嘗試不跳行讀取再自行處理
                        df = pd.read_csv(z.open(csv_name), index_col=0)

                    # 數據清理標準化
                    if df is not None:
                        # 1. 確保 Index 是日期格式 (YYYYMM)
                        # 過濾掉非數字的 Index (例如檔案底部的版權聲明或 Annual Factors)
                        rows_to_keep = []
                        for idx in df.index:
                            s_idx = str(idx).strip()
                            # 檢查是否為 6 位數日期 (192701)
                            if s_idx.isdigit() and len(s_idx) == 6:
                                rows_to_keep.append(True)
                            else:
                                rows_to_keep.append(False)
                        
                        df = df[rows_to_keep]
                        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m", errors='coerce')
                        
                        # 2. 轉換數值 (原始數據通常是百分比，需除以 100)
                        df = df.astype(float) / 100
                        
                        results[key] = df
                        success = True
                        break # 下載成功，跳出檔名迴圈
            
            except Exception as e:
                print(f"嘗試下載 {url} 失敗: {e}")
                continue
        
        if not success:
            st.error(f"❌ 無法下載數據: {key} (已嘗試所有備援網址，請稍後再試)")
            return None, None, None

    return results.get("25_Portfolios"), results.get("Momentum"), results.get("5_Factors")

# --- 執行數據獲取 ---
with st.spinner('正在連線至 Kenneth French 資料庫 (Dartmouth) ...'):
    df_25, df_mom, df_ff5 = get_fama_french_data()

if df_25 is None:
    st.stop()

# --- 數據處理邏輯 ---
try:
    # 統一時間軸
    start_date = str(start_year)
    df_25 = df_25[start_date:]
    df_mom = df_mom[start_date:]
    df_ff5 = df_ff5[start_date:]

    # 1. 建立九宮格 DataFrame
    # 欄位映射表 (Kenneth French 的欄位名稱 -> 九宮格名稱)
    # 注意：需處理欄位名稱可能帶有的空白
    df_25.columns = [c.strip() for c in df_25.columns]
    
    style_map = {
        "Large Growth": "BIG LoBM", "Large Blend": "BIG 3", "Large Value": "BIG HiBM",
        "Mid Growth": "ME3 LoBM", "Mid Blend": "ME3 3", "Mid Value": "ME3 HiBM",
        "Small Growth": "SMALL LoBM", "Small Blend": "SMALL 3", "Small Value": "SMALL HiBM"
    }
    
    df_final = pd.DataFrame(index=df_25.index)
    for ui_name, csv_name in style_map.items():
        if csv_name in df_25.columns:
            df_final[ui_name] = df_25[csv_name]

    # 2. 加入動能 (Momentum)
    # 動能通常在 "Hi PRIOR" 或 "10" (第10組，最高動能)
    df_mom.columns = [c.strip() for c in df_mom.columns]
    # 嘗試抓取最高動能組別
    if "Hi PRIOR" in df_mom.columns:
        df_final["Momentum"] = df_mom["Hi PRIOR"]
    elif "10" in df_mom.columns:
        df_final["Momentum"] = df_mom["10"]
    elif "High" in df_mom.columns: 
        df_final["Momentum"] = df_mom["High"]

    # 3. 加入市場 (Market)
    df_ff5.columns = [c.strip() for c in df_ff5.columns]
    df_final["Market"] = df_ff5["Mkt-RF"] + df_ff5["RF"]

    # --- 計算財務指標 ---
    metrics = []
    for col in df_final.columns:
        series = df_final[col]
        # 累積報酬
        total_ret = (1 + series).prod()
        # 年化報酬 CAGR
        months = len(series)
        cagr = (total_ret ** (12/months)) - 1 if months > 0 else 0
        # 年化波動率
        vol = series.std() * np.sqrt(12)
        # 夏普值 (Risk Free 設為 0 簡化比較)
        sharpe = cagr / vol if vol != 0 else 0
        
        metrics.append({
            "Asset": col, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe
        })

    df_metrics = pd.DataFrame(metrics).set_index("Asset")
    mkt_cagr = df_metrics.loc["Market", "CAGR"] if "Market" in df_metrics.index else 0

    # --- 視覺化呈現 ---

    # [區塊 1] 九宮格績效
    st.subheader(f"📊 投資風格九宮格 (CAGR, {start_year}-Present)")
    
    # 自定義 CSS 讓 Metric 更好看
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
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
                # 若績效優於大盤顯示火焰，否則顯示雪花
                icon = "🔥" if val["CAGR"] > mkt_cagr else "❄️"
                with cols_ui[idx]:
                    st.metric(
                        label=name,
                        value=f"{val['CAGR']:.2%}",
                        delta=f"Sharpe: {val['Sharpe']:.2f} {icon}",
                        delta_color="off"
                    )

    # [區塊 2] 淨值走勢圖
    st.divider()
    st.subheader("📈 財富累積走勢 (Log Scale)")
    
    # 選擇要畫圖的欄位 (預設選幾個代表性的)
    plot_cols = ["Small Value", "Momentum", "Large Growth", "Market"]
    available_plot_cols = [c for c in plot_cols if c in df_final.columns]
    
    if available_plot_cols:
        df_cum = (1 + df_final[available_plot_cols]).cumprod() * initial_capital
        fig = px.line(df_cum, log_y=True, title="假設初始投入 $10,000 之資產成長")
        st.plotly_chart(fig, use_container_width=True)

    # [區塊 3] Fama-French 因子溢酬
    st.divider()
    st.subheader("📐 因子溢酬 (Factor Premia)")
    st.caption("解釋：SMB (小公司效應), HML (價值股效應), RMW (獲利能力), CMA (投資保守度)")
    
    factors_to_plot = ["SMB", "HML", "RMW", "CMA"]
    available_factors = [c for c in factors_to_plot if c in df_ff5.columns]
    
    if available_factors:
        df_factors_cum = (1 + df_ff5[available_factors]).cumprod()
        fig2 = px.line(df_factors_cum, log_y=True, title="多空因子累積報酬 (Long-Short Returns)")
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"數據處理過程發生錯誤: {e}")
    st.warning("建議：點擊右下角 'Manage app' -> 'Reboot app' 重啟應用程式")
