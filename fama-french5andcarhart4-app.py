import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 頁面設定 ---
st.set_page_config(
    page_title="Fama-French 因子戰情室 (Pro)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS 美化 (專業暗黑風格) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #ffffff; }
    div[data-testid="stMetric"] {
        background-color: #1e2530 !important;
        border: 1px solid #364156;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="stMetric"] label { color: #a0aab9 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #ffffff !important; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 智能讀檔函數 (自動略過檔頭廢話) ---
@st.cache_data
def load_smart_csv(filename):
    """
    自動偵測 CSV 表頭位置，無論前面有多少行說明文字都能讀取。
    """
    try:
        # 1. 先掃描檔案，尋找特徵關鍵字來決定從哪一行開始讀
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        start_row = 0
        target_found = False
        
        # 關鍵字特徵庫
        keywords = ["Mkt-RF", "SMALL LoBM", "Mom", "SMALL HiBM"]
        
        for i, line in enumerate(lines):
            # 如果這一行包含關鍵字，且有逗號，那就是表頭
            if any(k in line for k in keywords) and "," in line:
                start_row = i
                target_found = True
                break
        
        if not target_found:
            return None

        # 2. 正式讀取
        df = pd.read_csv(filename, skiprows=start_row, index_col=0)
        
        # 3. 清洗數據
        # 濾掉非日期的列 (有些檔案結尾有 Copyright)
        df = df[df.index.astype(str).str.len() == 6]
        # 轉換日期索引
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m")
        # 轉換數值 (原始數據是百分比整數，除以 100 變小數)
        df = df.astype(float) / 100
        # 清除欄位名稱空格
        df.columns = [c.strip() for c in df.columns]
        
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"讀取 {filename} 時發生錯誤: {e}")
        return None

# --- 4. 側邊欄控制 ---
with st.sidebar:
    st.title("⚙️ 參數設定")
    start_year = st.slider("📅 回測起始年份", 1927, 2024, 1990)
    initial_capital = st.number_input("💰 初始本金", value=10000, step=1000)
    
    st.divider()
    st.info("✅ 系統已鎖定本地 CSV 檔案")

# --- 5. 主程式載入數據 ---
st.title("🚀 Fama-French 深度因子分析")

# 硬編碼你的檔案名稱 (請確保檔案在同目錄)
file_25 = "25_Portfolios_5x5.csv"
file_mom = "F-F_Momentum_Factor.csv"
file_ff5 = "F-F_Research_Data_5_Factors_2x3.csv"

# 載入
df_25 = load_smart_csv(file_25)
df_mom = load_smart_csv(file_mom)
df_ff5 = load_smart_csv(file_ff5)

# 檢查檔案是否齊全
missing_files = []
if df_25 is None: missing_files.append(file_25)
if df_mom is None: missing_files.append(file_mom)
if df_ff5 is None: missing_files.append(file_ff5)

if missing_files:
    st.error("❌ 找不到以下檔案，請確認它們跟 app.py 在同一個資料夾內：")
    for f in missing_files:
        st.code(f)
    st.stop()

# --- 6. 數據整合與計算 ---

# 時間過濾
mask = df_25.index.year >= start_year
df_25 = df_25[mask]
df_mom = df_mom[mask] if df_mom is not None else None
df_ff5 = df_ff5[mask]

# 建立總表
df_final = pd.DataFrame(index=df_25.index)

# 映射 25 Portfolios 到 風格箱 (Size-Value)
# 根據 Fama-French 定義：
# Small = SMALL, Big = BIG
# Value = HiBM, Growth = LoBM, Blend = BM3
mapping = {
    "Large Growth": "BIG LoBM", 
    "Large Blend": "BIG BM3",
    "Large Value": "BIG HiBM",
    "Mid Growth": "ME3 LoBM", # 近似中型
    "Mid Blend": "ME3 BM3",
    "Mid Value": "ME3 HiBM",
    "Small Growth": "SMALL LoBM",
    "Small Blend": "SMALL BM3",
    "Small Value": "SMALL HiBM"
}

for ui_name, col_name in mapping.items():
    if col_name in df_25.columns:
        df_final[ui_name] = df_25[col_name]

# 處理動能 (Momentum)
if df_mom is not None:
    # 通常欄位叫 'Mom'，有時候叫 '10' 或 'Hi PRIOR'，這裡做容錯
    mom_col = "Mom" if "Mom" in df_mom.columns else df_mom.columns[-1]
    df_final["Momentum"] = df_mom[mom_col]

# 處理市場 (Market)
mkt_col = "Mkt-RF"
rf_col = "RF"
df_final["Market"] = df_ff5[mkt_col] + df_ff5[rf_col] # 還原市場總報酬

# 計算統計數據
metrics = []
for col in df_final.columns:
    s = df_final[col]
    
    # 累積報酬
    total_ret = (1 + s).prod()
    # 年化報酬 (CAGR)
    cagr = (total_ret ** (12 / len(s))) - 1
    # 年化波動率
    vol = s.std() * np.sqrt(12)
    # 夏普值 (假設無風險利率已內含或簡化計算)
    sharpe = cagr / vol if vol > 0 else 0
    # 最大回撤 (MaxDD)
    cum_returns = (1 + s).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    
    metrics.append({
        "Asset": col, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": max_dd
    })

df_metrics = pd.DataFrame(metrics).set_index("Asset")
mkt_cagr = df_metrics.loc["Market", "CAGR"]

# --- 7. 視覺化呈現 ---

tab1, tab2, tab3 = st.tabs(["📊 風格九宮格", "📈 財富曲線", "📋 詳細數據"])

with tab1:
    st.markdown("### 🇺🇸 美股風格箱績效矩陣")
    st.markdown(f"*(基準: {start_year} - Present)*")
    
    rows = ["Large", "Mid", "Small"]
    types = ["Value", "Blend", "Growth"]
    
    for r in rows:
        cols = st.columns(3)
        for i, t in enumerate(types):
            name = f"{r} {t}"
            if name in df_metrics.index:
                d = df_metrics.loc[name]
                
                # 顏色邏輯：贏大盤用綠色/火焰，輸大盤用灰色
                is_winner = d['CAGR'] > mkt_cagr
                delta_val = f"{d['CAGR'] - mkt_cagr:.1%} vs Mkt"
                
                cols[i].metric(
                    label=name,
                    value=f"{d['CAGR']:.1%}",
                    delta=delta_val,
                    delta_color="normal" if is_winner else "off"
                )

with tab2:
    st.markdown("### 💰 10,000 美元投資累積價值 (對數座標)")
    
    # 預設挑選幾個關鍵資產畫圖
    selected_assets = st.multiselect(
        "選擇比較資產", 
        df_final.columns, 
        default=["Small Value", "Market", "Momentum", "Large Growth"]
    )
    
    if selected_assets:
        df_cum = (1 + df_final[selected_assets]).cumprod() * initial_capital
        
        fig = px.line(df_cum, log_y=True, template="plotly_dark")
        fig.update_layout(
            xaxis_title="年份",
            yaxis_title="資產淨值 ($)",
            legend_title="策略/資產",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 🔢 完整風險報酬表")
    
    # 格式化表格
    st.dataframe(
        df_metrics.style.format({
            "CAGR": "{:.2%}", 
            "Vol": "{:.2%}", 
            "Sharpe": "{:.2f}", 
            "MaxDD": "{:.2%}"
        }).background_gradient(subset=["CAGR"], cmap="Greens")
          .background_gradient(subset=["MaxDD"], cmap="Reds_r"),
        use_container_width=True,
        height=600
    )

    st.markdown("---")
    st.caption(f"資料來源: Kenneth R. French Data Library | 處理檔案: {file_25}, {file_mom}, {file_ff5}")
