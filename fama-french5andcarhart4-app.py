import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="學術級：Fama-French 九大風格與動能", layout="wide")
st.title("🎓 學術級：九大風格 & Fama-French 五因子回測系統")
st.caption("數據來源: Kenneth R. French Data Library | 涵蓋範圍: 1927 年至今 | 模型: Fama-French 5-Factor + Carhart Momentum")

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    # 學術數據通常是月資料
    start_year = st.slider("回測起始年份", 1927, 2023, 1927, help="Fama-French 數據最早可追溯至 1927 年")
    initial_capital = st.number_input("初始本金 (假設)", value=10000)
    
    st.divider()
    st.info("""
    **📚 數據構建原理**
    
    * **九宮格 (Style Box)**: 
      源自 [25 Portfolios on Size & B/M]。
      利用 Size (規模) 與 Book-to-Market (價值) 的 5x5 交叉組合萃取。
    
    * **動能 (Momentum)**: 
      源自 [10 Portfolios on Momentum] 的贏家組合 (Prior 10)。
      
    * **獲利 (RMW) & 投資 (CMA)**:
      源自 Fama-French 5因子模型資料。
    """)

# --- 核心數據抓取 ---
@st.cache_data(ttl=86400) # 每日更新一次即可
def get_academic_data():
    try:
        # 1. 抓取 25 Portfolios (用於九宮格: Size x Value)
        ds_25 = web.DataReader('25_Portfolios_Formed_on_Size_and_Book-to-Market', 'famafrench', start='1900-01-01')
        df_25 = ds_25[0] 

        # 2. 抓取 Momentum Portfolios (用於動能)
        ds_mom = web.DataReader('10_Portfolios_Prior_12_2', 'famafrench', start='1900-01-01')
        df_mom = ds_mom[0]

        # 3. 抓取 5-Factor Data (用於 RMW, CMA, Market)
        ds_ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='1900-01-01')
        df_ff5 = ds_ff5[0]

        return df_25, df_mom, df_ff5
    
    except Exception as e:
        return None, None, None

# 顯示讀取狀態
with st.spinner('正在連線至 Dartmouth College 抓取百年學術資料...'):
    df_25_raw, df_mom_raw, df_ff5_raw = get_academic_data()

if df_25_raw is None:
    st.error("⚠️ 無法連線至 Kenneth French 資料庫。可能是網路問題或資料源暫時無法存取。")
    st.stop()

# --- 數據清理 ---
# 資料為百分比 (例如 5.0 代表 5%)，需除以 100
start_date = str(start_year)
df_25 = df_25_raw[start_date:] / 100
df_mom = df_mom_raw[start_date:] / 100
df_ff5 = df_ff5_raw[start_date:] / 100

# --- 構建九宮格 (Mapping) ---
# Fama-French 原始欄位命名規則: Small/Big + LoBM/HiBM
style_map = {
    # Size Quintile 5 (Big)
    "Large Growth": "BIG LoBM",
    "Large Blend":  "BIG 3",
    "Large Value":  "BIG HiBM",
    
    # Size Quintile 3 (Mid - approximate)
    "Mid Growth":   "ME3 LoBM",
    "Mid Blend":    "ME3 3",
    "Mid Value":    "ME3 HiBM",
    
    # Size Quintile 1 (Small)
    "Small Growth": "SMALL LoBM",
    "Small Blend":  "SMALL 3",
    "Small Value":  "SMALL HiBM"
}

df_final = pd.DataFrame(index=df_25.index)

# 1. 填入風格因子
for name, col_name in style_map.items():
    if col_name in df_25.columns:
        df_final[name] = df_25[col_name]

# 2. 填入動能 (Momentum) - 取最高動能組
mom_col = "Hi PRIOR" if "Hi PRIOR" in df_mom.columns else "10"
df_final["Momentum"] = df_mom[mom_col]

# 3. 填入 FF5 因子 (Market, RMW, CMA)
# 注意：這些是因子溢酬 (Long - Short)，我們需要還原成多頭策略表現比較困難，
# 這裡我們直接展示因子本身的累積溢酬 (Cumulative Premium)
df_factors = df_ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].copy()
df_final["Market"] = df_factors["Mkt-RF"] + df_factors["RF"] # 市場投組

# 轉換索引
df_final.index = df_final.index.to_timestamp()
df_factors.index = df_factors.index.to_timestamp()

# --- 計算績效指標 ---
def calculate_metrics(series):
    # 幾何平均年化報酬
    total_ret = (1 + series).prod()
    months = len(series)
    if months == 0: return 0, 0, 0
    cagr = (total_ret) ** (12 / months) - 1
    
    # 年化波動率
    vol = series.std() * np.sqrt(12)
    
    # 夏普值 (簡化版)
    sharpe = cagr / vol if vol != 0 else 0
    
    return cagr, vol, sharpe

metrics_data = []
for col in df_final.columns:
    c, v, s = calculate_metrics(df_final[col])
    metrics_data.append({"Asset": col, "CAGR": c, "Vol": v, "Sharpe": s})

df_metrics = pd.DataFrame(metrics_data).set_index("Asset")

# --- 介面呈現 ---

# 1. 九宮格熱圖 (The 9-Box Grid)
st.subheader(f"📊 投資風格九宮格 (CAGR 年化報酬, {start_year}-Present)")
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #f0f2f6;
    border: 1px solid #d1d5db;
    padding: 10px; border-radius: 5px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

rows = ["Large", "Mid", "Small"]
cols = ["Value", "Blend", "Growth"]

c1, c2, c3 = st.columns(3)
cols_ui = [c1, c2, c3]

mkt_cagr = df_metrics.loc["Market", "CAGR"]

for i, size in enumerate(rows):
    row_cols = st.columns(3)
    for j, style in enumerate(cols):
        name = f"{size} {style}"
        if name in df_metrics.index:
            val = df_metrics.loc[name]
            
            # 視覺提示：高於大盤顯示 🔥
            emoji = "🔥" if val["CAGR"] > mkt_cagr else "❄️"
            
            with row_cols[j]:
                st.metric(
                    label=name,
                    value=f"{val['CAGR']:.2%}",
                    delta=f"{emoji} Sharpe: {val['Sharpe']:.2f}",
                    help=f"年化波動率: {val['Vol']:.2%}"
                )

# 2. 因子與動能的世紀對決
st.divider()
st.subheader("🚀 因子大亂鬥：Fama-French 5因子 + Momentum")
st.caption("此圖顯示各策略的 **淨值成長 (Log Scale)**。這就是你要的「最完整」因子對決。")

# 選擇要比較的選手
# 包含：動能、小型價值(最强風格)、獲利因子(RMW)、投資因子(CMA)、大盤
comparison_cols = ["Momentum", "Small Value", "Market"]
df_plot = df_final[comparison_cols].copy()

# 由於 RMW 和 CMA 是因子溢酬 (多-空)，我們模擬一個「純多頭」因子投組 (Market + Factor) 來讓比較有意義
# 或者直接畫因子累積溢酬。為了直觀，我們畫原始定義的資產：Small Value vs Momentum vs Market
# 另外把 RMW (獲利) 的概念加上去 -> 這裡用 "Large Growth" 其實某種程度代表了高獲利成長
df_plot["Large Growth"] = df_final["Large Growth"]

# 計算淨值
df_cum = (1 + df_plot).cumprod() * initial_capital

fig = px.line(df_cum, log_y=True, title=f"資產淨值走勢 ({start_year}-Present)")
st.plotly_chart(fig, use_container_width=True)

# 3. Fama-French 5 因子溢酬檢視
st.divider()
st.subheader("📐 Fama-French 5 因子溢酬 (Factor Premia)")
st.caption("這裡展示 5 個因子的純溢酬 (Long - Short) 累積表現。向上代表該因子有效。")

# 計算累積溢酬
cum_factors = (1 + df_factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]).cumprod()

col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown("**傳統三因子 (Market, Size, Value)**")
    fig_3f = px.line(cum_factors[["Mkt-RF", "SMB", "HML"]], log_y=True)
    st.plotly_chart(fig_3f, use_container_width=True)

with col_f2:
    st.markdown("**新五因子 (Profitability, Investment)**")
    st.write("* **RMW (Profitability)**: 高獲利 vs 低獲利 (Quality)")
    st.write("* **CMA (Investment)**: 保守投資 vs 積極擴張")
    fig_5f = px.line(cum_factors[["RMW", "CMA"]], log_y=True)
    st.plotly_chart(fig_5f, use_container_width=True)

# 4. 詳細數據
with st.expander("📋 查看詳細年化數據"):
    st.dataframe(df_metrics.style.format("{:.2%}"), use_container_width=True)
