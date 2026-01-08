import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 頁面設定 ---
st.set_page_config(page_title="九大風格因子與動能儀表板", layout="wide")
st.title("📊 投資風格九宮格 & 動能因子儀表板")
st.caption("基於 Fama-French 五因子與 Carhart 四因子模型概念 | 數據來源: Vanguard & iShares ETFs")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    period = st.selectbox("回測時間範圍", ["5y", "10y", "max"], index=2, help="選擇 max 以取得最久遠資料 (約 2004 年起)")
    initial_capital = st.number_input("初始本金 (USD)", value=10000, step=1000)
    
    st.divider()
    st.info("ℹ️ **關於模型**\n\n此 App 使用具代表性的 ETF 作為因子代理：\n\n* **SMB (規模)**: Small vs Large\n* **HML (價值)**: Value vs Growth\n* **MOM (動能)**: Momentum Strategy")

# --- 1. 定義九宮格與動能代號 (使用歷史最悠久的 Vanguard 系列) ---
# Vanguard 的風格 ETF 大多成立於 2004/01，是目前美股最標準的風格歷史數據
tickers_map = {
    # --- 大型股 (Large Cap) ---
    "Large Growth (大型成長)": "VUG",
    "Large Blend (大型混合)": "VOO", # 使用 VOO 代表大盤/混合
    "Large Value (大型價值)": "VTV",
    
    # --- 中型股 (Mid Cap) ---
    "Mid Growth (中型成長)": "VOT",
    "Mid Blend (中型混合)": "VO",
    "Mid Value (中型價值)": "VOE",
    
    # --- 小型股 (Small Cap) ---
    "Small Growth (小型成長)": "VBK",
    "Small Blend (小型混合)": "VB",
    "Small Value (小型價值)": "VBR",
    
    # --- 動能 (Momentum) - Carhart 4因子 ---
    # MTUM 成立於 2013，PDP 成立於 2007。為了數據長度，我們這裡加入 PDP 作參考，但主要顯示 MTUM
    "Momentum (動能)": "MTUM" 
}

# 反向查詢表
code_to_name = {v: k for k, v in tickers_map.items()}
all_tickers = list(tickers_map.values())

# --- 2. 數據下載與處理 ---
@st.cache_data(ttl=3600)
def get_data(period_str):
    try:
        # 下載數據
        df = yf.download(all_tickers, period=period_str, progress=False)['Adj Close']
        
        # 簡單清理
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
            
        return df
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

df_raw = get_data(period)

if df_raw.empty:
    st.stop()

# --- 3. 計算邏輯 ---
# 找出所有 ETF 共同存在的起始日 (因為 MTUM 比較年輕，若選 max，九宮格會從 2004 開始，但 MTUM 會從 2013 加入)
# 為了公平比較九宮格，我們先計算九宮格的起點
nine_box_tickers = [t for t in all_tickers if t != "MTUM"]
df_9box = df_raw[nine_box_tickers].dropna()
start_date_9box = df_9box.index[0]

# 動能數據單獨處理
df_mom = df_raw[["MTUM"]].dropna()

# 計算績效指標函數
def calculate_metrics(series):
    if series.empty: return 0, 0, 0
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    
    # 年化報酬 (CAGR)
    days = (series.index[-1] - series.index[0]).days
    cagr = (1 + total_ret) ** (365.25 / days) - 1
    
    # 波動率
    daily_ret = series.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    
    # 夏普 (假設無風險利率 0)
    sharpe = cagr / vol if vol != 0 else 0
    
    return cagr, vol, sharpe

# 預先計算所有指標
metrics = {}
for ticker in all_tickers:
    # 針對每個標的，取其有效數據區間
    series = df_raw[ticker].dropna()
    metrics[ticker] = calculate_metrics(series)

# --- 4. 介面佈局：九宮格熱圖 (The 9-Box Grid) ---
st.subheader("🏁 投資風格九宮格 (The Style Box)")
st.caption(f"數據起點: {start_date_9box.date()} (Vanguard 系列 ETF) | 顯示數據：年化報酬率 (CAGR)")

# 使用 Streamlit 的 Columns 模擬九宮格佈局
# CSS 樣式注入，讓它看起來更像一個風格箱
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    border: 1px solid #d1d5db;
}
div[data-testid="stMetric"]:hover {
    background-color: #e0e7ff;
    border-color: #6366f1;
}
</style>
""", unsafe_allow_html=True)

# 定義九宮格的數據
rows = ["Large", "Mid", "Small"]
cols = ["Value", "Blend", "Growth"]

# 建立 Grid
c1, c2, c3 = st.columns(3)
columns_ui = [c1, c2, c3]

# 繪製矩陣
for i, size in enumerate(rows):
    cols_ui = st.columns(3)
    for j, style in enumerate(cols):
        key = f"{size} {style}" # 產生類似 "Large Value" 的 key
        # 轉換成中文 Key
        full_key_zh = [k for k in tickers_map.keys() if key in k][0]
        ticker = tickers_map[full_key_zh]
        
        cagr, vol, sharpe = metrics[ticker]
        
        with cols_ui[j]:
            # 根據報酬率給予簡單的顏色標記 (視覺輔助)
            color_emoji = "🔥" if cagr > 0.10 else "😐" if cagr > 0.07 else "💧"
            st.metric(
                label=f"{size} {style} ({ticker})",
                value=f"{cagr:.2%}",
                delta=f"Sharpe: {sharpe:.2f}",
                help=f"年化波動率: {vol:.2%}"
            )

st.divider()

# --- 5. 動能因子 vs 大盤 ---
st.subheader("🚀 Carhart 動能因子 (Momentum) 挑戰賽")
col_mom1, col_mom2 = st.columns([1, 3])

with col_mom1:
    mtum_ticker = tickers_map["Momentum (動能)"]
    m_cagr, m_vol, m_sharpe = metrics[mtum_ticker]
    st.metric(
        label="Momentum (MTUM)",
        value=f"{m_cagr:.2%}",
        delta=f"Sharpe: {m_sharpe:.2f}",
        help="注意：MTUM 成立時間較短 (2013起)，CAGR 僅反映該段區間。"
    )
    st.caption("動能因子通常在趨勢明確時表現優異，但在震盪市或反轉時會有較大回撤。")

with col_mom2:
    # 繪製標準化比較圖 (以 MTUM 成立日為基準)
    df_compare = df_raw[["MTUM", "VOO", "VBR", "VUG"]].dropna()
    if not df_compare.empty:
        # 歸一化
        df_norm = df_compare / df_compare.iloc[0] * initial_capital
        st.line_chart(df_norm)
    else:
        st.warning("動能數據不足，無法繪製比較圖。")

# --- 6. Fama-French 因子溢酬分析 ---
st.divider()
st.subheader("📐 Fama-French 因子溢酬視覺化")
st.caption("透過 ETF 績效差值，觀察市場偏好 (Factor Premia)。")

col_ff1, col_ff2 = st.columns(2)

# 準備數據 (確保日期對齊)
df_ff = df_9box.pct_change().dropna()

# 計算累積報酬
cum_ret = (1 + df_ff).cumprod()

with col_ff1:
    st.markdown("#### 1️⃣ 規模因子 (SMB: Small Minus Big)")
    st.write("理論：長期而言，小型股應有高於大型股的溢酬。")
    # SMB Proxy: Small Blend (VB) - Large Blend (VOO)
    smb_series = cum_ret["VB"] / cum_ret["VOO"]
    
    fig_smb = px.line(smb_series, title="小型股相對大型股強弱 (VB / VOO)")
    fig_smb.add_hline(y=1, line_dash="dash", line_color="gray")
    fig_smb.update_layout(yaxis_title="相對強度 (數值上升代表小型股強)")
    st.plotly_chart(fig_smb, use_container_width=True)

with col_ff2:
    st.markdown("#### 2️⃣ 價值因子 (HML: High Minus Low)")
    st.write("理論：長期而言，價值股 (低 P/B) 應有高於成長股的溢酬。")
    # HML Proxy: Large Value (VTV) - Large Growth (VUG)
    # 這裡我們用比較純粹的 Value ETF vs Growth ETF
    hml_series = cum_ret["VTV"] / cum_ret["VUG"]
    
    fig_hml = px.line(hml_series, title="價值股相對成長股強弱 (VTV / VUG)")
    fig_hml.add_hline(y=1, line_dash="dash", line_color="gray")
    fig_hml.update_layout(yaxis_title="相對強度 (數值上升代表價值股強)")
    st.plotly_chart(fig_hml, use_container_width=True)

# --- 7. 詳細數據表 ---
with st.expander("📋 查看完整詳細數據"):
    # 製作一個 Summary Table
    summary_data = []
    for name, ticker in tickers_map.items():
        c, v, s = metrics[ticker]
        summary_data.append({
            "風格/因子": name,
            "代號": ticker,
            "年化報酬 (CAGR)": f"{c:.2%}",
            "波動率 (Vol)": f"{v:.2%}",
            "夏普值 (Sharpe)": f"{s:.2f}"
        })
    st.dataframe(pd.DataFrame(summary_data))
