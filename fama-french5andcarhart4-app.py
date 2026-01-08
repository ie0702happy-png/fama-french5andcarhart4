import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Fama-French 因子回測神器", layout="wide")

# --- 2. 核心讀檔函數 (自動跳過開頭的說明文字) ---
@st.cache_data
def load_ff_csv(filepath, keywords):
    """
    讀取 Fama-French CSV，自動偵測表頭位置
    keywords: 用來辨識表頭的關鍵字列表
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        header_row = None
        for i, line in enumerate(lines):
            # 只要該行包含關鍵字且有逗號，就認定是表頭
            if any(k in line for k in keywords) and "," in line:
                header_row = i
                break
        
        if header_row is None:
            return None

        # 讀取資料
        df = pd.read_csv(filepath, skiprows=header_row, index_col=0)
        
        # 清洗資料
        df = df[df.index.astype(str).str.len() == 6] # 只留 YYYYMM 格式的行
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m") # 轉成日期物件
        df = df.apply(pd.to_numeric, errors='coerce') # 轉成數字
        df = df / 100.0 # 原始數據是百分比(5.0)，轉成小數(0.05)
        df.columns = [c.strip() for c in df.columns] # 去除欄位空白
        return df
    except Exception as e:
        st.error(f"讀取 {filepath} 失敗: {e}")
        return None

# --- 3. 主程式 ---
st.title("🚀 Fama-French 因子投資回測系統")

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 回測設定")
    start_year = st.slider("開始年份", 1963, 2024, 1990)
    initial_money = st.number_input("初始本金 (USD)", value=10000, step=1000)
    st.info("請確保 csv 檔案與程式在同一目錄")

# 定義檔名 (對應你下載的檔案)
file_25 = "25_Portfolios_5x5.csv"
file_mom = "F-F_Momentum_Factor.csv"
file_ff5 = "F-F_Research_Data_5_Factors_2x3.csv"

# 載入數據
df_25 = load_ff_csv(file_25, ["SMALL LoBM", "BIG HiBM"])
df_mom = load_ff_csv(file_mom, ["Mom"])
df_ff5 = load_ff_csv(file_ff5, ["Mkt-RF", "RF"])

# 檢查檔案是否都讀到了
if df_25 is None or df_ff5 is None:
    st.error("❌ 找不到檔案！請確認目錄下有 `25_Portfolios_5x5.csv` 和 `F-F_Research_Data_5_Factors_2x3.csv`")
    st.stop()

# --- 4. 數據整理 ---
# 找出共同時間段
common_idx = df_25.index.intersection(df_ff5.index)
if df_mom is not None:
    common_idx = common_idx.intersection(df_mom.index)

# 篩選年份
common_idx = common_idx[common_idx.year >= start_year]

# 建立總表
data = pd.DataFrame(index=common_idx)

# (1) 定義主要策略 (從 25 Portfolios 挑選)
# 對照表: 
# SMALL LoBM = 小盤成長 (Small Growth)
# SMALL HiBM = 小盤價值 (Small Value)
# BIG LoBM   = 大盤成長 (Large Growth)
# BIG HiBM   = 大盤價值 (Large Value)
data["Small Value"] = df_25.loc[common_idx, "SMALL HiBM"]
data["Small Growth"] = df_25.loc[common_idx, "SMALL LoBM"]
data["Large Value"] = df_25.loc[common_idx, "BIG HiBM"]
data["Large Growth"] = df_25.loc[common_idx, "BIG LoBM"]

# (2) 加入大盤 (Mkt = Mkt-RF + RF)
data["Market (S&P500)"] = df_ff5.loc[common_idx, "Mkt-RF"] + df_ff5.loc[common_idx, "RF"]

# (3) 加入動能 (如果有)
if df_mom is not None:
    mom_col = "Mom" if "Mom" in df_mom.columns else df_mom.columns[0]
    data["Momentum"] = df_mom.loc[common_idx, mom_col]

# --- 5. 計算績效 ---
# 財富曲線 (累計報酬)
wealth = (1 + data).cumprod() * initial_money

# 績效指標表
metrics = []
for col in data.columns:
    # CAGR
    total_ret = (1 + data[col]).prod()
    years = len(data) / 12
    cagr = (total_ret ** (1/years)) - 1
    # Volatility
    vol = data[col].std() * np.sqrt(12)
    # Sharpe (假設無風險利率簡化為0或內含)
    sharpe = cagr / vol if vol > 0 else 0
    # Max Drawdown
    cum_ret = (1 + data[col]).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()
    
    metrics.append({
        "策略": col,
        "年化報酬 (CAGR)": f"{cagr:.2%}",
        "波動率 (Vol)": f"{vol:.2%}",
        "夏普值 (Sharpe)": f"{sharpe:.2f}",
        "最大回撤 (MaxDD)": f"{max_dd:.2%}"
    })

df_metrics = pd.DataFrame(metrics).set_index("策略")

# --- 6. 視覺化儀表板 ---
tab1, tab2, tab3 = st.tabs(["📈 財富曲線", "📊 績效指標", "🔥 風格九宮格"])

with tab1:
    st.subheader(f"💰 {initial_money:,} 美元投入後的資產變化")
    fig = px.line(wealth, log_y=True, title="資產成長 (對數座標)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📋 詳細風險報酬表")
    st.dataframe(df_metrics.style.highlight_max(axis=0, color='darkgreen'), use_container_width=True)

with tab3:
    st.subheader("🇺🇸 美股風格績效矩陣 (Size vs Value)")
    # 這裡我們手動抓 25 Portfolios 的 9 個代表點來畫九宮格
    # 矩陣: 3x3
    # Rows: Large(Big), Mid(ME3), Small
    # Cols: Value(HiBM), Blend(BM3), Growth(LoBM)
    
    # 準備九宮格資料
    matrix_data = {
        "Small Value": df_25.loc[common_idx, "SMALL HiBM"].mean() * 12,
        "Small Blend": df_25.loc[common_idx, "SMALL BM3"].mean() * 12,
        "Small Growth": df_25.loc[common_idx, "SMALL LoBM"].mean() * 12,
        
        "Mid Value": df_25.loc[common_idx, "ME3 HiBM"].mean() * 12,
        "Mid Blend": df_25.loc[common_idx, "ME3 BM3"].mean() * 12,
        "Mid Growth": df_25.loc[common_idx, "ME3 LoBM"].mean() * 12,
        
        "Large Value": df_25.loc[common_idx, "BIG HiBM"].mean() * 12,
        "Large Blend": df_25.loc[common_idx, "BIG BM3"].mean() * 12,
        "Large Growth": df_25.loc[common_idx, "BIG LoBM"].mean() * 12,
    }
    
    col1, col2, col3 = st.columns(3)
    
    def box(title, val, benchmark):
        delta = val - benchmark
        color = "green" if delta > 0 else "red"
        return f"""
        <div style="background-color: #262730; padding: 20px; border-radius: 10px; margin: 5px; text-align: center; border: 1px solid #4F4F4F;">
            <h4 style="margin:0; color: #FAFAFA;">{title}</h4>
            <h2 style="margin:10px 0; color: #FFF;">{val:.1%}</h2>
            <p style="margin:0; color: {color}; font-size: 0.9em;">vs Mkt {delta:+.1%}</p>
        </div>
        """

    mkt_ret = data["Market (S&P500)"].mean() * 12
    
    with col1:
        st.markdown("**Value (價值)**")
        st.markdown(box("Large Value", matrix_data["Large Value"], mkt_ret), unsafe_allow_html=True)
        st.markdown(box("Mid Value", matrix_data["Mid Value"], mkt_ret), unsafe_allow_html=True)
        st.markdown(box("Small Value", matrix_data["Small Value"], mkt_ret), unsafe_allow_html=True)
        
    with col2:
        st.markdown("**Blend (混合)**")
        st.markdown(box("Large Blend", matrix_data["Large Blend"], mkt_ret), unsafe_allow_html=True)
        st.markdown(box("Mid Blend", matrix_data["Mid Blend"], mkt_ret), unsafe_allow_html=True)
        st.markdown(box("Small Blend", matrix_data["Small Blend"], mkt_ret), unsafe_allow_html=True)
        
    with col3:
        st.markdown("**Growth (成長)**")
        st.markdown(box("Large Growth", matrix_data["Large Growth"], mkt_ret), unsafe_allow_html=True)
        st.markdown(box("Mid Growth", matrix_data["Mid Growth"], mkt_ret), unsafe_allow_html=True)
        st.markdown(box("Small Growth", matrix_data["Small Growth"], mkt_ret), unsafe_allow_html=True)
