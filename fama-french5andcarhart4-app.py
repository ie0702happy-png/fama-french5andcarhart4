import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. 頁面配置 ---
st.set_page_config(page_title="九大風格與動能全歷史分析", layout="wide", page_icon="📈")

# --- 2. 智慧讀檔函數 ---
@st.cache_data
def load_data():
    files = {
        "25_Portfolios": "25_Portfolios_5x5.csv",
        "Momentum": "F-F_Momentum_Factor.csv",
        "Factors": "F-F_Research_Data_5_Factors_2x3.csv"
    }
    
    data_dict = {}
    
    for key, filename in files.items():
        try:
            # 預讀檔案尋找表頭
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            skip_rows = 0
            for i, line in enumerate(lines):
                # 簡單判斷：包含逗號且有年份特徵或特定關鍵字
                if "," in line and ("LoBM" in line or "Mom" in line or "Mkt-RF" in line):
                    skip_rows = i
                    break
            
            # 讀取數據
            df = pd.read_csv(filename, skiprows=skip_rows, index_col=0)
            
            # 清洗索引 (保留 YYYYMM)
            df = df[df.index.astype(str).str.len() == 6]
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m")
            
            # 轉數值並處理百分比
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df / 100.0
            
            # 去除欄位空白
            df.columns = [c.strip() for c in df.columns]
            
            data_dict[key] = df
            
        except Exception as e:
            st.error(f"讀取 {filename} 失敗: {str(e)}")
            return None
            
    return data_dict

# --- 3. 數據處理與合成 ---
data_raw = load_data()

if data_raw:
    df_25 = data_raw["25_Portfolios"]
    df_mom = data_raw["Momentum"]
    df_ff = data_raw["Factors"]
    
    # 取時間交集 (受限於 5-Factor 資料起始點，通常為 1963)
    common_index = df_25.index.intersection(df_ff.index).intersection(df_mom.index)
    
    # 建立分析用 DataFrame
    df_analysis = pd.DataFrame(index=common_index)
    
    # --- 建構九宮格 (Nine-Box) ---
    # 對應邏輯：
    # Large (Big): Size 5
    # Mid: Size 3
    # Small: Size 1
    # Value: HiBM (BM 5)
    # Blend: BM 3
    # Growth: LoBM (BM 1)
    
    # Large Cap Row
    df_analysis["Large Value"] = df_25.loc[common_index, "BIG HiBM"]
    df_analysis["Large Blend"] = df_25.loc[common_index, "ME5 BM3"]
    df_analysis["Large Growth"] = df_25.loc[common_index, "BIG LoBM"]
    
    # Mid Cap Row
    df_analysis["Mid Value"]   = df_25.loc[common_index, "ME3 BM5"]
    df_analysis["Mid Blend"]   = df_25.loc[common_index, "ME3 BM3"]
    df_analysis["Mid Growth"]  = df_25.loc[common_index, "ME3 LoBM"]
    
    # Small Cap Row
    df_analysis["Small Value"] = df_25.loc[common_index, "SMALL HiBM"]
    df_analysis["Small Blend"] = df_25.loc[common_index, "ME1 BM3"]
    df_analysis["Small Growth"]= df_25.loc[common_index, "SMALL LoBM"]
    
    # --- 加入動能與大盤 ---
    # Momentum 因子通常是多空對沖 (Winners - Losers)，這裡直接呈現因子回報
    df_analysis["Momentum"] = df_mom.loc[common_index, "Mom"]
    
    # Market (Mkt-RF + RF) 還原市場總報酬
    df_analysis["Market"] = df_ff.loc[common_index, "Mkt-RF"] + df_ff.loc[common_index, "RF"]
    
    # 無風險利率 (算 Sharpe 用)
    rf = df_ff.loc[common_index, "RF"]

    # --- 4. 主介面 ---
    st.title("🏛️ Fama-French 九大風格與動能全歷史回測")
    st.markdown(f"**數據區間：** {common_index.min().strftime('%Y-%m')} 至 {common_index.max().strftime('%Y-%m')} (共 {len(common_index)/12:.1f} 年)")
    
    # 設定面板
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ 參數設定")
        initial_investment = st.number_input("初始本金 ($)", value=10000, step=1000)
        
        # 選擇要顯示的線圖
        all_strategies = df_analysis.columns.tolist()
        selected_strategies = st.multiselect(
            "選擇比較策略", 
            all_strategies, 
            default=["Small Value", "Large Growth", "Market", "Momentum", "Small Growth"]
        )
    
    # 計算累積報酬
    df_wealth = (1 + df_analysis).cumprod() * initial_investment
    
    with col2:
        st.subheader("📈 財富累積曲線 (對數座標)")
        fig = px.line(df_wealth[selected_strategies], log_y=True)
        fig.update_layout(height=500, xaxis_title="年份", yaxis_title="資產價值 (USD)")
        st.plotly_chart(fig, use_container_width=True)

    # --- 5. 績效統計表 ---
    st.markdown("---")
    st.subheader("📊 歷史績效詳細數據")
    
    metrics_list = []
    for col in df_analysis.columns:
        # 年化報酬
        total_ret = (1 + df_analysis[col]).prod()
        years = len(df_analysis) / 12
        cagr = (total_ret ** (1/years)) - 1
        
        # 波動率 (年化)
        vol = df_analysis[col].std() * np.sqrt(12)
        
        # 夏普值 (Excess Return / Vol)
        excess_ret = df_analysis[col] - rf
        sharpe = (excess_ret.mean() * 12) / vol
        
        # 最大回撤
        cum_ret = (1 + df_analysis[col]).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()
        
        metrics_list.append({
            "策略": col,
            "年化報酬 (CAGR)": cagr,
            "波動率 (Vol)": vol,
            "夏普值 (Sharpe)": sharpe,
            "最大回撤 (MaxDD)": max_dd
        })
    
    df_metrics = pd.DataFrame(metrics_list).set_index("策略")
    # 格式化顯示
    st.dataframe(
        df_metrics.style.format({
            "年化報酬 (CAGR)": "{:.2%}",
            "波動率 (Vol)": "{:.2%}",
            "夏普值 (Sharpe)": "{:.2f}",
            "最大回撤 (MaxDD)": "{:.2%}"
        }).background_gradient(subset=["年化報酬 (CAGR)"], cmap="Greens"),
        use_container_width=True
    )

    # --- 6. 風格九宮格視覺化 ---
    st.markdown("---")
    st.subheader("🇺🇸 風格九宮格 (Nine-Box) 年化報酬熱力圖")
    
    # 準備 3x3 矩陣數據
    box_data = np.array([
        [df_metrics.loc["Large Value", "年化報酬 (CAGR)"], df_metrics.loc["Large Blend", "年化報酬 (CAGR)"], df_metrics.loc["Large Growth", "年化報酬 (CAGR)"]],
        [df_metrics.loc["Mid Value", "年化報酬 (CAGR)"],   df_metrics.loc["Mid Blend", "年化報酬 (CAGR)"],   df_metrics.loc["Mid Growth", "年化報酬 (CAGR)"]],
        [df_metrics.loc["Small Value", "年化報酬 (CAGR)"], df_metrics.loc["Small Blend", "年化報酬 (CAGR)"], df_metrics.loc["Small Growth", "年化報酬 (CAGR)"]]
    ])
    
    box_text = np.array([
        [f"Large Value\n{box_data[0,0]:.2%}", f"Large Blend\n{box_data[0,1]:.2%}", f"Large Growth\n{box_data[0,2]:.2%}"],
        [f"Mid Value\n{box_data[1,0]:.2%}",   f"Mid Blend\n{box_data[1,1]:.2%}",   f"Mid Growth\n{box_data[1,2]:.2%}"],
        [f"Small Value\n{box_data[2,0]:.2%}", f"Small Blend\n{box_data[2,1]:.2%}", f"Small Growth\n{box_data[2,2]:.2%}"]
    ])
    
    fig_box = go.Figure(data=go.Heatmap(
        z=box_data,
        x=["Value", "Blend", "Growth"],
        y=["Large", "Mid", "Small"],
        text=box_text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        showscale=False
    ))
    
    fig_box.update_layout(
        height=500,
        width=600,
        title_text="風格箱年化報酬 (CAGR)",
        xaxis_side="top"
    )
    
    col_box1, col_box2 = st.columns([1,1])
    with col_box1:
        st.plotly_chart(fig_box, use_container_width=True)
    with col_box2:
        st.info("""
        **九宮格解讀：**
        * **左下角 (Small Value)**：歷史上報酬最高的區域。
        * **右下角 (Small Growth)**：歷史上表現最差的區域 (顏色最紅)。
        * **左側 (Value Column)**：整體表現通常優於右側 (Growth Column)。
        """)

else:
    st.warning("請確認目錄下是否有 `25_Portfolios_5x5.csv`, `F-F_Momentum_Factor.csv`, `F-F_Research_Data_5_Factors_2x3.csv` 檔案。")
