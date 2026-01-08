import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import plotly.express as px
import numpy as np
from datetime import datetime

# --- 頁面設定 (Dashboard 模式) ---
st.set_page_config(
    page_title="Fama-French 因子戰情室",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定義 CSS (UI 美化) ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 核心邏輯 ---

def generate_dummy_data():
    """生成高擬真模擬數據"""
    dates = pd.date_range(start="1927-01-01", end=datetime.today(), freq="M")
    n = len(dates)
    
    # 模擬 25 Portfolios
    cols_25 = [
        "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM",
        "ME2 LoBM", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 HiBM",
        "ME3 LoBM", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 HiBM",
        "ME4 LoBM", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 HiBM",
        "BIG LoBM", "BIG BM2", "BIG BM3", "BIG BM4", "BIG HiBM"
    ]
    data_25 = np.random.normal(0.008, 0.05, size=(n, 25)) 
    # 調整 Small Value 讓它表現好一點
    data_25[:, 4] = data_25[:, 4] + 0.002 
    df_25 = pd.DataFrame(data_25, index=dates, columns=cols_25)

    # 模擬 Momentum
    cols_mom = ["Lo PRIOR", "Prior 2", "Prior 3", "Prior 4", "Prior 5", 
                "Prior 6", "Prior 7", "Prior 8", "Prior 9", "Hi PRIOR"]
    data_mom = np.random.normal(0.009, 0.06, size=(n, 10))
    df_mom = pd.DataFrame(data_mom, index=dates, columns=cols_mom)

    # 模擬 5 Factors
    cols_ff = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    data_ff = np.random.normal(0.005, 0.03, size=(n, 6))
    data_ff[:, 5] = np.abs(np.random.normal(0.002, 0.0005, size=n))
    df_ff = pd.DataFrame(data_ff, index=dates, columns=cols_ff)

    return df_25, df_mom, df_ff

@st.cache_data(ttl=86400)
def get_fama_french_safe():
    base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
    }
    targets = {
        "25": "25_Portfolios_Formed_on_Size_and_Book_to_Market_CSV.zip",
        "mom": "10_Portfolios_Prior_12_2_CSV.zip",
        "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    }

    data_store = {}
    
    for key, fname in targets.items():
        try:
            r = requests.get(f"{base_url}/{fname}", headers=headers, timeout=3)
            if r.status_code != 200: return None, None, None, False
            
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
            try:
                df = pd.read_csv(z.open(csv_name), skiprows=3, index_col=0)
            except:
                df = pd.read_csv(z.open(csv_name), index_col=0)

            df = df[df.index.astype(str).str.len() == 6]
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m")
            df = df.astype(float) / 100
            data_store[key] = df
        except:
            return None, None, None, False

    return data_store.get("25"), data_store.get("mom"), data_store.get("ff5"), True

# --- 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 策略參數")
    start_year = st.slider("📅 回測起始年份", 1930, 2023, 2000)
    initial_capital = st.number_input("💰 初始本金 ($)", value=10000, step=1000)
    
    st.markdown("### 📊 資料源狀態")
    status_box = st.empty()

# --- 資料載入 ---
with st.spinner('🚀 系統初始化中...'):
    df_25, df_mom, df_ff5, is_real = get_fama_french_safe()

if not is_real:
    df_25, df_mom, df_ff5 = generate_dummy_data()
    status_box.warning("模擬數據模式")
    st.warning("⚠️ **網路連線限制提示**：已切換至「演示模式」。當前數據為演算法生成。")
else:
    status_box.success("真實數據連線")
    st.success("✅ **連線成功**：成功獲取 Kenneth R. French 原始數據庫。")

# --- 數據處理 ---
try:
    mask = df_25.index.year >= start_year
    df_25 = df_25[mask]
    df_mom = df_mom[mask]
    df_ff5 = df_ff5[mask]

    df_25.columns = [c.strip() for c in df_25.columns]
    df_mom.columns = [c.strip() for c in df_mom.columns]
    df_ff5.columns = [c.strip() for c in df_ff5.columns]

    df_final = pd.DataFrame(index=df_25.index)
    
    style_map = {
        "Large Growth": ["BIG LoBM", "BIG Lo"], 
        "Large Blend": ["BIG BM2", "BIG 2", "BIG 3"],
        "Large Value": ["BIG HiBM", "BIG Hi"],
        "Mid Growth": ["ME3 LoBM", "ME3 Lo"], 
        "Mid Blend": ["ME3 BM3", "ME3 3"], 
        "Mid Value": ["ME3 HiBM", "ME3 Hi"],
        "Small Growth": ["SMALL LoBM", "SMALL Lo"], 
        "Small Blend": ["SMALL BM3", "SMALL 3"], 
        "Small Value": ["SMALL HiBM", "SMALL Hi"]
    }

    for ui_name, possible_names in style_map.items():
        found = False
        for pname in possible_names:
            if pname in df_25.columns:
                df_final[ui_name] = df_25[pname]
                found = True
                break
        if not found:
             df_final[ui_name] = df_25.iloc[:, 0]

    mom_col = "Hi PRIOR" if "Hi PRIOR" in df_mom.columns else df_mom.columns[-1]
    df_final["Momentum"] = df_mom[mom_col]
    
    mkt_col = "Mkt-RF" if "Mkt-RF" in df_ff5.columns else df_ff5.columns[0]
    rf_col = "RF" if "RF" in df_ff5.columns else df_ff5.columns[-1]
    df_final["Market"] = df_ff5[mkt_col] + df_ff5[rf_col]

    metrics = []
    for col in df_final.columns:
        s = df_final[col]
        tot_ret = (1 + s).prod()
        ann_ret = (tot_ret ** (12/len(s))) - 1
        ann_vol = s.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd = (s + 1).cumprod().div((s + 1).cumprod().cummax()).sub(1).min()
        
        metrics.append({
            "Asset": col, "CAGR": ann_ret, "Vol": ann_vol, 
            "Sharpe": sharpe, "MaxDD": max_dd
        })
    df_metrics = pd.DataFrame(metrics).set_index("Asset")
    mkt_cagr = df_metrics.loc["Market", "CAGR"]

    # --- UI 主體 ---
    
    st.markdown(f"### 📈 市場回測分析報告 ({start_year} - Present)")
    
    tab1, tab2, tab3 = st.tabs(["🧩 風格九宮格", "🚀 淨值與因子走勢", "📋 詳細統計數據"])

    # === Tab 1 ===
    with tab1:
        st.markdown("#### 美股風格績效矩陣 (Size vs. Value)")
        rows = ["Large", "Mid", "Small"]
        cols = ["Value", "Blend", "Growth"]
        
        for r in rows:
            c1, c2, c3 = st.columns(3)
            col_order = [c1, c2, c3]
            types = ["Value", "Blend", "Growth"]
            
            for idx, t in enumerate(types):
                name = f"{r} {t}"
                if name in df_metrics.index:
                    d = df_metrics.loc[name]
                    is_outperform = d["CAGR"] > mkt_cagr
                    icon = "🔥" if is_outperform else "❄️"
                    delta_color = "normal" if is_outperform else "off"
                    
                    with col_order[idx]:
                        st.metric(
                            label=name,
                            value=f"{d['CAGR']:.1%}",
                            delta=f"Sharpe: {d['Sharpe']:.2f} {icon}",
                            delta_color=delta_color
                        )

    # === Tab 2 ===
    with tab2:
        col_charts_1, col_charts_2 = st.columns([2, 1])
        with col_charts_1:
            st.markdown("#### 💰 財富累積曲線 (Log Scale)")
            plot_assets = ["Small Value", "Momentum", "Large Growth", "Market"]
            valid_plot = [x for x in plot_assets if x in df_final.columns]
            df_cum = (1 + df_final[valid_plot]).cumprod() * initial_capital
            fig = px.line(df_cum, log_y=True, color_discrete_sequence=px.colors.qualitative.G10)
            fig.update_layout(xaxis_title="", yaxis_title="資產淨值", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_charts_2:
            st.markdown("#### 📐 因子表現")
            factors = ["SMB", "HML", "RMW", "CMA"]
            valid_factors = [x for x in factors if x in df_ff5.columns]
            if valid_factors:
                df_fac_cum = (1 + df_ff5[valid_factors]).cumprod()
                fig2 = px.line(df_fac_cum, log_y=True)
                fig2.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2), height=400)
                st.plotly_chart(fig2, use_container_width=True)

    # === Tab 3 (修復崩潰點) ===
    with tab3:
        st.markdown("#### 📊 各類資產風險報酬統計表")
        
        display_df = df_metrics.copy()
        
        # 這裡加上 Try-Except，如果 matplotlib 沒裝好，就顯示普通表格，不要報錯
        try:
            import matplotlib
            st.dataframe(
                display_df.style.format({
                    "CAGR": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}"
                }).background_gradient(subset=["CAGR", "Sharpe"], cmap="Greens")
                  .background_gradient(subset=["MaxDD"], cmap="Reds_r"),
                use_container_width=True, 
                height=400
            )
        except ImportError:
            # 降級處理：只顯示格式化後的表格，不顯示顏色
            st.warning("⚠️ 系統檢測到缺少 matplotlib 繪圖庫，表格將以純文字顯示。")
            st.dataframe(display_df, use_container_width=True, height=400)

except Exception as e:
    st.error("系統運算錯誤，請刷新頁面。")
    st.exception(e)
