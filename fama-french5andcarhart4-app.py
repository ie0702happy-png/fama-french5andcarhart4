import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# 設定繪圖風格
plt.style.use('dark_background')

def load_smart_csv_content(file_path, keywords):
    """
    讀取上傳的檔案，自動尋找表頭
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        start_row = 0
        found = False
        for i, line in enumerate(lines):
            if any(k in line for k in keywords) and "," in line:
                start_row = i
                found = True
                break
        
        if not found: return None
        
        # 讀取數據
        df = pd.read_csv(file_path, skiprows=start_row, index_col=0)
        
        # 清洗數據
        # 1. 確保 Index 是字串且長度為 6 (YYYYMM)
        df = df[df.index.astype(str).str.len() == 6]
        # 2. 轉日期
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m")
        # 3. 轉數值 (除以100)
        df = df.apply(pd.to_numeric, errors='coerce') / 100
        # 4. 去除欄位空白
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        return None

# 1. 讀取數據
df_25 = load_smart_csv_content("25_Portfolios_5x5.csv", ["SMALL LoBM", "BIG HiBM"])
df_mom = load_smart_csv_content("F-F_Momentum_Factor.csv", ["Mom"])
df_ff5 = load_smart_csv_content("F-F_Research_Data_5_Factors_2x3.csv", ["Mkt-RF", "RF"])

# 2. 數據合併
# 時間對齊 (取交集)
common_index = df_25.index.intersection(df_ff5.index)
if df_mom is not None:
    common_index = common_index.intersection(df_mom.index)

df_final = pd.DataFrame(index=common_index)

# 映射 25 Portfolios (簡化版：取極端值做代表)
# Small Value = SMALL HiBM (小盤價值)
# Large Growth = BIG LoBM (大盤成長)
# Market = Mkt-RF + RF
df_final["Small Value"] = df_25.loc[common_index, "SMALL HiBM"]
df_final["Large Growth"] = df_25.loc[common_index, "BIG LoBM"]
df_final["Small Growth"] = df_25.loc[common_index, "SMALL LoBM"]
df_final["Large Value"] = df_25.loc[common_index, "BIG HiBM"]

# 加入動能
if df_mom is not None:
    col = "Mom" if "Mom" in df_mom.columns else df_mom.columns[0]
    df_final["Momentum"] = df_mom.loc[common_index, col]

# 加入大盤 (Mkt-RF + RF)
df_final["Market (S&P500 Proxy)"] = df_ff5.loc[common_index, "Mkt-RF"] + df_ff5.loc[common_index, "RF"]

# 3. 過濾年份 (1990 至今，比較符合現代結構)
start_date = "1990-01-01"
df_calc = df_final[df_final.index >= start_date].copy()

# 4. 計算累積報酬 (假設本金 10000)
initial_capital = 10000
df_wealth = (1 + df_calc).cumprod() * initial_capital

# 5. 計算績效指標
metrics = []
for col in df_calc.columns:
    tot_ret = (1 + df_calc[col]).prod()
    n_years = len(df_calc) / 12
    cagr = (tot_ret ** (1 / n_years)) - 1
    vol = df_calc[col].std() * np.sqrt(12)
    sharpe = cagr / vol
    
    # Max DD
    cum = (1 + df_calc[col]).cumprod()
    dd = (cum / cum.cummax()) - 1
    max_dd = dd.min()
    
    metrics.append([col, cagr, vol, sharpe, max_dd])

df_metrics = pd.DataFrame(metrics, columns=["Strategy", "CAGR", "Volatility", "Sharpe", "MaxDD"])
df_metrics.set_index("Strategy", inplace=True)

# 6. 繪圖
fig, ax = plt.subplots(figsize=(12, 6))
for col in df_wealth.columns:
    # 畫線
    ax.plot(df_wealth.index, df_wealth[col], label=col, linewidth=2)

ax.set_yscale('log') # 對數座標看長期複利
ax.set_title(f'Wealth Accumulation ($10k Initial) - Log Scale ({start_date[:4]}-Present)', fontsize=14, color='white')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
ax.legend()
ax.grid(True, which="both", ls="-", alpha=0.2)

# 輸出結果
print(df_metrics.sort_values("CAGR", ascending=False).to_markdown(floatfmt=".2%"))
plt.show()
